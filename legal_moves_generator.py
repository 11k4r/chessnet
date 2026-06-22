import json
import random
import chess
from typing import List, Dict, Tuple, Set

from prompt_builder import PromptBuilder, ELO_TIERS, CoachMode


class LegalMovesGenerator:
    """
    Generates Phase 2 training data for chess legal move evaluation.
    Builds a 5-step atomic CoT: [Intent] [Perception] [Trajectory Evaluation]
    [Resolution] [Pedagogical Plan].

    Query types
    -----------
    1. Square-Targeted Move Query   - "What are the legal moves for the piece on e4?"
    2. Piece-Targeted Move Query    - "Where can the White Bishop move?"
    3. Color-Agnostic Move Query    - "What are the moves for all the rooks?"
    4. False Premise Verification   - "What are the moves for the Black King on d2?"
    5. Blocked / Pinned Query       - "Where can the Knight on f3 go?" (0 moves)

    Design notes
    ------------
    - [Trajectory Evaluation] shows genuine reasoning rather than restating
      [Resolution]: it surfaces the pseudo-legal -> legal delta when pins
      exclude squares, and definitively distinguishes "pinned" from "blocked"
      via board.is_pinned() rather than hedging with "or".
    - Promotion-capable pawn moves are deduplicated (python-chess emits one
      Move object per promotion piece, all sharing the same to_square) and
      surfaced explicitly as a pedagogical note rather than silently dropped.
    - elo_tier and mode are sampled per record and stored alongside "fen" so
      a downstream consumer can reconstruct the matching [ENVIRONMENT] block
      via PromptBuilder.build_environment_block. Answer wording adapts at the
      beginner tier (plain-language "why"); coach mode appends <ask> when
      there is genuine choice among legal moves. <|wait|> is intentionally
      NOT used here -- each of these answers is a single self-contained idea,
      and <|wait|> is for chunking multiple sequential ideas within one turn,
      not a turn-end marker (the chat interface already provides that).
    """

    PIECE_NAMES = {
        chess.PAWN:   "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK:   "Rook",
        chess.QUEEN:  "Queen",
        chess.KING:   "King",
    }

    def __init__(self, output_file: str):
        self.output_file   = output_file
        self.system_prompt = PromptBuilder.get_system_prompt()

    # -------------------------------------------------------------------------
    # Core Helpers
    # -------------------------------------------------------------------------

    def _piece_str(self, piece_type: int, color: chess.Color) -> str:
        c_str = "White" if color == chess.WHITE else "Black"
        return f"{c_str} {self.PIECE_NAMES[piece_type]}"

    def _join_squares(self, names: list) -> str:
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    def _format_sharegpt(
        self,
        fen: str,
        question: str,
        cot: str,
        answer: str,
        elo_tier: str = "club",
        mode: str = "coach",
    ) -> Dict:
        """
        Package a sample in the ShareGPT format expected by dataset.py.
        elo_tier/mode are stored as top-level fields (alongside fen) so the
        downstream pipeline can render the matching [ENVIRONMENT] block at
        train time via PromptBuilder.build_environment_block.
        """
        return {
            "fen":      fen,
            "elo_tier": elo_tier,
            "mode":     mode,
            "messages": [
                {"role": "system",    "content": self.system_prompt},
                {"role": "user",      "content": f"Question: {question}"},
                {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n{answer}"},
            ],
        }

    def _maybe_add_ask(self, mode: str, num_options: int) -> str:
        """
        Coach mode only: invite engagement when there is genuine choice among
        legal moves. Not <|wait|> -- this is a single trailing engagement
        hook, not mid-response chunking, so it doesn't require multiple
        sequential ideas to justify its use.
        """
        if mode == CoachMode.COACH and num_options > 1:
            return " <ask>Which of these would you play?</ask>"
        return ""

    # -------------------------------------------------------------------------
    # Move computation helpers
    # -------------------------------------------------------------------------

    def _get_legal_moves(
        self, board: chess.Board, sq: chess.Square
    ) -> Tuple[List[chess.Square], Set[chess.Square]]:
        """
        Returns (legal_destination_squares, promotion_squares).

        Destinations are deduplicated -- a promoting pawn generates one Move
        object per promotion piece (queen/rook/bishop/knight), all sharing
        the same to_square. Naive collection would report e.g. 4 "different"
        legal moves to e8 when there is really one destination with four
        promotion choices.

        Temporarily swaps board.turn so pieces of either color can be
        evaluated regardless of whose actual turn it is (a coaching tool
        needs "what could this piece do" even when it isn't that side's
        move). King safety and pin filtering remain valid under this swap
        since they are geometric properties of the position, not the move
        history -- the one caveat is a stale en-passant square in rare edge
        cases, which is a known, low-impact limitation of this approach.
        """
        piece = board.piece_at(sq)
        if not piece:
            return [], set()

        original_turn = board.turn
        try:
            board.turn = piece.color
            seen: Set[chess.Square] = set()
            moves: List[chess.Square] = []
            promo_squares: Set[chess.Square] = set()

            for m in board.legal_moves:
                if m.from_square != sq:
                    continue
                if m.promotion is not None:
                    promo_squares.add(m.to_square)
                if m.to_square not in seen:
                    seen.add(m.to_square)
                    moves.append(m.to_square)
        finally:
            board.turn = original_turn

        return moves, promo_squares

    def _get_pseudo_legal_destinations(
        self, board: chess.Board, sq: chess.Square
    ) -> Set[chess.Square]:
        """Unique pseudo-legal destinations for the piece on sq (set dedupes promotions)."""
        piece = board.piece_at(sq)
        if not piece:
            return set()

        original_turn = board.turn
        try:
            board.turn = piece.color
            dests = {m.to_square for m in board.pseudo_legal_moves if m.from_square == sq}
        finally:
            board.turn = original_turn

        return dests

    def _find_pinning_piece(
        self, board: chess.Board, color: chess.Color, sq: chess.Square
    ):
        """
        For a piece on sq that is genuinely pinned (board.is_pinned), find
        the enemy slider doing the pinning by walking the pin ray returned
        by board.pin(). Returns (piece, square) or None if it can't be
        identified (board.is_pinned already guarantees sq is pinned, but
        this lets us ground the claim in a named, located piece rather than
        the abstract assertion "would expose the King").
        """
        pin_ray = board.pin(color, sq)
        for s in pin_ray:
            p = board.piece_at(s)
            if p and p.color != color and p.piece_type in (chess.ROOK, chess.BISHOP, chess.QUEEN):
                return p, s
        return None

    def _king_safety_detail(
        self, board: chess.Board, king_color: chess.Color, excluded: List[chess.Square]
    ) -> List[str]:
        """
        For squares excluded from a King's legal moves, name the specific
        enemy piece controlling each one. A King is never "pinned" -- pins
        only restrict non-King pieces shielding their own King from a
        slider. A King simply cannot move into a square any enemy piece
        attacks, regardless of piece type (slider, knight, pawn, or king).

        board.attackers() can return empty even for a genuinely-excluded
        square: if the King is retreating directly along a slider's attack
        ray (e.g. King on e2 in check from a Queen on e8, trying to retreat
        to e1), the King's own current position blocks attackers()'s ray
        check, since the King hasn't actually moved yet when this is
        evaluated. Confirmed empirically -- attackers(e1) returns empty even
        though e1 is correctly excluded as still-in-check. The fallback
        below describes the rule rather than asserting a specific square is
        "controlled," which would be false in this case.
        """
        parts = []
        for ex_sq in excluded:
            ex_name   = chess.square_name(ex_sq)
            attackers = board.attackers(not king_color, ex_sq)
            if attackers:
                atk_sq    = next(iter(attackers))
                atk_piece = board.piece_at(atk_sq)
                atk_name  = self._piece_str(atk_piece.piece_type, atk_piece.color)
                parts.append(f"{ex_name} (controlled by the {atk_name} on {chess.square_name(atk_sq)})")
            else:
                # Genuine fallback: e.g. retreating along the same ray as the
                # checking slider, where the King's current square shadows
                # attackers()'s view of the destination square.
                parts.append(f"{ex_name} (still within the line of attack, even after the King moves)")
        return parts

    def _evaluate_piece_trajectory(
        self, board: chess.Board, sq: chess.Square
    ) -> Tuple[List[chess.Square], Set[chess.Square], str]:
        """
        Compute legal moves for a piece and build a natural-language
        description of the trajectory evaluation that shows genuine,
        board-grounded reasoning rather than restating the resolution:

          - zero moves -> definitively distinguishes pin vs block (non-King)
            or "no safe square" (King) via board.is_pinned(), instead of
            hedging with "blocked or pinned"
          - some moves -> shows the pseudo-legal -> legal delta when
            exclusions actually occurred, naming the SPECIFIC enemy piece
            responsible rather than asserting an abstract rule
          - King vs non-King are NEVER conflated: a King can never be
            "pinned" (board.is_pinned is always False for a king square --
            pins protect a King by restricting OTHER pieces, not the King
            itself). King exclusions are always "the King cannot move into
            check," attributed to whichever enemy piece controls that square.
          - promotions -> surfaced explicitly rather than silently collapsed

        Returns (legal_moves, promo_squares, detail_text). detail_text has
        no "[Trajectory Evaluation]" label -- callers compose the final
        sentence so multi-piece queries can join several details together.
        """
        piece   = board.piece_at(sq)
        sq_name = chess.square_name(sq)
        p_name  = self._piece_str(piece.piece_type, piece.color)
        is_king = piece.piece_type == chess.KING

        legal_moves, promo_squares = self._get_legal_moves(board, sq)
        pseudo   = self._get_pseudo_legal_destinations(board, sq)
        excluded = sorted(pseudo - set(legal_moves))

        # ------------------------------------------------------------------
        # Zero legal moves: determine the real cause, grounded in the board
        # ------------------------------------------------------------------
        if not legal_moves:
            if is_king:
                cause = (
                    "boxed in -- every available square is either occupied by a "
                    "friendly piece or controlled by an enemy piece, so the King "
                    "has no safe square to move to"
                )
            elif board.is_pinned(piece.color, sq):
                pinner = self._find_pinning_piece(board, piece.color, sq)
                if pinner:
                    pin_piece, pin_sq = pinner
                    pinner_name = self._piece_str(pin_piece.piece_type, pin_piece.color)
                    cause = (
                        f"absolutely pinned against its own King by the {pinner_name} "
                        f"on {chess.square_name(pin_sq)} -- any move would expose the King to check"
                    )
                else:
                    cause = "absolutely pinned against its own King -- any move would expose the King to check"
            else:
                cause = "fully blocked by surrounding pieces, with no legal destinations available"
            detail = f"the {p_name} on {sq_name} is {cause} (0 valid destinations)"
            return legal_moves, promo_squares, detail

        # ------------------------------------------------------------------
        # Has legal moves: show pseudo-legal -> legal delta, grounded
        # ------------------------------------------------------------------
        m_names   = [chess.square_name(m) for m in legal_moves]
        dest_word = "destination" if len(legal_moves) == 1 else "destinations"

        if excluded:
            pseudo_names = [chess.square_name(s) for s in sorted(pseudo)]
            pseudo_word  = "candidate" if len(pseudo) == 1 else "candidates"

            if is_king:
                # King: never a pin. Name the controlling enemy piece per square.
                excl_parts = self._king_safety_detail(board, piece.color, excluded)
                detail = (
                    f"the {p_name} on {sq_name} has {len(pseudo)} pseudo-legal {pseudo_word} "
                    f"({self._join_squares(pseudo_names)}), but king-safety filtering removes "
                    f"{self._join_squares(excl_parts)} -- the King cannot move into check -- "
                    f"leaving {len(legal_moves)} legal {dest_word}: {self._join_squares(m_names)}"
                )
            else:
                excluded_names = [chess.square_name(s) for s in excluded]
                pinner = self._find_pinning_piece(board, piece.color, sq)
                if pinner:
                    pin_piece, pin_sq = pinner
                    pinner_name = self._piece_str(pin_piece.piece_type, pin_piece.color)
                    pin_reason  = f"would expose the King to the {pinner_name} on {chess.square_name(pin_sq)}"
                else:
                    pin_reason = "would expose the King to check"
                detail = (
                    f"the {p_name} on {sq_name} has {len(pseudo)} pseudo-legal {pseudo_word} "
                    f"({self._join_squares(pseudo_names)}), but pin-constraint filtering removes "
                    f"{self._join_squares(excluded_names)} -- moving there {pin_reason} -- "
                    f"leaving {len(legal_moves)} legal {dest_word}: {self._join_squares(m_names)}"
                )
        else:
            exclusion_kind = "king-safety" if is_king else "pin-based"
            detail = (
                f"the {p_name} on {sq_name} has {len(legal_moves)} legal {dest_word} "
                f"with no {exclusion_kind} exclusions: {self._join_squares(m_names)}"
            )

        if promo_squares:
            promo_names = sorted(chess.square_name(s) for s in promo_squares)
            detail += f" (reaching {self._join_squares(promo_names)} triggers promotion)"

        return legal_moves, promo_squares, detail

    def _generate_ui_tags(
        self, board: chess.Board, origin: chess.Square, dests: List[chess.Square]
    ) -> str:
        """
        Generates XML UI tags for moves based on piece type.
        Jumps (N, K, P) use circles. Slides (B, R, Q) use arrows.
        Captures are marked with a danger circle -- 'danger' is semantically
        correct here, since a capture is an actual realized threat, unlike
        the earlier misuse of 'danger' to mean "opponent's piece."
        """
        piece = board.piece_at(origin)
        if not piece:
            return ""

        tags = []
        origin_name = chess.square_name(origin)

        for d in dests:
            d_name = chess.square_name(d)
            is_capture = board.piece_at(d) is not None

            if piece.piece_type in [chess.KNIGHT, chess.KING, chess.PAWN]:
                color = "danger" if is_capture else "good"
                tags.append(f'<circle square="{d_name}" color="{color}"/>')
            else:
                tags.append(f'<arrow from="{origin_name}" to="{d_name}" color="plan" style="solid"/>')
                if is_capture:
                    tags.append(f'<circle square="{d_name}" color="danger"/>')

        return " ".join(tags) + (" " if tags else "")

    def _turn_note(self, piece_color: chess.Color, board: chess.Board, p_name: str) -> str:
        """
        When evaluating a piece whose color isn't actually on move, flag that
        the moves shown are hypothetical -- otherwise the model risks telling
        a user "you can play Nf3" for a piece that can't move yet.
        """
        if piece_color == board.turn:
            return ""
        actual_side = "White" if board.turn == chess.WHITE else "Black"
        mover_side  = p_name.split()[0]
        return (
            f" Note: it is currently {actual_side}'s turn, so these are "
            f"hypothetical moves as if it were {mover_side}'s move."
        )

    # =========================================================================
    # QUERY TYPE 1: Square-Targeted Move Query
    # =========================================================================
    def generate_square_move_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        sq      = random.choice(range(64))
        sq_name = chess.square_name(sq)
        piece   = board.piece_at(sq)

        questions = [
            f"What are the legal moves for the piece on {sq_name}?",
            f"Where can the piece on {sq_name} go?",
            f"Show me the possible moves for {sq_name}.",
            f"If I want to move the piece on {sq_name}, what are my options?",
        ]

        intent = (
            f"[Intent] The user is requesting all legal destination squares "
            f"for the piece currently occupying {sq_name}."
        )

        if not piece:
            perception = f"[Perception] The <board_state> token for {sq_name} resolves as an empty square."
            trajectory = "[Trajectory Evaluation] N/A -- no piece is present to evaluate."
            resolution = f"[Resolution] {sq_name} is empty; there are no legal moves to compute."
            plan       = "[Pedagogical Plan] I will state clearly that the square is empty."
            answer     = f"The {sq_name} square is actually empty, so there are no moves to show!"
        else:
            p_name = self._piece_str(piece.piece_type, piece.color)
            perception = f"[Perception] The <board_state> token for {sq_name} resolves to a {p_name} signature."

            intent += self._turn_note(piece.color, board, p_name)

            legal_moves, promo_squares, detail = self._evaluate_piece_trajectory(board, sq)
            trajectory = (
                f"[Trajectory Evaluation] Consulting <board_state> tokens across the board "
                f"to evaluate {p_name} mobility from {sq_name}: {detail}."
            )

            if not legal_moves:
                resolution = f"[Resolution] The {p_name} on {sq_name} has 0 legal moves."
                plan = (
                    f"[Pedagogical Plan] I will highlight {sq_name} in 'info' and explain "
                    f"that the piece is completely stuck."
                )

                # A King can never be "pinned" -- pins restrict OTHER pieces
                # from exposing their own King, not the King itself. A King
                # with 0 moves is simply boxed in by occupied/attacked squares.
                is_king   = piece.piece_type == chess.KING
                is_pinned = (not is_king) and board.is_pinned(piece.color, sq)

                if elo_tier == "beginner":
                    if is_king:
                        cause_text = "every square around it is either occupied or attacked by an enemy piece, so it has nowhere safe to go"
                    elif is_pinned:
                        cause_text = "it's pinned to your King -- moving it would put your King in check, so the rules don't allow it"
                    else:
                        cause_text = "every square around it is occupied, so it has nowhere to go"
                    answer = (
                        f'<highlight square="{sq_name}" color="info"/> '
                        f"The {p_name} on {sq_name} can't move right now -- {cause_text}."
                    )
                else:
                    cause_word = "boxed in" if is_king else ("pinned" if is_pinned else "blocked")
                    answer = (
                        f'<highlight square="{sq_name}" color="info"/> '
                        f"The {p_name} on {sq_name} is currently stuck and has no legal "
                        f"moves ({cause_word})."
                    )
            else:
                m_names    = [chess.square_name(m) for m in legal_moves]
                resolution = (
                    f"[Resolution] Legal moves for {sq_name} resolve to: "
                    f"{self._join_squares(m_names)}."
                )

                ui_tags = self._generate_ui_tags(board, sq, legal_moves)
                plan = (
                    f"[Pedagogical Plan] I will highlight the origin {sq_name} in 'info' "
                    f"and use structural UI tags (arrows/circles) to map out the "
                    f"{len(legal_moves)} legal destinations."
                )

                dest_word = "square" if len(legal_moves) == 1 else "squares"
                answer = (
                    f'<highlight square="{sq_name}" color="info"/> {ui_tags}'
                    f"The {p_name} on {sq_name} can move to {len(legal_moves)} {dest_word}: "
                    f"{self._join_squares(m_names)}."
                )
                if promo_squares:
                    promo_names = sorted(chess.square_name(s) for s in promo_squares)
                    answer += f" Reaching {self._join_squares(promo_names)} also lets it promote!"

                answer += self._maybe_add_ask(mode, len(legal_moves))

        cot = f"{intent}\n{perception}\n{trajectory}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 2: Piece-Targeted Move Query (Single Color)
    # =========================================================================
    def generate_piece_move_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        color      = random.choice([chess.WHITE, chess.BLACK])
        piece_type = random.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        p_name     = self._piece_str(piece_type, color)

        questions = [
            f"Where can the {p_name} move?",
            f"What are the legal moves for the {p_name}?",
            f"Show me all the squares the {p_name} can reach.",
        ]

        sqs        = list(board.pieces(piece_type, color))
        piece_word = p_name if len(sqs) <= 1 else f"{p_name}s"
        intent     = (
            f"[Intent] The user is requesting all legal destination squares "
            f"for the {piece_word}."
        )
        if sqs:
            intent += self._turn_note(color, board, p_name)

        if not sqs:
            perception = f"[Perception] Filtering the 64 <board_state> visual tokens reveals no {p_name} signatures."
            trajectory = "[Trajectory Evaluation] N/A -- the piece has been captured."
            resolution = f"[Resolution] 0 {p_name} signatures found."
            plan       = "[Pedagogical Plan] I will inform the user that the piece is no longer on the board."
            answer     = f"You don't have a {p_name} on the board; it has been captured."
        else:
            sq_names   = [chess.square_name(s) for s in sqs]
            perception = (
                f"[Perception] Filtering the 64 <board_state> visual tokens reveals {len(sqs)} {piece_word} "
                f"at {self._join_squares(sq_names)}."
            )

            per_piece = []   # (s_name, legal_moves, promo_squares)
            details   = []
            res_parts = []
            ans_parts = []
            ui_color  = "info" if color == chess.WHITE else "key"
            ui_tags   = ""

            for s in sqs:
                s_name = chess.square_name(s)
                legal_moves, promo_squares, detail = self._evaluate_piece_trajectory(board, s)
                per_piece.append((s_name, legal_moves, promo_squares))
                details.append(detail)

                ui_tags += f'<highlight square="{s_name}" color="{ui_color}"/> '
                ui_tags += self._generate_ui_tags(board, s, legal_moves)

                if not legal_moves:
                    res_parts.append(f"{s_name} (0 moves)")
                    ans_parts.append(f"From {s_name}: stuck, 0 moves")
                else:
                    m_names = [chess.square_name(m) for m in legal_moves]
                    res_parts.append(f"{s_name} -> {self._join_squares(m_names)}")
                    ans_parts.append(f"From {s_name}: {self._join_squares(m_names)}")

            trajectory = (
                f"[Trajectory Evaluation] Consulting <board_state> tokens to evaluate "
                f"trajectories for each {p_name}: " + "; ".join(details) + "."
            )
            piece_eval_word = "piece" if len(sqs) == 1 else "pieces"
            resolution = (
                f"[Resolution] {len(sqs)} {piece_eval_word} evaluated. "
                f"Legal moves: {self._join_squares(res_parts)}."
            )
            plan = (
                f"[Pedagogical Plan] I will highlight the origin square(s) in '{ui_color}' "
                f"and emit structural UI tags for all legal paths."
            )

            if len(sqs) == 1:
                s_name, legal_moves, promo_squares = per_piece[0]
                if not legal_moves:
                    answer = f"{ui_tags}The {p_name} on {s_name} is completely stuck and cannot move."
                else:
                    m_names = [chess.square_name(m) for m in legal_moves]
                    answer = f"{ui_tags}The {p_name} on {s_name} can move to: {self._join_squares(m_names)}."
                    if promo_squares:
                        promo_names = sorted(chess.square_name(s) for s in promo_squares)
                        answer += f" Reaching {self._join_squares(promo_names)} also lets it promote!"
                    answer += self._maybe_add_ask(mode, len(legal_moves))
            else:
                answer = (
                    f"{ui_tags}You have {len(sqs)} {p_name}s. Here are their moves:\n- "
                    + "\n- ".join(ans_parts)
                )

        cot = f"{intent}\n{perception}\n{trajectory}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 3: Color-Agnostic Move Query
    # =========================================================================
    def generate_agnostic_move_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        piece_type = random.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK])
        base_name  = self.PIECE_NAMES[piece_type]

        questions = [
            f"What are the moves for all the {base_name}s?",
            f"Show me where every {base_name} can move.",
            f"I want to see the legal moves for both White and Black {base_name}s.",
        ]

        w_sqs   = list(board.pieces(piece_type, chess.WHITE))
        b_sqs   = list(board.pieces(piece_type, chess.BLACK))
        all_sqs = w_sqs + b_sqs

        intent = f"[Intent] The user is requesting legal moves for all {base_name}s across both colors."

        if not all_sqs:
            perception = f"[Perception] Broadcasting attention across all 64 <board_state> visual tokens detects no {base_name} signatures for either side."
            trajectory = "[Trajectory Evaluation] N/A."
            resolution = f"[Resolution] 0 {base_name} signatures found total."
            plan       = f"[Pedagogical Plan] I will state clearly that there are no {base_name}s left on the board."
            answer     = f"There are no {base_name}s left on the board for either side."
        else:
            w_names = [chess.square_name(s) for s in w_sqs]
            b_names = [chess.square_name(s) for s in b_sqs]
            perception = (
                f"[Perception] Scanning the 64 <board_state> visual tokens for {base_name} signatures: "
                f"White resolves at {self._join_squares(w_names) or 'none'}. "
                f"Black resolves at {self._join_squares(b_names) or 'none'}."
            )

            details   = []
            res_parts = []
            ans_parts = []
            ui_tags   = ""

            for c_color, sqs, c_str, ui_color in [
                (chess.WHITE, w_sqs, "White", "info"),
                (chess.BLACK, b_sqs, "Black", "key"),
            ]:
                for s in sqs:
                    s_name = chess.square_name(s)
                    legal_moves, promo_squares, detail = self._evaluate_piece_trajectory(board, s)
                    # detail already names the piece, color, and square --
                    # appended directly rather than re-prefixed to avoid
                    # redundant repetition (e.g. "White e4: the White Knight on e4...").
                    details.append(detail)

                    ui_tags += f'<highlight square="{s_name}" color="{ui_color}"/> '
                    ui_tags += self._generate_ui_tags(board, s, legal_moves)

                    if not legal_moves:
                        res_parts.append(f"{s_name} (0 moves)")
                        ans_parts.append(f"{c_str} {base_name} on {s_name}: stuck")
                    else:
                        m_names = [chess.square_name(m) for m in legal_moves]
                        res_parts.append(f"{s_name} -> {self._join_squares(m_names)}")
                        ans_parts.append(f"{c_str} {base_name} on {s_name}: {self._join_squares(m_names)}")

            trajectory = (
                "[Trajectory Evaluation] Consulting <board_state> tokens to evaluate "
                "trajectories for both sides: " + "; ".join(details) + "."
            )
            piece_eval_word = "piece" if len(all_sqs) == 1 else "pieces"
            resolution = (
                f"[Resolution] {len(all_sqs)} {piece_eval_word} evaluated. "
                f"{self._join_squares(res_parts)}."
            )
            plan = (
                "[Pedagogical Plan] I will highlight White origins in 'info' and Black "
                "origins in 'key', emit UI move tags, and list the destinations clearly."
            )

            answer = (
                f"{ui_tags}Here are the legal moves for all the {base_name}s:\n- "
                + "\n- ".join(ans_parts)
            )

        cot = f"{intent}\n{perception}\n{trajectory}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 4: False Premise Verification
    # =========================================================================
    def generate_verification_move_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        sq           = random.choice(range(64))
        sq_name      = chess.square_name(sq)
        actual_piece = board.piece_at(sq)

        # Guarantee a mismatch by picking a piece type/color not on the square
        expected_color = random.choice([chess.WHITE, chess.BLACK])
        expected_type  = random.choice(
            [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        )
        if (actual_piece and actual_piece.color == expected_color
                and actual_piece.piece_type == expected_type):
            expected_type = chess.QUEEN if expected_type == chess.KING else chess.KING

        expected_name = self._piece_str(expected_type, expected_color)

        questions = [
            f"What are the moves for the {expected_name} on {sq_name}?",
            f"Where can my {expected_name} on {sq_name} go?",
            f"Show me legal moves for the {sq_name} {expected_name}.",
        ]

        intent = (
            f"[Intent] The user is asking for the moves of a {expected_name}, "
            f"explicitly asserting it is on {sq_name}."
        )

        # Perception stays purely extractive/comparative -- the verdict
        # ("Mismatch") belongs in Resolution, not here. Folding the
        # conclusion into Perception duplicates it across two steps.
        perception = (
            f"[Perception] Extracting the <board_state> token at {sq_name} and "
            f"comparing it against the expected {expected_name} class."
        )
        actual_str = "an empty square" if not actual_piece else (
            f"a {self._piece_str(actual_piece.piece_type, actual_piece.color)}"
        )

        trajectory = "[Trajectory Evaluation] N/A -- premise verification precedes move calculation."
        resolution = (
            f"[Resolution] Mismatch -- {sq_name} contains {actual_str}, not a {expected_name}. "
            f"The user's premise is false."
        )

        actual_expected_sqs = list(board.pieces(expected_type, expected_color))

        if not actual_expected_sqs:
            plan = (
                f"[Pedagogical Plan] I will use an 'info' highlight on {sq_name} to correct "
                f"the user, and inform them the {expected_name} is no longer on the board."
            )
            answer = (
                f'<highlight square="{sq_name}" color="info"/> '
                f"Actually, there is no {expected_name} on {sq_name} (it is {actual_str}). "
                f"Furthermore, your {expected_name} has been captured."
            )
        else:
            actual_sq_names = [chess.square_name(s) for s in actual_expected_sqs]
            is_are     = "is" if len(actual_expected_sqs) == 1 else "are"
            piece_word = expected_name if len(actual_expected_sqs) == 1 else f"{expected_name}s"
            loc_word   = "location" if len(actual_expected_sqs) == 1 else "locations"

            plan = (
                f"[Pedagogical Plan] I will use an 'info' highlight on {sq_name} to correct "
                f"the identity, and a 'key' highlight on the actual {expected_name} {loc_word}. "
                f"I will withhold move calculations until the confusion is cleared."
            )
            hl_actual = "".join(f'<highlight square="{s}" color="key"/> ' for s in actual_sq_names)
            answer = (
                f'<highlight square="{sq_name}" color="info"/> {hl_actual}'
                f"Wait, there is {actual_str} on {sq_name}. Your {piece_word} {is_are} "
                f"actually on {self._join_squares(actual_sq_names)}! Which piece did you want to move?"
            )

        cot = f"{intent}\n{perception}\n{trajectory}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 5: Blocked / Pinned Query (0 Moves)
    # =========================================================================
    def generate_blocked_move_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        # Scan the board for a piece that exists but has 0 legal moves
        blocked_sqs = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type != chess.KING:
                legal_moves, _ = self._get_legal_moves(board, sq)
                if len(legal_moves) == 0:
                    blocked_sqs.append(sq)

        # If no completely blocked pieces exist in this FEN, fall back to Q1
        if not blocked_sqs:
            return self.generate_square_move_query(board, elo_tier, mode)

        sq      = random.choice(blocked_sqs)
        sq_name = chess.square_name(sq)
        piece   = board.piece_at(sq)
        p_name  = self._piece_str(piece.piece_type, piece.color)

        questions = [
            f"Where can the {p_name} on {sq_name} go?",
            f"What are the moves for the {sq_name} piece?",
            f"Show me the legal squares for the {p_name} on {sq_name}.",
        ]

        intent = f"[Intent] The user is requesting legal moves for the {p_name} on {sq_name}."
        intent += self._turn_note(piece.color, board, p_name)

        perception = f"[Perception] The <board_state> token for {sq_name} resolves correctly to a {p_name} signature."

        _, _, detail = self._evaluate_piece_trajectory(board, sq)
        trajectory = (
            f"[Trajectory Evaluation] Consulting <board_state> tokens across the board: {detail}."
        )
        resolution = f"[Resolution] Legal moves for {sq_name} resolve to 0."
        plan = (
            f"[Pedagogical Plan] I will highlight the origin {sq_name} in 'info' and "
            f"explicitly explain that the piece is immobilized."
        )

        is_pinned = board.is_pinned(piece.color, sq)
        if elo_tier == "beginner":
            cause_text = (
                "it's pinned to your King, so moving it would put your King in check"
                if is_pinned else
                "every square around it is occupied, so it has nowhere to go"
            )
            answer = (
                f'<highlight square="{sq_name}" color="info"/> '
                f"The {p_name} on {sq_name} can't move right now -- {cause_text}."
            )
        else:
            cause_word = "pinned" if is_pinned else "blocked"
            answer = (
                f'<highlight square="{sq_name}" color="info"/> '
                f"The {p_name} on {sq_name} actually has zero legal moves right now! "
                f"It is completely {cause_word}."
            )

        cot = f"{intent}\n{perception}\n{trajectory}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # BATCH GENERATION
    # =========================================================================
    def generate_dataset(self, fens: List[str]) -> None:
        """
        Run all five legal move query generators over a list of FENs.
        elo_tier and mode are randomized per sample so the dataset actually
        demonstrates the skill/mode-adaptive behavior the system prompt
        requires, rather than producing identical phrasing regardless of
        who's asking.
        """
        query_types = ["square", "piece", "agnostic", "verification", "blocked"]
        elo_tiers   = list(ELO_TIERS.keys())
        modes       = [CoachMode.COACH, CoachMode.COMMENTATOR, CoachMode.GAME]

        count = 0
        with open(self.output_file, "w") as f:
            for fen in fens:
                board      = chess.Board(fen)
                query_type = random.choice(query_types)
                elo_tier   = random.choice(elo_tiers)
                mode       = random.choice(modes)

                if query_type == "square":
                    record = self.generate_square_move_query(board, elo_tier, mode)
                elif query_type == "piece":
                    record = self.generate_piece_move_query(board, elo_tier, mode)
                elif query_type == "agnostic":
                    record = self.generate_agnostic_move_query(board, elo_tier, mode)
                elif query_type == "verification":
                    record = self.generate_verification_move_query(board, elo_tier, mode)
                else:
                    record = self.generate_blocked_move_query(board, elo_tier, mode)

                f.write(json.dumps(record) + "\n")
                count += 1

        print(f"✅ Generated {count} legal move samples -> {self.output_file}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    sample_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          # Start
        "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",  # Middlegame
        "8/8/8/4k3/8/2K5/1P6/8 w - - 0 1",                                    # Sparse endgame
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",                                    # Promotion-rich endgame
        "rnbq1rk1/pppp1ppp/5n2/2b5/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQ - 4 7",  # Sicilian
    ]

    generator = LegalMovesGenerator(output_file="phase2_legal_moves.jsonl")
    generator.generate_dataset(sample_fens * 25)   # 125 samples
