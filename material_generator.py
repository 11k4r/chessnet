import json
import random
import chess
from typing import List, Dict, Tuple

from prompt_builder import PromptBuilder, ELO_TIERS, CoachMode


class MaterialGenerator:
    """
    Generates Phase 2 training data for chess material evaluation: piece
    presence, grouped piece counts, material point totals, material
    advantage/comparison, and trade evaluation.

    Query types
    -----------
    1. Piece Presence       - "Does White still have both Bishops?"
    2. Grouped Piece Count  - "How many minor pieces does Black have?"
    3. Material Point Total - "What's White's total material value?"
    4. Material Advantage   - "Who has more material right now?"
    5. Trade Evaluation     - "Would trading my Rook for a Bishop and a pawn be fair?"

    CoT labels
    ----------
    [Intent] [Perception] [Material Calculation] [Resolution] [Pedagogical Plan]

    [Material Calculation] is the domain-specific "show your work" step --
    the material-evaluation analog of [Trajectory Evaluation] in the legal
    moves generator. It stays light for simple presence/count queries
    (mirroring how [Trajectory Evaluation] was minimal for some Phase 2
    cases) but does genuine point-value arithmetic for total/advantage/trade
    queries, so the step always contributes something [Resolution] doesn't
    already say -- the historical loss count, the summed arithmetic, or the
    point-by-point trade comparison, never just a restated final number.

    elo_tier / mode are sampled per record and stored alongside "fen" (same
    convention as legal_moves_generator.py). Point values are exactly the
    kind of explicit knowledge a beginner may not have yet, so beginner-tier
    answers spell them out; higher tiers state the number directly.
    """

    PIECE_NAMES = {
        chess.PAWN:   "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK:   "Rook",
        chess.QUEEN:  "Queen",
        chess.KING:   "King",
    }

    # Standard material point values. King is excluded from material totals
    # by chess convention -- it has no trade value and is always present.
    PIECE_VALUES = {
        chess.PAWN:   1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK:   5,
        chess.QUEEN:  9,
    }

    # High-to-low display order, consistent with the "logical ordering"
    # convention used elsewhere (e.g. Full Side Enumeration in Phase 1).
    VALUE_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

    MINOR_PIECES = [chess.BISHOP, chess.KNIGHT]
    MAJOR_PIECES = [chess.QUEEN, chess.ROOK]  # High-to-Low, consistent with VALUE_ORDER

    # Standard starting counts, used for "still have both X" / loss framing.
    STARTING_COUNTS = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 2, chess.KNIGHT: 2}

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

    def _maybe_add_ask(self, mode: str, text: str) -> str:
        """Coach mode only: a single trailing engagement hook, not mid-response chunking."""
        if mode == CoachMode.COACH:
            return f" <ask>{text}</ask>"
        return ""

    def _pt(self, n: int) -> str:
        """'point' for 1, 'points' otherwise -- avoids repeating this check everywhere."""
        return "point" if n == 1 else "points"

    def _square_is_light(self, sq: chess.Square) -> bool:
        return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1

    def _bishop_pair_note(self, sqs: List[chess.Square]) -> str:
        """
        If exactly 2 Bishops remain, note whether they form the bishop pair
        (one per square color, full diagonal coverage) -- a genuine chess
        concept worth surfacing rather than treating bishops as
        interchangeable with this presence check.
        """
        if len(sqs) != 2:
            return ""
        shades = {self._square_is_light(s) for s in sqs}
        if len(shades) == 2:
            return " They form the bishop pair -- one on each square color, covering both diagonals."
        shade_word = "light" if True in shades else "dark"
        return f" Both happen to be on {shade_word} squares, so the bishop pair isn't intact."

    def _material_breakdown(
        self, board: chess.Board, color: chess.Color
    ) -> Tuple[int, Dict[int, List[chess.Square]]]:
        """Returns (total_points, {piece_type: [squares]}) for a color. King excluded."""
        by_type: Dict[int, List[chess.Square]] = {}
        total = 0
        for pt in self.VALUE_ORDER:
            sqs = list(board.pieces(pt, color))
            if sqs:
                by_type[pt] = sqs
                total += len(sqs) * self.PIECE_VALUES[pt]
        return total, by_type

    def _format_breakdown(self, by_type: Dict[int, List[chess.Square]]) -> str:
        """e.g. '1 Queen (9), 2 Rooks (10), 2 Bishops (6), 2 Knights (6), and 8 Pawns (8)'"""
        if not by_type:
            return "no material"
        parts = []
        for pt in self.VALUE_ORDER:
            if pt in by_type:
                n    = len(by_type[pt])
                name = self.PIECE_NAMES[pt] + ("s" if n > 1 else "")
                val  = self.PIECE_VALUES[pt] * n
                parts.append(f"{n} {name} ({val})")
        return self._join_squares(parts)

    def _calc_terms(self, by_type: Dict[int, List[chess.Square]]) -> str:
        """e.g. '1x9 + 2x5 + 2x3 + 2x3 + 8x1' -- same VALUE_ORDER as the breakdown."""
        if not by_type:
            return "0"
        return " + ".join(
            f"{len(by_type[pt])}x{self.PIECE_VALUES[pt]}" for pt in self.VALUE_ORDER if pt in by_type
        )

    # =========================================================================
    # QUERY TYPE 1: Piece Presence
    # =========================================================================
    def generate_presence_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        color      = random.choice([chess.WHITE, chess.BLACK])
        c_str      = "White" if color == chess.WHITE else "Black"
        piece_type = random.choice([chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT])
        p_name     = self.PIECE_NAMES[piece_type]
        plural     = f"{p_name}s"

        start_n    = self.STARTING_COUNTS[piece_type]
        start_word = p_name if start_n == 1 else plural  # describes the STARTING set (no numeral attached)

        sqs   = list(board.pieces(piece_type, color))
        cur_n = len(sqs)
        cur_word = p_name if cur_n == 1 else plural       # describes the CURRENT count -- always paired with a numeral

        if start_n == 1:
            questions = [
                f"Does {c_str} still have their {p_name}?",
                f"Has {c_str} lost their {p_name}?",
                f"Is {c_str}'s {p_name} still on the board?",
            ]
        else:
            questions = [
                f"Does {c_str} still have both {plural}?",
                f"Has {c_str} lost any {plural}?",
                f"How many {plural} does {c_str} still have on the board?",
            ]

        intent = f"[Intent] The user is asking about the presence of {c_str}'s {start_word} on the board."
        # Genuine full-board search: piece locations are unknown in advance,
        # unlike a square-occupancy query where the target square is already given.
        perception = (
            f"[Perception] Scanning the 64 <board_state> visual tokens for {c_str} {p_name} signatures: "
            f"{cur_n} found"
            + (f" at {self._join_squares([chess.square_name(s) for s in sqs])}" if sqs else "")
            + "."
        )

        diff = start_n - cur_n
        if diff > 0:
            remain_word = "remains" if cur_n == 1 else "remain"
            calc_text = (
                f"started with {start_n} {start_word}; {cur_n} {remain_word}, "
                f"so {diff} {'has' if diff == 1 else 'have'} been captured"
            )
        elif diff < 0:
            extra = -diff
            calc_text = (
                f"started with {start_n} {start_word}; {cur_n} {cur_word} are present now "
                f"-- {extra} extra, most likely gained via pawn promotion"
            )
        else:
            calc_text = f"started with {start_n} {start_word}; all {start_n} remain on the board"

        calculation = f"[Material Calculation] Counting {c_str} {p_name} signatures: {calc_text}."

        if not sqs:
            resolution = f"[Resolution] {c_str} has 0 {plural} remaining; all have been captured."
            plan = f"[Pedagogical Plan] I will state clearly that the {start_word} {'is' if start_n == 1 else 'are'} gone."
            answer = (
                f"No, {c_str} has lost {'their' if start_n > 1 else 'the'} "
                f"{p_name}{'s' if start_n > 1 else ''} -- none remain on the board."
            )
        else:
            sq_names = [chess.square_name(s) for s in sqs]
            ui_color = "info" if color == chess.WHITE else "key"
            hl_tags  = "".join(f'<highlight square="{s}" color="{ui_color}"/> ' for s in sq_names)
            bishop_note = self._bishop_pair_note(sqs) if piece_type == chess.BISHOP else ""

            resolution = (
                f"[Resolution] {c_str} has {cur_n} {cur_word} on the board: "
                f"{self._join_squares(sq_names)}.{bishop_note}"
            )
            plan = (
                f"[Pedagogical Plan] I will highlight {'the' if cur_n == 1 else 'all'} "
                f"{cur_word} in '{ui_color}' and confirm the count."
            )

            if diff == 0 and start_n > 1:
                status = f"Yes, {c_str} still has both their {plural}, on {self._join_squares(sq_names)}"
            elif diff == 0:
                status = f"Yes, {c_str} still has their {p_name}, on {sq_names[0]}"
            elif diff > 0:
                status = (
                    f"{c_str} has {cur_n} {cur_word} left, on {self._join_squares(sq_names)} "
                    f"-- {diff} {'has' if diff == 1 else 'have'} been lost"
                )
            else:
                status = (
                    f"{c_str} actually has {cur_n} {cur_word}, on {self._join_squares(sq_names)} "
                    f"(extra from promotion)"
                )

            answer = f"{hl_tags}{status}.{bishop_note}"

        cot = f"{intent}\n{perception}\n{calculation}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 2: Grouped Piece Count (minor / major / total)
    # =========================================================================
    def generate_grouped_count_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        color = random.choice([chess.WHITE, chess.BLACK])
        c_str = "White" if color == chess.WHITE else "Black"
        group = random.choice(["minor", "major", "total"])

        if group == "minor":
            types       = self.MINOR_PIECES
            group_label = "minor pieces"
            questions = [
                f"How many minor pieces does {c_str} have?",
                f"Count {c_str}'s Bishops and Knights.",
                f"How many Bishops and Knights does {c_str} have left?",
            ]
        elif group == "major":
            types       = self.MAJOR_PIECES
            group_label = "major pieces"
            questions = [
                f"How many major pieces does {c_str} have?",
                f"Count {c_str}'s Rooks and Queens.",
                f"How many Rooks and Queens does {c_str} have left?",
            ]
        else:
            types       = self.VALUE_ORDER  # excludes King, High-to-Low
            group_label = "pieces (excluding the King)"
            questions = [
                f"How many pieces does {c_str} have left in total?",
                f"What's {c_str}'s total piece count?",
                f"How many non-King pieces does {c_str} have on the board?",
            ]

        intent = f"[Intent] The user is asking for {c_str}'s total count of {group_label}."

        per_type: Dict[int, List[chess.Square]] = {}
        sqs_all: List[chess.Square] = []
        for pt in types:
            sqs = list(board.pieces(pt, color))
            if sqs:
                per_type[pt] = sqs
                sqs_all.extend(sqs)

        type_strs = [
            f"{len(sqs)} {self.PIECE_NAMES[pt]}{'s' if len(sqs) > 1 else ''}"
            for pt, sqs in per_type.items()
        ]
        perception = (
            f"[Perception] Scanning the 64 <board_state> visual tokens for {c_str} {group_label}: "
            + (self._join_squares(type_strs) if type_strs else "none found") + "."
        )

        calculation = (
            "[Material Calculation] Summing across piece types: "
            + (
                " + ".join(str(len(sqs)) for sqs in per_type.values()) + f" = {len(sqs_all)}"
                if per_type else "0"
            )
            + "."
        )

        if not sqs_all:
            resolution = f"[Resolution] {c_str} has 0 {group_label}."
            plan = f"[Pedagogical Plan] I will state clearly that {c_str} has none of these pieces left."
            answer = f"{c_str} has no {group_label} remaining on the board."
        else:
            sq_names = [chess.square_name(s) for s in sqs_all]
            ui_color = "info" if color == chess.WHITE else "key"
            hl_tags  = "".join(f'<highlight square="{s}" color="{ui_color}"/> ' for s in sq_names)

            resolution = (
                f"[Resolution] {c_str} has {len(sqs_all)} {group_label} total, "
                f"on {self._join_squares(sq_names)}."
            )
            plan = (
                f"[Pedagogical Plan] I will highlight all {len(sqs_all)} pieces in '{ui_color}' "
                f"and break the count down by type."
            )
            breakdown = self._join_squares(type_strs)
            answer = f"{hl_tags}{c_str} has {len(sqs_all)} {group_label}: {breakdown}."

        cot = f"{intent}\n{perception}\n{calculation}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 3: Material Point Total
    # =========================================================================
    def generate_material_total_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        color = random.choice([chess.WHITE, chess.BLACK])
        c_str = "White" if color == chess.WHITE else "Black"

        questions = [
            f"What's {c_str}'s total material value?",
            f"How many points of material does {c_str} have?",
            f"What is {c_str}'s material count?",
        ]

        intent = f"[Intent] The user is asking for {c_str}'s total material value in standard chess points."
        perception = (
            f"[Perception] Scanning the 64 <board_state> visual tokens for all {c_str} piece signatures "
            f"(King excluded from material totals)."
        )

        total, by_type = self._material_breakdown(board, color)
        breakdown      = self._format_breakdown(by_type)
        calc_terms     = self._calc_terms(by_type)

        calculation = (
            f"[Material Calculation] Standard point values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9. "
            f"{c_str} has {breakdown}. Summing: {calc_terms} = {total} {self._pt(total)}."
        )
        resolution = f"[Resolution] {c_str}'s total material value is {total} {self._pt(total)}."

        sqs_all  = [s for sqs in by_type.values() for s in sqs]
        ui_color = "info" if color == chess.WHITE else "key"
        hl_tags  = "".join(f'<highlight square="{chess.square_name(s)}" color="{ui_color}"/> ' for s in sqs_all)
        plan = (
            f"[Pedagogical Plan] I will highlight all of {c_str}'s pieces in '{ui_color}' "
            f"and state the point total clearly."
        )

        if elo_tier == "beginner":
            answer = (
                f"{hl_tags}In chess, pieces are worth points: pawn=1, knight=3, bishop=3, rook=5, "
                f"queen=9 (the King isn't counted). Adding up {c_str}'s pieces -- {breakdown} -- "
                f"gives a total of {total} {self._pt(total)}."
            )
        else:
            answer = f"{hl_tags}{c_str} has {total} {self._pt(total)} of material ({breakdown})."

        cot = f"{intent}\n{perception}\n{calculation}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 4: Material Advantage / Comparison
    # =========================================================================
    def generate_material_advantage_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        questions = [
            "Who has more material right now?",
            "Is the material balanced?",
            "How much material is White up by?",
            "What's the material difference in this position?",
            "Does either side have a material advantage?",
        ]

        intent = "[Intent] The user is asking which side holds a material advantage, and by how much."
        perception = (
            "[Perception] Scanning the 64 <board_state> visual tokens for all piece signatures on both sides "
            "(King excluded from material totals)."
        )

        w_total, w_by_type = self._material_breakdown(board, chess.WHITE)
        b_total, b_by_type = self._material_breakdown(board, chess.BLACK)
        diff = w_total - b_total

        w_breakdown = self._format_breakdown(w_by_type)
        b_breakdown = self._format_breakdown(b_by_type)
        calculation = (
            f"[Material Calculation] White: {w_breakdown} = {w_total} {self._pt(w_total)}. "
            f"Black: {b_breakdown} = {b_total} {self._pt(b_total)}. Difference: {w_total} - {b_total} = {diff}."
        )

        if diff == 0:
            resolution = "[Resolution] Material is exactly balanced between both sides."
            plan = "[Pedagogical Plan] I will state clearly that material is equal, without highlighting either side."
            if elo_tier == "beginner":
                answer = (
                    f"The material is balanced right now -- both sides have {w_total} points worth "
                    f"of pieces (pawn=1, knight=3, bishop=3, rook=5, queen=9, King not counted)."
                )
            else:
                answer = f"Material is even: both sides have {w_total} points."
        else:
            leader      = "White" if diff > 0 else "Black"
            margin      = abs(diff)
            margin_word = "point" if margin == 1 else "points"
            ui_color    = "info" if diff > 0 else "key"
            lead_by_type = w_by_type if diff > 0 else b_by_type
            lead_sqs     = [s for sqs in lead_by_type.values() for s in sqs]
            sq_names     = [chess.square_name(s) for s in lead_sqs]
            hl_tags      = "".join(f'<highlight square="{s}" color="{ui_color}"/> ' for s in sq_names)

            resolution = f"[Resolution] {leader} is ahead in material by {margin} {margin_word}."
            plan = (
                f"[Pedagogical Plan] I will highlight {leader}'s pieces in '{ui_color}' "
                f"and state the material difference clearly."
            )

            if elo_tier == "beginner":
                answer = (
                    f"{hl_tags}{leader} has more material -- they're up by {margin} {margin_word} "
                    f"(White has {w_total}, Black has {b_total}; pawn=1, knight=3, bishop=3, "
                    f"rook=5, queen=9)."
                )
            else:
                answer = (
                    f"{hl_tags}{leader} is up {margin} {margin_word} of material "
                    f"(White {w_total} - Black {b_total})."
                )

            answer += self._maybe_add_ask(mode, "How would you use that advantage?")

        cot = f"{intent}\n{perception}\n{calculation}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # QUERY TYPE 5: Trade Evaluation
    # =========================================================================
    def generate_trade_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        white_pieces = [(pt, sq) for pt in self.PIECE_VALUES for sq in board.pieces(pt, chess.WHITE)]
        black_pieces = [(pt, sq) for pt in self.PIECE_VALUES for sq in board.pieces(pt, chess.BLACK)]

        if not white_pieces or not black_pieces:
            # Not enough material on the board to construct a trade scenario;
            # fall back to a material advantage query instead.
            return self.generate_material_advantage_query(board, elo_tier, mode)

        give_color = random.choice([chess.WHITE, chess.BLACK])
        give_pool, get_pool = (
            (white_pieces, black_pieces) if give_color == chess.WHITE else (black_pieces, white_pieces)
        )
        give_color_str = "White" if give_color == chess.WHITE else "Black"

        give_pt, give_sq = random.choice(give_pool)
        give_val = self.PIECE_VALUES[give_pt]

        # A 2-for-1 "receive" package only makes realistic chess sense as an
        # exchange sacrifice: giving up a higher-value piece (Rook/Queen) for
        # a lower piece plus pawn(s) -- e.g. "Rook for Bishop and 2 Pawns".
        # A low-value give piece (e.g. a Pawn) should never appear to "win"
        # a Knight AND a Rook -- that's not a trade any player would face.
        can_do_multi = give_val >= 5 and len(get_pool) >= 2
        n_get = min(random.choice([1, 1, 2]) if can_do_multi else 1, len(get_pool))

        if n_get == 2:
            pawn_items  = [item for item in get_pool if item[0] == chess.PAWN]
            other_items = [item for item in get_pool if item[0] != chess.PAWN]
            if pawn_items and other_items:
                # Bias toward the realistic "piece + pawn(s)" pattern
                get_items = [random.choice(pawn_items), random.choice(other_items)]
            else:
                get_items = random.sample(get_pool, n_get)
        else:
            get_items = random.sample(get_pool, n_get)

        give_name    = self._piece_str(give_pt, give_color)
        give_sq_name = chess.square_name(give_sq)
        get_names    = [self._piece_str(pt, not give_color) for pt, sq in get_items]
        get_sq_names = [chess.square_name(sq) for pt, sq in get_items]
        get_desc     = self._join_squares([f"{n} on {s}" for n, s in zip(get_names, get_sq_names)])

        questions = [
            f"Would trading my {give_name} on {give_sq_name} for {get_desc} be a fair material trade?",
            f"Is giving up the {give_name} on {give_sq_name} for {get_desc} a good trade?",
            f"If I traded my {give_name} on {give_sq_name} for {get_desc}, would I come out ahead materially?",
        ]

        intent = (
            f"[Intent] The user is asking whether giving up their {give_name} on {give_sq_name} "
            f"in exchange for {get_desc} would be material-favorable."
        )
        perception = (
            f"[Perception] Confirming via <board_state> tokens: the {give_name} is on {give_sq_name}; "
            f"{get_desc} {'is' if len(get_items) == 1 else 'are'} confirmed on the board."
        )

        get_val = sum(self.PIECE_VALUES[pt] for pt, sq in get_items)
        net     = get_val - give_val

        get_calc = " + ".join(
            f"{self.PIECE_VALUES[pt]} ({self._piece_str(pt, not give_color)})" for pt, sq in get_items
        )
        calculation = (
            f"[Material Calculation] Giving up: {give_name} = {give_val} {self._pt(give_val)}. "
            f"Receiving: {get_calc} = {get_val} {self._pt(get_val)}. "
            f"Net: {get_val} - {give_val} = {'+' if net >= 0 else ''}{net}."
        )

        if net == 0:
            resolution = "[Resolution] The trade is exactly material-even."
            verdict    = "an exactly even trade, material-wise"
        elif net > 0:
            net_word   = "point" if net == 1 else "points"
            resolution = f"[Resolution] The trade favors {give_color_str} by {net} {net_word}."
            verdict    = f"a good trade for {give_color_str} -- you'd gain {net} {net_word} of material"
        else:
            loss      = -net
            loss_word = "point" if loss == 1 else "points"
            resolution = f"[Resolution] The trade costs {give_color_str} {loss} {loss_word}."
            verdict    = f"a bad trade for {give_color_str} materially -- you'd lose {loss} {loss_word}"

        give_ui = "info" if give_color == chess.WHITE else "key"
        get_ui  = "key" if give_color == chess.WHITE else "info"
        hl_tags = f'<highlight square="{give_sq_name}" color="{give_ui}"/> '
        hl_tags += "".join(f'<highlight square="{s}" color="{get_ui}"/> ' for s in get_sq_names)

        plan = (
            f"[Pedagogical Plan] I will highlight the {give_name} being given up in '{give_ui}' "
            f"and the piece(s) received in '{get_ui}', then state the material verdict clearly."
        )

        if elo_tier == "beginner":
            answer = (
                f"{hl_tags}Let's add it up: your {give_name} is worth {give_val} {self._pt(give_val)}, and "
                f"{get_desc} {'is' if len(get_items) == 1 else 'are'} worth {get_val} {self._pt(get_val)} total "
                f"(pawn=1, knight=3, bishop=3, rook=5, queen=9). That makes this {verdict}."
            )
        else:
            answer = f"{hl_tags}{give_name} ({give_val}) for {get_desc} ({get_val}) -- {verdict}."

        if net != 0:
            answer += self._maybe_add_ask(
                mode, "Would the resulting position favor you beyond the raw point count?"
            )

        cot = f"{intent}\n{perception}\n{calculation}\n{resolution}\n{plan}"
        return self._format_sharegpt(
            board.fen(), random.choice(questions), cot, answer, elo_tier, mode
        )

    # =========================================================================
    # BATCH GENERATION
    # =========================================================================
    def generate_dataset(self, fens: List[str]) -> None:
        """Run all five material query generators over a list of FENs."""
        query_types = ["presence", "grouped_count", "material_total", "advantage", "trade"]
        elo_tiers   = list(ELO_TIERS.keys())
        modes       = [CoachMode.COACH, CoachMode.COMMENTATOR, CoachMode.GAME]

        count = 0
        with open(self.output_file, "w") as f:
            for fen in fens:
                board      = chess.Board(fen)
                query_type = random.choice(query_types)
                elo_tier   = random.choice(elo_tiers)
                mode       = random.choice(modes)

                if query_type == "presence":
                    record = self.generate_presence_query(board, elo_tier, mode)
                elif query_type == "grouped_count":
                    record = self.generate_grouped_count_query(board, elo_tier, mode)
                elif query_type == "material_total":
                    record = self.generate_material_total_query(board, elo_tier, mode)
                elif query_type == "advantage":
                    record = self.generate_material_advantage_query(board, elo_tier, mode)
                else:
                    record = self.generate_trade_query(board, elo_tier, mode)

                f.write(json.dumps(record) + "\n")
                count += 1

        print(f"✅ Generated {count} material samples -> {self.output_file}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    sample_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          # Start
        "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",  # Middlegame
        "8/8/8/4k3/8/2K5/1P6/8 w - - 0 1",                                    # Sparse endgame
        "r3k2r/ppp2ppp/2n5/3qp3/3P4/2N5/PPP2PPP/R2QK2R w KQkq - 0 1",       # Material imbalance
        "rnbq1rk1/pppp1ppp/5n2/2b5/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQ - 4 7",  # Sicilian
    ]

    generator = MaterialGenerator(output_file="phase2_material.jsonl")
    generator.generate_dataset(sample_fens * 25)   # 125 samples
