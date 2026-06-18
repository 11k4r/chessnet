import json
import random
import chess
from typing import List, Dict

from prompt_builder import PromptBuilder


class PiecePlacementGenerator:
    """
    Generates Phase 1 training data for chess piece placement.
    Builds strict 4-step atomic CoT focusing on visual token 'Perception'
    and pedagogical UI planning.

    Query types
    -----------
    1. Direct Piece Query        — "Where is the White Queen?"
    2. Square Occupancy Query    — "What piece is on e4?"
    3. Advanced Pawn Query       — "Where is my most advanced pawn?"
    4. Color-Agnostic Query      — "Where are all the knights?"
    5. Spatial / Region Query    — "Is there a knight on the queenside?"
    6. Identity Verification     — "Is there a White Knight on e4?"
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
    # Helpers
    # -------------------------------------------------------------------------

    def _piece_str(self, piece_type: int, color: chess.Color) -> str:
        c_str = "White" if color == chess.WHITE else "Black"
        return f"{c_str} {self.PIECE_NAMES[piece_type]}"

    def _ordinal(self, n: int) -> str:
        """Return the ordinal string for n (e.g. 1 -> '1st', 2 -> '2nd')."""
        if 11 <= n % 100 <= 13:
            return f"{n}th"
        return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"

    def _join_squares(self, names: list) -> str:
        """Join a list of strings with Oxford-comma style for 3+ items."""
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    def _format_sharegpt(
        self, fen: str, question: str, cot: str, answer: str
    ) -> Dict:
        """Package a sample in the ShareGPT format expected by dataset.py."""
        return {
            "fen": fen,
            "messages": [
                {"role": "system",    "content": self.system_prompt},
                {"role": "user",      "content": f"Question: {question}"},
                {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n{answer}"},
            ],
        }

    # =========================================================================
    # QUERY TYPE 1: Direct Piece Query  ("Where is the White Queen?")
    # =========================================================================
    def generate_direct_query(self, board: chess.Board) -> Dict:
        color      = random.choice([chess.WHITE, chess.BLACK])
        piece_type = random.choice(
            [chess.QUEEN, chess.KING, chess.KNIGHT, chess.BISHOP, chess.ROOK]
        )
        piece_name = self._piece_str(piece_type, color)

        squares  = list(board.pieces(piece_type, color))
        sq_names = [chess.square_name(sq) for sq in squares]

        # ------------------------------------------------------------------
        # Singular / zero-piece path
        # ------------------------------------------------------------------
        if len(sq_names) <= 1:
            questions = [
                f"Where is the {piece_name}?",
                f"Find the {piece_name}.",
                f"Locate the {piece_name} for me.",
                f"Show me the {piece_name}.",
                f"Where exactly is the {piece_name}?",
                f"What square is the {piece_name} on?",
                f"What is the coordinate of the {piece_name}?",
                f"Which square currently holds the {piece_name}?",
                f"Tell me the position of the {piece_name}.",
                f"Identify the location of the {piece_name}.",
                f"Can you tell me what square the {piece_name} is on?",
                f"Could you point out the {piece_name}?",
                f"Please tell me where the {piece_name} is located.",
                f"I need help finding the {piece_name}.",
                f"Can you spot the {piece_name} on the board?",
                f"Where do you see the {piece_name} on the board?",
                f"Point me to the {piece_name}.",
            ]
            intent     = f"[Intent] The user is requesting the exact board coordinate of the {piece_name}."
            perception = (
                f"[Perception] Focusing perception across the 64 <board_state> visual tokens "
                f"to identify the feature signature of the {piece_name}."
            )

            if len(sq_names) == 0:
                resolution = f"[Resolution] No {piece_name} signatures resolve within the tensor."
                plan       = (
                    f"[Pedagogical Plan] The piece is missing. I will state clearly that it has "
                    f"been captured and is no longer on the board."
                )
                answer     = f"The {piece_name} is no longer on the board; it has been captured."
            else:
                sq         = sq_names[0]
                resolution = (
                    f"[Resolution] The representation resolves uniquely to the token corresponding "
                    f"to the {sq} square. No other {piece_name} signatures are present in the tensor."
                )
                plan       = (
                    f'[Pedagogical Plan] I will emit <highlight square="{sq}" color="info"/> '
                    f"to visually pinpoint the piece for the user, and state its location clearly in the text."
                )
                answer     = f'<highlight square="{sq}" color="info"/> The {piece_name} is on {sq}.'

        # ------------------------------------------------------------------
        # Plural path (2+ pieces - possible after promotion)
        # ------------------------------------------------------------------
        else:
            both_all  = "both" if len(sq_names) == 2 else "all"
            questions = [
                f"Where are my {piece_name}s?",
                f"Can you show me where {both_all} {piece_name}s are?",
                f"Locate the {piece_name}s for me.",
                f"Where do you see {piece_name}s on the board?",
                f"Can you spot all {piece_name}s for me?",
                f"Point me to {both_all} {piece_name}s.",
                f"Identify the locations of the {piece_name}s.",
            ]
            intent     = f"[Intent] The user is requesting the exact board coordinates of the {piece_name}s."
            perception = (
                f"[Perception] Focusing perception across the 64 <board_state> visual tokens "
                f"to identify the feature signatures of the {piece_name}s."
            )

            sq_str  = self._join_squares(sq_names)
            hl_tags = "".join(f'<highlight square="{s}" color="info"/> ' for s in sq_names)
            cmds    = self._join_squares(
                [f'<highlight square="{s}" color="info"/>' for s in sq_names]
            )

            resolution = f"[Resolution] {piece_name} representations resolve at {sq_str}."
            plan       = (
                f"[Pedagogical Plan] I will emit {cmds} to mark all {piece_name} locations "
                f"for the user."
            )
            answer     = (
                f"{hl_tags}You have {len(sq_names)} {piece_name}s remaining, "
                f"located on {sq_str}."
            )

        cot = f"{intent}\n{perception}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

    # =========================================================================
    # QUERY TYPE 2: Square Occupancy Query  ("What piece is on e4?")
    # =========================================================================
    def generate_square_query(self, board: chess.Board) -> Dict:
        sq      = random.randint(0, 63)
        sq_name = chess.square_name(sq)
        piece   = board.piece_at(sq)

        questions = [
            f"Is there a piece on {sq_name}?",
            f"What piece is sitting on {sq_name}?",
            f"Check {sq_name} and tell me what is there.",
            f"What's on {sq_name}?",
            f"Is {sq_name} occupied?",
            f"Tell me what occupies {sq_name}.",
            f"Can you identify what's on {sq_name}?",
        ]

        intent     = f"[Intent] The user is inquiring about the occupancy of the {sq_name} square."
        perception = (
            f"[Perception] Focusing perception on the <board_state> token corresponding to {sq_name}."
        )

        if piece is None:
            resolution = (
                f"[Resolution] The token for {sq_name} resolves entirely as an empty square feature."
            )
            plan   = (
                f'[Pedagogical Plan] The square is empty. To visually ground the conversation, '
                f'I will use <circle square="{sq_name}" color="info"/> to draw their eye to the '
                f"specific square, and state clearly that no piece is there."
            )
            answer = (
                f'<circle square="{sq_name}" color="info"/> '
                f"The {sq_name} square is completely empty."
            )
        else:
            piece_name = self._piece_str(piece.piece_type, piece.color)
            resolution = (
                f"[Resolution] The token for {sq_name} resolves to the feature signature "
                f"of a {piece_name}."
            )
            plan   = (
                f"[Pedagogical Plan] The square is occupied. I will highlight the square using "
                f'<highlight square="{sq_name}" color="info"/> and inform the user of the piece.'
            )
            answer = (
                f'<highlight square="{sq_name}" color="info"/> '
                f"There is a {piece_name} on {sq_name}."
            )

        cot = f"{intent}\n{perception}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

    # =========================================================================
    # QUERY TYPE 3: Advanced Pawn Query  ("Where is my most advanced pawn?")
    # =========================================================================
    def generate_advanced_pawn_query(self, board: chess.Board) -> Dict:
        color     = random.choice([chess.WHITE, chess.BLACK])
        color_str = "White" if color == chess.WHITE else "Black"
        pawns     = list(board.pieces(chess.PAWN, color))

        questions = [
            f"Where is my most advanced {color_str} pawn?",
            f"Which {color_str} pawn is furthest up the board?",
            f"What is my most advanced {color_str} pawn?",
            f"Which of my {color_str} pawns has pushed the furthest?",
            f"Find the most advanced {color_str} pawn for me.",
        ]

        intent     = (
            "[Intent] The user wants to identify their most advanced pawn, "
            "requiring an evaluation of relative rank depth."
        )
        perception = (
            f"[Perception] Attending to the <board_state> tensor in parallel "
            f"to extract all {color_str} Pawn feature signatures."
        )

        # ------------------------------------------------------------------
        # No pawns remaining
        # ------------------------------------------------------------------
        if not pawns:
            resolution = f"[Resolution] No {color_str} Pawn signatures resolve within the tensor."
            plan       = (
                f"[Pedagogical Plan] I will inform the user that they have no pawns "
                f"remaining on the board."
            )
            answer     = f"You don't have any {color_str} pawns left on the board."
            cot        = f"{intent}\n{perception}\n{resolution}\n{plan}"
            return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

        # ------------------------------------------------------------------
        # Determine most-advanced rank
        # For White: highest rank number. For Black: lowest rank number.
        # ------------------------------------------------------------------
        sq_info = [
            (chess.square_name(sq), chess.square_rank(sq) + 1)
            for sq in pawns
        ]

        adv_rank     = max(r[1] for r in sq_info) if color == chess.WHITE \
                       else min(r[1] for r in sq_info)
        advanced_sqs = [r[0] for r in sq_info if r[1] == adv_rank]
        other_sqs    = [r for r in sq_info if r[0] not in advanced_sqs]

        all_sq_names = self._join_squares([r[0] for r in sq_info])
        adv_sq_names = self._join_squares(advanced_sqs)

        is_are = "is" if len(advanced_sqs) == 1 else "are"
        pawn_s = "pawn" if len(advanced_sqs) == 1 else "pawns"

        # ------------------------------------------------------------------
        # Resolution
        # ------------------------------------------------------------------
        if len(pawns) == 1:
            resolution = (
                f"[Resolution] Only one {color_str} Pawn signature resolves: "
                f"{all_sq_names}, on rank {adv_rank}."
            )
        elif not other_sqs:
            resolution = (
                f"[Resolution] {color_str} pawn representations resolve at {all_sq_names}. "
                f"All reside equally on rank {adv_rank}."
            )
        else:
            adv_str   = self._join_squares(
                [f"{sq} (rank {adv_rank})" for sq in advanced_sqs]
            )
            other_str = self._join_squares(
                [f"{r[0]} (rank {r[1]})" for r in other_sqs]
            )

            if color == chess.WHITE:
                comparison = (
                    f"{adv_str} > {other_str}. "
                    f"Therefore, {adv_sq_names} {is_are} the most advanced."
                )
            else:
                comparison = (
                    f"For Black, advancement is toward rank 1. "
                    f"{adv_str} {is_are} closest to rank 1 among {all_sq_names}. "
                    f"Therefore, {adv_sq_names} {is_are} the most advanced."
                )

            resolution = (
                f"[Resolution] {color_str} pawn representations resolve at {all_sq_names}. "
                f"Rank comparison: {comparison}"
            )

        hl_plan = self._join_squares(
            [f'<highlight square="{sq}" color="key"/>' for sq in advanced_sqs]
        )
        plan    = (
            f"[Pedagogical Plan] I will highlight the advanced {pawn_s} using {hl_plan}. "
            f"In my explanation, I will note the rank to reinforce board geography."
        )

        hl_tags = "".join(
            f'<highlight square="{sq}" color="key"/> ' for sq in advanced_sqs
        )
        answer  = (
            f"{hl_tags}Your most advanced {color_str} {pawn_s} {is_are} located on "
            f"{adv_sq_names}, sitting on the {self._ordinal(adv_rank)} rank."
        )

        cot = f"{intent}\n{perception}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

    # =========================================================================
    # QUERY TYPE 4: Color-Agnostic Query  ("Where are all the knights?")
    # =========================================================================
    def generate_agnostic_piece_query(self, board: chess.Board) -> Dict:
        piece_type  = random.choice(
            [chess.QUEEN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.PAWN]
        )
        base_name   = self.PIECE_NAMES[piece_type]
        plural_name = f"{base_name}s"

        questions = [
            f"Where are all the {plural_name}?",
            f"Can you point out every {base_name} on the board?",
            f"Show me all the {plural_name}, regardless of whose they are.",
            f"Just highlight every {base_name} right now.",
            f"Identify all {plural_name}.",
            f"I completely lost track of the {plural_name}. Where are they?",
            f"Are there any {plural_name} left in this game?",
            f"Do we have any {plural_name} on the board right now?",
            f"Where do you see {plural_name} on the board?",
            f"Can you spot all the {plural_name} for me?",
        ]

        white_sqs   = list(board.pieces(piece_type, chess.WHITE))
        black_sqs   = list(board.pieces(piece_type, chess.BLACK))
        white_names = [chess.square_name(sq) for sq in white_sqs]
        black_names = [chess.square_name(sq) for sq in black_sqs]

        intent     = (
            f"[Intent] The user is requesting the locations of all {plural_name} on the board, "
            f"explicitly omitting color specificity to query both White and Black pieces."
        )
        perception = (
            f"[Perception] Broadcasting attention across all 64 <board_state> tokens to detect "
            f"feature signatures matching the {base_name} class, bypassing the color embedding filter."
        )

        hl_tags      = ""
        answer_parts = []

        if not white_names and not black_names:
            resolution = (
                f"[Resolution] No {base_name} signatures (White or Black) resolve within the tensor."
            )
            plan   = (
                f"[Pedagogical Plan] I will plainly inform the user that all {plural_name} "
                f"have been captured."
            )
            answer = (
                f"There are no {plural_name} left on the board for either side; "
                f"they have all been captured."
            )

        else:
            res_parts = []

            if white_names:
                w_str   = self._join_squares(white_names)
                w_piece = base_name if len(white_names) == 1 else plural_name
                res_parts.append(f"White representations at {w_str}")
                hl_tags += "".join(
                    f'<highlight square="{s}" color="info"/> ' for s in white_names
                )
                answer_parts.append(f"White has {len(white_names)} {w_piece} on {w_str}")
            else:
                res_parts.append("no White representations")
                answer_parts.append("White has none remaining")

            if black_names:
                b_str   = self._join_squares(black_names)
                b_piece = base_name if len(black_names) == 1 else plural_name
                res_parts.append(f"Black representations at {b_str}")
                # 'key' (yellow) for Black distinguishes from White (info/blue)
                # without implying threat, which 'danger' would incorrectly suggest.
                hl_tags += "".join(
                    f'<highlight square="{s}" color="key"/> ' for s in black_names
                )
                answer_parts.append(f"Black has {len(black_names)} {b_piece} on {b_str}")
            else:
                res_parts.append("no Black representations")
                answer_parts.append("Black has none remaining")

            white_cmds = self._join_squares(
                [f'<highlight square="{s}" color="info"/>' for s in white_names]
            )
            black_cmds = self._join_squares(
                [f'<highlight square="{s}" color="key"/>' for s in black_names]
            )
            all_cmds   = self._join_squares(
                [c for c in [white_cmds, black_cmds] if c]
            )

            resolution = (
                f"[Resolution] Feature extraction complete: {'; '.join(res_parts)}."
            )
            plan   = (
                f"[Pedagogical Plan] I will emit {all_cmds} to mark all {plural_name}. "
                f"White pieces use 'info' (blue) and Black pieces use 'key' (yellow) "
                f"to visually differentiate them. I will break down locations by color in text."
            )
            answer = f"{hl_tags}Here are the {plural_name}: {'; '.join(answer_parts)}."

        cot = f"{intent}\n{perception}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

    # =========================================================================
    # QUERY TYPE 5: Spatial / Region Query  ("Is there a knight on the queenside?")
    # =========================================================================
    def generate_region_query(self, board: chess.Board) -> Dict:
        regions = {
            "bottom half": {
                "squares":   [sq for sq in range(64) if chess.square_rank(sq) < 4],
                "aliases":   ["bottom of the board", "bottom half", "first few rows",
                              "down low", "ranks 1 to 4"],
                "zone_cmd":  None,
                "grounding": "ranks 1 through 4",
            },
            "top half": {
                "squares":   [sq for sq in range(64) if chess.square_rank(sq) >= 4],
                "aliases":   ["top of the board", "top half", "highest ranks",
                              "up top", "ranks 5 to 8"],
                "zone_cmd":  None,
                "grounding": "ranks 5 through 8",
            },
            "left side": {
                "squares":   [sq for sq in range(64) if chess.square_file(sq) < 4],
                "aliases":   ["left side", "queenside", "over on the left",
                              "a through d files"],
                "zone_cmd":  '<highlight_zone zone="queenside" color="info"/>',
                "grounding": "files a through d",
            },
            "right side": {
                "squares":   [sq for sq in range(64) if chess.square_file(sq) >= 4],
                "aliases":   ["right side", "kingside", "over on the right",
                              "near the h-file"],
                "zone_cmd":  '<highlight_zone zone="kingside" color="info"/>',
                "grounding": "files e through h",
            },
            "center": {
                "squares":   [chess.D4, chess.D5, chess.E4, chess.E5],
                "aliases":   ["dead center", "middle of the board",
                              "central four squares"],
                "zone_cmd":  '<highlight_zone zone="center_squares" color="info"/>',
                # Ordered to match sorted() square-index output: d4, e4, d5, e5
                "grounding": "the d4, e4, d5, and e5 squares",
            },
        }

        region_key    = random.choice(list(regions.keys()))
        region_data   = regions[region_key]
        region_phrase = random.choice(region_data["aliases"])

        color      = random.choice([chess.WHITE, chess.BLACK])
        piece_type = random.choice(
            [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN]
        )
        piece_str  = self._piece_str(piece_type, color)
        plural_str = f"{piece_str}s"

        questions = [
            f"Is there a {piece_str} anywhere on the {region_phrase}?",
            f"Are there any {plural_str} hiding in the {region_phrase}?",
            f"Do I have a {piece_str} over in the {region_phrase}?",
            f"Check the {region_phrase}. Any {plural_str} there?",
            f"Tell me if a {piece_str} is located on the {region_phrase}.",
            f"Can you spot any {plural_str} in the {region_phrase}?",
        ]

        intent     = (
            f"[Intent] The user is inquiring about the presence of {plural_str} specifically "
            f"within a geometric subset of the board: the {region_key} ({region_phrase})."
        )
        # Grounding step: translate the casual phrase to concrete coordinates first,
        # then describe the masking operation on the tensor.
        perception = (
            f"[Perception] Translating '{region_phrase}' to concrete board coordinates "
            f"({region_data['grounding']}). Masking the <board_state> tensor to isolate "
            f"the corresponding tokens, filtering this spatial subset for {piece_str} "
            f"feature signatures."
        )

        # sorted() ensures deterministic square ordering across runs
        piece_sqs  = set(board.pieces(piece_type, color))
        region_sqs = set(region_data["squares"])
        found_sqs  = sorted(piece_sqs.intersection(region_sqs))

        if not found_sqs:
            resolution = (
                f"[Resolution] Evaluating the requested spatial slice yields no matching tokens. "
                f"There are no {piece_str} signatures in the {region_key}."
            )
            plan   = (
                f"[Pedagogical Plan] I will confirm that the region is clear of that piece. "
                f"No highlights are needed since the query resolves negatively."
            )
            answer = f"No, there are no {plural_str} located in the {region_phrase}."

        else:
            found_names = [chess.square_name(sq) for sq in found_sqs]
            sq_str      = self._join_squares(found_names)
            is_are      = "is" if len(found_sqs) == 1 else "are"
            piece_noun  = piece_str if len(found_sqs) == 1 else plural_str

            resolution = (
                f"[Resolution] The spatial mask successfully captures {piece_str} signatures "
                f"resolving at {sq_str} within the {region_key}."
            )

            hl_tags = "".join(
                f'<highlight square="{s}" color="key"/> ' for s in found_names
            )

            if region_data["zone_cmd"]:
                plan   = (
                    f"[Pedagogical Plan] I will emit {region_data['zone_cmd']} to show the region "
                    f"conceptually, then use <highlight> to pinpoint the exact square(s) occupied."
                )
                answer = (
                    f"{region_data['zone_cmd']} {hl_tags}"
                    f"Yes, there {is_are} {len(found_sqs)} {piece_noun} in the {region_phrase}, "
                    f"specifically on {sq_str}."
                )
            else:
                plan   = (
                    f"[Pedagogical Plan] I will emit <highlight> commands to pinpoint the piece(s) "
                    f"found in the specified geometric area."
                )
                answer = (
                    f"{hl_tags}Yes, there {is_are} {len(found_sqs)} {piece_noun} in the "
                    f"{region_phrase}, specifically on {sq_str}."
                )

        cot = f"{intent}\n{perception}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

    # =========================================================================
    # QUERY TYPE 6: Identity Verification Query  ("Is there a White Knight on e4?")
    # =========================================================================
    def generate_identity_verification_query(self, board: chess.Board) -> Dict:
        sq      = random.randint(0, 63)
        sq_name = chess.square_name(sq)

        expected_color = random.choice([chess.WHITE, chess.BLACK])
        expected_type  = random.choice(
            [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN]
        )
        expected_name  = self._piece_str(expected_type, expected_color)

        actual_piece   = board.piece_at(sq)

        # Colour-neutral phrasing — avoids "my" which implies ownership
        # and would be wrong when expected_color is the opponent's side.
        questions = [
            f"Is there a {expected_name} on {sq_name}?",
            f"Can you confirm whether there is a {expected_name} on {sq_name}?",
            f"I think there's a {expected_name} on {sq_name}. Am I right?",
            f"Verify if the piece on {sq_name} is a {expected_name}.",
            f"Is the piece sitting on {sq_name} a {expected_name}?",
            f"Check {sq_name} — is it occupied by a {expected_name}?",
        ]

        intent     = (
            f"[Intent] The user is verifying both the occupancy and the specific identity "
            f"({expected_name}) of the piece on the {sq_name} square."
        )
        perception = (
            f"[Perception] Focusing perception on the <board_state> token corresponding to {sq_name}, "
            f"comparing its extracted feature signature against the expected {expected_name} class."
        )

        # ------------------------------------------------------------------
        # Branch 1: Square is empty
        # ------------------------------------------------------------------
        if actual_piece is None:
            resolution = (
                f"[Resolution] Mismatch. The token for {sq_name} resolves as an empty square. "
                f"No {expected_name} signature is present."
            )
            plan   = (
                f'[Pedagogical Plan] I will use <circle square="{sq_name}" color="info"/> '
                f"to draw attention to the empty square and correct the user."
            )
            answer = (
                f'<circle square="{sq_name}" color="info"/> '
                f"No — the {sq_name} square is actually completely empty."
            )

        # ------------------------------------------------------------------
        # Branch 2: Exact match
        # ------------------------------------------------------------------
        elif (actual_piece.color == expected_color
              and actual_piece.piece_type == expected_type):
            resolution = (
                f"[Resolution] Match verified. The token for {sq_name} resolves exactly to the "
                f"feature signature of a {expected_name}."
            )
            plan   = (
                f'[Pedagogical Plan] I will use <highlight square="{sq_name}" color="good"/> '
                f"to confirm the user's assumption visually."
            )
            answer = (
                f'<highlight square="{sq_name}" color="good"/> '
                f"Yes, exactly right — there is a {expected_name} on {sq_name}."
            )

        # ------------------------------------------------------------------
        # Branch 3: Occupied but wrong piece
        # Uses 'info' (neutral blue) for all mismatch cases.
        # 'danger' is semantically wrong here — it implies threat rather
        # than identity difference. The text carries the correction.
        # ------------------------------------------------------------------
        else:
            actual_name = self._piece_str(actual_piece.piece_type, actual_piece.color)
            resolution  = (
                f"[Resolution] Identity mismatch. The token for {sq_name} resolves to a "
                f"{actual_name}, not the expected {expected_name}."
            )
            plan   = (
                f'[Pedagogical Plan] I will use <highlight square="{sq_name}" color="info"/> '
                f"to point out the square and correct the user's misconception by identifying "
                f"the actual piece."
            )
            answer = (
                f'<highlight square="{sq_name}" color="info"/> '
                f"Not quite — there is actually a {actual_name} on {sq_name}."
            )

        cot = f"{intent}\n{perception}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

    # =========================================================================
    # BATCH GENERATION
    # =========================================================================
    def generate_dataset(self, fens: List[str]) -> None:
        """Run all six query generators over a list of FENs and write JSONL."""
        query_types = [
            "direct",
            "square",
            "advanced_pawn",
            "agnostic_piece",
            "region",
            "identity_verification",
        ]
        count = 0
        with open(self.output_file, "w") as f:
            for fen in fens:
                board      = chess.Board(fen)
                query_type = random.choice(query_types)

                if query_type == "direct":
                    record = self.generate_direct_query(board)
                elif query_type == "square":
                    record = self.generate_square_query(board)
                elif query_type == "advanced_pawn":
                    record = self.generate_advanced_pawn_query(board)
                elif query_type == "agnostic_piece":
                    record = self.generate_agnostic_piece_query(board)
                elif query_type == "region":
                    record = self.generate_region_query(board)
                else:
                    record = self.generate_identity_verification_query(board)

                f.write(json.dumps(record) + "\n")
                count += 1

        print(f"✅ Generated {count} piece placement samples -> {self.output_file}")

