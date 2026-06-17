import json
import random
import chess
from typing import List, Dict, Optional

from prompt_builder import PromptBuilder


class PiecePlacementGenerator:
    """
    Generates Phase 1 training data for chess piece placement.
    Builds strict 4-step atomic CoT focusing on visual token 'Perception'
    and pedagogical UI planning.
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
        self.output_file = output_file
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
                # Direct & Simple
                f"Where is the {piece_name}?",
                f"Find the {piece_name}.",
                f"Locate the {piece_name} for me.",
                f"Show me the {piece_name}.",
                f"Where exactly is the {piece_name}?",
                
                # Coordinate & Position focused
                f"What square is the {piece_name} on?",
                f"What is the coordinate of the {piece_name}?",
                f"Which square currently holds the {piece_name}?",
                f"Tell me the position of the {piece_name}.",
                f"Identify the location of the {piece_name}.",
                
                # Conversational & Polite
                f"Can you tell me what square the {piece_name} is on?",
                f"Could you point out the {piece_name}?",
                f"Please tell me where the {piece_name} is located.",
                f"I need help finding the {piece_name}.",
                
                # Vision/Observation framed
                f"Do you see the {piece_name}?",
                f"Where do you see the {piece_name} on the board?",
                f"Scan the board and find the {piece_name}."
            ]
            intent    = f"[Intent] The user is requesting the exact board coordinate of the {piece_name}."
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
        # Plural path (2+ pieces — possible after promotion)
        # ------------------------------------------------------------------
        else:
            both_all  = "both" if len(sq_names) == 2 else "all"
            questions = [
                # Direct & Simple
                f"Where are my {piece_name}s?",
                f"Find the {piece_name}s.",
                f"Locate the {piece_name}s for me.",
                f"Show me where {both_all} {piece_name}s are.",
                
                # Coordinate & Position focused
                f"What squares are the {piece_name}s on?",
                f"What are the coordinates of the {piece_name}s?",
                f"Which squares hold the {piece_name}s?",
                f"Tell me the positions of {both_all} {piece_name}s.",
                f"Identify the locations of the {piece_name}s.",
                
                # Conversational & Polite
                f"Can you show me where {both_all} {piece_name}s are?",
                f"Could you point out the {piece_name}s?",
                f"Please tell me where the {piece_name}s are located.",
                f"I need help finding my {piece_name}s.",
                
                # Vision/Observation framed
                f"Do you see {both_all} the {piece_name}s?",
                f"Where do you see the {piece_name}s on the board?",
                f"Scan the board and find {both_all} {piece_name}s."
            ]
            intent    = f"[Intent] The user is requesting the exact board coordinates of the {piece_name}s."
            perception = (
                f"[Perception] Focusing perception across the 64 <board_state> visual tokens "
                f"to identify the feature signatures of the {piece_name}s."
            )

            sq_str   = self._join_squares(sq_names)
            hl_tags  = "".join(f'<highlight square="{s}" color="info"/> ' for s in sq_names)
            cmds     = self._join_squares(
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
    # QUERY TYPE 2: Empty / Specific Square Query  ("Is there a piece on e4?")
    # =========================================================================
    def generate_square_query(self, board: chess.Board) -> Dict:
        sq      = random.randint(0, 63)
        sq_name = chess.square_name(sq)
        piece   = board.piece_at(sq)

        questions = [
            f"Is there a piece on {sq_name}?",
            f"What piece is sitting on {sq_name}?",
            f"Check {sq_name} and tell me what is there.",
        ]

        intent     = f"[Intent] The user is inquiring about the occupancy of the {sq_name} square."
        perception = (
            f"[Perception] Focusing perception on the <board_state> token corresponding to {sq_name}."
        )

        if piece is None:
            resolution = (
                f"[Resolution] The token for {sq_name} resolves entirely as an empty square feature."
            )
            plan = (
                f'[Pedagogical Plan] The square is empty. To visually ground the conversation, '
                f'I will use <circle square="{sq_name}" color="info"/> to draw their eye to the '
                f"specific square, and state clearly that no piece is there."
            )
            answer = f'<circle square="{sq_name}" color="info"/> The {sq_name} square is completely empty.'
        else:
            piece_name = self._piece_str(piece.piece_type, piece.color)
            resolution = (
                f"[Resolution] The token for {sq_name} resolves to the feature signature "
                f"of a {piece_name}."
            )
            plan = (
                f"[Pedagogical Plan] The square is occupied. I will highlight the square using "
                f'<highlight square="{sq_name}" color="info"/> and inform the user of the piece.'
            )
            answer = (
                f'<highlight square="{sq_name}" color="info"/> '
                f"There is a {piece_name} currently sitting on {sq_name}."
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
            (chess.square_name(sq), chess.square_rank(sq) + 1)   # rank is 1-indexed
            for sq in pawns
        ]

        adv_rank     = max(r[1] for r in sq_info) if color == chess.WHITE \
                       else min(r[1] for r in sq_info)
        advanced_sqs = [r[0] for r in sq_info if r[1] == adv_rank]
        other_sqs    = [r for r in sq_info if r[0] not in advanced_sqs]

        all_sq_names = self._join_squares([r[0] for r in sq_info])
        adv_sq_names = self._join_squares(advanced_sqs)

        # Singular / plural helpers shared across resolution and answer
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
            # All pawns share the same rank
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
                # Higher rank number = more advanced for White
                comparison = (
                    f"{adv_str} > {other_str}. "
                    f"Therefore, {adv_sq_names} {is_are} the most advanced."
                )
            else:
                # Lower rank number = more advanced for Black (closer to rank 1)
                comparison = (
                    f"For Black, advancement is toward rank 1. "
                    f"{adv_str} {is_are} closest to rank 1 among {all_sq_names}. "
                    f"Therefore, {adv_sq_names} {is_are} the most advanced."
                )

            resolution = (
                f"[Resolution] {color_str} pawn representations resolve at {all_sq_names}. "
                f"Rank comparison: {comparison}"
            )

        # ------------------------------------------------------------------
        # Pedagogical plan and answer
        # ------------------------------------------------------------------
        hl_plan  = self._join_squares(
            [f'<highlight square="{sq}" color="key"/>' for sq in advanced_sqs]
        )
        plan     = (
            f"[Pedagogical Plan] I will highlight the advanced {pawn_s} using {hl_plan}. "
            f"In my explanation, I will note the rank to reinforce board geography."
        )

        hl_tags  = "".join(
            f'<highlight square="{sq}" color="key"/> ' for sq in advanced_sqs
        )
        answer   = (
            f"{hl_tags}Your most advanced {color_str} {pawn_s} {is_are} located on "
            f"{adv_sq_names}, sitting on the {self._ordinal(adv_rank)} rank."
        )

        cot = f"{intent}\n{perception}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer)

    # =========================================================================
    # BATCH GENERATION
    # =========================================================================
    def generate_dataset(self, fens: List[str]) -> None:
        """Run all three query generators over a list of FENs and write JSONL."""
        count = 0
        with open(self.output_file, "w") as f:
            for fen in fens:
                board      = chess.Board(fen)
                query_type = random.choice(["direct", "square", "advanced_pawn"])

                if query_type == "direct":
                    record = self.generate_direct_query(board)
                elif query_type == "square":
                    record = self.generate_square_query(board)
                else:
                    record = self.generate_advanced_pawn_query(board)

                f.write(json.dumps(record) + "\n")
                count += 1

        print(f"✅ Generated {count} piece placement samples → {self.output_file}")