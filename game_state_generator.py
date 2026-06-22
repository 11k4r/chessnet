import json
import random
import chess
from typing import List, Dict

from prompt_builder import PromptBuilder, ELO_TIERS, CoachMode


class GameStateGenerator:
    """
    Generates training data for chess game state evaluation.
    Bifurcates reasoning into two distinct pathways to train the model on WHEN
    to use vision and WHEN to read text metadata.

    Path A: Metadata (Turn, Counters, Castling)
    [Intent] [Source Check] [State Extraction] [Resolution] [Pedagogical Plan]
    -> Bypasses vision entirely. Explicitly flags that the query is answerable
       via the FEN string in the [ENVIRONMENT] block. [Perception] is
       deliberately NOT used here -- that label is reserved exclusively for
       genuine <board_state> visual token consultation everywhere else in
       this project. Reusing it for text-parsing would blur the very
       distinction this generator exists to teach.

    Path B/C: Vision (Checks, Terminal States)
    [Intent] [Perception] [Trajectory Evaluation] [Resolution] [Pedagogical Plan]
    -> Explicitly scans the 64 <board_state> visual tokens.

    Note on terminal-state detection: every Board in this generator (and
    every other generator in this project) is constructed fresh from a
    snapshot FEN via chess.Board(fen), never via accumulated board.push()
    history. Threefold/fivefold repetition detection fundamentally requires
    move-stack history to compare prior positions, so those terminations
    can structurally never trigger here -- only Checkmate, Stalemate,
    Insufficient Material, the automatic 75-move rule, and the claimable
    50-move rule (via claim_draw=True) are reachable.
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
        self, fen: str, question: str, cot: str, answer: str, elo_tier: str, mode: str
    ) -> Dict:
        return {
            "fen": fen,
            "elo_tier": elo_tier,
            "mode": mode,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n{answer}"},
            ],
        }

    def _maybe_add_ask(self, mode: str, text: str) -> str:
        """Coach mode only: a trailing engagement hook, used only where it's genuinely motivated by the answer."""
        if mode == CoachMode.COACH:
            return f" <ask>{text}</ask>"
        return ""

    # =========================================================================
    # QUERY TYPE 1: Turn Query (Metadata)
    # =========================================================================
    def generate_turn_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        questions = [
            "Whose move is it?",
            "Who is to play?",
            "Is it White or Black to move?",
            "Whose turn is it right now?",
        ]

        active_color = "White" if board.turn == chess.WHITE else "Black"
        fen_char = "w" if board.turn == chess.WHITE else "b"

        intent = "[Intent] The user is asking whose turn it is to move."
        source_check = (
            "[Source Check] This is answerable directly from FEN text in [ENVIRONMENT] "
            "-- no <board_state> consultation needed."
        )
        state_ext  = f"[State Extraction] The active color field in the FEN is '{fen_char}'."
        resolution = f"[Resolution] It is {active_color}'s turn to move."
        plan       = "[Pedagogical Plan] I will state the active color clearly. No UI highlights are necessary."

        # No <ask> here -- "whose turn is it" is a basic orientation question,
        # not a moment with a genuine follow-up to invite.
        answer = f"It is currently {active_color}'s turn to move."

        cot = f"{intent}\n{source_check}\n{state_ext}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer, elo_tier, mode)

    # =========================================================================
    # QUERY TYPE 2: Move Counters Query (Metadata)
    # =========================================================================
    def generate_counters_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        questions = [
            "What move number are we on?",
            "How many moves have been played?",
            "Are we close to the 50-move rule draw?",
            "What is the halfmove clock at?",
        ]

        intent = "[Intent] The user is asking about the current move number or the 50-move rule status."
        source_check = (
            "[Source Check] This is answerable directly from FEN text in [ENVIRONMENT] "
            "-- no <board_state> consultation needed."
        )

        halfmove = board.halfmove_clock
        fullmove = board.fullmove_number

        state_ext  = (
            f"[State Extraction] Parsing the FEN string: the halfmove clock is {halfmove} "
            f"and the fullmove number is {fullmove}."
        )
        resolution = (
            f"[Resolution] We are on move {fullmove}, with {halfmove} halfmoves since the "
            f"last pawn advance or capture."
        )
        plan = "[Pedagogical Plan] I will state the move number and explain the halfmove clock status."

        if elo_tier == "beginner":
            answer = (
                f"We are currently on move {fullmove}. The 'halfmove clock' (which tracks "
                f"progress toward a 50-move automatic draw) is at {halfmove}. This resets to 0 "
                f"every time a pawn moves or a piece is captured."
            )
        else:
            answer = f"We are on move {fullmove}. The halfmove clock is currently at {halfmove}/100."

        cot = f"{intent}\n{source_check}\n{state_ext}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer, elo_tier, mode)

    # =========================================================================
    # QUERY TYPE 3: Castling Rights Query (Metadata)
    # =========================================================================
    def generate_castling_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        color = random.choice([chess.WHITE, chess.BLACK])
        c_str = "White" if color == chess.WHITE else "Black"

        # Every question is color-specific, matching the color-specific answer.
        questions = [
            f"Can {c_str} still castle?",
            f"Has {c_str} lost the right to castle?",
            f"What are {c_str}'s castling rights?",
            f"Is castling still legal for {c_str} in this game?",
        ]

        intent = f"[Intent] The user is inquiring about absolute castling legality for {c_str} based on game history."
        source_check = (
            "[Source Check] This is answerable directly from FEN text in [ENVIRONMENT] "
            "-- no <board_state> consultation needed."
        )

        fen_castling = board.fen().split()[2]
        state_ext = f"[State Extraction] The FEN castling field is '{fen_castling}'."

        kingside  = board.has_kingside_castling_rights(color)
        queenside = board.has_queenside_castling_rights(color)

        if kingside and queenside:
            res_text = f"{c_str} retains both kingside and queenside castling rights."
            ans_text = f"Yes, {c_str} still has the right to castle on either the kingside or the queenside."
        elif kingside:
            res_text = f"{c_str} retains kingside rights, but has lost queenside rights."
            ans_text = f"{c_str} can still castle kingside, but has lost the right to castle queenside."
        elif queenside:
            res_text = f"{c_str} retains queenside rights, but has lost kingside rights."
            ans_text = f"{c_str} can still castle queenside, but has lost the right to castle kingside."
        else:
            res_text = f"{c_str} has completely lost the right to castle."
            ans_text = f"No, {c_str} has completely lost the right to castle for the rest of the game."

        resolution = f"[Resolution] {res_text}"
        plan = "[Pedagogical Plan] I will explicitly state which castling directions remain legally available based on history."

        if elo_tier == "beginner":
            ans_text += (
                " (Remember, you permanently lose the right to castle if the King or the "
                "respective Rook has previously moved!)."
            )

        cot = f"{intent}\n{source_check}\n{state_ext}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, ans_text, elo_tier, mode)

    # =========================================================================
    # QUERY TYPE 4: Check Identification (Vision)
    # =========================================================================
    def generate_check_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        active_color = "White" if board.turn == chess.WHITE else "Black"
        king_sq      = board.king(board.turn)
        king_sq_name = chess.square_name(king_sq) if king_sq is not None else "unknown"

        questions = [
            f"Is {active_color} in check?",
            "Am I in check right now?",
            "Is the King under attack?",
            "What piece is giving check?",
        ]

        intent = f"[Intent] The user is asking if the {active_color} King is currently in check, and by which piece."
        perception = (
            f"[Perception] Scanning the 64 <board_state> visual tokens to locate the "
            f"{active_color} King and identify incoming attack signatures."
        )

        checkers = board.checkers()

        if not checkers:
            trajectory = (
                f"[Trajectory Evaluation] The {active_color} King is on {king_sq_name}. "
                f"Tracing enemy attack rays reveals no pieces controlling this square."
            )
            resolution = f"[Resolution] {active_color} is not in check."
            plan       = "[Pedagogical Plan] I will confirm the King is safe. No highlights are needed."
            answer     = f"No, the {active_color} King is perfectly safe right now."
        else:
            checker_list   = list(checkers)
            names_and_sqs  = [
                f"{self._piece_str(board.piece_at(sq).piece_type, board.piece_at(sq).color)} on {chess.square_name(sq)}"
                for sq in checker_list
            ]
            check_str = self._join_squares(names_and_sqs)

            trajectory = (
                f"[Trajectory Evaluation] The {active_color} King is on {king_sq_name}. "
                f"Tracing attack rays reveals direct threats from: {check_str}."
            )
            resolution = f"[Resolution] {active_color} is in check from the {check_str}."

            hl_tags  = "".join(f'<circle square="{chess.square_name(sq)}" color="danger"/> ' for sq in checker_list)
            hl_tags += f'<highlight square="{king_sq_name}" color="bad"/> '

            plan = (
                "[Pedagogical Plan] I will use <circle> in 'danger' (red) on the checking piece(s) "
                "and highlight the King's square in 'bad' to visually map the immediate threat."
            )

            answer  = f"{hl_tags}Yes, {active_color} is in check! It's coming from the {check_str}."
            answer += self._maybe_add_ask(mode, "How would you get out of this check?")

        cot = f"{intent}\n{perception}\n{trajectory}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer, elo_tier, mode)

    # =========================================================================
    # QUERY TYPE 5: Terminal State (Vision/Complex)
    # =========================================================================
    def generate_terminal_query(
        self, board: chess.Board, elo_tier: str = "club", mode: str = "coach"
    ) -> Dict:
        questions = [
            "Is the game over?",
            "Is this checkmate?",
            "Is it a stalemate or can I move?",
            "Did someone win?",
        ]

        active_color = "White" if board.turn == chess.WHITE else "Black"
        # claim_draw=True enables detection of the claimable 50-move rule
        # (purely a halfmove_clock check, no history needed). Repetition-based
        # draws remain structurally unreachable -- see class docstring.
        outcome = board.outcome(claim_draw=True)

        intent = "[Intent] The user is asking if the current position is a terminal state (Checkmate, Stalemate, or Draw)."
        perception = (
            "[Perception] Scanning the 64 <board_state> visual tokens to assess King safety "
            "and overall piece mobility for the active color."
        )

        if outcome is None:
            trajectory = (
                "[Trajectory Evaluation] Evaluating legal moves and king safety: the King is "
                "not mated and valid destinations exist for the active player. The state is non-terminal."
            )
            resolution = "[Resolution] The game is ongoing."
            plan       = "[Pedagogical Plan] I will inform the user the game is still active."
            answer     = "The game is not over yet! There are still legal moves available."

        elif outcome.termination == chess.Termination.CHECKMATE:
            winner_str = "White" if outcome.winner == chess.WHITE else "Black"
            trajectory = (
                f"[Trajectory Evaluation] The {active_color} King is in check, and calculating "
                f"all legal trajectories yields 0 valid moves. The attack cannot be blocked, "
                f"captured, or evaded."
            )
            resolution = f"[Resolution] Checkmate. {winner_str} wins."

            checkers = board.checkers()
            hl_tags  = "".join(f'<circle square="{chess.square_name(sq)}" color="danger"/> ' for sq in checkers)
            king_sq  = chess.square_name(board.king(board.turn))
            hl_tags += f'<highlight square="{king_sq}" color="bad"/> '

            plan = (
                "[Pedagogical Plan] I will highlight the checking pieces and the mated King "
                "in 'danger'/'bad', stating clearly that the game is over."
            )
            answer = (
                f"{hl_tags}Yes, it is Checkmate! {winner_str} has won the game. The "
                f"{active_color} King is attacked and has absolutely no safe squares left."
            )

        elif outcome.termination == chess.Termination.STALEMATE:
            trajectory = (
                f"[Trajectory Evaluation] The {active_color} King is NOT in check, but "
                f"calculating all legal trajectories across all pieces yields 0 valid moves. "
                f"The player is completely boxed in."
            )
            resolution = "[Resolution] Stalemate. The game is drawn."
            plan       = "[Pedagogical Plan] I will state clearly that it is a stalemate and explain the rule."
            answer     = (
                f"The game is a draw by Stalemate. {active_color} is not in check, but has "
                f"absolutely no legal moves left on the board."
            )

        else:
            # Reachable here: INSUFFICIENT_MATERIAL, SEVENTYFIVE_MOVES (automatic),
            # and FIFTY_MOVES (claimable, via claim_draw=True -- a pure halfmove_clock
            # check needing no history). THREEFOLD_REPETITION and FIVEFOLD_REPETITION
            # can never appear: every Board here is built fresh from a snapshot FEN
            # with no move_stack, and repetition detection fundamentally requires
            # comparing against accumulated push() history.
            term_name = outcome.termination.name.replace('_', ' ').lower()
            trajectory = f"[Trajectory Evaluation] Evaluating board state reveals a drawn condition ({term_name})."
            resolution = f"[Resolution] The game is drawn by {term_name}."
            plan       = "[Pedagogical Plan] I will inform the user that the game ended in a draw."
            answer     = f"The game is over. It's a draw by {term_name}."

        cot = f"{intent}\n{perception}\n{trajectory}\n{resolution}\n{plan}"
        return self._format_sharegpt(board.fen(), random.choice(questions), cot, answer, elo_tier, mode)

    # =========================================================================
    # BATCH GENERATION
    # =========================================================================
    def generate_dataset(self, fens: List[str]) -> None:
        """Run all five game state query generators over a list of FENs."""
        query_types = ["turn", "counters", "castling", "check", "terminal"]
        elo_tiers   = list(ELO_TIERS.keys())
        modes       = [CoachMode.COACH, CoachMode.COMMENTATOR, CoachMode.GAME]

        count = 0
        with open(self.output_file, "w") as f:
            for fen in fens:
                board      = chess.Board(fen)
                query_type = random.choice(query_types)
                elo_tier   = random.choice(elo_tiers)
                mode       = random.choice(modes)

                if query_type == "turn":
                    record = self.generate_turn_query(board, elo_tier, mode)
                elif query_type == "counters":
                    record = self.generate_counters_query(board, elo_tier, mode)
                elif query_type == "castling":
                    record = self.generate_castling_query(board, elo_tier, mode)
                elif query_type == "check":
                    record = self.generate_check_query(board, elo_tier, mode)
                else:
                    record = self.generate_terminal_query(board, elo_tier, mode)

                f.write(json.dumps(record) + "\n")
                count += 1

        print(f"✅ Generated {count} game state samples -> {self.output_file}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    sample_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          # Start
        "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",  # Middlegame
        "8/8/8/4k3/8/2K5/1P6/8 w - - 0 1",                                    # Sparse endgame
        "rnbq1rk1/pppp1ppp/5n2/2b5/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQ - 4 7",  # Sicilian
        "7k/8/8/8/8/8/8/R6K w - - 0 1",                                       # Checkmate/Stalemate potential
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 100 60",                          # Triggers FIFTY_MOVES
    ]

    generator = GameStateGenerator(output_file="phase2_game_state.jsonl")
    generator.generate_dataset(sample_fens * 25)
