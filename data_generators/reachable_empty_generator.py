"""
reachable_empty_generator.py
────────────────────────────
Offline synthetic data generator for quiet (non-capture) movement tasks.

Produces three task types to build robust spatial awareness and filtering:

  SPECIFIC    — "Where can the knight on f3 go without capturing?"
  AMBIGUOUS   — "What empty squares can my knight reach?" 
  INVALID     — "Where can the bishop on e4 step?" (wrong piece/empty square)

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think>. Prompts are built via PromptBuilder.

Usage
-----
    python reachable_empty_generator.py                      # single example
    python reachable_empty_generator.py --n 50000 --out data/reachable_empty.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Dict, List, Optional, Tuple

import chess

from config import ChessCoachConfig
from prompt_builder import PromptBuilder

# ── Module-level singletons ───────────────────────────────────────────────────
_config  = ChessCoachConfig()
_builder = PromptBuilder(_config)   # tokenizer=None is fine for offline use

# ── Formatting helpers ────────────────────────────────────────────────────────
_PIECE_NAME = {
    chess.PAWN:   "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK:   "rook",
    chess.QUEEN:  "queen",
    chess.KING:   "king",
}

def _pname(piece_type: int) -> str:
    return _PIECE_NAME[piece_type]

def _color(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"

def _join_targets(targets: List[str]) -> str:
    if not targets:
        return ""
    if len(targets) == 1:
        return targets[0]
    if len(targets) == 2:
        return f"{targets[0]} and {targets[1]}"
    return ", ".join(targets[:-1]) + f", and {targets[-1]}"


# ══════════════════════════════════════════════════════════════════════════════
# Question Templates
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {piece} = "knight", {piece_name} = "knight", {sq} = "f3"
SPECIFIC_Q: List[str] = [
    # Formal & Descriptive
    "Which empty squares can the {piece} on {sq} move to (excluding captures)?",
    "What unoccupied squares can the {piece} at {sq} reach?",
    "List the empty squares available to the {piece} on {sq}.",
    "Where can the {piece} on {sq} move without capturing anything?",
    "Which vacant squares is the {piece} on {sq} able to reach?",
    "What non-capture moves does the {piece} on {sq} have?",
    "List all quiet moves (no captures) for the {piece} on {sq}.",
    "Identify the unoccupied squares that the {piece} on {sq} can legally transition to.",
    "Determine all quiet moves available to the {piece} located at {sq}.",
    "Which squares devoid of pieces can the {piece_name} on {sq} access?",
    "Provide a comprehensive list of non-capture destinations for the {piece} on {sq}.",
    "What are the legal destinations for the {piece} on {sq} that do not involve a capture?",
    "Enumerate the vacant squares accessible by the {piece} currently occupying {sq}.",
    "To which open squares is the {piece} on {sq} permitted to move?",
    "What are the quiet move options for the {piece_name} at {sq}?",

    # Casual & Conversational
    "where can the {piece} on {sq} go without taking anything?",
    "what empty squares can the {piece_name} on {sq} reach?",
    "show me the quiet moves for {piece} on {sq}",
    "where can the {piece_name} on {sq} go without capturing?",
    "which free squares can the {piece} at {sq} land on?",
    "tell me the non-capture moves for the {piece} on {sq}",
    "Where can the {piece} on {sq} step without hitting anyone?",
    "Can the {piece} on {sq} just move without taking a piece?",
    "What open spots can the {piece} on {sq} slide into?",
    "Tell me where the {piece_name} on {sq} can go that's empty.",
    "Are there any free spaces for the {piece} on {sq} to land on?",
    "Where's an open square for my {piece} on {sq}?",
    "If I don't want to capture, where can the {piece} on {sq} go?",
    "Does the {piece} on {sq} have any safe, empty squares to move to?",
    "What are the empty spaces looking like for the {piece} on {sq}?",

    # Short & Terse
    "Quiet moves {piece} {sq}.",
    "Empty squares for {piece} {sq}.",
    "{piece} on {sq} non-captures.",
    "Open spots {piece} {sq}.",
    "{piece_name} {sq} free moves.",
    "No-capture moves {piece} {sq}.",
    "Vacant squares from {sq} for {piece}.",

    # Typos & Informal
    "which emtpy squares can {piece} on {sq} reach?",
    "waht empty sqaures can the {piece_name} on {sq} go to?",
    "quite moves for {piece} on {sq}?",
    "non captur moves for {piece_name} at {sq}?",
    "were can the {piece} go witout capturing on {sq}?",
    "emty squares for {piece} {sq}",
    "wher can {piece} on {sq} go with no capture?",
    "wat quiet moves does {piece} {sq} hav?",
    "list unoccpied sqaures 4 {piece} at {sq}",
    "where can {piece} on {sq} move 2 dat is empty?",
    "free spots 4 {piece_name} on {sq}?",
    "can the {piece} on {sq} just move to a blank square?",
    "shw me empty sqs for {piece} on {sq}",
    "no capture moves 4 {piece} {sq} plz",
    "is there anywhere empty for {piece} at {sq} to go",
    "wut free spaces can {piece} hit from {sq}?",
]

# Placeholders: {piece_name} = "knight"
AMBIGUOUS_Q: List[str] = [
    "What empty squares can my {piece_name} reach?",
    "Where can the {piece_name} go without capturing anything?",
    "List the quiet moves for the {piece_name}.",
    "Are there any free squares for the {piece_name} to land on?",
    "Show me the non-capture moves for my {piece_name}.",
]

# Placeholders: {piece} = wrong piece, {sq} = wrong sq
INVALID_Q: List[str] = [
    "Which empty squares can the {piece} on {sq} move to?",
    "What unoccupied squares can the {piece} at {sq} reach?",
    "List the empty squares available to the {piece} on {sq}.",
    "Where can the {piece} on {sq} move without capturing?",
    "Tell me where the {piece} on {sq} can go that's empty.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer Templates
# ══════════════════════════════════════════════════════════════════════════════

REACHABLE_EMPTY_A: List[str] = [
    # Formal & Direct
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} can move to these empty squares: {targets}.",
    "<think>\n{thought}\n</think>\nWithout capturing, the {piece} at {sq} can reach: {targets}.",
    "<think>\n{thought}\n</think>\nQuiet (non-capture) moves for the {piece} on {sq}: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} can land on the following empty squares: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} can legally move to the following unoccupied squares: {targets}.",
    "<think>\n{thought}\n</think>\nNon-capturing destinations for the {piece} at {sq} include: {targets}.",
    "<think>\n{thought}\n</think>\nThe empty squares accessible to the {piece} on {sq} are: {targets}.",
    "<think>\n{thought}\n</think>\nA complete list of quiet moves for the {piece_name} at {sq} is as follows: {targets}.",
    "<think>\n{thought}\n</think>\nVacant squares within reach of the {piece} on {sq} are: {targets}.",
    "<think>\n{thought}\n</think>\nThe legal, non-capture moves for the {piece} located at {sq} are: {targets}.",

    # Casual & Conversational
    "<think>\n{thought}\n</think>\nYou can move the {piece} on {sq} to these open spots: {targets}.",
    "<think>\n{thought}\n</think>\nThese squares are empty and ready for the {piece} on {sq}: {targets}.",
    "<think>\n{thought}\n</think>\nWithout taking anything, your {piece} on {sq} can slide to: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} has these free spaces available: {targets}.",
    "<think>\n{thought}\n</think>\nHere are the empty squares the {piece} on {sq} can step to: {targets}.",
    "<think>\n{thought}\n</think>\nIf you just want to move into open space, the {piece} on {sq} can go to: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece_name} on {sq} can safely land on these empty squares: {targets}."
]

REACHABLE_EMPTY_A_NONE: List[str] = [
    # Formal & Direct
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} has no quiet moves — it can only capture or is completely blocked.",
    "<think>\n{thought}\n</think>\nThere are no empty squares the {piece} on {sq} can move to without capturing.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} currently has no valid moves to unoccupied squares.",
    "<think>\n{thought}\n</think>\nAll legal moves for the {piece} at {sq} result in a capture; there are no quiet moves.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} cannot access any empty squares at this time.",
    "<think>\n{thought}\n</think>\nZero vacant squares are reachable by the {piece_name} on {sq}.",
    "<think>\n{thought}\n</think>\nNo non-capture moves are available for the {piece} positioned on {sq}.",

    # Casual & Conversational
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} doesn't have any open spaces to move into.",
    "<think>\n{thought}\n</think>\nYou can't make any quiet moves with the {piece} on {sq} right now.",
    "<think>\n{thought}\n</think>\nEvery move for the {piece} on {sq} is blocked or involves a capture.",
    "<think>\n{thought}\n</think>\nNo empty spots available for the {piece} on {sq}.",
    "<think>\n{thought}\n</think>\nThe {piece_name} on {sq} is either stuck or has to take something.",
    "<think>\n{thought}\n</think>\nThere are no free squares for the {piece} on {sq} to slide into.",
    "<think>\n{thought}\n</think>\nUnfortunately, the {piece} on {sq} cannot reach any empty squares."
]


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought Generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_thought(board: chess.Board, origin: int, piece: chess.Piece) -> Tuple[str, List[str]]:
    """
    Generate CoT narrating the search for non-capturing legal moves.
    Returns the CoT string and the deduplicated list of empty target squares.
    """
    origin_name = chess.square_name(origin)
    p_name      = _pname(piece.piece_type)
    c_name      = _color(piece.color)

    parts = [
        f"The user is asking for reachable empty squares (quiet moves) for the {c_name} {p_name} on {origin_name}.",
        f"Verifying the <board_state> token: confirmed {c_name} {p_name} resides at {origin_name}."
    ]

    # 1. Retrieve all legal moves
    legal_moves = [m for m in board.legal_moves if m.from_square == origin]

    if not legal_moves:
        parts.append("The piece is completely immobilized (blocked or absolutely pinned). Zero legal moves available.")
        return " ".join(parts), []

    parts.append(f"Found {len(legal_moves)} total legal move(s). Filtering out any captures.")

    # 2. Filter out captures
    quiet_targets = []
    capture_count = 0

    for m in legal_moves:
        dest_name = chess.square_name(m.to_square)
        if board.is_capture(m):
            if board.is_en_passant(m):
                parts.append(f"Note: Moving to {dest_name} is an en passant capture. Discarding.")
            capture_count += 1
        else:
            quiet_targets.append(dest_name)

    if capture_count > 0:
        parts.append(f"Discarded {capture_count} capture move(s).")

    # 3. Deduplicate (Pawns promoting on an empty square create 4 moves to the same square)
    unique_quiet_targets = sorted(list(set(quiet_targets)))

    if not unique_quiet_targets:
        parts.append("After filtering, no quiet moves to empty squares remain.")
    else:
        parts.append(f"Final quiet destinations: {', '.join(unique_quiet_targets)}.")

    return " ".join(parts), unique_quiet_targets


# ══════════════════════════════════════════════════════════════════════════════
# Packaging
# ══════════════════════════════════════════════════════════════════════════════

def _package(board: chess.Board, raw_q: str, answer: str) -> Dict:
    user_content = _builder.build_text_prompt(
        board           = board,
        user_message    = raw_q,
        candidate_moves = None,
    )
    return {
        "fen": board.fen(),
        "messages": [
            {"role": "system",    "content": _config.system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": answer},
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Sampler Functions
# ══════════════════════════════════════════════════════════════════════════════

def sample_specific_quiet_moves(board: chess.Board) -> Optional[Dict]:
    """SPECIFIC task: What empty squares can piece X on square Y move to?"""
    active_squares = [s for s in chess.SQUARES if (p := board.piece_at(s)) and p.color == board.turn]
    if not active_squares:
        return None

    origin = random.choice(active_squares)
    piece  = board.piece_at(origin)
    sq_name = chess.square_name(origin)
    p_name  = _pname(piece.piece_type)

    q = random.choice(SPECIFIC_Q).format(piece=p_name, piece_name=p_name, sq=sq_name)
    thought_str, targets = _generate_thought(board, origin, piece)

    fmt_kw = dict(thought=thought_str, piece=p_name, piece_name=p_name, sq=sq_name, targets=_join_targets(targets))

    if targets:
        a = random.choice(REACHABLE_EMPTY_A).format(**fmt_kw)
    else:
        a = random.choice(REACHABLE_EMPTY_A_NONE).format(**fmt_kw)

    return _package(board, q, a)


def sample_ambiguous_quiet_moves(board: chess.Board) -> Optional[Dict]:
    """AMBIGUOUS task: What empty squares can my piece type X reach?"""
    piece_type = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING])
    p_name = _pname(piece_type)
    color  = board.turn
    c_name = _color(color)
    
    instances = [s for s in chess.SQUARES if (p := board.piece_at(s)) and p.color == color and p.piece_type == piece_type]
    
    q = random.choice(AMBIGUOUS_Q).format(piece_name=p_name)
    
    thought = [
        f"The user asks for quiet moves (empty squares) for their {p_name}s without specifying a starting square.",
        f"Scanning the <board_state> for {c_name} {p_name}s..."
    ]
    
    # CASE 0: No pieces
    if not instances:
        thought.append("None found.")
        ans_body = f"You do not have any {p_name}s on the board right now."
        ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
        return _package(board, q, ans)

    # CASE 1: Exactly one piece
    if len(instances) == 1:
        origin = instances[0]
        origin_name = chess.square_name(origin)
        piece = board.piece_at(origin)
        
        thought.append(f"Found exactly one on {origin_name}. Evaluating quiet moves.")
        sub_thought, targets = _generate_thought(board, origin, piece)
        thought.append(sub_thought)
        
        if targets:
            ans_body = f"You have one {p_name} on {origin_name}. It can move to these empty squares: {_join_targets(targets)}."
        else:
            ans_body = f"You have one {p_name} on {origin_name}, but it has no empty squares to move to (it is blocked or must capture)."
            
        ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
        return _package(board, q, ans)

    # CASE 2: Multiple pieces
    thought.append(f"Found {len(instances)} {p_name}s. I will evaluate each one separately.")
    
    piece_reports = []
    total_targets = 0
    
    for origin in instances:
        origin_name = chess.square_name(origin)
        piece = board.piece_at(origin)
        sub_thought, targets = _generate_thought(board, origin, piece)
        thought.append(f"[{origin_name}] {sub_thought}")
        
        if targets:
            piece_reports.append(f"The {p_name} on {origin_name} can move to: {_join_targets(targets)}.")
            total_targets += len(targets)
        else:
            piece_reports.append(f"The {p_name} on {origin_name} has no quiet moves.")

    ans_body = f"There are {len(instances)} {c_name} {p_name}s on the board. "
    
    if total_targets == 0:
        ans_body += "None of them can reach an empty square this turn."
    else:
        ans_body += "Here are their options:\n- " + "\n- ".join(piece_reports)

    ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
    return _package(board, q, ans)


def sample_invalid_premise(board: chess.Board) -> Optional[Dict]:
    """INVALID task: Ask about quiet moves for a piece that does not exist on the given square."""
    origin_sq = random.choice(chess.SQUARES)
    actual_p  = board.piece_at(origin_sq)
    
    types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    if actual_p:
        types.remove(actual_p.piece_type)
        
    fake_type = random.choice(types)
    fake_name = _pname(fake_type)
    origin_name = chess.square_name(origin_sq)
    
    q = random.choice(INVALID_Q).format(piece=fake_name, sq=origin_name)
    
    thought = [
        f"The user is asking about empty destinations for a {fake_name} on {origin_name}.",
        f"Step 1: Check the <board_state> visual token at {origin_name}."
    ]
    
    if actual_p:
        actual_name = _pname(actual_p.piece_type)
        actual_col  = _color(actual_p.color)
        thought.append(f"The token encodes a {actual_col} {actual_name}, not a {fake_name}. The premise is flawed.")
        ans_body = f"There is no {fake_name} on {origin_name}. That square currently holds a {actual_col} {actual_name}."
    else:
        thought.append("The token is empty. The user is asking about a ghost piece.")
        ans_body = f"There is no {fake_name} on {origin_name} — that square is completely empty."
        
    ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
    return _package(board, q, ans)


# ══════════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLERS = [
    sample_specific_quiet_moves,
    sample_specific_quiet_moves,
    sample_specific_quiet_moves,
    sample_ambiguous_quiet_moves,
    sample_ambiguous_quiet_moves,
    sample_invalid_premise,
]

def generate_sample(board: chess.Board) -> Optional[Dict]:
    return random.choice(_SAMPLERS)(board)

def _random_board() -> chess.Board:
    board = chess.Board()
    for _ in range(random.randint(4, 40)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Reachable Empty Squares data generator")
    parser.add_argument("--n",   type=int, default=1, help="Number of examples")
    parser.add_argument("--out", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--fen", type=str, default=None, help="Specific FEN to use")
    args = parser.parse_args()

    out_file = open(args.out, "w") if args.out else sys.stdout
    generated, attempts = 0, 0

    while generated < args.n and attempts < args.n * 10:
        attempts += 1
        board = chess.Board(args.fen) if args.fen else _random_board()
        sample = generate_sample(board)
        if sample:
            out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            generated += 1

    if args.out:
        out_file.close()
        print(f"Generated {generated} examples → {args.out}", file=sys.stderr)