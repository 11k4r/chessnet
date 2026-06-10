"""
piece_count_generator.py
────────────────────────
Offline synthetic data generator for global piece counting tasks.

Produces two task types to build robust board-scanning and tallying capabilities:

  ACTIVE_COUNT  — "How many White knights are there?" (count > 0)
  MISSING_COUNT — "How many Black queens are left?"   (count == 0, piece captured)

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think>. Prompts are built via PromptBuilder.

Usage
-----
    python piece_count_generator.py                      # single example
    python piece_count_generator.py --n 50000 --out data/piece_count.jsonl
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

def _join_squares(squares: List[str]) -> str:
    if not squares:
        return ""
    if len(squares) == 1:
        return squares[0]
    if len(squares) == 2:
        return f"{squares[0]} and {squares[1]}"
    return ", ".join(squares[:-1]) + f", and {squares[-1]}"


# ══════════════════════════════════════════════════════════════════════════════
# Question Templates
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {color} = "White", {piece_name} = "knight"
PIECE_COUNT_Q: List[str] = [
    # Formal & Descriptive
    "How many {piece_name}s does {color} have?",
    "What is the count of {color}'s {piece_name}s?",
    "How many {piece_name}s does {color} still have on the board?",
    "Count the number of {color} {piece_name}s.",
    "How many {piece_name}s are left for {color}?",
    "What is {color}'s {piece_name} count?",
    "Determine the total quantity of {color} {piece_name}s currently active.",
    "Provide the total number of {piece_name}s belonging to {color}.",
    "Calculate the remaining {color} {piece_name}s in the current position.",
    "State the exact number of {piece_name}s that {color} controls.",
    "Enumerate the {color} {piece_name}s left on the board.",
    "How many of {color}'s {piece_name}s are still in play?",

    # Casual & Conversational
    "how many {piece_name}s does {color} have left?",
    "how many {color} {piece_name}s are on the board?",
    "count {color}'s {piece_name}s",
    "number of {color} {piece_name}s?",
    "tell me how many {piece_name}s {color} has",
    "What's the total for {color} {piece_name}s?",
    "How many {piece_name}s is {color} playing with right now?",
    "Are there many {color} {piece_name}s left?",
    "Can you check how many {piece_name}s {color} has?",
    "What's the {piece_name} situation for {color}?",
    "I need the count of {color}'s {piece_name}s.",

    # Short & Terse
    "{color} {piece_name} count.",
    "{color}'s {piece_name}s?",
    "Number of {color} {piece_name}s.",
    "Count {color} {piece_name}s.",
    "{color} {piece_name}s left?",
    "How many {color} {piece_name}s?",
    "Total {color} {piece_name}s.",

    # Typos & Informal
    "how mny {piece_name}s does {color} hav?",
    "how many {piece_name}z does {color} stil have?",
    "cont {color} {piece_name}s plz",
    "how meny {color} {piece_name}s r there?",
    "{color} {piece_name} count??",
    "numbr of {piece_name}s 4 {color}?",
    "wat is the count of {color} {piece_name}s",
    "how much {piece_name}s {color} got?",
    "hw many {piece_name}s 4 {color}?",
    "how many {color} {piece_name} are left",
    "cnt {color} {piece_name}s",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer Templates
# ══════════════════════════════════════════════════════════════════════════════

# Used when count > 0
PIECE_COUNT_A_HAS: List[str] = [
    "<think>\n{thought}\n</think>\n{color} has {count} {piece_name}{plural} on the board.",
    "<think>\n{thought}\n</think>\nThere {verb} {count} {color} {piece_name}{plural} remaining.",
    "<think>\n{thought}\n</think>\nRight now, {color} controls {count} {piece_name}{plural}.",
    "<think>\n{thought}\n</think>\nI count {count} {color} {piece_name}{plural}.",
    "<think>\n{thought}\n</think>\n{color} has exactly {count} {piece_name}{plural} left in play.",
]

# Used when count > 0 (includes explicitly listing the squares for added grounding)
PIECE_COUNT_A_HAS_DETAILED: List[str] = [
    "<think>\n{thought}\n</think>\n{color} has {count} {piece_name}{plural}. {they_are} located on: {squares}.",
    "<think>\n{thought}\n</think>\nThere {verb} {count} {color} {piece_name}{plural} left on the board, located at {squares}.",
    "<think>\n{thought}\n</think>\nI see {count} {color} {piece_name}{plural} in this position ({squares}).",
]

# Used when count == 0
PIECE_COUNT_A_ZERO: List[str] = [
    "<think>\n{thought}\n</think>\n{color} has 0 {piece_name}s left on the board.",
    "<think>\n{thought}\n</think>\nThere are no {color} {piece_name}s remaining.",
    "<think>\n{thought}\n</think>\n{color} does not have any {piece_name}s currently in play.",
    "<think>\n{thought}\n</think>\nZero. All of {color}'s {piece_name}s have been captured.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought Generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_thought(board: chess.Board, color: chess.Color, piece_type: int) -> Tuple[str, int, List[str]]:
    """
    Generate CoT, total count, and the exact squares where the pieces are located.
    """
    c_name = _color(color)
    p_name = _pname(piece_type)

    parts = [
        f"The user asks for the total count of {c_name} {p_name}s.",
        f"Scanning the <board_state> visual tokens for matches..."
    ]

    found_squares = []
    
    # Iterate through all squares sequentially to simulate a visual scan
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type == piece_type:
            found_squares.append(chess.square_name(sq))

    count = len(found_squares)

    if count == 0:
        parts.append(f"Scan complete. No {c_name} {p_name}s were detected anywhere on the board.")
    else:
        parts.append(f"Found {count} match{'es' if count > 1 else ''} located at: {', '.join(found_squares)}.")
        parts.append(f"Final tally is {count}.")

    return " ".join(parts), count, found_squares


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

def sample_active_piece_count(board: chess.Board) -> Optional[Dict]:
    """Task: Count a piece type that DOES exist on the board."""
    # Find all piece types currently on the board
    present_pieces = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            present_pieces.append((piece.color, piece.piece_type))
            
    if not present_pieces:
        return None

    color, piece_type = random.choice(list(set(present_pieces)))
    
    return _build_sample(board, color, piece_type)


def sample_missing_piece_count(board: chess.Board) -> Optional[Dict]:
    """Task: Count a piece type that has been completely captured (count = 0)."""
    all_combinations = [(c, pt) for c in [chess.WHITE, chess.BLACK] for pt in chess.PIECE_TYPES]
    
    present_pieces = set()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            present_pieces.add((piece.color, piece.piece_type))
            
    missing_pieces = list(set(all_combinations) - present_pieces)
    
    if not missing_pieces:
        # Very early game, no pieces captured yet
        return None
        
    color, piece_type = random.choice(missing_pieces)
    
    return _build_sample(board, color, piece_type)


def _build_sample(board: chess.Board, color: chess.Color, piece_type: int) -> Dict:
    """Helper to assemble the final Q/A pairs based on the selected target."""
    c_name = _color(color)
    p_name = _pname(piece_type)

    q = random.choice(PIECE_COUNT_Q).format(color=c_name, piece_name=p_name)
    thought_str, count, squares = _generate_thought(board, color, piece_type)

    # Grammar resolution
    plural   = "s" if count != 1 else ""
    verb     = "are" if count != 1 else "is"
    they_are = "They are" if count != 1 else "It is"

    fmt_kw = dict(
        thought=thought_str, 
        color=c_name, 
        piece_name=p_name, 
        count=count, 
        plural=plural, 
        verb=verb, 
        they_are=they_are,
        squares=_join_squares(squares)
    )

    if count == 0:
        a = random.choice(PIECE_COUNT_A_ZERO).format(**fmt_kw)
    else:
        # 50/50 split between giving just the number, or the number + the exact squares
        if random.random() < 0.5:
            a = random.choice(PIECE_COUNT_A_HAS).format(**fmt_kw)
        else:
            a = random.choice(PIECE_COUNT_A_HAS_DETAILED).format(**fmt_kw)

    return _package(board, q, a)


# ══════════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLERS = [
    sample_active_piece_count,
    sample_active_piece_count,
    sample_active_piece_count,
    sample_missing_piece_count, # 25% chance to test 0-count detection
]

def generate_sample(board: chess.Board) -> Optional[Dict]:
    return random.choice(_SAMPLERS)(board)

def _random_board() -> chess.Board:
    board = chess.Board()
    for _ in range(random.randint(4, 60)): # extended to 60 to ensure more pieces are captured
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Piece Count data generator")
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
