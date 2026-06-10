"""
piece_on_square_generator.py
────────────────────────────
Offline synthetic data generator for direct square-querying tasks.

Produces two task types to build strict visual grounding and coordinate 
resolution capabilities:

  OCCUPIED_SQUARE — "What is on e4?" (Returns the piece and color)
  EMPTY_SQUARE    — "What is on d5?" (Returns empty)

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think>. Prompts are built via PromptBuilder.

Usage
-----
    python piece_on_square_generator.py                      # single example
    python piece_on_square_generator.py --n 50000 --out data/piece_on_square.jsonl
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


# ══════════════════════════════════════════════════════════════════════════════
# Question Templates
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {sq} = "e4"
PIECE_ON_SQUARE_Q: List[str] = [
    # Formal & Descriptive
    "What piece is on {sq}?",
    "Which piece occupies the square {sq}?",
    "What is on {sq}?",
    "Is there a piece on {sq}, and if so, what is it?",
    "Identify the piece on {sq}.",
    "Tell me what piece sits on {sq}.",
    "What piece can be found on {sq}?",
    "Describe what is occupying {sq}.",
    "What piece is currently positioned on {sq}?",
    "Please identify the occupant of the square {sq}.",
    "State the piece located at {sq}.",
    "Is {sq} currently occupied by any piece?",
    "Determine the piece residing on {sq}.",
    "What type of piece is occupying {sq}?",
    "Can you tell me what piece is placed on {sq}?",
    "Verify the presence of any piece on {sq}.",
    "Which color and piece type currently hold the square {sq}?",
    "Please check the board and tell me what is on {sq}.",

    # Casual & Conversational
    "what's on {sq}?",
    "what piece is sitting on {sq}?",
    "anything on {sq}?",
    "what's sitting at {sq}?",
    "which piece is at {sq}?",
    "tell me whats on {sq}",
    "hey whats on {sq}?",
    "Who is on {sq} right now?",
    "Any piece at {sq}?",
    "What do we have on {sq}?",
    "Whose piece is on {sq}?",
    "Is there anything hanging out on {sq}?",
    "Check {sq} for me, what's there?",
    "Do you know what piece is on {sq}?",
    "Take a look at {sq}, what piece is that?",
    "Is {sq} taken by anything?",
    "What's occupying {sq} at the moment?",

    # Short & Terse
    "{sq} piece?",
    "What is on {sq}?",
    "Occupant of {sq}?",
    "Check {sq}.",
    "Piece at {sq}.",
    "Who is at {sq}?",
    "{sq} contents.",
    "Status of {sq}?",

    # Typos, Slang, & Informal
    "waht pice is on {sq}?",
    "wats on sqaure {sq}?",
    "which peice is on {sq}?",
    "wuts at {sq}?",
    "wats occypying {sq}?",
    "piece on {sq}??",
    "is there a piece at {sq}?",
    "whos at {sq}?",
    "peice on {sq} pls",
    "tell me wut is at {sq}",
    "pice on {sq}",
    "wht is chillin on {sq}?",
    "anythin on {sq} rn?",
    "wat is on {sq}",
    "chk {sq} for a piece",
    "is {sq} empty or nah?",
    "whose guy is on {sq}?",
    "sumthing on {sq}?",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer Body Templates (Wrapped in CoT during assembly)
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {sq} = "e4", {color} = "White", {piece} = "knight"
PIECE_ON_SQUARE_A_OCCUPIED: List[str] = [
    # Formal & Direct
    "{sq} is occupied by a {color} {piece}.",
    "There is a {color} {piece} on {sq}.",
    "The {color} {piece} sits on {sq}.",
    "On {sq} you will find a {color} {piece}.",
    "{sq} has a {color} {piece} on it.",
    "The square {sq} is currently occupied by a {color} {piece}.",
    "A {color} {piece} is positioned on {sq}.",
    "You will find the {color} {piece} located at {sq}.",
    "The piece residing on {sq} is a {color} {piece}.",
    "According to the current position, a {color} {piece} occupies {sq}.",

    # Casual & Conversational
    "Looks like a {color} {piece} is on {sq}.",
    "That's a {color} {piece} sitting on {sq}.",
    "You've got a {color} {piece} right there on {sq}.",
    "It's a {color} {piece} on {sq}.",
    "Right now, {sq} belongs to a {color} {piece}.",
    "Checking the board... yep, there's a {color} {piece} on {sq}.",
    "The {color} {piece} is holding down {sq}."
]

# Placeholders: {sq} = "e4"
PIECE_ON_SQUARE_A_EMPTY: List[str] = [
    # Formal & Direct
    "{sq} is empty.",
    "There is no piece on {sq}.",
    "The square {sq} is unoccupied.",
    "Nothing is on {sq}.",
    "{sq} has no piece on it.",
    "{sq} is completely clear of any pieces.",
    "No piece currently resides on {sq}.",
    "The square {sq} remains vacant at this time.",
    "There are zero pieces located on {sq}.",
    "An inspection of {sq} reveals it is empty.",

    # Casual & Conversational
    "Nothing to see on {sq}, it's empty.",
    "{sq} is completely clear.",
    "Looks like {sq} is vacant right now.",
    "There's nothing sitting on {sq}.",
    "You won't find any pieces on {sq}.",
    "{sq} is wide open.",
    "Nobody is on {sq} at the moment."
]


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought Generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_thought(board: chess.Board, sq: int) -> Tuple[str, Optional[chess.Piece]]:
    """
    Generate CoT narrating the visual indexing into the specified square.
    Returns the CoT string and the piece found (if any).
    """
    sq_name = chess.square_name(sq)
    piece = board.piece_at(sq)

    parts = [
        f"The user asks for the contents of square {sq_name}.",
        f"Indexing into the <board_state> visual tokens to inspect the coordinate {sq_name}."
    ]

    if piece:
        c_name = _color(piece.color)
        p_name = _pname(piece.piece_type)
        parts.append(f"The token at {sq_name} encodes a {c_name} {p_name}.")
        parts.append(f"Confirmed: {sq_name} is occupied.")
    else:
        parts.append(f"The token at {sq_name} is empty.")
        parts.append(f"Confirmed: no piece is present on this square.")

    return " ".join(parts), piece


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

def sample_occupied_square(board: chess.Board) -> Optional[Dict]:
    """Task: Query a square that currently holds a piece."""
    occupied_squares = [sq for sq in chess.SQUARES if board.piece_at(sq)]
    if not occupied_squares:
        return None

    sq = random.choice(occupied_squares)
    sq_name = chess.square_name(sq)

    q = random.choice(PIECE_ON_SQUARE_Q).format(sq=sq_name)
    thought_str, piece = _generate_thought(board, sq)

    c_name = _color(piece.color)
    p_name = _pname(piece.piece_type)

    ans_body = random.choice(PIECE_ON_SQUARE_A_OCCUPIED).format(
        sq=sq_name, 
        color=c_name, 
        piece=p_name
    )
    
    a = f"<think>\n{thought_str}\n</think>\n{ans_body}"

    return _package(board, q, a)


def sample_empty_square(board: chess.Board) -> Optional[Dict]:
    """Task: Query a square that is currently empty."""
    empty_squares = [sq for sq in chess.SQUARES if not board.piece_at(sq)]
    if not empty_squares:
        return None  # Highly unlikely unless the board is completely full

    sq = random.choice(empty_squares)
    sq_name = chess.square_name(sq)

    q = random.choice(PIECE_ON_SQUARE_Q).format(sq=sq_name)
    thought_str, _ = _generate_thought(board, sq)

    ans_body = random.choice(PIECE_ON_SQUARE_A_EMPTY).format(sq=sq_name)
    
    a = f"<think>\n{thought_str}\n</think>\n{ans_body}"

    return _package(board, q, a)


# ══════════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLERS = [
    sample_occupied_square,
    sample_occupied_square, # Weighted to test occupied squares more often
    sample_empty_square,
]

def generate_sample(board: chess.Board) -> Optional[Dict]:
    return random.choice(_SAMPLERS)(board)

def _random_board() -> chess.Board:
    board = chess.Board()
    for _ in range(random.randint(4, 60)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Piece On Square data generator")
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