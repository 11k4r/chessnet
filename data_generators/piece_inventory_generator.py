"""
piece_inventory_generator.py
────────────────────────────
Offline synthetic data generator for global piece inventory tasks.

Produces task types to build robust board-scanning, categorization, 
and spatial enumeration capabilities for the model.

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think>. Prompts are built via PromptBuilder.

Usage
-----
    python piece_inventory_generator.py                      # single example
    python piece_inventory_generator.py --n 50000 --out data/piece_inventory.jsonl
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
    chess.KING:   "king",
    chess.QUEEN:  "queen",
    chess.ROOK:   "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
    chess.PAWN:   "pawn",
}

# The order we want to scan and report pieces in (highest value to lowest)
_PIECE_ORDER = [
    chess.KING, chess.QUEEN, chess.ROOK, 
    chess.BISHOP, chess.KNIGHT, chess.PAWN
]

def _pname(piece_type: int) -> str:
    return _PIECE_NAME[piece_type]

def _color(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"

def _grammatical_join(items: List[str]) -> str:
    """Grammatically join a list of strings (e.g., 'A, B, and C')."""
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ══════════════════════════════════════════════════════════════════════════════
# Question Templates
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {color} = "White"
PIECE_INVENTORY_Q: List[str] = [
    # Formal & Descriptive
    "List all of {color}'s pieces and their squares.",
    "What pieces does {color} have on the board, and where are they?",
    "Give me a complete inventory of {color}'s pieces.",
    "Enumerate every {color} piece and its current square.",
    "Where are all of {color}'s pieces right now?",
    "List every piece belonging to {color} along with its location.",
    "What are {color}'s remaining pieces and their positions?",
    "Provide a comprehensive breakdown of {color}'s current piece locations.",
    "Identify the positions of all remaining pieces controlled by {color}.",
    "Detail the current board state regarding {color}'s forces.",
    "What is the exact distribution of {color}'s army?",
    "State the coordinates of every piece belonging to {color}.",
    "Catalog all {color} pieces currently in play.",
    "What is the positional layout of {color}'s remaining units?",
    "Please provide the full setup of {color}'s pieces at this moment.",

    # Casual & Conversational
    "where are all {color}'s pieces?",
    "show me all of {color}'s pieces",
    "what pieces does {color} still have?",
    "list {color} pieces please",
    "tell me all {color} pieces and where they are",
    "give me {color}'s piece list",
    "what does {color} have left on the board?",
    "What's {color}'s army looking like?",
    "Can you tell me where {color}'s stuff is?",
    "Give me a rundown of {color}'s pieces.",
    "Where is all of {color}'s army?",
    "What's the status of {color}'s pieces?",
    "Which squares have {color} pieces on them?",
    "Can you map out {color}'s pieces for me?",
    "What does {color} currently have in play?",

    # Short & Terse
    "{color} piece inventory.",
    "All {color} pieces.",
    "{color} pieces and squares.",
    "Positions of {color} pieces.",
    "{color}'s remaining pieces.",
    "List {color} army.",
    "{color} layout.",
    "Where are {color} pieces?",

    # Typos, Slang, & Informal
    "were are all {color}s pieces?",
    "list al {color} peices and ther sqaures",
    "waht pieces does {color} hav?",
    "{color} pieces pls?",
    "waht does {color} stil have on bord?",
    "invetory of {color} pieces?",
    "wht does {color} got left?",
    "list {color} army rn",
    "wer are {color}s guys?",
    "sho me {color} pieces",
    "all {color} peices?",
    "wut does {color} have left on the board?",
    "gimme {color} piece locations",
    "tel me what {color} is playing with",
    "whre are {color}s units",
    "is {color} missing pieces? what do they have?",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer Templates
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {thought}, {color}, {inventory}
PIECE_INVENTORY_A: List[str] = [
    # Formal & Direct
    "<think>\n{thought}\n</think>\n{color} has the following pieces: {inventory}.",
    "<think>\n{thought}\n</think>\n{color}'s pieces: {inventory}.",
    "<think>\n{thought}\n</think>\nHere is {color}'s complete piece inventory: {inventory}.",
    "<think>\n{thought}\n</think>\n{color} currently has: {inventory}.",
    "<think>\n{thought}\n</think>\nThe current positions of {color}'s pieces are: {inventory}.",
    "<think>\n{thought}\n</think>\nAn enumeration of {color}'s remaining pieces yields: {inventory}.",
    "<think>\n{thought}\n</think>\nHere are the locations of all {color} pieces: {inventory}.",
    "<think>\n{thought}\n</think>\nThe {color} forces are distributed as follows: {inventory}.",
    "<think>\n{thought}\n</think>\nThis is the complete list of {color}'s active pieces and their squares: {inventory}.",
    "<think>\n{thought}\n</think>\n{color}'s remaining material on the board consists of: {inventory}.",

    # Casual & Conversational
    "<think>\n{thought}\n</think>\n{color} is currently playing with: {inventory}.",
    "<think>\n{thought}\n</think>\nRight now, {color} has these pieces on the board: {inventory}.",
    "<think>\n{thought}\n</think>\nHere's what {color} has left: {inventory}.",
    "<think>\n{thought}\n</think>\nThese are the pieces {color} still has in play: {inventory}.",
    "<think>\n{thought}\n</think>\nTaking a look, {color}'s army consists of: {inventory}.",
    "<think>\n{thought}\n</think>\nYou can find {color}'s pieces here: {inventory}.",
    "<think>\n{thought}\n</think>\nHere is a quick rundown of {color}'s board presence: {inventory}."
]


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought Generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_inventory_data(board: chess.Board, color: chess.Color) -> Tuple[str, str]:
    """
    Generate CoT and the formatted natural language inventory string.
    """
    c_name = _color(color)
    
    parts = [
        f"The user is requesting a full inventory of all {c_name} pieces currently on the board.",
        f"I will systematically scan the <board_state> tokens and categorize the findings by piece type."
    ]

    # 1. Gather all pieces into a dictionary grouped by piece type
    inventory_dict: Dict[int, List[str]] = {pt: [] for pt in _PIECE_ORDER}
    total_pieces = 0
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            sq_name = chess.square_name(sq)
            inventory_dict[piece.piece_type].append(sq_name)
            total_pieces += 1

    parts.append(f"Scan complete. Found a total of {total_pieces} {c_name} piece{'s' if total_pieces != 1 else ''}.")

    # 2. Build the CoT breakdown and the final narrative string
    inventory_phrases = []
    
    for pt in _PIECE_ORDER:
        sqs = inventory_dict[pt]
        if not sqs:
            continue
            
        p_name = _pname(pt)
        count = len(sqs)
        sqs_joined = _grammatical_join(sqs)
        
        # Add to CoT
        parts.append(f"- {p_name.capitalize()}s ({count}): {sqs_joined}.")
        
        # Add to the final text string formulation
        if count == 1:
            inventory_phrases.append(f"1 {p_name} on {sqs_joined}")
        else:
            inventory_phrases.append(f"{count} {p_name}s on {sqs_joined}")

    parts.append("Compiling the final formatted inventory string.")
    
    # E.g., "1 king on e1, 1 queen on d1, and 2 rooks on a1 and h1"
    final_inventory_string = _grammatical_join(inventory_phrases)

    return " ".join(parts), final_inventory_string


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

def sample_piece_inventory(board: chess.Board) -> Optional[Dict]:
    """Task: Detail every piece a specific color has on the board."""
    # Randomly pick White or Black to query
    color = random.choice([chess.WHITE, chess.BLACK])
    c_name = _color(color)
    
    q = random.choice(PIECE_INVENTORY_Q).format(color=c_name)
    thought_str, inventory_str = _generate_inventory_data(board, color)

    fmt_kw = dict(
        thought=thought_str, 
        color=c_name, 
        inventory=inventory_str
    )

    a = random.choice(PIECE_INVENTORY_A).format(**fmt_kw)

    return _package(board, q, a)


# ══════════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLERS = [
    sample_piece_inventory,
]

def generate_sample(board: chess.Board) -> Optional[Dict]:
    return random.choice(_SAMPLERS)(board)

def _random_board() -> chess.Board:
    board = chess.Board()
    # Randomize the game phase (from opening to deep endgame)
    for _ in range(random.randint(4, 80)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Piece Inventory data generator")
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

if __name__ == "__main__":
    _run_cli()