"""
capture_targets_generator.py
────────────────────────────
Offline synthetic data generator for capture target identification tasks.

Produces three task types to build robust spatial awareness and threat detection:

  SPECIFIC    — "What can the knight on f3 capture?"    (explicit piece + square)
  AMBIGUOUS   — "What can my knight take right now?"    (piece type only)
  INVALID     — "What can the bishop on e4 take?"       (wrong piece / empty square)

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think>. Prompts are built via PromptBuilder.

Usage
-----
    python capture_targets_generator.py                      # single example
    python capture_targets_generator.py --n 50000 --out data/capture_targets.jsonl
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
    """Grammatically join a list of target strings (e.g., 'rook on d4 and pawn on e5')."""
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
    "What pieces can the {piece} on {sq} capture?",
    "Which enemy pieces can the {piece} at {sq} take?",
    "List all capture moves available to the {piece} on {sq}.",
    "What can the {piece} on {sq} take this turn?",
    "Which pieces are capturable by the {piece} at {sq}?",
    "Enumerate the pieces the {piece} on {sq} can capture.",
    "What enemy pieces are within capture range of the {piece} on {sq}?",
    "Identify the pieces that the {piece} on {sq} is currently attacking.",
    "Determine all valid capture targets for the {piece} located at {sq}.",
    "Provide a list of opponent pieces threatened by the {piece_name} on {sq}.",
    "What are the legal capturing options for the {piece} on {sq}?",
    "Which enemy units are under attack by the {piece} currently at {sq}?",
    "State the pieces that can be legally removed from the board by the {piece} on {sq}.",

    # Casual & Conversational
    "what can the {piece} on {sq} eat?",
    "which pieces can the {piece_name} on {sq} take right now?",
    "show me what the {piece} at {sq} can capture",
    "can the {piece} on {sq} take anything?",
    "what's the {piece_name} on {sq} able to capture?",
    "tell me the captures for the {piece} on {sq}",
    "what enemy pieces does the {piece} on {sq} threaten to take?",
    "Who can the {piece} on {sq} take out?",
    "Is there anything the {piece} on {sq} can kill?",
    "What's in danger from the {piece} on {sq}?",
    "Can my {piece} on {sq} grab any pieces?",
    "Does the {piece} on {sq} have any targets?",
    "What can I take with the {piece_name} on {sq}?",
    "Are there any captures available for my {piece} on {sq}?",

    # Short & Terse
    "Captures for {piece} on {sq}?",
    "{piece} at {sq} captures.",
    "{piece} {sq} targets?",
    "Takeable pieces {piece} {sq}.",
    "What can {piece} {sq} take?",
    "{piece_name} {sq} hit list.",

    # Typos & Informal
    "waht can the {piece} on {sq} caputre?",
    "wich peices can {piece_name} on {sq} take?",
    "captur targets for {piece} at {sq}?",
    "what can {piece_name} on {sq} eat?",
    "what peices is {piece} on {sq} able 2 take?",
    "captures 4 the {piece_name} on {sq}?",
    "wat can the {piece} on {sq} capture",
    "can {piece} on {sq} kill anything?",
    "who is {piece} at {sq} attacking rn?",
    "what can the {piece_name} on {sq} snatch?",
    "gimme captures 4 {piece} at {sq}",
    "wut can {piece} {sq} take",
]

# Placeholders: {piece_name} = "knight"
AMBIGUOUS_Q: List[str] = [
    "What can my {piece_name} capture?",
    "Are there any targets for the {piece_name}?",
    "Which enemy pieces can a {piece_name} take right now?",
    "List the captures available to the {piece_name}.",
    "Can any {piece_name} grab an enemy piece?",
    "What is the {piece_name} attacking?",
    "Show me the {piece_name} captures.",
]

# Placeholders: {piece} = wrong piece, {sq} = wrong sq
INVALID_Q: List[str] = [
    "What can the {piece} on {sq} capture?",
    "Are there any targets for the {piece} at {sq}?",
    "Which enemy pieces can the {piece} on {sq} take right now?",
    "List the captures available to the {piece} on {sq}.",
    "Can the {piece} on {sq} grab an enemy piece?",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer Templates
# ══════════════════════════════════════════════════════════════════════════════

SPECIFIC_A_HAS_TARGETS: List[str] = [
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} can capture the following: {targets}.",
    "<think>\n{thought}\n</think>\nFrom {sq}, the {piece} can legally take: {targets}.",
    "<think>\n{thought}\n</think>\nCapture options for the {piece} on {sq} include: {targets}.",
    "<think>\n{thought}\n</think>\nYou can grab these pieces with the {piece} on {sq}: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} is currently attacking: {targets}.",
    "<think>\n{thought}\n</think>\nYour {piece_name} on {sq} is eyeing these targets: {targets}.",
]

SPECIFIC_A_NO_TARGETS: List[str] = [
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} cannot capture any pieces this turn.",
    "<think>\n{thought}\n</think>\nThere are no legal captures available for the {piece} on {sq}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} isn't currently attacking any enemy pieces.",
    "<think>\n{thought}\n</think>\nYou can't take anything with the {piece} on {sq} right now.",
    "<think>\n{thought}\n</think>\nThere's nothing for the {piece} on {sq} to take this turn.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought Generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_thought(
    board: chess.Board,
    origin: int,
    piece: chess.Piece
) -> Tuple[str, List[str]]:
    """
    Generate CoT and a list of target strings (e.g., "rook on d4").
    """
    origin_name = chess.square_name(origin)
    p_name      = _pname(piece.piece_type)
    c_name      = _color(piece.color)
    
    parts = [
        f"The user is asking about capture targets for the {c_name} {p_name} on {origin_name}.",
        f"Grounding check: verifying the <board_state> token at {origin_name}. Confirmed {c_name} {p_name}.",
        f"Scanning movement rays and jump patterns for enemy pieces."
    ]
    
    # 1. Gather pseudo-legal captures (simulating visual scanning)
    pseudo_captures = [m for m in board.pseudo_legal_moves if m.from_square == origin and board.is_capture(m)]
    
    if not pseudo_captures:
        parts.append(f"No enemy pieces lie along the {p_name}'s attack trajectories. Geometrically, there are zero potential targets.")
        return " ".join(parts), []
        
    parts.append(f"Found {len(pseudo_captures)} potential target(s) geometrically.")
    
    # 2. Filter for legality (simulating pin checks and king safety)
    legal_targets = []
    removed_by_pins = 0
    
    for m in pseudo_captures:
        is_ep = board.is_en_passant(m)
        
        # Determine what piece is actually being captured
        if is_ep:
            # En passant captures a pawn on the rank of the capturing pawn
            ep_rank = chess.square_rank(origin)
            ep_file = chess.square_file(m.to_square)
            target_sq = chess.square(ep_file, ep_rank)
            target_piece_name = "pawn"
            ep_note = " (via en passant)"
        else:
            target_sq = m.to_square
            target_piece = board.piece_at(target_sq)
            target_piece_name = _pname(target_piece.piece_type) if target_piece else "piece"
            ep_note = ""
            
        t_sq_name = chess.square_name(target_sq)
        target_string = f"{target_piece_name} on {t_sq_name}{ep_note}"
        
        parts.append(f"Tracing line to {t_sq_name}: found enemy {target_piece_name}.")
        
        if m in board.legal_moves:
            legal_targets.append(target_string)
            parts.append(f"Move to capture {t_sq_name} is legal (no absolute pin).")
        else:
            removed_by_pins += 1
            parts.append(f"Wait, capturing on {t_sq_name} is illegal. The {p_name} is absolutely pinned to the king, or the move exposes the king to check.")

    # 3. Finalize CoT
    if removed_by_pins > 0:
        parts.append(f"King-safety filter removed {removed_by_pins} target(s).")
        
    if not legal_targets:
        parts.append("After legality checks, zero valid captures remain.")
    else:
        parts.append(f"Final valid targets: {', '.join(legal_targets)}.")
        
    return " ".join(parts), legal_targets


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

def sample_specific_captures(board: chess.Board) -> Optional[Dict]:
    """SPECIFIC task: What can a named piece on a named square capture?"""
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
        a = random.choice(SPECIFIC_A_HAS_TARGETS).format(**fmt_kw)
    else:
        a = random.choice(SPECIFIC_A_NO_TARGETS).format(**fmt_kw)

    return _package(board, q, a)


def sample_ambiguous_captures(board: chess.Board) -> Optional[Dict]:
    """AMBIGUOUS task: What can a piece type (no square specified) capture?"""
    piece_type = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    p_name = _pname(piece_type)
    color  = board.turn
    c_name = _color(color)
    
    instances = [s for s in chess.SQUARES if (p := board.piece_at(s)) and p.color == color and p.piece_type == piece_type]
    
    q = random.choice(AMBIGUOUS_Q).format(piece_name=p_name)
    
    thought = [
        f"The user asks what their {p_name} can capture without specifying a starting square.",
        f"Scanning the <board_state> for {c_name} {p_name}s..."
    ]
    
    # CASE 0: No pieces
    if not instances:
        thought.append("None found.")
        ans_body = f"You do not have any {p_name}s on the board, so there are no captures to make."
        ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
        return _package(board, q, ans)

    # CASE 1: Exactly one piece
    if len(instances) == 1:
        origin = instances[0]
        origin_name = chess.square_name(origin)
        piece = board.piece_at(origin)
        
        thought.append(f"Found exactly one on {origin_name}. Evaluating threats.")
        sub_thought, targets = _generate_thought(board, origin, piece)
        thought.append(sub_thought)
        
        if targets:
            ans_body = f"You have one {p_name} on {origin_name}. It can capture: {_join_targets(targets)}."
        else:
            ans_body = f"You have one {p_name} on {origin_name}, but it cannot capture any pieces right now."
            
        ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
        return _package(board, q, ans)

    # CASE 2: Multiple pieces
    thought.append(f"Found {len(instances)} {p_name}s. I will evaluate the attack rays for each one separately.")
    
    piece_reports = []
    total_targets = 0
    
    for origin in instances:
        origin_name = chess.square_name(origin)
        piece = board.piece_at(origin)
        sub_thought, targets = _generate_thought(board, origin, piece)
        thought.append(f"[{origin_name}] {sub_thought}")
        
        if targets:
            piece_reports.append(f"The {p_name} on {origin_name} can capture: {_join_targets(targets)}.")
            total_targets += len(targets)
        else:
            piece_reports.append(f"The {p_name} on {origin_name} has no legal captures.")

    ans_body = f"There are {len(instances)} {c_name} {p_name}s on the board. "
    
    if total_targets == 0:
        ans_body += "None of them can capture any enemy pieces this turn."
    else:
        ans_body += "Here are their options:\n- " + "\n- ".join(piece_reports)

    ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
    return _package(board, q, ans)


def sample_invalid_premise(board: chess.Board) -> Optional[Dict]:
    """INVALID task: Ask about captures for a piece that does not exist on the given square."""
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
        f"The user asks what the {fake_name} on {origin_name} can capture.",
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
    sample_specific_captures,
    sample_specific_captures,
    sample_specific_captures,
    sample_ambiguous_captures,
    sample_ambiguous_captures,
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
    parser = argparse.ArgumentParser(description="Capture Targets data generator")
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