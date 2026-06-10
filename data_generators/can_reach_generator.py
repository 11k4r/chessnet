"""
can_reach_generator.py
──────────────────────
Offline synthetic data generator for targeted movement validation tasks.

Produces three task types to build robust spatial reasoning and ray-tracing:

  SPECIFIC    — "Can the knight on f3 reach d4?"       (explicit piece + target)
  AMBIGUOUS   — "Can my knight reach d4?"              (piece type + target)
  INVALID     — "Can the bishop on e4 reach f5?"       (wrong piece / empty origin)

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think>. Prompts are built via PromptBuilder.

Usage
-----
    python can_reach_generator.py                      # single example
    python can_reach_generator.py --n 50000 --out data/can_reach.jsonl
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

# Placeholders: {piece_name} = "knight", {origin} = "f3", {target} = "d4"
SPECIFIC_Q: List[str] = [
    # Formal
    "Can the {piece_name} on {origin} legally move to {target}?",
    "Is {target} reachable by the {piece_name} currently on {origin}?",
    "Evaluate if the {piece_name} on {origin} can land on {target}.",
    "Is the trajectory from {origin} to {target} valid for the {piece_name}?",
    "Does the {piece_name} on {origin} have {target} as a legal destination?",
    # Casual
    "Can the {piece_name} on {origin} go to {target}?",
    "Is {target} a valid move for the {piece_name} on {origin}?",
    "Could I move my {piece_name} from {origin} to {target}?",
    "Does the {piece_name} at {origin} have a clear path to {target}?",
    # Terse / Informal
    "move {piece_name} {origin} to {target}?",
    "{origin} to {target} legal for {piece_name}?",
    "can {piece_name} jump from {origin} to {target}?",
    "is {target} open for {piece_name} on {origin}?",
    "can {piece_name} on {origin} land on {target} rn?",
]

# Placeholders: {piece_name} = "knight", {target} = "d4"
AMBIGUOUS_Q: List[str] = [
    "Can the {piece_name} move to {target}?",
    "Is {target} reachable by my {piece_name}?",
    "Can any {piece_name} go to {target} right now?",
    "Is it possible to play a {piece_name} to {target}?",
    "Can a {piece_name} land on {target}?",
    "{piece_name} to {target}?",
    "Is {target} a legal destination for the {piece_name}?",
]

# Placeholders: {piece_name} = wrong piece, {origin} = wrong sq, {target} = "d4"
INVALID_Q: List[str] = [
    "Can the {piece_name} on {origin} jump to {target}?",
    "Is it legal to move the {piece_name} on {origin} to {target}?",
    "Can my {piece_name} on {origin} go to {target}?",
    "What about the {piece_name} on {origin} to {target}?",
    "Can the {piece_name} at {origin} reach {target}?",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer Templates
# ══════════════════════════════════════════════════════════════════════════════

SPECIFIC_A_YES: List[str] = [
    "<think>\n{thought}\n</think>\nYes, the {piece_name} on {origin} can move to {target}.",
    "<think>\n{thought}\n</think>\nYes — {target} is a legal move for the {piece_name} on {origin}.",
    "<think>\n{thought}\n</think>\nThat is a valid maneuver. The {piece_name} on {origin} can reach {target}.",
]

SPECIFIC_A_NO: List[str] = [
    "<think>\n{thought}\n</think>\nNo, the {piece_name} on {origin} cannot reach {target}.",
    "<think>\n{thought}\n</think>\nNo — {target} is not a legal move for the {piece_name} on {origin}.",
]

SPECIFIC_A_NO_REASON: List[str] = [
    "<think>\n{thought}\n</think>\nNo, the {piece_name} on {origin} cannot reach {target} because {reason}.",
    "<think>\n{thought}\n</think>\nSorry, {target} isn't a legal destination for that {piece_name}. The move is invalid because {reason}.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought Generation
# ══════════════════════════════════════════════════════════════════════════════

def _get_path(sq1: int, sq2: int) -> List[int]:
    """Return square indices between sq1 and sq2 (exclusive), or empty if not a straight/diagonal line."""
    f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
    f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
    if f1 != f2 and r1 != r2 and abs(f1 - f2) != abs(r1 - r2):
        return []
    df = 1 if f2 > f1 else (-1 if f2 < f1 else 0)
    dr = 1 if r2 > r1 else (-1 if r2 < r1 else 0)
    path = []
    curr_f, curr_r = f1 + df, r1 + dr
    while (curr_f, curr_r) != (f2, r2):
        path.append(chess.square(curr_f, curr_r))
        curr_f += df
        curr_r += dr
    return path


def _generate_thought(board: chess.Board, origin: int, target: int, piece: chess.Piece) -> Tuple[str, bool, str]:
    """
    Generate CoT, final boolean, and failure reason.
    Returns: (thought_string, can_reach_boolean, reason_string)
    """
    origin_name = chess.square_name(origin)
    target_name = chess.square_name(target)
    p_name      = _pname(piece.piece_type)
    c_name      = _color(piece.color)
    
    parts = [
        f"The user is asking if the {c_name} {p_name} on {origin_name} can move to {target_name}.",
        f"Verifying board state at {origin_name}: confirmed, {c_name} {p_name} is present.",
        f"Let's trace the trajectory to {target_name}."
    ]
    
    f1, r1 = chess.square_file(origin), chess.square_rank(origin)
    f2, r2 = chess.square_file(target), chess.square_rank(target)
    file_diff = abs(f1 - f2)
    rank_diff = abs(r1 - r2)
    
    is_straight = (file_diff == 0 or rank_diff == 0)
    is_diagonal = (file_diff == rank_diff)
    is_knight   = (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)
    
    valid_geometry = False
    path: List[int] = []
    reason = ""
    promotion = None

    # 1. Geometry Check
    if piece.piece_type == chess.ROOK:
        parts.append("Rooks move orthogonally.")
        if is_straight:
            valid_geometry, path = True, _get_path(origin, target)
        else:
            reason = "that square does not align horizontally or vertically"
            
    elif piece.piece_type == chess.BISHOP:
        parts.append("Bishops slide diagonally.")
        if is_diagonal:
            valid_geometry, path = True, _get_path(origin, target)
        else:
            reason = "that square is not on a valid diagonal"
            
    elif piece.piece_type == chess.QUEEN:
        parts.append("Queens combine orthogonal and diagonal movement.")
        if is_straight or is_diagonal:
            valid_geometry, path = True, _get_path(origin, target)
        else:
            reason = "that square does not align straight or diagonally"
            
    elif piece.piece_type == chess.KNIGHT:
        parts.append("Knights jump in an L-shape.")
        if is_knight:
            valid_geometry = True
        else:
            reason = "that is not a valid L-shaped jump"
            
    elif piece.piece_type == chess.KING:
        parts.append("Kings step exactly one adjacent square.")
        if file_diff <= 1 and rank_diff <= 1:
            valid_geometry = True
        elif file_diff == 2 and rank_diff == 0:
            parts.append("This looks like a castling attempt.")
            valid_geometry = True
        else:
            reason = "that square is out of range for a single king step"
            
    elif piece.piece_type == chess.PAWN:
        direction = 1 if piece.color == chess.WHITE else -1
        parts.append(f"Pawns advance {'up' if direction == 1 else 'down'} the board.")
        
        if file_diff == 0 and r2 - r1 == direction:
            parts.append("Single step forward geometry matches.")
            valid_geometry = True
        elif file_diff == 0 and r2 - r1 == 2 * direction and ((piece.color == chess.WHITE and r1 == 1) or (piece.color == chess.BLACK and r1 == 6)):
            parts.append("Double-step from the starting rank geometry matches.")
            valid_geometry, path = True, [chess.square(f1, r1 + direction)]
        elif file_diff == 1 and r2 - r1 == direction:
            parts.append("Diagonal capture vector matches.")
            valid_geometry = True
        else:
            reason = "that violates the pawn's directional mechanics"

        if valid_geometry and r2 in [0, 7]:
            parts.append("This move reaches the back rank, requiring a promotion.")
            promotion = chess.QUEEN

    if not valid_geometry:
        parts.append(f"Geometry fails: {reason}.")
        return " ".join(parts), False, reason

    # 2. Collision Check
    if path:
        path_names = [chess.square_name(s) for s in path]
        parts.append(f"Scanning the ray through: {', '.join(path_names)}.")
        for sq_idx in path:
            occupant = board.piece_at(sq_idx)
            if occupant:
                occ_tag = _pname(occupant.piece_type)
                occ_color = "friendly" if occupant.color == piece.color else "enemy"
                parts.append(f"Collision detected! A {occ_color} {occ_tag} blocks {chess.square_name(sq_idx)}.")
                reason = f"the path is blocked by a {occ_color} {occ_tag} on {chess.square_name(sq_idx)}"
                return " ".join(parts), False, reason
        parts.append("Ray is clear.")

    # 3. Destination Check
    target_occupant = board.piece_at(target)
    if target_occupant:
        occ_tag = _pname(target_occupant.piece_type)
        if target_occupant.color == piece.color:
            parts.append(f"Destination {target_name} is occupied by our own {occ_tag}.")
            reason = f"the destination is occupied by your own {occ_tag}"
            return " ".join(parts), False, reason
        else:
            if piece.piece_type == chess.PAWN and file_diff == 0:
                parts.append(f"Pawns cannot capture straight ahead. The enemy {occ_tag} blocks it.")
                reason = f"pawns cannot capture straight ahead, and an enemy {occ_tag} is blocking it"
                return " ".join(parts), False, reason
            parts.append(f"Enemy {occ_tag} detected on {target_name}. Valid capture destination.")
    else:
        if piece.piece_type == chess.PAWN and file_diff == 1:
            if board.has_legal_en_passant() and board.ep_square == target:
                parts.append(f"{target_name} is an active en passant target. Valid capture.")
            else:
                parts.append(f"{target_name} is empty. Pawns can only move diagonally to capture.")
                reason = "pawns can only move diagonally to capture, and that square is empty"
                return " ".join(parts), False, reason
        else:
            parts.append(f"{target_name} is unoccupied.")

    # 4. Engine Legality Check (Pins and Checks)
    # The CoT narrative traced the geometry; the engine validates absolute rules.
    move = chess.Move(origin, target, promotion=promotion)
    is_legal = move in board.legal_moves

    if is_legal:
        parts.append("Filtering against king-safety: the piece is not absolutely pinned. Move is fully legal.")
        return " ".join(parts), True, ""
    else:
        parts.append("Wait, king-safety filter triggers. Executing this move would leave or expose the King to check.")
        reason = "moving that piece would leave or place your King in check (absolute pin or active check)"
        return " ".join(parts), False, reason


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

def sample_specific_reach(board: chess.Board) -> Optional[Dict]:
    """SPECIFIC task: Can piece X on square Y reach square Z?"""
    active_squares = [s for s in chess.SQUARES if (p := board.piece_at(s)) and p.color == board.turn]
    if not active_squares:
        return None

    origin = random.choice(active_squares)
    piece  = board.piece_at(origin)
    sq_name = chess.square_name(origin)
    p_name  = _pname(piece.piece_type)

    legal_dests = {m.to_square for m in board.legal_moves if m.from_square == origin}

    # 50/50 split between asking about a legal vs illegal destination
    if legal_dests and random.random() < 0.5:
        target_sq = random.choice(list(legal_dests))
    else:
        all_squares = set(chess.SQUARES) - {origin}
        unreachable = list(all_squares - legal_dests)
        if not unreachable:
            return None
        target_sq = random.choice(unreachable)

    target_name = chess.square_name(target_sq)

    q = random.choice(SPECIFIC_Q).format(piece_name=p_name, origin=sq_name, target=target_name)
    thought_str, can_reach, reason = _generate_thought(board, origin, target_sq, piece)

    fmt_kw = dict(thought=thought_str, piece_name=p_name, origin=sq_name, target=target_name, reason=reason)

    if can_reach:
        a = random.choice(SPECIFIC_A_YES).format(**fmt_kw)
    else:
        if reason and random.random() < 0.6:
            a = random.choice(SPECIFIC_A_NO_REASON).format(**fmt_kw)
        else:
            a = random.choice(SPECIFIC_A_NO).format(**fmt_kw)

    return _package(board, q, a)


def sample_ambiguous_reach(board: chess.Board) -> Optional[Dict]:
    """AMBIGUOUS task: Can my piece type X reach square Z?"""
    piece_type = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    target_sq  = random.choice(chess.SQUARES)
    target_name = chess.square_name(target_sq)
    
    p_name = _pname(piece_type)
    color  = board.turn
    c_name = _color(color)
    
    instances = [s for s in chess.SQUARES if (p := board.piece_at(s)) and p.color == color and p.piece_type == piece_type]
    
    q = random.choice(AMBIGUOUS_Q).format(piece_name=p_name, target=target_name)
    
    thought = [
        f"The user asks if a {c_name} {p_name} can move to {target_name} without specifying the starting square.",
        f"Scanning the <board_state> for {c_name} {p_name}s..."
    ]
    
    # CASE 0: No pieces
    if not instances:
        thought.append("None found.")
        ans_body = f"You do not have any {p_name}s on the board, so no {p_name} can reach {target_name}."
        ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
        return _package(board, q, ans)

    # CASE 1: Exactly one piece
    if len(instances) == 1:
        origin = instances[0]
        origin_name = chess.square_name(origin)
        piece = board.piece_at(origin)
        
        thought.append(f"Found exactly one on {origin_name}. Let's evaluate it.")
        sub_thought, can_reach, reason = _generate_thought(board, origin, target_sq, piece)
        thought.append(sub_thought)
        
        if can_reach:
            ans_body = f"You have one {p_name} on {origin_name}, and yes, it can reach {target_name}."
        else:
            if reason and random.random() < 0.6:
                ans_body = f"You have one {p_name} on {origin_name}, but no, it cannot reach {target_name} because {reason}."
            else:
                ans_body = f"You have one {p_name} on {origin_name}, but it cannot reach {target_name}."
                
        ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
        return _package(board, q, ans)

    # CASE 2: Multiple pieces
    thought.append(f"Found {len(instances)} {p_name}s. I will evaluate each one separately.")
    
    can_reach_sqs = []
    cannot_reach_sqs = []
    
    for origin in instances:
        origin_name = chess.square_name(origin)
        piece = board.piece_at(origin)
        sub_thought, can_reach, reason = _generate_thought(board, origin, target_sq, piece)
        thought.append(f"[{origin_name}] {sub_thought}")
        
        if can_reach:
            can_reach_sqs.append(origin_name)
        else:
            cannot_reach_sqs.append(origin_name)

    ans_body = f"There are {len(instances)} {c_name} {p_name}s on the board. "
    if not can_reach_sqs:
        ans_body += f"None of them can reach {target_name}."
    elif not cannot_reach_sqs:
        ans_body += f"All of them ({', '.join(can_reach_sqs)}) can legally reach {target_name}."
    else:
        can_str = ", ".join(can_reach_sqs)
        cant_str = ", ".join(cannot_reach_sqs)
        ans_body += f"The {p_name} on {can_str} can reach {target_name}, but the one on {cant_str} cannot."

    ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
    return _package(board, q, ans)


def sample_invalid_premise(board: chess.Board) -> Optional[Dict]:
    """INVALID task: Ask about a piece that does not exist on the given square."""
    origin_sq = random.choice(chess.SQUARES)
    actual_p  = board.piece_at(origin_sq)
    
    types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    if actual_p:
        types.remove(actual_p.piece_type)
        
    fake_type = random.choice(types)
    fake_name = _pname(fake_type)
    
    target_sq = random.choice([s for s in chess.SQUARES if s != origin_sq])
    
    origin_name = chess.square_name(origin_sq)
    target_name = chess.square_name(target_sq)
    
    q = random.choice(INVALID_Q).format(piece_name=fake_name, origin=origin_name, target=target_name)
    
    thought = [
        f"The user asks if a {fake_name} on {origin_name} can move to {target_name}.",
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
    sample_specific_reach,
    sample_specific_reach,
    sample_specific_reach,
    sample_ambiguous_reach,
    sample_ambiguous_reach,
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
    parser = argparse.ArgumentParser(description="Can-Reach data generator")
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
