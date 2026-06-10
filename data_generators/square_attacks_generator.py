"""
square_attacks_generator.py
───────────────────────────
Offline synthetic data generator for square attack / board control tasks.

Produces three task types to build robust spatial awareness and threat detection:

  SPECIFIC    — "What squares does the knight on f3 attack?"
  AMBIGUOUS   — "What squares does my knight attack?" 
  INVALID     — "What does the bishop on e4 attack?" (wrong piece/empty square)

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think>. Prompts are built via PromptBuilder.

Usage
-----
    python square_attacks_generator.py                      # single example
    python square_attacks_generator.py --n 50000 --out data/square_attacks.jsonl
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
    "What squares does the {piece} on {sq} attack?",
    "Which squares are attacked by the {piece} on {sq}?",
    "List all squares the {piece} at {sq} controls or attacks.",
    "What is the attack range of the {piece} on {sq}?",
    "Which squares fall under the influence of the {piece} on {sq}?",
    "Enumerate every square attacked by the {piece} on {sq}.",
    "What squares can the {piece} on {sq} threaten?",
    "List the squares that the {piece} at {sq} puts under attack.",
    "Identify all squares currently threatened by the {piece} located at {sq}.",
    "Determine the complete set of squares under attack by the {piece_name} on {sq}.",
    "Provide the coordinates of the squares controlled by the {piece} on {sq}.",
    "What is the field of influence for the {piece} situated at {sq}?",
    "Detail the squares that are within the attacking radius of the {piece} on {sq}.",
    "Which squares are dominated by the presence of the {piece_name} on {sq}?",
    "State the specific squares the {piece} on {sq} is attacking right now.",
    "What sectors of the board does the {piece} on {sq} cover?",

    # Casual & Conversational
    "what squares does the {piece_name} on {sq} cover?",
    "where does the {piece} on {sq} attack?",
    "which squares is the {piece_name} on {sq} eyeing?",
    "tell me what squares the {piece} at {sq} attacks",
    "what does the {piece} on {sq} control?",
    "show me the attack squares for {piece} on {sq}",
    "what's the {piece_name} on {sq} attacking?",
    "which squares does the {piece} on {sq} threaten right now?",
    "Where is the {piece} on {sq} aiming?",
    "What spots are covered by the {piece} on {sq}?",
    "Which squares should I be careful of because of the {piece} on {sq}?",
    "What's in the firing line of the {piece_name} on {sq}?",
    "What ground does the {piece} on {sq} hold?",
    "Can you tell me what squares the {piece} on {sq} has its eyes on?",
    "Where is the {piece} on {sq} putting the pressure?",
    "Does the {piece} on {sq} look at any important squares?",

    # Short & Terse
    "Attack squares {piece} {sq}.",
    "{piece} on {sq} attacks.",
    "Squares threatened by {piece} {sq}.",
    "{piece_name} {sq} control map.",
    "What does {piece} {sq} hit?",
    "Range of {piece} on {sq}.",
    "{piece} at {sq} targets.",
    "Threats from {piece} {sq}.",

    # Typos, Slang, & Informal
    "waht squars does the {piece} on {sq} atack?",
    "wich squares is {piece_name} on {sq} atacking?",
    "sqaures attacked by {piece} on {sq}?",
    "what squres does {piece_name} at {sq} threatin?",
    "list atacked squares for {piece} on {sq}",
    "wher does the {piece_name} on {sq} look?",
    "wat spots does the {piece} {sq} cover?",
    "whch squares do {piece} on {sq} control?",
    "is {piece} on {sq} attacking anything?",
    "wut squares r hit by {piece} on {sq}?",
    "tell me squares thretend by {piece} {sq}",
    "atk range of {piece} {sq} pls",
    "where does {piece} {sq} aim at?",
    "list da squares dat {piece} on {sq} attacks",
    "shw me wat {piece_name} on {sq} attacks",
    "sqaures covered by {piece} {sq}?",
]

# Placeholders: {piece_name} = "knight"
AMBIGUOUS_Q: List[str] = [
    "What squares does my {piece_name} attack?",
    "Where does the {piece_name} exert control?",
    "List the attacked squares for the {piece_name}.",
    "Are there any specific squares my {piece_name} is eyeing?",
    "Show me the attack map for my {piece_name}.",
]

# Placeholders: {piece} = wrong piece, {sq} = wrong sq
INVALID_Q: List[str] = [
    "What squares does the {piece} on {sq} attack?",
    "Which squares are attacked by the {piece} on {sq}?",
    "List all squares the {piece} at {sq} controls or attacks.",
    "What is the attack range of the {piece} on {sq}?",
    "Tell me what squares the {piece} at {sq} attacks.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer Templates
# ══════════════════════════════════════════════════════════════════════════════

SQUARE_ATTACKS_A: List[str] = [
    # Formal & Direct
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} attacks the following squares: {targets}.",
    "<think>\n{thought}\n</think>\nFrom {sq}, the {piece} attacks: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} at {sq} controls or attacks: {targets}.",
    "<think>\n{thought}\n</think>\nSquares attacked by the {piece} on {sq}: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} located at {sq} exercises control over: {targets}.",
    "<think>\n{thought}\n</think>\nThe squares currently threatened by the {piece} on {sq} are: {targets}.",
    "<think>\n{thought}\n</think>\nAn analysis of the {piece} on {sq} reveals it attacks: {targets}.",
    "<think>\n{thought}\n</think>\nThe sphere of influence for the {piece_name} on {sq} encompasses: {targets}.",
    "<think>\n{thought}\n</think>\nThe following squares are in the attack range of the {piece} at {sq}: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} projects its attack onto: {targets}.",

    # Casual & Conversational
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} is eyeing these spots: {targets}.",
    "<think>\n{thought}\n</think>\nYou'll find that the {piece} on {sq} covers: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece_name} on {sq} is keeping an eye on: {targets}.",
    "<think>\n{thought}\n</think>\nRight now, the {piece} on {sq} is attacking: {targets}.",
    "<think>\n{thought}\n</think>\nHere's what the {piece} on {sq} is threatening: {targets}.",
    "<think>\n{thought}\n</think>\nThese are the squares the {piece} on {sq} holds down: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} puts pressure on: {targets}.",
    "<think>\n{thought}\n</think>\nIt targets the following squares: {targets}."
]

SQUARE_ATTACKS_A_NONE: List[str] = [
    # Formal & Direct
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} does not attack any squares.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} has no attack squares in this position.",
    "<think>\n{thought}\n</think>\nThere are no squares currently threatened by the {piece} on {sq}.",
    "<think>\n{thought}\n</think>\nThe {piece} at {sq} exerts zero attacks on the board.",
    "<think>\n{thought}\n</think>\nNo squares fall under the influence of the {piece_name} on {sq}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} is not attacking anything right now.",
    "<think>\n{thought}\n</think>\nIn this configuration, the {piece} located at {sq} controls no squares.",
    "<think>\n{thought}\n</think>\nZero squares are being targeted by the {piece} on {sq}.",

    # Casual & Conversational
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} isn't attacking any squares.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} has no control over any squares right now.",
    "<think>\n{thought}\n</think>\nIt looks like the {piece} on {sq} isn't threatening anything.",
    "<think>\n{thought}\n</think>\nThe {piece_name} on {sq} is totally boxed in and attacks nothing.",
    "<think>\n{thought}\n</think>\nThere are zero squares covered by the {piece} on {sq}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} doesn't cover a single square.",
    "<think>\n{thought}\n</think>\nNothing is being hit by the {piece} on {sq}."
]


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought Generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_thought(board: chess.Board, origin: int, piece: chess.Piece) -> Tuple[str, List[str]]:
    """
    Generate CoT narrating the geometric mapping of attacked squares.
    Returns the CoT string and the list of attacked square names.
    """
    origin_name = chess.square_name(origin)
    p_name      = _pname(piece.piece_type)
    c_name      = _color(piece.color)

    parts = [
        f"The user is asking about the squares attacked (controlled) by the {c_name} {p_name} on {origin_name}.",
        f"Verifying the <board_state> token: confirmed {c_name} {p_name} resides at {origin_name}.",
        f"It's important to note that 'attacks' are different from 'legal moves'. A piece attacks a square even if it is occupied by a friendly piece (defending it), and absolute pins do not prevent a piece from exerting control over a square."
    ]

    # Rule recitation based on piece type
    if piece.piece_type == chess.PAWN:
        parts.append(f"Pawns attack diagonally forward one square. They do not attack straight ahead.")
    elif piece.piece_type == chess.KNIGHT:
        parts.append(f"Knights attack in an L-shape pattern, ignoring intervening pieces.")
    elif piece.piece_type == chess.BISHOP:
        parts.append(f"Bishops attack along diagonal rays until blocked by any piece.")
    elif piece.piece_type == chess.ROOK:
        parts.append(f"Rooks attack along horizontal and vertical rays until blocked by any piece.")
    elif piece.piece_type == chess.QUEEN:
        parts.append(f"Queens attack along orthogonal and diagonal rays until blocked by any piece.")
    elif piece.piece_type == chess.KING:
        parts.append(f"Kings attack all adjacent squares (horizontal, vertical, and diagonal).")

    # Get attacked squares using python-chess engine
    attacks_mask = board.attacks(origin)
    attacked_squares = [chess.square_name(sq) for sq in attacks_mask]

    if not attacked_squares:
        # Note: Mathematically in chess, pieces almost always attack at least 1 square unless 
        # completely trapped by board edges/own pieces in highly specific edge cases.
        parts.append("Tracing trajectories... Zero squares are currently being attacked by this piece.")
    else:
        parts.append(f"Mapping the attack rays: found {len(attacked_squares)} attacked square(s).")
        parts.append(f"Final evaluated targets: {', '.join(attacked_squares)}.")

    return " ".join(parts), attacked_squares


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

def sample_specific_attacks(board: chess.Board) -> Optional[Dict]:
    """SPECIFIC task: What squares does piece X on square Y attack?"""
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
        a = random.choice(SQUARE_ATTACKS_A).format(**fmt_kw)
    else:
        a = random.choice(SQUARE_ATTACKS_A_NONE).format(**fmt_kw)

    return _package(board, q, a)


def sample_ambiguous_attacks(board: chess.Board) -> Optional[Dict]:
    """AMBIGUOUS task: What squares does my piece type X attack?"""
    piece_type = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING])
    p_name = _pname(piece_type)
    color  = board.turn
    c_name = _color(color)
    
    instances = [s for s in chess.SQUARES if (p := board.piece_at(s)) and p.color == color and p.piece_type == piece_type]
    
    q = random.choice(AMBIGUOUS_Q).format(piece_name=p_name)
    
    thought = [
        f"The user asks for the attacked squares controlled by their {p_name}s without specifying a starting square.",
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
        
        thought.append(f"Found exactly one on {origin_name}. Evaluating attack map.")
        sub_thought, targets = _generate_thought(board, origin, piece)
        thought.append(sub_thought)
        
        if targets:
            ans_body = f"You have one {p_name} on {origin_name}. It attacks: {_join_targets(targets)}."
        else:
            ans_body = f"You have one {p_name} on {origin_name}, but it is not attacking any squares."
            
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
            piece_reports.append(f"The {p_name} on {origin_name} attacks: {_join_targets(targets)}.")
            total_targets += len(targets)
        else:
            piece_reports.append(f"The {p_name} on {origin_name} does not attack any squares.")

    ans_body = f"There are {len(instances)} {c_name} {p_name}s on the board. "
    
    if total_targets == 0:
        ans_body += "None of them currently exert control over any squares."
    else:
        ans_body += "Here is their attack coverage:\n- " + "\n- ".join(piece_reports)

    ans = f"<think>\n{' '.join(thought)}\n</think>\n{ans_body}"
    return _package(board, q, ans)


def sample_invalid_premise(board: chess.Board) -> Optional[Dict]:
    """INVALID task: Ask about attacks for a piece that does not exist on the given square."""
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
        f"The user is asking about the attack map for a {fake_name} on {origin_name}.",
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
    sample_specific_attacks,
    sample_specific_attacks,
    sample_specific_attacks,
    sample_ambiguous_attacks,
    sample_ambiguous_attacks,
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
    parser = argparse.ArgumentParser(description="Square Attacks data generator")
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
