"""
legal_moves_generator.py
────────────────────────
Offline synthetic data generator for legal-move grounding tasks.

Produces three task types to build robust rule-following behaviour:

  SPECIFIC    — "Where can the knight on f3 go?"   (explicit piece + square)
  AMBIGUOUS   — "Where can my knight go?"           (piece type only)
  INVALID     — "Where can the bishop on e4 go?"   (wrong piece / empty square)

All examples are packaged in ShareGPT / Qwen JSONL format with full CoT
enclosed in <think>...</think> as required by the training format.

The prompt structure is generated via PromptBuilder.build_text_prompt so
training examples are guaranteed to match live inference prompts exactly.

Usage
-----
    python legal_moves_generator.py                        # single example
    python legal_moves_generator.py --n 50000 --out data/legal_moves.jsonl
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

# ── Module-level singletons (offline — no tokenizer needed) ───────────────────
_config  = ChessCoachConfig()
_builder = PromptBuilder(_config)   # tokenizer=None is fine for offline use

# ── Piece name helpers ────────────────────────────────────────────────────────
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
# Question templates
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {piece} = "knight", {piece_name} = "knight", {sq} = "f3"
SPECIFIC_Q: List[str] = [
    # Formal / descriptive
    "What are the legal moves for the {piece} on {sq}?",
    "List all legal moves available to the {piece} on {sq}.",
    "Which squares can the {piece} on {sq} legally move to?",
    "Provide a complete list of legal moves for the {piece_name} on {sq}.",
    "According to the rules of chess, where can the {piece} on {sq} move?",
    "What squares can the {piece} on {sq} reach from its current position?",
    "Enumerate all valid destinations for the {piece} at {sq}.",
    "What are all the squares the {piece} on {sq} is able to move to?",
    "Where exactly can the {piece} on {sq} legally go in this position?",
    "Could you tell me the full set of legal moves for the {piece} on {sq}?",
    # Casual / conversational
    "Where can the {piece} on {sq} go?",
    "What moves can the {piece} on {sq} make right now?",
    "Where is the {piece} on {sq} allowed to go?",
    "If I want to move the {piece} on {sq}, what are my options?",
    "I have a {piece_name} on {sq} — where can it go?",
    "My {piece_name} is sitting on {sq}. What can it do?",
    "Tell me where the {piece_name} on {sq} can move.",
    "What are my choices if I move the {piece_name} from {sq}?",
    "Can you show me where the {piece} on {sq} can go?",
    "What moves does the {piece} on {sq} have available?",
    "I'm thinking about moving my {piece_name} on {sq}. Where can it go?",
    "Looking at the {piece} on {sq} — what are its legal destinations?",
    # Interrogative / yes-no framing
    "Can the {piece} on {sq} move anywhere?",
    "Does the {piece} on {sq} have any legal moves?",
    "Is the {piece} on {sq} stuck, or can it move?",
    "Are there any moves available for the {piece} on {sq}?",
    "Is there anywhere the {piece} on {sq} can legally go?",
    # Short / terse
    "Moves for {piece} on {sq}?",
    "{piece} at {sq} — destinations?",
    "Legal moves: {piece} {sq}?",
    "{piece_name} on {sq}, where can it go?",
    "Where to move {piece} from {sq}?",
    "Options for {piece} at {sq}?",
    "{piece} {sq} legal destinations?",
    # Typos / informal (robustness)
    "waht are the legel moves for the {piece} on {sq}?",
    "wher can the {piece} at {sq} go to?",
    "gimme legal moves for {piece} {sq}",
    "legal mv for {piece} on {sq}",
    "is there any moves for {piece} on {sq}",
    "shw me moves 4 {piece} at {sq}",
    "wats the posible moves for {piece} on {sq}?",
    "can u tell me were {piece} on {sq} can move",
    "wat can the {piece} on {sq} do rn?",
    "moves 4 the {piece} at {sq} pls",
]

# Placeholders: {piece_name} = "knight"
AMBIGUOUS_Q: List[str] = [
    "What are the legal moves for the {piece_name}?",
    "Where can my {piece_name} move?",
    "List the moves for the {piece_name}.",
    "Show me all destinations for the {piece_name}.",
    "What options does my {piece_name} have?",
    "Where can the {piece_name} go in this position?",
    "What can my {piece_name} do right now?",
    "List all legal moves for my {piece_name}.",
    "What moves are available to the {piece_name}?",
    "Where would the {piece_name} go if I moved it?",
    "Can you list every square my {piece_name} can reach?",
    "What are the {piece_name}'s legal options here?",
    "Tell me all the places my {piece_name} can go.",
    "I want to move a {piece_name} — what are the options?",
    "Where are the {piece_name} destinations?",
    "{piece_name} moves?",
    "Moves for my {piece_name}?",
    "Legal destinations for the {piece_name}?",
    "What does my {piece_name} control right now?",
    "Where can I send the {piece_name}?",
]

# Placeholders: {piece} / {piece_name} = the WRONG piece, {sq} = the square
INVALID_Q: List[str] = [
    "What are the moves for the {piece} on {sq}?",
    "Where can the {piece} on {sq} go?",
    "List the destinations for the {piece_name} at {sq}.",
    "Give me the options for the {piece} on {sq}.",
    "What legal moves does the {piece_name} on {sq} have?",
    "Can the {piece} on {sq} move anywhere?",
    "Tell me the {piece}'s moves from {sq}.",
    "Where can my {piece_name} on {sq} go?",
    "What can the {piece} on {sq} do?",
    "Moves for {piece} at {sq}?",
    "Legal destinations for {piece_name} on {sq}?",
    "{piece} on {sq} — where can it move?",
    "I want to move my {piece_name} on {sq}. Where can it go?",
    "What are the {piece}'s options from {sq}?",
]


# ══════════════════════════════════════════════════════════════════════════════
# Answer templates
# ══════════════════════════════════════════════════════════════════════════════

# Placeholders: {thought}, {piece}, {piece_name}, {sq}, {targets}, {n}
SPECIFIC_A_HAS_MOVES: List[str] = [
    "<think>\n{thought}\n</think>\nThe legal moves for the {piece} on {sq} are: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} can move to: {targets}.",
    "<think>\n{thought}\n</think>\nLegal moves for the {piece} at {sq}: {targets}.",
    "<think>\n{thought}\n</think>\nYou can move the {piece} on {sq} to: {targets}.",
    "<think>\n{thought}\n</think>\nFrom {sq}, the {piece} has {n} legal move(s): {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} can reach the following squares: {targets}.",
    "<think>\n{thought}\n</think>\nMoving the {piece_name} from {sq}, your options are: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece_name} at {sq} has {n} legal destination(s): {targets}.",
    "<think>\n{thought}\n</think>\nHere are all legal moves for the {piece} on {sq}: {targets}.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} is free to go to: {targets}.",
]

SPECIFIC_A_NO_MOVES: List[str] = [
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} has no legal moves.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} cannot move — it has no legal moves available.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} is stuck and has zero legal moves right now.",
    "<think>\n{thought}\n</think>\nUnfortunately, the {piece} on {sq} has no legal moves in this position.",
    "<think>\n{thought}\n</think>\nNo legal moves exist for the {piece_name} on {sq}.",
    "<think>\n{thought}\n</think>\nThe {piece_name} on {sq} is completely immobilized — zero legal moves.",
    "<think>\n{thought}\n</think>\nThe {piece} on {sq} cannot go anywhere legally from here.",
]

_PINNED_NO_MOVES = [
    "The piece has open squares, but it's caught in an absolute pin. Moving it would expose the king to check, so it's completely paralyzed.",
    "It looks like it can move, but this piece is doing a vital job shielding the king. Because of the absolute pin, it can't legally step out of the way.",
    "While there are normal moves available, the piece is absolutely pinned against the king. Not a single one of those moves is legal.",
    "Geometrically it has options, but tactically it's frozen by an absolute pin. Any move leaves the king in check, so zero moves are permitted.",
    "The piece is absolutely pinned. Even though its path is clear, moving it would put the king in check, which is strictly illegal."
]

_BLOCKED_NO_MOVES = [
    "The piece is completely boxed in. It has absolutely no room to breathe.",
    "Look at that piece—it's entirely trapped. There isn't a single open square for it to step to.",
    "This piece is totally hemmed in by the current position. It simply cannot budge.",
    "Every single direction is blocked off. The piece is completely paralyzed right now.",
    "It's completely stuck. The surrounding pieces and the edge of the board have left it with nowhere to go."
]

_EN_PASSANT_NOTICES = [
    "En passant capture available: {targets}.",
    "You have an en passant capture on {targets}.",
    "Ooh, an en passant capture is possible on {targets}!",
    "En passant capture available: {targets}. Did you know that en passant is forced? :) (Just kidding, but it is a fun rule.)",
    "You can capture en passant on {targets}. Mandatory 'en passant is forced' joke goes here! :)",
]

# ══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _join_targets(targets: List[str]) -> str:
    """Grammatically join a list of square names."""
    if not targets:
        return ""
    if len(targets) == 1:
        return targets[0]
    if len(targets) == 2:
        return f"{targets[0]} and {targets[1]}"
    return ", ".join(targets[:-1]) + f", and {targets[-1]}"


def _plural(n: int, singular: str, plural: Optional[str] = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


# ══════════════════════════════════════════════════════════════════════════════
# Chain-of-Thought generation
# ══════════════════════════════════════════════════════════════════════════════

# Per-component phrase pools — randomly sampled to avoid template lock-in
_OPEN_TEMPLATES = [
    "The user asks about legal moves for the {color} {piece_name} on {sq}.",
    "I need to find all legal moves for the {color} {piece_name} sitting on {sq}.",
    "This is a legal-move query for the {color} {piece_name} at {sq}.",
    "Let me work out every legal destination for the {color} {piece_name} on {sq}.",
    "Finding legal moves: {color} {piece_name} on {sq}.",
]

_GROUND_TEMPLATES = [
    "Checking the <board_state> visual tokens at square {sq} — confirmed, the {color} {piece_name} is there.",
    "The board encoding at {sq} confirms the presence of a {color} {piece_name}.",
    "Visual token at {sq} encodes a {color} {piece_name}. Grounding verified.",
    "Cross-referencing {sq} against the <board_state> sequence — {color} {piece_name} found.",
    "The <board_state> latent at {sq} corresponds to the {color} {piece_name}. Confirmed.",
]

_RULE_TEMPLATES = {
    chess.PAWN: [
        "Pawns advance one square forward (two from the starting rank) and capture one square diagonally. En passant and promotion are special cases to watch for.",
        "A pawn on its starting rank can push one or two squares forward. It captures diagonally one square ahead. I must also check for en passant eligibility.",
        "Pawn movement: one square forward (two if unmoved), diagonal captures only. Promotions occur on the last rank.",
    ],
    chess.KNIGHT: [
        "Knights jump in an L-shape: two squares along one axis and one square perpendicular. They ignore intervening pieces.",
        "Knight movement is an L-shape — 2+1 squares. Knights leap over other pieces, so only board edges and friendly pieces block them.",
        "The knight hops in eight possible L-patterns. Friendly pieces block landing squares; the knight cannot be blocked mid-jump.",
    ],
    chess.BISHOP: [
        "Bishops slide diagonally any number of squares. They are blocked by the first piece on each diagonal ray.",
        "Bishop movement: unlimited diagonal slides along four rays, blocked by any intervening piece.",
        "Bishops move diagonally and can travel any distance, stopping at the first piece in their path.",
    ],
    chess.ROOK: [
        "Rooks slide orthogonally (horizontally and vertically) any number of squares, blocked by intervening pieces.",
        "Rook movement: unlimited horizontal and vertical slides. Castling rights do not affect move generation here.",
        "Rooks travel along ranks and files until hitting a piece or the board edge.",
    ],
    chess.QUEEN: [
        "The queen combines rook and bishop movement — she slides orthogonally or diagonally any number of squares.",
        "Queen movement: eight directional slides (rank, file, diagonal), blocked by the first piece in each direction.",
        "As the most powerful piece, the queen slides in any of eight directions until blocked.",
    ],
    chess.KING: [
        "The king moves exactly one square in any of eight directions. Castling must also be checked.",
        "Kings step one square in any direction. Squares controlled by the opponent are excluded. Castling is a separate check.",
        "One-square movement in all eight directions; squares under attack are disallowed.",
    ],
}

_FILTER_PIN = [
    "Cross-referencing against king-safety: any move that exposes the king to check is removed.",
    "Filtering for absolute pins — moves that leave the king in check are illegal.",
    "Applying king-safety filter: trajectories that expose the king are pruned.",
]

_FILTER_CHECK = [
    "The side is currently in check, so only moves that resolve the check are legal.",
    "In-check position: only moves that block, capture the attacker, or move the king are included.",
]


def _generate_thought(
    board:  chess.Board,
    origin: int,
    piece:  chess.Piece,
) -> Tuple[str, List[str]]:
    """
    Generate a CoT inner-monologue for a legal-moves query.

    Returns
    -------
    thought  : str  — the full thought string (goes inside <think>)
    targets  : list[str]  — unique destination square names (legal moves only)
    """
    origin_name = chess.square_name(origin)
    color_name  = _color(piece.color)
    piece_name  = _pname(piece.piece_type)

    parts: List[str] = []

    # 1. Opening
    parts.append(
        random.choice(_OPEN_TEMPLATES).format(
            color=color_name, piece_name=piece_name, sq=origin_name
        )
    )

    # 2. Visual grounding
    parts.append(
        random.choice(_GROUND_TEMPLATES).format(
            sq=origin_name, color=color_name, piece_name=piece_name
        )
    )

    # 3. Movement rules
    parts.append(random.choice(_RULE_TEMPLATES[piece.piece_type]))

    # 4. In-check notice
    if board.is_check():
        parts.append(random.choice(_FILTER_CHECK))

    # 5. Collect legal moves from this square
    legal_from_here = [m for m in board.legal_moves if m.from_square == origin]

    if not legal_from_here:
        # Explain WHY there are no moves
        pseudo = [m for m in board.pseudo_legal_moves if m.from_square == origin]
        if pseudo:
            parts.append(random.choice(_PINNED_NO_MOVES))
        else:
            parts.append(random.choice(_BLOCKED_NO_MOVES))
        return " ".join(parts), []

    # 6. Categorise moves
    quiets:      List[str] = []
    captures:    List[str] = []
    promotions:  List[str] = []   # destination squares with promotions (deduplicated)
    ep_squares:  List[str] = []
    castle_sqs:  List[str] = []
    seen_promo:  set        = set()

    for m in legal_from_here:
        dest = chess.square_name(m.to_square)
        if board.is_castling(m):
            castle_sqs.append(dest)
        elif m.promotion:
            if dest not in seen_promo:
                seen_promo.add(dest)
                promotions.append(dest)
        elif board.is_en_passant(m):
            ep_squares.append(dest)
        elif board.is_capture(m):
            captures.append(dest)
        else:
            quiets.append(dest)

    # 7. Narrate findings
    if quiets:
        parts.append(
            f"Found {len(quiets)} quiet "
            f"{_plural(len(quiets), 'destination')}: {', '.join(quiets)}."
        )
    if captures:
        parts.append(
            f"Found {len(captures)} capture "
            f"{_plural(len(captures), 'target')}: {', '.join(captures)}."
        )
    if promotions:
        parts.append(
            f"Promotion available on "
            f"{'square' if len(promotions) == 1 else 'squares'}: "
            f"{', '.join(promotions)}. "
            f"Each promotion square counts as one destination "
            f"(piece choice handled separately)."
        )
    if ep_squares:
        parts.append(
            random.choice(_EN_PASSANT_NOTICES).format(targets=", ".join(ep_squares))
        )
    if castle_sqs:
        parts.append(
            f"Castling {'is' if len(castle_sqs) == 1 else 'options are'} "
            f"available: {', '.join(castle_sqs)}."
        )

    # 8. Pin-filter notice (only if some pseudo-legal moves were removed)
    pseudo_count = sum(1 for _ in board.pseudo_legal_moves if _.from_square == origin)
    # pseudo-legal counts each promotion separately, so normalise
    pseudo_unique_dests = len({m.to_square for m in board.pseudo_legal_moves if m.from_square == origin})
    legal_unique_dests  = len({m.to_square for m in legal_from_here})
    if legal_unique_dests < pseudo_unique_dests:
        filtered = pseudo_unique_dests - legal_unique_dests
        parts.append(
            random.choice(_FILTER_PIN) +
            f" {filtered} destination(s) were removed."
        )

    # 9. Final target list (deduplicated, sorted for readability)
    all_targets = sorted(set(quiets + captures + promotions + ep_squares + castle_sqs))
    return " ".join(parts), all_targets


# ══════════════════════════════════════════════════════════════════════════════
# Packaging
# ══════════════════════════════════════════════════════════════════════════════

def _package(
    board:  chess.Board,
    raw_q:  str,
    answer: str,
) -> Dict:
    """
    Build the final JSONL dict using PromptBuilder as the single source of truth.

    Offline generation never has Maia running, so candidate_moves=None.
    The model must handle positions with and without the <candidate_moves>
    section — the XML structure makes the presence/absence explicit.
    """
    user_content = _builder.build_text_prompt(
        board           = board,
        user_message    = raw_q,
        candidate_moves = None,      # no Maia inference in offline mode
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
# Sampling functions
# ══════════════════════════════════════════════════════════════════════════════

def sample_specific_piece(board: chess.Board) -> Optional[Dict]:
    """
    SPECIFIC task: legal moves for a named piece on a named square.
    """
    active_squares = [
        s for s in chess.SQUARES
        if (p := board.piece_at(s)) and p.color == board.turn
    ]
    if not active_squares:
        return None

    origin = random.choice(active_squares)
    piece  = board.piece_at(origin)
    sq_name = chess.square_name(origin)
    p_name  = _pname(piece.piece_type)

    # Build question
    fmt = dict(piece=p_name, piece_name=p_name, sq=sq_name)
    q   = random.choice(SPECIFIC_Q).format(**fmt)

    # Build thought + target list
    thought_str, targets = _generate_thought(board, origin, piece)

    # Build answer
    if not targets:
        a = random.choice(SPECIFIC_A_NO_MOVES).format(
            thought=thought_str, piece=p_name, piece_name=p_name, sq=sq_name
        )
    else:
        a = random.choice(SPECIFIC_A_HAS_MOVES).format(
            thought   = thought_str,
            piece     = p_name,
            piece_name = p_name,
            sq        = sq_name,
            targets   = _join_targets(targets),
            n         = len(targets),
        )

    return _package(board, q, a)


def sample_ambiguous_piece(board: chess.Board) -> Optional[Dict]:
    """
    AMBIGUOUS task: legal moves for a piece type with no square specified.
    The model must identify all instances and enumerate each.
    """
    # Pick a random piece type (exclude king — its moves are less interesting
    # to enumerate and it is always present)
    piece_type = random.choice([
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN,
    ])
    p_name = _pname(piece_type)
    color  = board.turn

    q = random.choice(AMBIGUOUS_Q).format(piece_name=p_name)

    # Find all instances of this piece for the side to move
    instances = [
        s for s in chess.SQUARES
        if (p := board.piece_at(s))
        and p.color == color
        and p.piece_type == piece_type
    ]

    thought_parts: List[str] = [
        f"The user asks for legal moves of the {_color(color)} {p_name} "
        f"without specifying a square.",
        f"Scanning the <board_state> visual tokens for {_color(color)} {p_name}s...",
    ]

    # No instances
    if not instances:
        thought_parts.append(
            f"None found. The {_color(color)} side has no {p_name}s on the board."
        )
        ans = (
            f"<think>\n{' '.join(thought_parts)}\n</think>\n"
            f"You don't have any {p_name}s on the board right now."
        )
        return _package(board, q, ans)

    # Single instance — treat like a specific query
    if len(instances) == 1:
        origin    = instances[0]
        sq_name   = chess.square_name(origin)
        piece     = board.piece_at(origin)
        thought_parts.append(f"Found exactly one: {sq_name}.")
        sub_thought, targets = _generate_thought(board, origin, piece)
        thought_parts.append(sub_thought)

        if not targets:
            ans_body = (
                f"You have one {p_name} on {sq_name}, "
                f"but it has no legal moves in this position."
            )
        else:
            ans_body = (
                f"You have one {p_name} on {sq_name}. "
                f"Its legal moves are: {_join_targets(targets)}."
            )
        ans = f"<think>\n{' '.join(thought_parts)}\n</think>\n{ans_body}"
        return _package(board, q, ans)

    # Multiple instances — process each
    thought_parts.append(
        f"Found {len(instances)} {p_name}s. "
        f"I will evaluate each one separately."
    )
    ans_lines = [
        f"There are {len(instances)} {_color(color)} {p_name}s on the board:"
    ]

    for origin in instances:
        sq_name    = chess.square_name(origin)
        piece      = board.piece_at(origin)
        sub_thought, targets = _generate_thought(board, origin, piece)
        thought_parts.append(f"[{sq_name}] {sub_thought}")

        if not targets:
            ans_lines.append(f"- {p_name.capitalize()} on {sq_name}: no legal moves.")
        else:
            ans_lines.append(
                f"- {p_name.capitalize()} on {sq_name}: {_join_targets(targets)}."
            )

    ans = (
        f"<think>\n{' '.join(thought_parts)}\n</think>\n"
        + "\n".join(ans_lines)
    )
    return _package(board, q, ans)


def sample_invalid_premise(board: chess.Board) -> Optional[Dict]:
    """
    INVALID PREMISE task: user asks about a piece that is not on the given square.
    Teaches the model to be grounded rather than confabulate.
    Two sub-types:
      (a) Square has a DIFFERENT piece than asked about.
      (b) Square is completely EMPTY.
    """
    target_sq   = random.choice(chess.SQUARES)
    sq_name     = chess.square_name(target_sq)
    actual      = board.piece_at(target_sq)

    # Choose a piece type that is NOT the actual one on this square
    all_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                 chess.ROOK, chess.QUEEN, chess.KING]
    if actual:
        wrong_types = [t for t in all_types if t != actual.piece_type]
    else:
        wrong_types = all_types
    fake_type   = random.choice(wrong_types)
    fake_name   = _pname(fake_type)

    q = random.choice(INVALID_Q).format(
        piece=fake_name, piece_name=fake_name, sq=sq_name
    )

    thought_parts = [
        f"The user asks about a {fake_name} on {sq_name}.",
        f"Step 1: Check the <board_state> visual token at {sq_name}...",
    ]

    if actual:
        actual_name  = _pname(actual.piece_type)
        actual_color = _color(actual.color)
        thought_parts.append(
            f"The token at {sq_name} encodes a {actual_color} {actual_name}, "
            f"not a {fake_name}. The user's premise is incorrect."
        )
        ans_body = (
            f"There is no {fake_name} on {sq_name}. "
            f"That square has a {actual_color} {actual_name} on it."
        )
    else:
        thought_parts.append(
            f"The token at {sq_name} is empty — no piece is present there at all. "
            f"The user is asking about a piece that does not exist."
        )
        ans_body = (
            f"There is no {fake_name} on {sq_name} — that square is completely empty."
        )

    ans = f"<think>\n{' '.join(thought_parts)}\n</think>\n{ans_body}"
    return _package(board, q, ans)


# ══════════════════════════════════════════════════════════════════════════════
# Master entry point
# ══════════════════════════════════════════════════════════════════════════════

# Task weights: specific appears most (core grounding), others for diversity
_SAMPLERS = [
    sample_specific_piece,
    sample_specific_piece,
    sample_specific_piece,
    sample_ambiguous_piece,
    sample_ambiguous_piece,
    sample_invalid_premise,
]


def generate_sample(board: chess.Board) -> Optional[Dict]:
    """
    Generate one training example from a random task type.
    Returns None on degenerate positions (e.g. no active pieces).
    """
    return random.choice(_SAMPLERS)(board)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _random_board() -> chess.Board:
    """Return a random non-trivial board position via a short random game."""
    board = chess.Board()
    n_moves = random.randint(4, 40)
    for _ in range(n_moves):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Legal moves data generator")
    parser.add_argument("--n",   type=int, default=1,
                        help="Number of examples to generate (default: 1)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSONL file path (default: stdout)")
    parser.add_argument("--fen", type=str, default=None,
                        help="Use a specific FEN instead of random boards")
    args = parser.parse_args()

    out_file = open(args.out, "w") if args.out else sys.stdout

    generated = 0
    attempts   = 0
    max_attempts = args.n * 10

    while generated < args.n and attempts < max_attempts:
        attempts += 1
        board  = chess.Board(args.fen) if args.fen else _random_board()
        sample = generate_sample(board)
        if sample is None:
            continue
        out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        generated += 1

    if args.out:
        out_file.close()
        print(f"Generated {generated} examples → {args.out}", file=sys.stderr)