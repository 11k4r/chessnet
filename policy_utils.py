"""
policy_utils.py
───────────────
Candidate move extraction from Maia-3 logits.

All outputs are in *real* board coordinates: if it is Black's turn the
mirroring that Maia applies internally is undone before returning, so
callers and the LLM prompt always work in standard chess notation.
"""

from __future__ import annotations

import chess
import torch
from typing import List, NamedTuple, Optional

from maia3.utils import get_all_possible_moves, mirror_move

# ── Move index table (built once at import time) ─────────────────────────────

_ALL_MOVES       = get_all_possible_moves()          # list[str], length 4352
_ALL_MOVES_INDEX = {uci: i for i, uci in enumerate(_ALL_MOVES)}


# ── Data contract ─────────────────────────────────────────────────────────────

class CandidateMove(NamedTuple):
    """
    A single candidate move returned for one board position.

    Two sets of square indices are carried because they serve different roles
    inside build_policy_tokens:

    • maia_from_sq / maia_to_sq  — index into the hooked sq_from / sq_to
      tensors, which live in Maia's (possibly mirrored) square space.

    • real_from_sq / real_to_sq  — index used for the learned square-identity
      embeddings, always in standard board coordinates (a1=0 … h8=63).
    """
    real_uci:     str    # standard UCI for prompt text, e.g. "e2e4"
    prob:         float  # softmax probability from Maia
    maia_from_sq: int    # from-square in Maia's space  (feature lookup)
    maia_to_sq:   int    # to-square   in Maia's space  (feature lookup)
    real_from_sq: int    # from-square in real space     (sq embedding)
    real_to_sq:   int    # to-square   in real space     (sq embedding)


# ── Public API ────────────────────────────────────────────────────────────────

def get_candidate_moves(
    board:              chess.Board,
    logits_move:        torch.Tensor,       # (4352,) — single board, raw logits
    user_queried_ucis:   Optional[str] = None,
    max_moves:          int   = 8,
    min_moves:          int   = 2,
    relative_threshold: float = 0.10,
) -> List[CandidateMove]:
    """
    Select top candidate moves for one position.

    Steps
    -----
    1. Build a legal-move mask in Maia's (possibly mirrored) move space.
    2. Mask illegal moves with -inf, softmax in fp32.
    3. Take top-k; prune moves whose probability falls below
       ``relative_threshold * best_prob`` (keeping at least ``min_moves``).
    4. Guarantee the user-queried move is included if it is legal.
    5. Return everything in real board coordinates.
    """
    is_black = board.turn == chess.BLACK

    # 1. Legal mask
    mask = _build_legal_mask(board, is_black, logits_move.device)
    if mask.sum() == 0:
        return []                            # checkmate / stalemate

    # 2. Mask + softmax (fp32 for numerical safety)
    logits_fp32 = logits_move.float()[: len(_ALL_MOVES)]
    logits_fp32 = logits_fp32.masked_fill(~mask, float("-inf"))
    probs       = torch.softmax(logits_fp32, dim=-1)

    # 3. Top-k with relative threshold pruning
    k = min(max_moves, int(mask.sum().item()))
    top_probs, top_indices = torch.topk(probs, k=k)
    best_prob = top_probs[0].item()

    candidates: List[CandidateMove] = []
    for prob_val, idx in zip(top_probs.tolist(), top_indices.tolist()):
        if prob_val == 0.0:
            break
        if prob_val < best_prob * relative_threshold and len(candidates) >= min_moves:
            break
        candidates.append(_make_candidate(_ALL_MOVES[idx], prob_val, is_black))

    # 4. Inject user-queried move if missing
    if user_queried_ucis:
        legal_real = {m.uci() for m in board.legal_moves}
        for uci in user_queried_ucis: # 2. Iterate through the list
            if uci in legal_real:
                already_present = any(c.real_uci == uci for c in candidates)
                if not already_present:
                    maia_uci = mirror_move(uci) if is_black else uci
                    idx      = _ALL_MOVES_INDEX.get(maia_uci)
                    prob_val = probs[idx].item() if idx is not None else 0.0
                    candidates.append(_make_candidate(maia_uci, prob_val, is_black))

    return candidates


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_legal_mask(
    board:    chess.Board,
    is_black: bool,
    device:   torch.device,
) -> torch.Tensor:
    """Boolean mask over _ALL_MOVES; True where a move is legal."""
    mask = torch.zeros(len(_ALL_MOVES), device=device, dtype=torch.bool)
    for move in board.legal_moves:
        uci = mirror_move(move.uci()) if is_black else move.uci()
        idx = _ALL_MOVES_INDEX.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def _make_candidate(
    maia_uci: str,
    prob:     float,
    is_black: bool,
) -> CandidateMove:
    """Build a CandidateMove, converting Maia-space UCI to real coordinates."""
    real_uci     = mirror_move(maia_uci) if is_black else maia_uci
    maia_from_sq = chess.parse_square(maia_uci[:2])
    maia_to_sq   = chess.parse_square(maia_uci[2:4])
    real_from_sq = chess.parse_square(real_uci[:2])
    real_to_sq   = chess.parse_square(real_uci[2:4])
    return CandidateMove(real_uci, prob, maia_from_sq, maia_to_sq, real_from_sq, real_to_sq)
