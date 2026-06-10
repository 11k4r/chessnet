"""
chess_tokens.py
───────────────
LLaVA-style visual token extraction from a frozen Maia-3 model.

Produces eight types of token that are fed to Qwen alongside the text prompt:

  BOARD_SQUARE    (B, 64, D)   One token per square — full transformer output
  EVAL            (B,  1, D)   Position evaluation hidden state
  PONDER          (B,  1, D)   Move-complexity / think-time signal
  ELO_SELF        (B,  1, D)   Skill embedding for the side to move
  ELO_OPPO        (B,  1, D)   Skill embedding for the opponent
  TENSION_GLOBAL  (B,  1, D)   Mean residual-stream delta (layers 3 → 7)
  TENSION_PEAK    (B,  1, D)   Max  residual-stream delta (layers 3 → 7)
  POLICY_MOVE     (B, ≤8, D)   One token per candidate move (padded)

D = qwen_dim (4096 for Qwen3-8B)

Usage
-----
    extractor  = MaiaFeatureExtractor(maia_model, freeze_maia=True)
    projectors = MaiaProjectors()
    ...
    logits_move, _, _ = extractor(maia_tokens, self_elos, oppo_elos)
    visual = extract_all_tokens(boards, extractor, maia_tokens,
                                self_elos, oppo_elos, projectors, device)
"""

from __future__ import annotations

import chess
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

from policy_utils import CandidateMove, get_candidate_moves


# ─────────────────────────────────────────────────────────────────────────────
# 1. Token Type Registry
# ─────────────────────────────────────────────────────────────────────────────

class TokenType(IntEnum):
    BOARD_SQUARE    = 0
    EVAL            = 1
    PONDER          = 2
    POLICY_MOVE     = 3
    ELO_SELF        = 4
    ELO_OPPO        = 5
    TENSION_GLOBAL  = 6
    TENSION_PEAK    = 7
    NUM_TYPES       = 8   # sentinel — not a real type


# ─────────────────────────────────────────────────────────────────────────────
# 2. Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """2-layer MLP with GELU — the LLaVA-1.5 bridge pattern."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenTypeEmbedding(nn.Module):
    """
    Learnable per-type embedding added to visual tokens after projection.
    Broadcasting handles any leading batch / sequence dimensions automatically.

        tokens : (..., qwen_dim)
        output : (..., qwen_dim)   same shape, type embedding added
    """

    def __init__(self, qwen_dim: int = 4096) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(TokenType.NUM_TYPES), qwen_dim)

    def forward(self, tokens: torch.Tensor, token_type: TokenType) -> torch.Tensor:
        type_emb = self.embedding(
            torch.tensor(int(token_type), device=tokens.device)
        )                                       # (qwen_dim,)
        return tokens + type_emb                # broadcasts over all leading dims


class PolicyBridge(nn.Module):
    """
    Projects a single candidate move into Qwen's embedding space.

    Combines:
      • from_feat   (1024,) — from-square latent, augmented with learned
                              square-identity embedding (real board coords)
      • to_feat     (1024,) — to-square latent, same treatment
      • promo_signal  (4,)  — promotion bias (zeros for non-promotions)

    The square-identity embeddings use *real* board coordinates so the
    representation is consistent regardless of whose turn it is.
    """

    _FEAT_DIM  = 1024
    _PROMO_DIM = 4
    _IN_DIM    = _FEAT_DIM + _FEAT_DIM + _PROMO_DIM   # 2052

    def __init__(self, qwen_dim: int = 4096) -> None:
        super().__init__()
        self.from_sq_emb = nn.Embedding(64, self._FEAT_DIM)
        self.to_sq_emb   = nn.Embedding(64, self._FEAT_DIM)
        self.projector   = MLP(self._IN_DIM, qwen_dim, qwen_dim)

    def forward(
        self,
        from_feat:    torch.Tensor,  # (1024,)
        to_feat:      torch.Tensor,  # (1024,)
        promo_signal: torch.Tensor,  # (4,)
        real_from_sq: int,
        real_to_sq:   int,
    ) -> torch.Tensor:               # (qwen_dim,)
        device   = from_feat.device
        from_emb = self.from_sq_emb(torch.tensor(real_from_sq, device=device))
        to_emb   = self.to_sq_emb(torch.tensor(real_to_sq,     device=device))
        combined = torch.cat(
            [from_feat + from_emb, to_feat + to_emb, promo_signal], dim=-1
        )                            # (2052,)
        return self.projector(combined)


# ─────────────────────────────────────────────────────────────────────────────
# 3. All Projectors (single trainable module — easy to checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class MaiaProjectors(nn.Module):
    """
    Holds every bridge and the token-type embedding table.
    This is the *only* thing that trains; Maia is frozen, Qwen
    may be LoRA-tuned separately.
    """

    def __init__(
        self,
        maia_dim: int = 1024,
        elo_dim:  int = 128,
        qwen_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.board          = MLP(maia_dim, qwen_dim, qwen_dim)
        self.value_proj     = MLP(maia_dim, qwen_dim, qwen_dim)
        self.ponder         = MLP(maia_dim, qwen_dim, qwen_dim)
        self.elo            = MLP(elo_dim,  qwen_dim, qwen_dim)  # shared self / oppo
        self.tension_global = MLP(maia_dim, qwen_dim, qwen_dim)
        self.tension_peak   = MLP(maia_dim, qwen_dim, qwen_dim)
        self.policy         = PolicyBridge(qwen_dim=qwen_dim)
        self.type_emb       = TokenTypeEmbedding(qwen_dim=qwen_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Extractor — hook registration
# ─────────────────────────────────────────────────────────────────────────────

class MaiaFeatureExtractor(nn.Module):
    """
    Wraps a (frozen) Maia-3 model and registers eight forward hooks.

    The hook sites are declared as a class-level dict so they are easy to audit
    and extend without touching the forward logic.

    Feature shapes after a forward pass
    ─────────────────────────────────────
    board_state  (B, 64, 1024)   transformer.norm output
    eval_hid     (B,     1024)   fc_value_hid  output  (pre-ReLU)
    ponder_hid   (B,     1024)   fc_ponder_hid output  (pre-ReLU)
    sq_from      (B, 64, 1024)   proj_sq_from  output
    sq_to        (B, 64, 1024)   proj_sq_to    output
    promo_bias   (B,  8,    4)   promo_bias_proj output
    layer3_out   (B, 64, 1024)   transformer.layers[3] output
    layer7_out   (B, 64, 1024)   transformer.layers[7] output
    """

    # Maps feature name → function(base_model) → nn.Module to hook
    _HOOK_SITES: Dict[str, callable] = {
        "board_state": lambda m: m.transformer.norm,
        "eval_hid":    lambda m: m.fc_value_hid,
        "ponder_hid":  lambda m: m.fc_ponder_hid,
        "sq_from":     lambda m: m.proj_sq_from,
        "sq_to":       lambda m: m.proj_sq_to,
        "promo_bias":  lambda m: m.promo_bias_proj,
        "layer3_out":  lambda m: m.transformer.layers[3],
        "layer7_out":  lambda m: m.transformer.layers[7],
    }

    def __init__(self, maia_model: nn.Module, freeze_maia: bool = True) -> None:
        super().__init__()
        self.model       = maia_model
        self.freeze_maia = freeze_maia
        self.features: Dict[str, torch.Tensor] = {}
        self._handles:  List = []

        if freeze_maia:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self._register_hooks()

    # ── hooks ─────────────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        base = getattr(self.model, "module", self.model)   # unwrap DDP if needed
        for name, selector in self._HOOK_SITES.items():
            handle = selector(base).register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def _hook(_, __, output: torch.Tensor) -> None:
            self.features[name] = output.detach() if self.freeze_maia else output
        return _hook

    def remove_hooks(self) -> None:
        """Call this when you need raw Maia inference without the overhead."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        tokens:    torch.Tensor,   # (B, 64, token_dim)
        self_elos: torch.Tensor,   # (B,)
        oppo_elos: torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_on = torch.is_grad_enabled() and not self.freeze_maia
        with torch.set_grad_enabled(grad_on):
            return self.model(tokens, self_elos, oppo_elos)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Token Builders  (pure functions — easy to unit-test individually)
# ─────────────────────────────────────────────────────────────────────────────

def build_board_tokens(
    extractor:  MaiaFeatureExtractor,
    projectors: MaiaProjectors,
    boards:     List[chess.Board],
) -> torch.Tensor:
    """
    (B, 64, qwen_dim) — one token per board square, always in real
    board coordinates (a1 = index 0, h8 = index 63).

    The problem this solves
    ───────────────────────
    Maia-3 internally rank-flips the board when it is Black's turn so it
    always evaluates from the side-to-move's perspective. This means the
    hooked `board_state` tensor is in Maia's coordinate space, not real
    chess coordinates:

        White to move: token[42] = c6 features  ← correct, no flip
        Black to move: token[18] = c6 features  ← wrong! 18 is c3 in Qwen's eyes

    Qwen reads the 64 tokens as a fixed sequence with 1-D position IDs, so
    without correction it sees a rank-flipped board for every Black position.
    The training text labels always use real coordinates ("knight on c6"),
    creating an inconsistency that causes the model to output the mirrored
    square for Black positions while being correct for White.

    The fix
    ───────
    After hooking but before projection, for every batch item where it is
    Black's turn we reshape the 64 tokens into an (8, 8) grid and flip along
    the rank axis (dim=1). This maps Maia's rank-3 slot back to the real
    rank-6 slot, restoring absolute orientation.

    We .clone() first so the hook cache is never mutated in-place.
    """
    x = extractor.features["board_state"].clone()  # (B, 64, 1024) — never mutate the cache

    # Build a boolean mask: True for every batch item where Black is to move
    black_mask = torch.tensor(
        [board.turn == chess.BLACK for board in boards],
        device = x.device,
        dtype  = torch.bool,
    )

    if black_mask.any():
        # Reshape to (B_black, 8, 8, 1024), flip ranks, flatten back to (B_black, 64, 1024)
        feat_dim = x.size(-1)
        grid     = x[black_mask].view(-1, 8, 8, feat_dim)
        x[black_mask] = torch.flip(grid, dims=[1]).view(-1, 64, feat_dim)

    x = projectors.board(x)                        # (B, 64, qwen_dim)
    return projectors.type_emb(x, TokenType.BOARD_SQUARE)


def build_eval_token(
    extractor:  MaiaFeatureExtractor,
    projectors: MaiaProjectors,
) -> torch.Tensor:
    """(B, 1, qwen_dim) — position evaluation hidden state."""
    x = extractor.features["eval_hid"]             # (B, 1024)
    x = projectors.value_proj(x).unsqueeze(1)      # (B, 1, qwen_dim)
    return projectors.type_emb(x, TokenType.EVAL)


def build_ponder_token(
    extractor:  MaiaFeatureExtractor,
    projectors: MaiaProjectors,
) -> torch.Tensor:
    """(B, 1, qwen_dim) — move-complexity / think-time signal."""
    x = extractor.features["ponder_hid"]           # (B, 1024)
    x = projectors.ponder(x).unsqueeze(1)          # (B, 1, qwen_dim)
    return projectors.type_emb(x, TokenType.PONDER)


def build_elo_tokens(
    extractor:  MaiaFeatureExtractor,
    self_elos:  torch.Tensor,
    oppo_elos:  torch.Tensor,
    projectors: MaiaProjectors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B, 1, qwen_dim) × 2 — skill embeddings for both players."""
    base    = getattr(extractor.model, "module", extractor.model)
    grad_on = torch.is_grad_enabled() and not extractor.freeze_maia

    with torch.set_grad_enabled(grad_on):
        self_emb = base.interpolate_elo(self_elos)  # (B, 128)
        oppo_emb = base.interpolate_elo(oppo_elos)  # (B, 128)

    self_tok = projectors.elo(self_emb).unsqueeze(1)   # (B, 1, qwen_dim)
    oppo_tok = projectors.elo(oppo_emb).unsqueeze(1)   # (B, 1, qwen_dim)

    self_tok = projectors.type_emb(self_tok, TokenType.ELO_SELF)
    oppo_tok = projectors.type_emb(oppo_tok, TokenType.ELO_OPPO)
    return self_tok, oppo_tok


def build_tension_tokens(
    extractor:  MaiaFeatureExtractor,
    projectors: MaiaProjectors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (B, 1, qwen_dim) × 2 — residual-stream delta between layers 3 and 7.

    global token : mean delta across all 64 squares → broad positional change
    peak   token : max  delta across all 64 squares → sharpest local change
    """
    l3    = extractor.features["layer3_out"]        # (B, 64, 1024)
    l7    = extractor.features["layer7_out"]        # (B, 64, 1024)
    delta = l7 - l3                                 # (B, 64, 1024)

    global_feat = delta.mean(dim=1)                 # (B, 1024)
    peak_feat,_ = delta.max(dim=1)                  # (B, 1024)

    global_tok = projectors.tension_global(global_feat).unsqueeze(1)
    peak_tok   = projectors.tension_peak(peak_feat).unsqueeze(1)

    global_tok = projectors.type_emb(global_tok, TokenType.TENSION_GLOBAL)
    peak_tok   = projectors.type_emb(peak_tok,   TokenType.TENSION_PEAK)
    return global_tok, peak_tok


def build_policy_tokens(
    extractor:       MaiaFeatureExtractor,
    projectors:      MaiaProjectors,
    candidate_moves: List[List[CandidateMove]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (B, ≤8, qwen_dim), (B, ≤8) bool mask — one token per candidate move.

    Feature lookup uses Maia-space square indices (correct for mirrored boards).
    Square-identity embeddings use real-space indices (consistent notation).
    """
    sq_from_all = extractor.features["sq_from"]    # (B, 64, 1024)
    sq_to_all   = extractor.features["sq_to"]      # (B, 64, 1024)
    promo_all   = extractor.features["promo_bias"] # (B,  8,    4)

    B      = sq_from_all.size(0)
    device = sq_from_all.device
    dtype  = sq_from_all.dtype
    out_dim = projectors.policy.projector.net[-1].out_features

    per_batch: List[torch.Tensor] = []

    for b in range(B):
        moves = candidate_moves[b]

        if not moves:
            per_batch.append(torch.empty(0, out_dim, device=device, dtype=dtype))
            continue

        tokens_for_b: List[torch.Tensor] = []
        for move in moves:
            from_feat = sq_from_all[b, move.maia_from_sq, :]   # (1024,)
            to_feat   = sq_to_all[b,   move.maia_to_sq,   :]   # (1024,)

            is_promo     = len(move.real_uci) == 5
            to_file      = chess.square_file(move.maia_to_sq)
            promo_signal = (
                promo_all[b, to_file, :]
                if is_promo
                else torch.zeros(4, device=device, dtype=dtype)
            )

            tok = projectors.policy(
                from_feat, to_feat, promo_signal,
                move.real_from_sq, move.real_to_sq,
            )                                                  # (qwen_dim,)
            tokens_for_b.append(tok.unsqueeze(0))              # (1, qwen_dim)

        batch_tensor = torch.cat(tokens_for_b, dim=0)          # (N, qwen_dim)
        batch_tensor = projectors.type_emb(batch_tensor, TokenType.POLICY_MOVE)
        per_batch.append(batch_tensor)

    return _pad_and_mask(per_batch, device, dtype, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Output Container + Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChessVisualTokens:
    """
    All visual tokens for one forward pass, ready to be injected into Qwen.

    Shapes
    ──────
    board          (B, 64, D)
    eval           (B,  1, D)
    ponder         (B,  1, D)
    elo_self       (B,  1, D)
    elo_oppo       (B,  1, D)
    tension_global (B,  1, D)
    tension_peak   (B,  1, D)
    policy_tokens  (B, ≤8, D)   padded
    policy_mask    (B, ≤8)      True for real tokens, False for padding
    candidate_moves             raw move data for prompt construction
    """
    board:           torch.Tensor
    value_proj:      torch.Tensor
    ponder:          torch.Tensor
    elo_self:        torch.Tensor
    elo_oppo:        torch.Tensor
    tension_global:  torch.Tensor
    tension_peak:    torch.Tensor
    policy_tokens:   torch.Tensor
    policy_mask:     torch.Tensor
    candidate_moves: List[List[CandidateMove]]


def extract_all_tokens(
    boards:            List[chess.Board],
    extractor:         MaiaFeatureExtractor,
    maia_input_tokens: torch.Tensor,           # (B, 64, token_dim)
    self_elos:         torch.Tensor,           # (B,)
    oppo_elos:         torch.Tensor,           # (B,)
    projectors:        MaiaProjectors,
    device:            torch.device,
    user_queried_ucis: Optional[List[Optional[str]]] = None,
    max_moves:         int = 8,
) -> ChessVisualTokens:
    """
    Single forward pass through Maia → all visual tokens.

    Parameters
    ----------
    boards             : python-chess Board objects, one per batch item
    extractor          : MaiaFeatureExtractor wrapping the frozen model
    maia_input_tokens  : pre-built input tensor (B, 64, token_dim)
    self_elos          : Elo ratings of the side to move  (B,)
    oppo_elos          : Elo ratings of the opponent      (B,)
    projectors         : MaiaProjectors (the trainable part)
    device             : target device
    user_queried_ucis  : optional per-batch move UCIs to guarantee inclusion
    max_moves          : maximum candidate moves per position
    """
    if user_queried_ucis is None:
        user_queried_ucis = [None] * len(boards)

    # ── single Maia forward pass — triggers all eight hooks ──────────────
    logits_move, _, _ = extractor(maia_input_tokens, self_elos, oppo_elos)

    # ── policy candidate selection (per board, CPU-side chess logic) ─────
    candidate_moves = [
        get_candidate_moves(board, logits_move[b], user_queried_ucis[b], max_moves)
        for b, board in enumerate(boards)
    ]

    # ── build all token types ─────────────────────────────────────────────
    elo_self, elo_oppo       = build_elo_tokens(extractor, self_elos, oppo_elos, projectors)
    tension_global, tension_peak = build_tension_tokens(extractor, projectors)
    policy_tokens, policy_mask   = build_policy_tokens(extractor, projectors, candidate_moves)

    return ChessVisualTokens(
        board           = build_board_tokens(extractor, projectors, boards),
        value_proj      = build_eval_token(extractor, projectors),
        ponder          = build_ponder_token(extractor, projectors),
        elo_self        = elo_self,
        elo_oppo        = elo_oppo,
        tension_global  = tension_global,
        tension_peak    = tension_peak,
        policy_tokens   = policy_tokens,
        policy_mask     = policy_mask,
        candidate_moves = candidate_moves,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pad_and_mask(
    sequences: List[torch.Tensor],   # each (N_i, out_dim), N_i may be 0
    device:    torch.device,
    dtype:     torch.dtype,
    out_dim:   int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a variable-length list of token sequences into a (B, max_N, out_dim)
    tensor with a corresponding boolean attention mask.
    Empty sequences (checkmate / stalemate) receive a zero sentinel token
    and a fully-False mask row so downstream code never sees empty rows.
    """
    sentinel = torch.zeros(1, out_dim, device=device, dtype=dtype)
    to_pad, lengths = [], []

    for seq in sequences:
        if seq.size(0) == 0:
            to_pad.append(sentinel)
            lengths.append(0)
        else:
            to_pad.append(seq)
            lengths.append(seq.size(0))

    padded = pad_sequence(to_pad, batch_first=True, padding_value=0.0)
    B, max_len = padded.size(0), padded.size(1)

    mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)
    for i, length in enumerate(lengths):
        if length > 0:
            mask[i, :length] = True

    return padded, mask