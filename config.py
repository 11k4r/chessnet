"""
config.py
─────────
All configuration for the ChessCoach model.
Two dataclasses: MaiaConfig (mirrors Cfg79m as a proper dataclass)
and ChessCoachConfig (top-level settings).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MaiaConfig:
    """
    Configuration for the frozen Maia-3 79M backbone.
    Mirrors the notebook's Cfg79m exactly so MAIA3Model accepts it directly.
    """
    checkpoint_path:     str   = "maia3-79/maia3-79m.pt"
    history:             int   = 8
    dim_vit:             int   = 1024
    dim_emb:             int   = 128
    num_blocks:          int   = 8
    num_heads:           int   = 32
    head_hid_dim:        int   = 1024
    mlp_ratio:           float = 2.0
    dropout:             float = 0.0
    use_gab:             bool  = True
    gab_gen_size:        int   = 128
    gab_per_square_dim:  int   = 32
    gab_intermediate_dim: int  = 128
    use_relative_bias:   bool  = False
    use_absolute_pe:     bool  = False
    activation:          str   = "gelu"
    use_rms_norm:        bool  = True
    omit_qkv_biases:     bool  = True
    include_time_info:   bool  = False
    device:              str   = "cpu"


@dataclass
class ChessCoachConfig:
    """Top-level configuration for the full ChessCoach model."""

    # ── Sub-configs ────────────────────────────────────────────────────────
    maia: MaiaConfig = field(default_factory=MaiaConfig)

    # ── Qwen backbone ──────────────────────────────────────────────────────
    qwen_path:  str = "Qwen3"
    qwen_dim:   int = 4096          # Qwen3-8B hidden dim

    # ── Projection ─────────────────────────────────────────────────────────
    maia_dim:   int = 1024          # dim_vit of Maia
    elo_dim:    int = 128           # dim_emb of Maia

    # ── Behaviour ──────────────────────────────────────────────────────────
    freeze_maia:         bool = True
    max_candidate_moves: int  = 8

    # ── Special tokens ────────────────
    vis_token: str = "<vis>"
    wait_token: str = "<|wait|>"

    # ── System prompt ──────────────────────────────────────────────────────
    system_prompt: str = (
        "You are an expert chess coach. "
        "You receive the board position encoded as visual tokens alongside "
        "position evaluation, move complexity, player ratings, board dynamics, "
        "and a ranked list of candidate moves. "
        "Provide clear, insightful analysis and coaching advice tailored to "
        "the player's skill level. Think step-by-step before giving your answer."
    )

    # ── Inference ──────────────────────────────────────────────────────────
    enable_thinking: bool  = True   # Qwen3 native CoT (<think> blocks)
    strip_thinking:  bool  = False  # If True, remove <think> from returned text
    max_new_tokens:  int   = 1024
    temperature:     float = 0.7
    top_p:           float = 0.9
    do_sample:       bool  = True
