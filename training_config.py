"""
training_config.py
──────────────────
All training hyperparameters in one dataclass.
Kept separate from ChessCoachConfig so model code and training code
are independently importable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrainingConfig:

    # ── Paths ─────────────────────────────────────────────────────────────
    train_jsonl:    str  = "data/train.jsonl"
    val_jsonl:      Optional[str] = None        # skip eval if None
    output_dir:     str  = "checkpoints"
    run_name:       str  = "chess_coach_v1"

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_rank:      int  = 64
    lora_alpha:     int  = 128                  # scaling = alpha / rank = 2.0
    lora_dropout:   float = 0.05
    # Target every projection in attention + both FFN gates of Qwen3's SwiGLU
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # FFN
    ])

    # ── Batch / gradient accumulation ────────────────────────────────────
    per_device_batch_size:      int  = 4
    gradient_accumulation_steps: int = 4    # effective batch = 4*4 = 16
    max_seq_len:                int  = 2048

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Two param groups: projectors (train from scratch) need a higher LR
    lr_projectors:  float = 2e-4
    lr_lora:        float = 1e-4
    weight_decay:   float = 0.01
    betas:          Tuple[float, float] = (0.9, 0.95)
    max_grad_norm:  float = 1.0

    # ── Scheduler ─────────────────────────────────────────────────────────
    num_epochs:     int   = 3
    warmup_ratio:   float = 0.03            # fraction of total steps for warmup

    # ── Precision / Efficiency ────────────────────────────────────────────
    dtype:                  str  = "bfloat16"   # BF16 is native on Blackwell
    gradient_checkpointing: bool = True
    compile_llm:            bool = False        # torch.compile — disable if hooks cause issues

    # ── Data loader ───────────────────────────────────────────────────────
    num_workers:    int  = 4
    prefetch_factor: int = 2
    # Default elos used when the JSONL example omits them
    default_self_elo: int = 1500
    default_oppo_elo: int = 1500

    # ── Logging / checkpointing ────────────────────────────────────────────
    log_every_n_steps:  int = 10
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 500
    keep_last_n_ckpts:  int = 3             # rotate old checkpoints
    use_wandb:          bool = True
