"""
train.py
────────
Full training script for ChessCoach.

Trainable parameters
────────────────────
  • MaiaProjectors  — trains from scratch (lr_projectors, higher LR)
  • Qwen3 LoRA      — rank-64 adapters on all attention + FFN projections

Everything else (Maia backbone, Qwen base weights) is frozen.

RTX 5090 optimisations
──────────────────────
  • BF16 autocast  — native on Blackwell, zero overhead vs FP32
  • Flash Attention 2  — loaded via attn_implementation arg
  • TF32 matmuls  — torch.set_float32_matmul_precision("high")
  • Gradient checkpointing  — trades recomputation for ~40% VRAM reduction
  • Fused AdamW  — single CUDA kernel, faster than vanilla AdamW
  • torch.compile  — optional; off by default because hooks interact with dynamo

Usage
─────
    # Quick test run
    python train.py --train data/train.jsonl --output checkpoints/run1

    # Full config override
    python train.py \
        --train data/train.jsonl \
        --val   data/val.jsonl \
        --output checkpoints/run1 \
        --run_name chess_coach_v1 \
        --epochs 3 \
        --batch 4 \
        --accum 4 \
        --lr_proj 2e-4 \
        --lr_lora 1e-4 \
        --no_wandb
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, ".")
from config import ChessCoachConfig, MaiaConfig
from dataset import ChessBatch, JSONLDataset, build_dataloader
from model import ChessCoach
from training_config import TrainingConfig

log = logging.getLogger("chess_coach.train")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        datefmt = "%H:%M:%S",
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "train.log")),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model setup
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    coach_cfg:    ChessCoachConfig,
    train_cfg:    TrainingConfig,
    device:       torch.device,
) -> ChessCoach:
    """
    Instantiate ChessCoach, apply LoRA to Qwen, and set up all efficiency flags.
    """
    log.info("Building ChessCoach model...")
    model = ChessCoach(coach_cfg)

    # ── Gradient checkpointing on Qwen ────────────────────────────────────
    if train_cfg.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing: enabled")

    # ── Apply LoRA via PEFT ───────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type        = TaskType.CAUSAL_LM,
        r                = train_cfg.lora_rank,
        lora_alpha       = train_cfg.lora_alpha,
        lora_dropout     = train_cfg.lora_dropout,
        target_modules   = train_cfg.lora_target_modules,
        bias             = "none",
        # Keep the base model in BF16; LoRA matrices initialised in BF16 too
        inference_mode   = False,
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    model.llm.print_trainable_parameters()

    # ── TF32 matmuls (free speedup on Ampere/Blackwell) ───────────────────
    torch.set_float32_matmul_precision("high")

    # ── Optional torch.compile ────────────────────────────────────────────
    if train_cfg.compile_llm:
        log.info("Compiling LLM with torch.compile (mode=reduce-overhead)…")
        model.llm = torch.compile(model.llm, mode="reduce-overhead")

    model = model.to(device)
    log.info("Model on device: %s", device)

    _log_param_counts(model)
    return model


def _log_param_counts(model: ChessCoach) -> None:
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable
    log.info(
        "Parameters — total: %s | trainable: %s | frozen: %s",
        _fmt(total), _fmt(trainable), _fmt(frozen),
    )


def _fmt(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Optimiser + scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: ChessCoach, cfg: TrainingConfig) -> AdamW:
    """
    Two parameter groups with different learning rates:
      - MaiaProjectors: train from scratch → higher LR
      - Qwen LoRA adapters: fine-tune  → lower LR

    fused=True uses a single CUDA kernel per step — measurably faster on GPU.
    """
    projector_ids = {id(p) for p in model.projectors.parameters()}

    projector_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) in projector_ids]
    lora_params      = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in projector_ids]

    param_groups = [
        {"params": projector_params, "lr": cfg.lr_projectors, "name": "projectors"},
        {"params": lora_params,      "lr": cfg.lr_lora,       "name": "lora"},
    ]

    optimizer = AdamW(
        param_groups,
        betas        = cfg.betas,
        weight_decay = cfg.weight_decay,
        fused        = True,    # RTX 5090: single-kernel AdamW
    )
    log.info(
        "Optimizer: fused AdamW | lr_projectors=%.2e | lr_lora=%.2e",
        cfg.lr_projectors, cfg.lr_lora,
    )
    return optimizer


def build_scheduler(
    optimizer:   AdamW,
    num_warmup:  int,
    num_total:   int,
) -> LambdaLR:
    """
    Cosine decay with linear warmup.
    Implemented as a LambdaLR so it works with multiple param groups.
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return step / max(1, num_warmup)
        progress = (step - num_warmup) / max(1, num_total - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Precision context
# ─────────────────────────────────────────────────────────────────────────────

def autocast_ctx(dtype_str: str) -> contextlib.AbstractContextManager:
    """Return an appropriate autocast context for the configured dtype."""
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:         ChessCoach,
    optimizer:     AdamW,
    scheduler:     LambdaLR,
    update_count:  int,          # optimizer steps taken (used for checkpoint naming)
    global_step:   int,          # total forward passes = update_count * grad_accum
    epoch:         int,          # current epoch index (0-based)
    epoch_step:    int,          # batches completed inside current epoch
    loss:          float,
    output_dir:    str,
    keep_last_n:   int = 3,
) -> None:
    """
    Save projector weights + LoRA adapter weights + full training state.

    Two step counters are saved because they track different things:
      update_count : optimizer steps — drives logging / eval / ckpt schedules
      global_step  : forward passes  — drives gradient accumulation modulo
      epoch_step   : batches done in the current epoch — lets us skip ahead
                     when resuming mid-epoch so no example is trained twice
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{update_count}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Projectors (custom bridges — the only non-PEFT trainable weights)
    torch.save(
        model.projectors.state_dict(),
        ckpt_dir / "projectors.pt",
    )

    # LoRA adapter (PEFT saves as adapter_model.safetensors + adapter_config.json)
    model.llm.save_pretrained(ckpt_dir / "lora")

    # Training state (everything needed for an exact resume)
    torch.save(
        {
            "update_count": update_count,
            "global_step":  global_step,
            "epoch":        epoch,
            "epoch_step":   epoch_step,
            "loss":         loss,
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
        },
        ckpt_dir / "training_state.pt",
    )

    log.info("Saved checkpoint → %s", ckpt_dir)
    _rotate_checkpoints(output_dir, keep_last_n)


def _rotate_checkpoints(output_dir: str, keep_last_n: int) -> None:
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    for old in checkpoints[:-keep_last_n]:
        shutil.rmtree(old, ignore_errors=True)
        log.info("Removed old checkpoint: %s", old)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Scan output_dir for checkpoint-N folders and return the path of the
    one with the highest N.  Returns None if no checkpoints exist.
    """
    pattern = os.path.join(output_dir, "checkpoint-*")
    candidates = [
        p for p in glob.glob(pattern)
        if os.path.isfile(os.path.join(p, "training_state.pt"))
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: int(p.rsplit("-", 1)[-1]))
    return latest


def load_checkpoint(
    model:      ChessCoach,
    optimizer:  AdamW,
    scheduler:  LambdaLR,
    ckpt_dir:   str,
    device:     torch.device,
) -> Dict:
    """
    Restore model weights, optimizer state, scheduler state, and all step
    counters from a saved checkpoint.

    Returns the full training_state dict so the caller can restore:
        update_count, global_step, epoch, epoch_step
    """
    ckpt_path = Path(ckpt_dir)

    # Projectors
    model.projectors.load_state_dict(
        torch.load(
            ckpt_path / "projectors.pt",
            map_location=device,
            weights_only=True,
        )
    )
    log.info("Loaded projectors from %s", ckpt_path / "projectors.pt")

    # LoRA adapters (PEFT)
    model.llm.load_adapter(str(ckpt_path / "lora"), adapter_name="default")
    log.info("Loaded LoRA adapter from %s", ckpt_path / "lora")

    # Training state
    state = torch.load(
        ckpt_path / "training_state.pt",
        map_location=device,
        weights_only=True,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    # ── Backward-compatibility with old checkpoint format ─────────────────
    # Old format had only "step" (= update_count). New format has
    # update_count, global_step, and epoch_step stored separately.
    if "update_count" not in state:
        old_step = state.get("step", 0)
        state["update_count"] = old_step
        state["global_step"]  = old_step   # best-effort; grad_accum unknown
        state["epoch_step"]   = 0          # re-trains a partial epoch but won't crash
        log.warning(
            "Old checkpoint format detected — 'update_count' missing. "
            "Inferred update_count=%d from 'step'. "
            "epoch_step set to 0 (first few batches of this epoch may be re-trained).",
            old_step,
        )

    log.info(
        "Resumed from %s — epoch %d, update_step %d, epoch_step %d",
        ckpt_dir,
        state["epoch"],
        state["update_count"],
        state["epoch_step"],
    )
    return state


def load_weights_for_finetune(
    model:    ChessCoach,
    ckpt_dir: str,
    device:   torch.device,
) -> None:
    """
    FINETUNE mode — restore weights only, discard optimizer/scheduler state.

    Use this when:
      • The original training run completed and you want more epochs
      • You found specific weaknesses and want targeted re-training on new data
      • You want a fresh LR schedule (cosine restarts from the configured LR,
        not from near-zero where the previous run left off)

    The optimizer and scheduler are freshly built by the caller with whatever
    lr / num_epochs you configure for this new run.
    """
    _load_weights(model, Path(ckpt_dir), device)
    log.info(
        "Fine-tune: loaded weights from %s  "
        "(optimizer + scheduler are freshly initialised for this run)",
        ckpt_dir,
    )


def _load_weights(model: ChessCoach, ckpt_path: Path, device: torch.device) -> None:
    """Shared weight-loading used by both resume and finetune."""
    model.projectors.load_state_dict(
        torch.load(
            ckpt_path / "projectors.pt",
            map_location = device,
            weights_only = True,
        )
    )
    log.info("Loaded projectors  ← %s", ckpt_path / "projectors.pt")
    model.llm.load_adapter(str(ckpt_path / "lora"), adapter_name="default")
    log.info("Loaded LoRA adapter ← %s", ckpt_path / "lora")




def train_step(
    model:     ChessCoach,
    batch:     ChessBatch,
    device:    torch.device,
    dtype_str: str,
    scaler:    Optional[GradScaler],
    accum_steps: int,
) -> torch.Tensor:
    """
    Forward pass for one batch. Returns unscaled loss (for logging).
    Backward is called here; optimizer step happens in the outer loop
    once gradient_accumulation_steps batches have been processed.
    """
    # Move tensors to device
    maia_tokens = batch.maia_tokens.to(device, non_blocking=True)
    self_elos   = batch.self_elos.to(device,   non_blocking=True)
    oppo_elos   = batch.oppo_elos.to(device,   non_blocking=True)

    with autocast_ctx(dtype_str):
        output = model(
            boards        = batch.boards,
            maia_tokens   = maia_tokens,
            self_elos     = self_elos,
            oppo_elos     = oppo_elos,
            user_messages = batch.user_messages,
            assistant_messages = batch.answers,
            histories     = batch.histories,
            queried_ucis  = None,
        )
        loss = output.loss
        scaled_loss = loss / accum_steps

    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    # Return the raw loss for accurate logging
    return loss.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    model:      ChessCoach,
    val_loader: torch.utils.data.DataLoader,
    device:     torch.device,
    dtype_str:  str,
    max_batches: int = 50,
) -> float:
    """Returns mean validation loss over up to max_batches batches."""
    model.eval()
    total_loss  = 0.0
    num_batches = 0

    for batch in val_loader:
        if num_batches >= max_batches:
            break
        maia_tokens = batch.maia_tokens.to(device, non_blocking=True)
        self_elos   = batch.self_elos.to(device,   non_blocking=True)
        oppo_elos   = batch.oppo_elos.to(device,   non_blocking=True)

        with autocast_ctx(dtype_str):
            output = model(
                boards        = batch.boards,
                maia_tokens   = maia_tokens,
                self_elos     = self_elos,
                oppo_elos     = oppo_elos,
                user_messages = batch.user_messages,
                histories     = batch.histories,
                queried_ucis  = None,
            )
        total_loss  += output.loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(1, num_batches)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:        ChessCoach,
    train_cfg:    TrainingConfig,
    device:       torch.device,
    resume_from:  Optional[str] = None,
    finetune_from: Optional[str] = None,
) -> None:
    """
    Full training loop.

    resume_from   : path to a checkpoint to resume exactly (restores optimizer
                    + scheduler state). Use for crash recovery.
    finetune_from : path to a checkpoint to load weights from, with a FRESH
                    optimizer and scheduler. Use for continued training after
                    a completed run, or targeted fine-tuning on new data.
    """

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = JSONLDataset(
        train_cfg.train_jsonl,
        default_self_elo = train_cfg.default_self_elo,
        default_oppo_elo = train_cfg.default_oppo_elo,
    )
    val_ds = (
        JSONLDataset(
            train_cfg.val_jsonl,
            default_self_elo = train_cfg.default_self_elo,
            default_oppo_elo = train_cfg.default_oppo_elo,
        )
        if train_cfg.val_jsonl
        else None
    )

    train_loader = build_dataloader(
        train_ds,
        batch_size      = train_cfg.per_device_batch_size,
        shuffle         = True,
        num_workers     = train_cfg.num_workers,
        prefetch_factor = train_cfg.prefetch_factor,
    )
    val_loader = (
        build_dataloader(val_ds, batch_size=train_cfg.per_device_batch_size,
                         shuffle=False, num_workers=train_cfg.num_workers,
                         prefetch_factor=train_cfg.prefetch_factor)
        if val_ds else None
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(model, train_cfg)

    steps_per_epoch = math.ceil(len(train_ds) / train_cfg.per_device_batch_size)
    total_steps     = steps_per_epoch * train_cfg.num_epochs
    update_steps    = math.ceil(total_steps / train_cfg.gradient_accumulation_steps)
    warmup_steps    = math.ceil(update_steps * train_cfg.warmup_ratio)

    scheduler = build_scheduler(optimizer, warmup_steps, update_steps)

    log.info(
        "Training: epochs=%d | steps/epoch=%d | total_updates=%d | warmup=%d",
        train_cfg.num_epochs, steps_per_epoch, update_steps, warmup_steps,
    )

    # ── Precision — BF16 on 5090 doesn't need a GradScaler ───────────────
    scaler = (
        GradScaler() if train_cfg.dtype == "float16" else None
    )

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_run = None
    if train_cfg.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project = "chess-coach",
                name    = train_cfg.run_name,
                config  = {**vars(train_cfg)},
            )
            log.info("W&B run: %s", wandb_run.url)
        except ImportError:
            log.warning("wandb not installed — skipping W&B logging")

    # ── Resume / finetune ─────────────────────────────────────────────────
    start_epoch    = 0
    global_step    = 0
    update_count   = 0
    skip_batches   = 0

    if resume_from:
        # RESUME: restore weights + optimizer + scheduler + counters exactly.
        # The LR schedule continues from where it left off.
        state        = load_checkpoint(model, optimizer, scheduler, resume_from, device)
        update_count = state["update_count"]
        global_step  = state["global_step"]
        start_epoch  = state["epoch"]
        skip_batches = state["epoch_step"]
        log.info(
            "Resuming: epoch=%d | update_count=%d | global_step=%d | "
            "skipping first %d batches of epoch",
            start_epoch, update_count, global_step, skip_batches,
        )

    elif finetune_from:
        # FINETUNE: restore weights only.
        # Optimizer and scheduler are fresh — LR starts from the configured
        # initial value, epochs count from 0, no batches are skipped.
        load_weights_for_finetune(model, finetune_from, device)

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, train_cfg.num_epochs):
        log.info("── Epoch %d / %d ──", epoch + 1, train_cfg.num_epochs)
        epoch_loss  = 0.0
        epoch_steps = 0
        t0          = time.perf_counter()

        # Skip batches that were already trained in this epoch.
        # We do this by fast-forwarding the iterator; the DataLoader still
        # loads those batches but we discard them without running forward.
        # This only applies to the first resumed epoch; subsequent epochs
        # run in full.
        loader_iter = iter(train_loader)
        if skip_batches > 0:
            log.info("Fast-forwarding %d already-trained batches…", skip_batches)
            for _ in range(skip_batches):
                try:
                    next(loader_iter)
                except StopIteration:
                    break
            skip_batches = 0    # only skip on the first (resumed) epoch

        for step, batch in enumerate(loader_iter):
            global_step += 1

            # ── Forward + backward ────────────────────────────────────────
            loss = train_step(model, batch, device, train_cfg.dtype, scaler, train_cfg.gradient_accumulation_steps)

            epoch_loss  += loss.item()
            epoch_steps += 1

            # ── Gradient accumulation ─────────────────────────────────────
            if global_step % train_cfg.gradient_accumulation_steps == 0:
                update_count += 1

                # Unscale before clipping (no-op when scaler is None)
                if scaler:
                    scaler.unscale_(optimizer)

                grad_norm = nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.max_grad_norm,
                )

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # ── Logging ───────────────────────────────────────────────
                if update_count % train_cfg.log_every_n_steps == 0:
                    elapsed  = time.perf_counter() - t0
                    avg_loss = epoch_loss / epoch_steps
                    lr_proj  = optimizer.param_groups[0]["lr"]
                    lr_lora  = optimizer.param_groups[1]["lr"]

                    log.info(
                        "step %6d | loss %.4f | grad_norm %.3f | "
                        "lr_proj %.2e | lr_lora %.2e | %.1f s/step",
                        update_count, avg_loss, grad_norm,
                        lr_proj, lr_lora,
                        elapsed / epoch_steps,
                    )

                    if wandb_run:
                        wandb_run.log({
                            "train/loss":      avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/lr_proj":   lr_proj,
                            "train/lr_lora":   lr_lora,
                            "train/step":      update_count,
                            "train/epoch":     epoch,
                        })

                # ── Evaluation ────────────────────────────────────────────
                if (
                    val_loader is not None
                    and update_count % train_cfg.eval_every_n_steps == 0
                ):
                    val_loss = evaluate(model, val_loader, device, train_cfg.dtype)
                    log.info("val/loss = %.4f", val_loss)
                    if wandb_run:
                        wandb_run.log({"val/loss": val_loss, "train/step": update_count})
                    model.train()

                # ── Checkpointing ─────────────────────────────────────────
                if update_count % train_cfg.save_every_n_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        update_count = update_count,
                        global_step  = global_step,
                        epoch        = epoch,
                        epoch_step   = step + 1,    # batches done in this epoch
                        loss         = epoch_loss / epoch_steps,
                        output_dir   = train_cfg.output_dir,
                        keep_last_n  = train_cfg.keep_last_n_ckpts,
                    )

        log.info(
            "Epoch %d complete — mean loss %.4f",
            epoch + 1, epoch_loss / max(1, epoch_steps),
        )

    # ── Final save ────────────────────────────────────────────────────────
    save_checkpoint(
        model, optimizer, scheduler,
        update_count = update_count,
        global_step  = global_step,
        epoch        = train_cfg.num_epochs,
        epoch_step   = 0,           # epoch complete — nothing to skip on resume
        loss         = epoch_loss / max(1, epoch_steps),
        output_dir   = train_cfg.output_dir,
        keep_last_n  = 999,         # always keep the final checkpoint
    )
    log.info("Training complete.")

    if wandb_run:
        wandb_run.finish()


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ChessCoach")
    p.add_argument("--train",       required=True, help="Path to training JSONL")
    p.add_argument("--val",         default=None,  help="Path to validation JSONL")
    p.add_argument("--output",      default="checkpoints", help="Checkpoint output dir")
    p.add_argument("--run_name",    default="chess_coach_v1")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch",       type=int,   default=4,   help="Per-device batch size")
    p.add_argument("--accum",       type=int,   default=4,   help="Gradient accumulation steps")
    p.add_argument("--lr_proj",     type=float, default=2e-4)
    p.add_argument("--lr_lora",     type=float, default=1e-4)
    p.add_argument("--lora_rank",   type=int,   default=64)
    p.add_argument("--max_seq_len", type=int,   default=2048)
    p.add_argument("--resume",        default=None,
                   help="Checkpoint dir to resume from (restores optimizer + scheduler). "
                        "Use for crash recovery.")
    p.add_argument("--resume_latest", action="store_true",
                   help="Auto-detect and resume from the latest checkpoint in --output")
    p.add_argument("--finetune",      default=None,
                   help="Checkpoint dir to load weights from for a NEW training run. "
                        "Optimizer and scheduler are freshly initialised — use this "
                        "after a completed run to add more epochs or train on new data.")
    p.add_argument("--no_compile",  action="store_true", help="Disable torch.compile")
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--workers",     type=int,   default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for training.")
    device = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("GPU: %s | VRAM: %.1f GB", gpu_name, vram_gb)

    # ── Configs ────────────────────────────────────────────────────────────
    coach_cfg = ChessCoachConfig()
    train_cfg = TrainingConfig(
        train_jsonl               = args.train,
        val_jsonl                 = args.val,
        output_dir                = args.output,
        run_name                  = args.run_name,
        num_epochs                = args.epochs,
        per_device_batch_size     = args.batch,
        gradient_accumulation_steps = args.accum,
        lr_projectors             = args.lr_proj,
        lr_lora                   = args.lr_lora,
        lora_rank                 = args.lora_rank,
        max_seq_len               = args.max_seq_len,
        compile_llm               = not args.no_compile,
        use_wandb                 = not args.no_wandb,
        num_workers               = args.workers,
    )

    setup_logging(train_cfg.output_dir)
    log.info("Run: %s", train_cfg.run_name)
    log.info("Output: %s", train_cfg.output_dir)

    # ── Resolve which checkpoint mode to use ──────────────────────────────
    # Precedence: --resume_latest > --resume > --finetune > fresh start
    resume_from   = None
    finetune_from = None

    if args.resume_latest:
        resume_from = find_latest_checkpoint(args.output)
        if resume_from:
            log.info("Auto-detected latest checkpoint: %s", resume_from)
        else:
            log.info("No checkpoint found in %s — starting fresh", args.output)
    elif args.resume:
        resume_from = args.resume
    elif args.finetune:
        finetune_from = args.finetune
        log.info("Fine-tune mode: loading weights from %s", finetune_from)
        log.info("  LR and schedule are freshly initialised (not restored from checkpoint)")

    # ── Build + train ──────────────────────────────────────────────────────
    model = build_model(coach_cfg, train_cfg, device)
    train(model, train_cfg, device,
          resume_from   = resume_from,
          finetune_from = finetune_from)


if __name__ == "__main__":
    main()