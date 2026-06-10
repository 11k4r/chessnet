"""
dataset.py
──────────
Dataset contract and JSONL implementation for ChessCoach training.

Design principle
────────────────
The dataset is responsible for:
  1. Reading Q&A pairs from disk (JSONL).
  2. Reconstructing chess.Board objects from FEN strings.
  3. Computing Maia-3 board tokens (the (64, 96) float tensor).
  4. Returning everything the model's forward() needs.

The dataset is NOT responsible for:
  - Prompt assembly (PromptBuilder owns that).
  - Visual token extraction (ChessCoach.forward() owns that).
  - Tokenisation (ChessCoach._build_batch_inputs() owns that).

JSONL schema
────────────
Each line must be a JSON object with:

  Required
  ────────
  "fen"         : str  — current board FEN
  "messages"    : list — ShareGPT format:
                    [{"role": "system",    "content": "..."},
                     {"role": "user",      "content": "..."},   # contains "Question: ..."
                     {"role": "assistant", "content": "..."}]

  Optional (fall back to TrainingConfig defaults if absent)
  ────────
  "self_elo"       : int  — Elo of the side to move
  "oppo_elo"       : int  — Elo of the opponent
  "history_fens"   : list[str] — prior FENs (oldest→newest, excluding current)
                      Used for Maia's 8-position history window.
                      If absent, the current position is replicated.

Extending
─────────
Subclass ChessDataset and override __getitem__ to add new data sources.
The collate_fn works with any subclass that returns ChessSample.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import chess
import torch
from torch.utils.data import DataLoader, Dataset

# Maia-3 board tokenisation
sys.path.insert(0, ".")
from maia3.dataset import get_historical_tokens, tokenize_board
from config import MaiaConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-example container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChessSample:
    """
    Everything the model needs for one training example.

    maia_tokens : (64, token_dim) float32 — board representation for Maia
    self_elo    : int — raw Elo value of the side to move
    oppo_elo    : int — raw Elo value of the opponent
    board       : chess.Board — reconstructed from FEN
    user_message: str — bare question text (no XML wrapper)
    answer      : str — full assistant response (including <think> block)
    history     : prior conversation turns [{role, content}]
    """
    maia_tokens:  torch.Tensor          # (64, token_dim)
    self_elo:     int
    oppo_elo:     int
    board:        chess.Board
    user_message: str
    answer:       str
    history:      List[Dict[str, str]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ChessDataset(Dataset, ABC):
    """Abstract base class. Subclass and implement __len__ and __getitem__."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> ChessSample: ...


# ─────────────────────────────────────────────────────────────────────────────
# 3. JSONL implementation
# ─────────────────────────────────────────────────────────────────────────────

class JSONLDataset(ChessDataset):
    """
    Reads ChessSample objects from a JSONL file produced by the data generators.

    Token computation is done once per example in __getitem__; for large
    datasets consider caching tokens to disk (e.g. with HDF5 or .pt files).
    """

    def __init__(
        self,
        jsonl_path:       str,
        maia_history:     int = 8,          # must match MaiaConfig.history
        default_self_elo: int = 1500,
        default_oppo_elo: int = 1500,
        max_examples:     Optional[int] = None,
    ) -> None:
        self.maia_history     = maia_history
        self.default_self_elo = default_self_elo
        self.default_oppo_elo = default_oppo_elo

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        self._records: List[Dict] = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                self._records.append(json.loads(line))
                if max_examples and i + 1 >= max_examples:
                    break

        print(f"[JSONLDataset] Loaded {len(self._records)} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> ChessSample:
        rec = self._records[idx]

        fen_str = rec.get("fen")

        if fen_str is not None:
            # ── Board reconstruction ──────────────────────────────────────────
            board = chess.Board(rec["fen"])
    
            # ── History boards for Maia's sliding window ──────────────────────
            history_fens = rec.get("history_fens", [])
            history_boards = [chess.Board(f) for f in history_fens]
            # Board list expected by get_historical_tokens: most recent first
            board_sequence = [board] + list(reversed(history_boards))
            tokenized_sequence = [tokenize_board(b) for b in board_sequence]

            maia_cfg = MaiaConfig(history=self.maia_history)
    
            # ── Maia board tokens ─────────────────────────────────────────────
            # get_historical_tokens returns a numpy array (64, 12 * history)
            # Padding (replicate earliest) is handled by maia3 internally
            # get_historical_tokens returns a numpy array (64, 12 * history)
            historical_tokens_tensor = get_historical_tokens(
                board_history=tokenized_sequence,
                cfg=maia_cfg,          
                base=0.0, 
                inc=0.0, 
                clk_left_before=0.0, 
                clk_ponder=0.0
            )
            maia_tokens = historical_tokens_tensor.float()
        else:
            board = None
            maia_tokens = torch.zeros((64, 96), dtype=torch.float32) 

        # ── Elo values ────────────────────────────────────────────────────
        self_elo = rec.get("self_elo", self.default_self_elo)
        oppo_elo = rec.get("oppo_elo", self.default_oppo_elo)

        # ── Parse messages ────────────────────────────────────────────────
        messages     = rec["messages"]
        user_content = _find_role(messages, "user")
        answer       = _find_role(messages, "assistant")

        # Extract the raw question from the XML-wrapped user content.
        # The template always ends with "Question: {raw_question}" on its own line.
        user_message = _extract_raw_question(user_content)

        return ChessSample(
            maia_tokens  = maia_tokens,
            self_elo     = self_elo,
            oppo_elo     = oppo_elo,
            board        = board,
            user_message = user_message,
            answer       = answer,
            history      = [],              # single-turn for now
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Collate function
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChessBatch:
    """
    Collated batch ready for ChessCoach.forward().

    Note: boards and string lists can't be stacked — they stay as Python lists.
    maia_tokens and elos are padded/stacked into tensors.
    """
    maia_tokens:   torch.Tensor           # (B, 64, token_dim)
    self_elos:     torch.Tensor           # (B,)   float32 raw elo values
    oppo_elos:     torch.Tensor           # (B,)   float32 raw elo values
    boards:        List[chess.Board]
    user_messages: List[str]
    answers:       List[str]
    histories:     List[List[Dict[str, str]]]


def collate_fn(samples: List[ChessSample]) -> ChessBatch:
    """
    Stack tensors; keep everything else as lists.
    maia_tokens are assumed to share the same shape (64, token_dim).
    """
    return ChessBatch(
        maia_tokens   = torch.stack([s.maia_tokens for s in samples]),   # (B, 64, 96)
        self_elos     = torch.tensor([s.self_elo for s in samples], dtype=torch.float32),
        oppo_elos     = torch.tensor([s.oppo_elo for s in samples], dtype=torch.float32),
        boards        = [s.board        for s in samples],
        user_messages = [s.user_message for s in samples],
        answers       = [s.answer       for s in samples],
        histories     = [s.history      for s in samples],
    )


def build_dataloader(
    dataset:        ChessDataset,
    batch_size:     int,
    shuffle:        bool = True,
    num_workers:    int  = 4,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Convenience wrapper that wires up the collate function."""
    return DataLoader(
        dataset,
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor if num_workers > 0 else None,
        collate_fn      = collate_fn,
        pin_memory      = True,
        persistent_workers = num_workers > 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_role(messages: List[Dict], role: str) -> str:
    """Return the content of the first message with the given role."""
    for m in messages:
        if m["role"] == role:
            return m["content"]
    raise ValueError(f"No message with role '{role}' found.")


def _extract_raw_question(user_content: str) -> str:
    """
    Pull the bare question text out of the XML-wrapped user content.

    The template always ends with 'Question: {text}', so we split on that
    marker and take everything after it.  Falls back to the full content
    if the marker is absent (e.g. plain-text turns in multi-turn data).
    """
    marker = "Question: "
    if marker in user_content:
        return user_content.split(marker, 1)[1].strip()
    return user_content.strip()
