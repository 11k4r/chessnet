"""
prompt_builder.py
─────────────────
Constructs structured Qwen3 chat prompts that interleave natural language
with XML tags and <vis> placeholders for each visual token type.

Single Source of Truth
──────────────────────
`build_text_prompt` is the canonical definition of the prompt structure.
It is safe to call offline (no tensors, no Maia) — the data generators use
it directly.  `position_segment` calls it and layers on the live tensors,
ensuring training data and live inference always see the same XML skeleton.

Visual token ordering contract
───────────────────────────────
The <vis> placeholders appear in build_text_prompt in this fixed order:
  1.  64 ×  board_state   (from visual_tokens.board)
  2.   1 ×  evaluation    (from visual_tokens.eval)
  3.   1 ×  complexity    (from visual_tokens.ponder)
  4.   1 ×  tension global(from visual_tokens.tension_global)
  5.   1 ×  tension peak  (from visual_tokens.tension_peak)
  6.   1 ×  elo self      (from visual_tokens.elo_self)
  7.   1 ×  elo oppo      (from visual_tokens.elo_oppo)
  8.   N ×  candidate     (from visual_tokens.policy_tokens[:valid_n])

position_segment collects tensors in exactly this order.
The runtime assertion enforces the invariant on every forward pass.
"""

from __future__ import annotations

import re
import chess
try:
    import torch  # not needed for offline data generation
except ImportError:
    torch = None  # offline mode — position_segment will raise if called without torch
from typing import Dict, List, Optional, Tuple

from chess_tokens import ChessVisualTokens
from config import ChessCoachConfig


class PromptBuilder:
    """
    Builds Qwen3-format conversation prompts with <vis> placeholders.
    """

    def __init__(self, config: ChessCoachConfig, tokenizer=None) -> None:
        self.config    = config
        self.tokenizer = tokenizer   # Optional — not needed for offline data gen
        self.vis       = config.vis_token

    # ─────────────────────────────────────────────────────────────────────
    # 1. Canonical prompt structure  (OFFLINE SAFE — no tensors required)
    # ─────────────────────────────────────────────────────────────────────

    def build_text_prompt(
        self,
        board:           chess.Board,
        user_message:    str,
        candidate_moves: Optional[List[Tuple[str, float]]] = None,
        # list of (real_uci, probability) — None means no policy section
    ) -> str:
        """
        Build the full XML-structured user-turn content string.

        This is the ONLY place the prompt structure is defined.
        Both offline data generators and live inference call this method.

        <vis> tokens appear in the order documented in the module docstring.
        """
        vis   = self.vis
        color = "White" if board.turn == chess.WHITE else "Black"

        lines = [
            "<chess_context>",

            # ── Board state (64 vis tokens) ───────────────────────────────
            "  <board_state>",
            f"    <color_to_move>{color}</color_to_move>",
            f"    {vis * 64}",
            "  </board_state>",

            # ── Game metrics (6 vis tokens, fixed order) ──────────────────
            "  <game_metrics>",
            f"    <evaluation>{vis}</evaluation>",
            f"    <complexity>{vis}</complexity>",
            f'    <tension type="global">{vis}</tension>',
            f'    <tension type="peak">{vis}</tension>',
            f'    <elo side="self">{vis}</elo>',
            f'    <elo side="oppo">{vis}</elo>',
            "  </game_metrics>",
        ]

        # ── Candidate moves (0-8 vis tokens, variable) ────────────────────
        if candidate_moves:
            lines.append(f'  <candidate_moves count="{len(candidate_moves)}">')
            for i, (uci, prob) in enumerate(candidate_moves):
                lines.append(
                    f'    <move rank="{i + 1}" uci="{uci}" '
                    f'prob="{prob * 100:.0f}%">{vis}</move>'
                )
            lines.append("  </candidate_moves>")

        lines += [
            "</chess_context>",
            "",
            f"Question: {user_message}",
        ]

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # 2. Live inference: build text + collect tensors in one call
    # ─────────────────────────────────────────────────────────────────────

    def position_segment(
        self,
        visual_tokens: ChessVisualTokens,
        board:         chess.Board,
        user_message:  str,
        batch_idx:     int = 0,
    ) -> Tuple[str, torch.Tensor]:
        """
        Build the user-turn content string AND collect the matching visual tensor.

        Returns
        -------
        content    : str with exactly N <vis> placeholders
        vis_tensor : (N, qwen_dim) — embeddings in the same order as placeholders
        """
        b       = batch_idx
        tensors : List[torch.Tensor] = []

        # ── Collect tensors in the order defined by build_text_prompt ─────
        tensors.append(visual_tokens.board[b])           # (64, D) board_state
        tensors.append(visual_tokens.value_proj[b])            # ( 1, D) evaluation
        tensors.append(visual_tokens.ponder[b])          # ( 1, D) complexity
        tensors.append(visual_tokens.tension_global[b])  # ( 1, D) tension global
        tensors.append(visual_tokens.tension_peak[b])    # ( 1, D) tension peak
        tensors.append(visual_tokens.elo_self[b])        # ( 1, D) elo self
        tensors.append(visual_tokens.elo_oppo[b])        # ( 1, D) elo oppo

        # ── Policy tokens (variable length) ──────────────────────────────
        valid_n   = int(visual_tokens.policy_mask[b].sum().item())
        candidates = visual_tokens.candidate_moves[b]
        move_meta : List[Tuple[str, float]] = []

        if candidates and valid_n > 0:
            move_meta = [(m.real_uci, m.prob) for m in candidates[:valid_n]]
            tensors.append(visual_tokens.policy_tokens[b, :valid_n, :])  # (N, D)

        # ── Build text via the shared offline method ──────────────────────
        content = self.build_text_prompt(
            board           = board,
            user_message    = user_message,
            candidate_moves = move_meta if move_meta else None,
        )

        vis_tensor = torch.cat(tensors, dim=0)   # (N_vis, D)

        # ── Runtime invariant check ───────────────────────────────────────
        n_placeholders = content.count(self.vis)
        n_tensors      = vis_tensor.size(0)
        if n_placeholders != n_tensors:
            raise AssertionError(
                f"[PromptBuilder] <vis> mismatch at batch item {b}: "
                f"{n_placeholders} placeholders in text, "
                f"{n_tensors} visual embeddings. "
                f"Check that build_text_prompt and position_segment "
                f"collect tokens in the same order."
            )

        return content, vis_tensor

    # ─────────────────────────────────────────────────────────────────────
    # 3. Conversation assembly
    # ─────────────────────────────────────────────────────────────────────

    def build_conversation(
        self,
        system_prompt:         str,
        history:               List[Dict[str, str]],
        current_content:       str,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Assemble a full Qwen3-format conversation string.

        Parameters
        ----------
        history         : Prior turns as [{role, content}] dicts (plain text).
                          No visual tokens are needed here — past responses
                          already described the position in natural language.
        current_content : Output of position_segment or build_text_prompt.
        """
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": current_content}]
        )
        return self.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt
        )

    def apply_chat_template(
        self,
        messages:              List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Thin wrapper around Qwen3's tokenizer chat template."""
        assert self.tokenizer is not None, (
            "apply_chat_template requires a tokenizer. "
            "Pass tokenizer= when constructing PromptBuilder for live inference."
        )
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize               = False,
                add_generation_prompt  = add_generation_prompt,
                enable_thinking        = self.config.enable_thinking,
            )
        except TypeError:
            # Older tokenizer versions without enable_thinking support
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize              = False,
                add_generation_prompt = add_generation_prompt,
            )

    # ─────────────────────────────────────────────────────────────────────
    # 4. Static helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from generated text."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()