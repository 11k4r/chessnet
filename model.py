"""
model.py
────────
ChessCoach: a LLaVA-style chess coaching model.

Architecture
────────────
  Frozen Maia-3 (79M)
      │
      ├── 8 hook sites → MaiaFeatureExtractor
      │
  MaiaProjectors (trainable)
      │  bridges + token-type embeddings
      │  maps Maia features → Qwen embedding space
      │
  PromptBuilder
      │  assembles Qwen3 chat-format string with <vis> placeholders
      │  returns ordered visual tensor (N_vis, 4096)
      │
  _inject_visual_tokens
      │  replaces <vis> embeddings in Qwen's input_embeds
      │
  Qwen3-8B (frozen / LoRA-tunable)
      │
  loss / generated text

Multi-turn
──────────
Each call to forward / generate accepts a `histories` argument: a list
(one per batch item) of prior turns as [{role, content}] dicts.  Only the
CURRENT turn has a board position and visual tokens.  Prior turns are
already in plain text (the coach's previous responses, which described the
position in words), so no re-extraction is needed.

Chain-of-Thought
────────────────
Qwen3's native thinking mode is used: with enable_thinking=True the model
produces <think>...</think> before its final answer.  The training loss is
computed over both the reasoning and final-answer tokens (trains full CoT).
strip_thinking=True in the config omits the <think> block from returned
strings during inference.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import chess
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import ChessCoachConfig, MaiaConfig
from chess_tokens import (
    ChessVisualTokens,
    MaiaFeatureExtractor,
    MaiaProjectors,
    extract_all_tokens,
)
from prompt_builder import PromptBuilder


class ChessCoach(nn.Module):
    """
    LLaVA-style chess coach that fuses Maia-3 visual tokens with Qwen3.

    Trainable parameters
    ────────────────────
    Only MaiaProjectors is trained by default.  Maia is frozen.  Qwen3
    should be LoRA-tuned externally (add adapters before passing to this
    class, or set requires_grad=True on selected layers).

    Parameters
    ----------
    config : ChessCoachConfig
    """

    def __init__(self, config: ChessCoachConfig) -> None:
        super().__init__()
        self.config = config

        # ── 1. Maia backbone (frozen) ─────────────────────────────────────
        maia_model      = self._load_maia(config.maia)
        self.extractor  = MaiaFeatureExtractor(maia_model, freeze_maia=config.freeze_maia)

        # ── 2. Trainable bridges ──────────────────────────────────────────
        self.projectors = MaiaProjectors(
            maia_dim = config.maia_dim,
            elo_dim  = config.elo_dim,
            qwen_dim = config.qwen_dim,
        )

        # ── 3. Qwen3 backbone + tokenizer ────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_path, trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # ── 4. Register <vis> special token ──────────────────────────────
        self.vis_token_id = self._setup_vis_token()

        # ── 5. Prompt builder ─────────────────────────────────────────────
        self.prompt_builder = PromptBuilder(config, self.tokenizer)

        # Cache the assistant-turn header token IDs for label building
        self._assistant_header_ids: List[int] = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._im_end_id: int = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    # ─────────────────────────────────────────────────────────────────────
    # Public: training forward pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        boards:        List[chess.Board],
        maia_tokens:   torch.Tensor,                    # (B, 64, token_dim)
        self_elos:     torch.Tensor,                    # (B,)
        oppo_elos:     torch.Tensor,                    # (B,)
        user_messages: List[str],                       # current-turn user text
        assistant_messages: Optional[List[str]] = None,
        histories:     Optional[List[List[Dict[str, str]]]] = None,
        # [{role, content}] per batch item — plain text, no visual tokens
        queried_ucis:  Optional[List[Optional[str]]]   = None,
        labels:        Optional[torch.Tensor]          = None,
    ) -> CausalLMOutputWithPast:
        """
        Full training forward pass.

        Returns a HuggingFace CausalLMOutputWithPast; if labels are provided
        (or auto-built) the .loss field contains the LM cross-entropy loss.
        """
        device   = maia_tokens.device
        histories = histories or [[] for _ in boards]

        # 1. Single Maia forward — triggers all eight hooks
        visual = extract_all_tokens(
            boards            = boards,
            extractor         = self.extractor,
            maia_input_tokens = maia_tokens,
            self_elos         = self_elos,
            oppo_elos         = oppo_elos,
            projectors        = self.projectors,
            device            = device,
            user_queried_ucis = queried_ucis,
            max_moves         = self.config.max_candidate_moves,
        )

        # 2. Build prompts + tokenise; collect ordered visual tensors
        input_ids, attention_mask, ordered_visuals, auto_labels = (
            self._build_batch_inputs(
                boards        = boards,
                visual        = visual,
                user_messages = user_messages,
                assistant_messages = assistant_messages,
                histories     = histories,
                device        = device,
                build_labels  = labels is None,
            )
        )

        # Use externally provided labels if given
        if labels is not None:
            auto_labels = labels

        # 3. Replace <vis> placeholders with actual visual embeddings
        inputs_embeds = self._inject_visual_tokens(
            input_ids, ordered_visuals, device
        )

        # 4. Qwen forward
        return self.llm(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            labels         = auto_labels,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public: inference
    # ─────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        boards:        List[chess.Board],
        maia_tokens:   torch.Tensor,
        self_elos:     torch.Tensor,
        oppo_elos:     torch.Tensor,
        user_messages: List[str],
        histories:     Optional[List[List[Dict[str, str]]]] = None,
        queried_ucis:  Optional[List[Optional[str]]]       = None,
        **generation_kwargs,
    ) -> List[str]:
        """
        Generate coaching responses for a batch of positions.

        Returns a list of decoded strings (one per batch item).
        If config.strip_thinking is True, <think>...</think> blocks are
        removed before returning.
        """
        device    = maia_tokens.device
        histories = histories or [[] for _ in boards]

        # 1. Maia forward + visual token extraction
        visual = extract_all_tokens(
            boards            = boards,
            extractor         = self.extractor,
            maia_input_tokens = maia_tokens,
            self_elos         = self_elos,
            oppo_elos         = oppo_elos,
            projectors        = self.projectors,
            device            = device,
            user_queried_ucis = queried_ucis,
            max_moves         = self.config.max_candidate_moves,
        )

        # 2. Build prompts (no label computation needed)
        input_ids, attention_mask, ordered_visuals, _ = self._build_batch_inputs(
            boards        = boards,
            visual        = visual,
            user_messages = user_messages,
            assistant_messages = None,
            histories     = histories,
            device        = device,
            build_labels  = False,
        )

        # 3. Inject visual tokens
        inputs_embeds = self._inject_visual_tokens(
            input_ids, ordered_visuals, device
        )

        # 4. Merge caller kwargs with config defaults
        gen_cfg = dict(
            max_new_tokens = self.config.max_new_tokens,
            temperature    = self.config.temperature,
            top_p          = self.config.top_p,
            do_sample      = self.config.do_sample,
        )
        gen_cfg.update(generation_kwargs)

        # 5. Generate — inputs_embeds encodes the full prompt,
        #    output_ids contains only the newly generated tokens.
        output_ids = self.llm.generate(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            **gen_cfg,
        )

        # 6. Decode
        responses = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        if self.config.strip_thinking:
            responses = [PromptBuilder.strip_thinking(r) for r in responses]

        return responses

    # ─────────────────────────────────────────────────────────────────────
    # Checkpointing (projectors only; Maia and Qwen live separately)
    # ─────────────────────────────────────────────────────────────────────

    def save_pretrained(self, path: str) -> None:
        """
        Save trainable components to disk.
        Maia and Qwen are NOT saved here — they are loaded from their
        original checkpoints and any LoRA adapters are handled externally.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.projectors.state_dict(),
            os.path.join(path, "projectors.pt"),
        )
        print(f"[ChessCoach] Projectors saved to {path}/projectors.pt")

    def load_pretrained_projectors(self, path: str) -> None:
        """Load previously saved projector weights into this model."""
        projectors_path = os.path.join(path, "projectors.pt")
        state = torch.load(projectors_path, map_location="cpu", weights_only=True)
        self.projectors.load_state_dict(state)
        print(f"[ChessCoach] Projectors loaded from {projectors_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Private: initialisation helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_maia(maia_cfg: MaiaConfig) -> nn.Module:
        """Load Maia-3 79M from checkpoint and return an eval-mode model."""
        import sys
        sys.path.insert(0, ".")
        from maia3.models import MAIA3Model

        model = MAIA3Model(maia_cfg)
        ckpt  = torch.load(
            maia_cfg.checkpoint_path, map_location="cpu", weights_only=True
        )
        state_dict = ckpt.get("model_state_dict", ckpt)
        renamed    = {k.replace("smolgen", "gab"): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(renamed, strict=False)
        if missing:
            print(f"[Maia] Missing keys  : {missing[:3]}")
        if unexpected:
            print(f"[Maia] Unexpected keys: {unexpected[:3]}")

        model.eval()
        print("[Maia] Loaded successfully.")
        return model

    def _setup_vis_token(self) -> int:
        """
        Register <vis> as a new special token, resize Qwen's embedding
        table to accommodate it, and return its token ID.
        """
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.config.vis_token]}
        )
        if added:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            print(f"[ChessCoach] Added {added} special token(s); "
                  f"vocab size now {len(self.tokenizer)}.")

        vis_id = self.tokenizer.convert_tokens_to_ids(self.config.vis_token)
        assert vis_id != self.tokenizer.unk_token_id, \
            "<vis> was not correctly registered in the tokenizer."
        return vis_id

    # ─────────────────────────────────────────────────────────────────────
    # Private: batch construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_batch_inputs(
        self,
        boards:        List[chess.Board],
        visual:        ChessVisualTokens,
        user_messages: List[str],
        assistant_messages: Optional[List[str]],
        histories:     List[List[Dict[str, str]]],
        device:        torch.device,
        build_labels:  bool,
    ) -> Tuple[
        torch.Tensor,           # input_ids      (B, L)
        torch.Tensor,           # attention_mask (B, L)
        List[torch.Tensor],     # ordered_visuals per item (N_vis_i, D)
        Optional[torch.Tensor], # labels         (B, L) or None
    ]:
        """
        For each item in the batch:
          1. Build position segment → (content_str, vis_tensor)
          2. Assemble full conversation and apply Qwen3 chat template
          3. Tokenise (no padding yet)

        Then pad the batch, optionally build labels.
        """
        per_item_ids: List[torch.Tensor] = []
        ordered_visuals: List[torch.Tensor] = []

        for b in range(len(boards)):

            if boards[b] is not None:
                # Build content and visual tensor for this turn
                content, vis_tensor = self.prompt_builder.position_segment(
                    visual_tokens = visual,
                    board         = boards[b],
                    user_message  = user_messages[b],
                    batch_idx     = b,
                )
            else:
                # Modality: Pure Text Conversation
                content = user_messages[b]
                # Return an empty tensor of shape (0, qwen_dim)
                vis_tensor = torch.empty((0, self.config.qwen_dim), device=device)

            # Assemble full conversation string
            prompt_str = self.prompt_builder.build_conversation(
                system_prompt         = self.config.system_prompt,
                history               = histories[b],
                current_content       = content,
                add_generation_prompt = True,
            )

            if assistant_messages is not None and assistant_messages[b]:
                prompt_str += assistant_messages[b] + "<|im_end|>\n"

            # Tokenise without padding (we pad across the batch below)
            token_ids = self.tokenizer(
                prompt_str,
                return_tensors     = "pt",
                add_special_tokens = False,
            ).input_ids[0]                              # (L_i,)

            per_item_ids.append(token_ids)
            ordered_visuals.append(vis_tensor.to(device))

        # Pad to batch max length
        input_ids, attention_mask = self._pad_sequences(per_item_ids, device)

        # Build labels
        labels = (
            self._build_labels(input_ids, attention_mask)
            if build_labels
            else None
        )

        
        return input_ids, attention_mask, ordered_visuals, labels

    # ─────────────────────────────────────────────────────────────────────
    # Private: visual token injection
    # ─────────────────────────────────────────────────────────────────────

    def _inject_visual_tokens(
        self,
        input_ids:       torch.Tensor,          # (B, L)
        ordered_visuals: List[torch.Tensor],    # per item (N_vis_i, D)
        device:          torch.device,
    ) -> torch.Tensor:
        """
        Replace <vis> placeholder embeddings with actual visual embeddings.

        Steps
        ─────
        1. Embed all tokens via Qwen's embedding layer.
        2. For each batch item, locate <vis> positions in input_ids.
        3. Overwrite those positions with the ordered visual tensor.

        Returns inputs_embeds (B, L, qwen_dim) ready for Qwen's forward.
        """
        embed_layer   = self.llm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()       # (B, L, D)

        for b, vis_tokens in enumerate(ordered_visuals):
            vis_positions = (input_ids[b] == self.vis_token_id).nonzero(
                as_tuple=True
            )[0]                                             # (N_vis,)

            if vis_positions.numel() != vis_tokens.size(0):
                raise ValueError(
                    f"Batch item {b}: found {vis_positions.numel()} <vis> "
                    f"placeholders in prompt but {vis_tokens.size(0)} visual "
                    f"embeddings were provided."
                )

            inputs_embeds[b, vis_positions] = vis_tokens.to(inputs_embeds.dtype)

        return inputs_embeds                                 # (B, L, D)

    # ─────────────────────────────────────────────────────────────────────
    # Private: label construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_labels(
        self,
        input_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """
        Build causal-LM labels for instruction tuning.

        Rules
        ─────
        • Padding positions               → -100
        • <vis> positions                 → -100  (visual tokens are inputs only)
        • System / user turn tokens       → -100  (not supervised)
        • ALL assistant response tokens   → kept  (supervised; includes <think>)

        The last rule covers every assistant segment in a multi-turn
        conversation, so partial conversations are fully supervised.
        """
        labels = torch.full_like(input_ids, -100)
        n_hdr  = len(self._assistant_header_ids)

        for b in range(input_ids.size(0)):
            seq = input_ids[b].tolist()
            i   = 0
            while i <= len(seq) - n_hdr:
                # Look for the assistant-turn header
                if seq[i : i + n_hdr] == self._assistant_header_ids:
                    resp_start = i + n_hdr          # first token of the response
                    resp_end   = resp_start
                    while resp_end < len(seq) and seq[resp_end] != self._im_end_id:
                        resp_end += 1
                    # Supervise every token in this response segment
                    labels[b, resp_start:resp_end] = input_ids[b, resp_start:resp_end]
                    i = resp_end + 1
                else:
                    i += 1

        # Mask <vis> tokens and padding regardless of segment
        labels[input_ids == self.vis_token_id] = -100
        labels[attention_mask == 0]            = -100

        return labels

    # ─────────────────────────────────────────────────────────────────────
    # Private: padding
    # ─────────────────────────────────────────────────────────────────────

    def _pad_sequences(
        self,
        sequences: List[torch.Tensor],  # list of 1-D token-ID tensors
        device:    torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Right-pad sequences to the longest in the batch.
        Returns (input_ids, attention_mask) both of shape (B, max_len).
        """
        pad_id  = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        max_len = max(s.size(0) for s in sequences)
        B       = len(sequences)

        input_ids      = torch.full((B, max_len), pad_id,  dtype=torch.long,  device=device)
        attention_mask = torch.zeros((B, max_len),          dtype=torch.long,  device=device)

        for i, seq in enumerate(sequences):
            L = seq.size(0)
            input_ids[i, :L]      = seq.to(device)
            attention_mask[i, :L] = 1

        return input_ids, attention_mask
