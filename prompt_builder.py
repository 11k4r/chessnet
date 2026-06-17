"""
prompt_builder.py
-----------------
Constructs structured Qwen3 chat prompts that interleave natural language
with XML board-command tags and <vis> placeholders for each visual token type.

Architecture
------------
System prompt  -- STATIC. No template variables. Cached across all turns.
                  Embed once via SYSTEM_PROMPT constant.

User turn      -- DYNAMIC per turn. Three parts assembled by build_text_prompt:
                    1. [ENVIRONMENT]     FEN, ELO tier, mode   (backend injects)
                    2. <chess_context>   <vis> placeholders     (Maia tokens)
                    3. Question          The user's message

Visual token ordering contract
------------------------------
<vis> placeholders appear in build_text_prompt in this fixed order:

    1.  64 x board_state      (visual_tokens.board)
    2.   1 x evaluation       (visual_tokens.value_proj)
    3.   1 x complexity       (visual_tokens.ponder)
    4.   1 x tension_global   (visual_tokens.tension_global)
    5.   1 x tension_peak     (visual_tokens.tension_peak)
    6.   1 x elo_self         (visual_tokens.elo_self)
    7.   1 x elo_oppo         (visual_tokens.elo_oppo)
    8.   N x candidate move   (visual_tokens.policy_tokens[:valid_n])

position_segment collects tensors in exactly this order.
The runtime assertion enforces the invariant on every forward pass.

Output tag conventions
----------------------
<move uci="g1f3"/>       -- single move, always UCI inside XML tags
<moves>g1f3 b8c6</moves> -- move sequence, UCI, space-separated
<branch id="..." ...>    -- variation block; </branch> pops position stack
<main_line/>             -- jump to root game line from any branch depth
SAN (e.g. "Nf3")        -- used only in natural language text, never in tags
"""

from __future__ import annotations

import re
import chess

try:
    import torch
except ImportError:
    torch = None  # offline mode -- position_segment will raise if called

from typing import Dict, List, Optional, Tuple

from chess_tokens import ChessVisualTokens
from config import ChessCoachConfig


# ---------------------------------------------------------------------------
# 1. Mode and ELO constants
# ---------------------------------------------------------------------------

class CoachMode:
    COACH       = "coach"        # interactive; pauses after each idea
    COMMENTATOR = "commentator"  # continuous flow; no pauses
    GAME        = "game"         # model plays moves and narrates thinking


ELO_TIERS: Dict[str, Tuple[int, int]] = {
    "beginner":     (0,    800),
    "club":         (800,  1500),
    "intermediate": (1500, 2000),
    "advanced":     (2000, 2400),
    "expert":       (2400, 9999),
}


def get_elo_tier(elo: int) -> str:
    """Map a numeric ELO rating to its tier string."""
    for tier, (lo, hi) in ELO_TIERS.items():
        if lo <= elo < hi:
            return tier
    return "expert"


# ---------------------------------------------------------------------------
# 2. Static system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
<role>
You are an expert, interactive chess coach with grandmaster-level understanding.
You control a live chess board UI by emitting structured XML commands inline with your explanations.
</role>

<visual_context>
Each user message contains an environment block and a <chess_context> block of neural visual tokens (<vis>) produced by the Maia-3 engine. Reference these signals by label in your internal <think> reasoning:

- <board_state> 64 per-square embeddings — piece placement & coordination
- <evaluation> Scalar position signal (positive = White better)
- <complexity> Move difficulty / sharpness of the position
- <tension type="global"> Mean residual pressure across the board
- <tension type="peak"> Sharpest local tactical tension on the board
- <elo side="self"> Skill embedding of the side to move
- <elo side="oppo"> Skill embedding of the opponent
- <candidate_moves> Maia's ranked top moves with probabilities

Use these signals to ground your coaching angle before writing a single word of response.
</visual_context>

<behavior_modifiers>
# SKILL-LEVEL BEHAVIOR
| Tier         | ELO       | Behavior                                                              |
|--------------|-----------|-----------------------------------------------------------------------|
| beginner     | < 800     | Name every piece and square. Avoid jargon. Explain every move.       |
| club         | 800-1500  | Explain key concepts. Skip obvious moves. Introduce long-term plans. |
| intermediate | 1500-2000 | Focus on critical positions. Assume opening knowledge.               |
| advanced     | 2000-2400 | Name opening theory. Assume tactical vision. Be concise on basics.   |
| expert       | 2400+     | Dense annotation. Engine depth. No hand-holding.                     |

# MODE BEHAVIOR
| Mode        | Behavior                                                                       |
|-------------|--------------------------------------------------------------------------------|
| coach       | Teach interactively. Use <|wait|> after each key idea. Use <ask> to engage.   |
| commentator | Flow continuously — never use <|wait|>. Advance with <next_move/> on cadence. |
| game        | You are playing. Announce your move first, then briefly explain your thinking. |
</behavior_modifiers>

<ui_commands>
Emit commands inline with your text — never in a silent trailing block.

## Navigation
<move uci="g1f3"/>                  Execute a single move using UCI notation
<move uci="e7e8q"/>                 Promotion: append target piece letter
<moves>e2e4 e7e5 g1f3 b8c6</moves>  Execute a move sequence without commentary between
<next_move/>  <prev_move/>          Step through a loaded game
<goto_move n="14" color="white"/>   Jump to move N in the loaded game
<reset_board/>                      Reset to session start position (clears annotations, branches, visibility)
<set_position fen="..."/>           Load a new FEN position directly
<flip_board side="black"/>          Flip board orientation

## Variations
<branch id="b1" label="The key idea">   Enter a variation — saves current board state
  <moves>f4f5 b7b6</moves>
  ... explanation ...
  <branch id="b1a" label="Sub-line">    Nesting is valid
    <moves>c2c4 d7d5</moves>
  </branch>
</branch>                               Exit — automatically restores the prior position

<main_line/>                            Return to the root game line from any branch depth
<preview_line color="plan">d4d5 c7c5</preview_line>   Ghost moves: displayed but not played
<clear_preview/>

## Visual Annotations
<highlight square="f7" color="danger"/>
<highlight_zone zone="kingside" color="danger"/>   zones: kingside | queenside | center | center_squares
<arrow from="c4" to="f7" color="attack" style="solid"/>
<circle square="d5" color="key"/>
<clear_highlight square="f7"/>   <clear_highlights/>
<clear_arrows/>   <clear_circles/>   <clear_annotations/>

## Piece Visibility
<hide pieces="bishops,queens"/>   options: all | pawns | knights | bishops | rooks | queens | kings
<hide color="black"/>             Hide all pieces of a specific side
<show pieces="all"/>   <show color="black"/>
<isolate squares="e4,d4"/>        Hide everything except pieces on the specified squares
<fade pieces="all"/>   <fade color="black"/>   <unfade pieces="all"/>

## Teaching & Interaction
<|wait|>                             Pause and await user interaction (coach mode only)
<ask>What would you play here?</ask>

## Color Semantics
Squares, zones, circles: danger=red | key=yellow | good=green | bad=red-pulse | info=blue
Arrows: attack=red | defend=green | plan=blue-dashed
</ui_commands>

<critical_rules>
IMPORTANT: You MUST adhere to these operational rules at all times:
1. Think first. Before every response, reason privately inside <think>...</think> — this block is internal and never shown to the user. Analyze the evaluation, evaluate candidate moves, choose your pedagogical angle, and list your planned commands.
2. Commands are inline. Weave commands into your sentences as you write, exactly where they belong. Never batch commands at the end of a response.
3. Explain before you act. Every command must be preceded by at least one sentence of natural language context.
4. Respect mode and skill level. Match vocabulary, depth, and use of <|wait|> strictly to the environment attributes in the user's message.
5. Branches must have IDs. Every <branch> requires a unique id attribute. </branch> restores the position that existed before that branch opened. <main_line/> returns to the root game line from any branch depth.
6. Only legal moves. Verify every <move> against the current board state inside <think> before emitting it. Use friendly SAN (e.g., "Nf3") in natural language, but strictly UCI (e.g., "g1f3") inside XML tags.
7. Keep annotations clean. Call <clear_annotations/> when moving to a new idea. Never leave more than three highlights active at once.
</critical_rules>"""


# ---------------------------------------------------------------------------
# 3. PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds Qwen3-format conversation prompts with <vis> placeholders.

    System prompt  -- static; retrieve once with get_system_prompt().
    User turn      -- dynamic; build per turn with build_text_prompt()
                      (offline) or position_segment() (live, with tensors).
    """

    def __init__(self, config: ChessCoachConfig, tokenizer=None) -> None:
        self.config    = config
        self.tokenizer = tokenizer  # optional -- not needed for offline data gen
        self.vis       = config.vis_token  # "<vis>"

    # -- System prompt --------------------------------------------------------

    @staticmethod
    def get_system_prompt() -> str:
        """
        Return the static system prompt string.

        Call once per session and pass the result to build_conversation.
        Contains no template variables -- safe to cache for KV-cache reuse.
        """
        return SYSTEM_PROMPT

    # -- Environment block ----------------------------------------------------

    @staticmethod
    def build_environment_block(
        board: chess.Board,
        elo_tier: str = "club",
        mode: str = CoachMode.COACH,
    ) -> str:
        """
        Build the per-turn [ENVIRONMENT] block prepended to the user turn.

        Contains the dynamic state the system prompt rules refer to as
        "the environment attributes in the user's message."
        The FEN here is ground truth -- the model must not infer board
        state from conversation history.
        """
        color = "White" if board.turn == chess.WHITE else "Black"
        return (
            "[ENVIRONMENT]\n"
            f"Current Board FEN : {board.fen()}\n"
            f"Side to Move      : {color}\n"
            f"User Skill Level  : {elo_tier}\n"
            f"Mode              : {mode}"
        )

    # -- Chess context block (OFFLINE SAFE -- no tensors required) ------------

    def build_text_prompt(
        self,
        board: chess.Board,
        user_message: str,
        candidate_moves: Optional[List[Tuple[str, float]]] = None,
        elo_tier: str = "club",
        mode: str = CoachMode.COACH,
    ) -> str:
        """
        Build the full user-turn content string.

        This is the ONLY place the prompt structure is defined.
        Both offline data generators and live inference call this method,
        guaranteeing training data and production always see the same layout.

        Parameters
        ----------
        board           : Current position.
        user_message    : The user's question or instruction.
        candidate_moves : List of (real_uci, probability) from policy_tokens.
                          None means no <candidate_moves> section is emitted.
        elo_tier        : One of the ELO_TIERS keys.
        mode            : CoachMode constant.

        Output structure
        ----------------
        [ENVIRONMENT]        <- dynamic state (FEN, ELO tier, mode)
        <chess_context>      <- <vis> placeholders
        Question: ...        <- user message
        """
        vis   = self.vis
        color = "White" if board.turn == chess.WHITE else "Black"

        lines = [
            self.build_environment_block(board, elo_tier, mode),
            "",
            "<chess_context>",

            # Board state: 64 vis tokens
            "  <board_state>",
            f"    <color_to_move>{color}</color_to_move>",
            f"    {vis * 64}",
            "  </board_state>",

            # Game metrics: 6 vis tokens (fixed order)
            "  <game_metrics>",
            f"    <evaluation>{vis}</evaluation>",
            f"    <complexity>{vis}</complexity>",
            f'    <tension type="global">{vis}</tension>',
            f'    <tension type="peak">{vis}</tension>',
            f'    <elo side="self">{vis}</elo>',
            f'    <elo side="oppo">{vis}</elo>',
            "  </game_metrics>",
        ]

        # Candidate moves: 0-8 vis tokens (variable)
        if candidate_moves:
            lines.append(f'  <candidate_moves count="{len(candidate_moves)}">')
            for i, (uci, prob) in enumerate(candidate_moves):
                lines.append(
                    f'    <move rank="{i + 1}" uci="{uci}"'
                    f' prob="{prob * 100:.0f}%">{vis}</move>'
                )
            lines.append("  </candidate_moves>")

        lines += [
            "</chess_context>",
            "",
            f"Question: {user_message}",
        ]

        return "\n".join(lines)

    # -- Live inference: text + tensors in one call ---------------------------

    def position_segment(
        self,
        visual_tokens: ChessVisualTokens,
        board: chess.Board,
        user_message: str,
        batch_idx: int = 0,
        elo_tier: str = "club",
        mode: str = CoachMode.COACH,
    ) -> Tuple[str, "torch.Tensor"]:
        """
        Build the user-turn content string AND collect the matching visual tensor.

        Returns
        -------
        content    : str with exactly N <vis> placeholders
        vis_tensor : (N, qwen_dim) -- embeddings in placeholder order
        """
        assert torch is not None, "position_segment requires torch."
        b = batch_idx

        # Collect tensors in the order defined by build_text_prompt
        tensors: List[torch.Tensor] = [
            visual_tokens.board[b],           # (64, D) board_state
            visual_tokens.value_proj[b],      # ( 1, D) evaluation
            visual_tokens.ponder[b],          # ( 1, D) complexity
            visual_tokens.tension_global[b],  # ( 1, D) tension global
            visual_tokens.tension_peak[b],    # ( 1, D) tension peak
            visual_tokens.elo_self[b],        # ( 1, D) elo self
            visual_tokens.elo_oppo[b],        # ( 1, D) elo oppo
        ]

        # Policy tokens: variable length
        valid_n    = int(visual_tokens.policy_mask[b].sum().item())
        candidates = visual_tokens.candidate_moves[b]
        move_meta: List[Tuple[str, float]] = []

        if candidates and valid_n > 0:
            move_meta = [(m.real_uci, m.prob) for m in candidates[:valid_n]]
            tensors.append(visual_tokens.policy_tokens[b, :valid_n, :])  # (N, D)

        # Build text via the shared offline method
        content = self.build_text_prompt(
            board           = board,
            user_message    = user_message,
            candidate_moves = move_meta or None,
            elo_tier        = elo_tier,
            mode            = mode,
        )

        vis_tensor = torch.cat(tensors, dim=0)  # (N_vis, D)

        # Invariant: placeholder count must match tensor count
        n_placeholders = content.count(self.vis)
        n_tensors      = vis_tensor.size(0)

        if n_placeholders != n_tensors:
            raise AssertionError(
                f"[PromptBuilder] <vis> mismatch at batch item {b}: "
                f"{n_placeholders} placeholders in text, "
                f"{n_tensors} visual embeddings. "
                f"Ensure build_text_prompt and position_segment collect "
                f"tokens in the same order."
            )

        return content, vis_tensor

    # -- Conversation assembly ------------------------------------------------

    def build_conversation(
        self,
        current_content: str,
        history: Optional[List[Dict[str, str]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Assemble a full Qwen3-format conversation string.

        The system prompt is static (no board/ELO/mode baked in).
        The dynamic environment block lives inside current_content,
        built by build_text_prompt or position_segment.

        Parameters
        ----------
        current_content : Output of position_segment or build_text_prompt.
        history         : Prior turns as [{role, content}] dicts.
                          Plain text -- no visual tokens needed, as past
                          responses already described the position in prose.
        """
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + (history or [])
            + [{"role": "user",  "content": current_content}]
        )
        return self.apply_chat_template(messages, add_generation_prompt)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Thin wrapper around Qwen3's tokenizer chat template."""
        assert self.tokenizer is not None, (
            "apply_chat_template requires a tokenizer. "
            "Pass tokenizer= to PromptBuilder for live inference."
        )
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize              = False,
                add_generation_prompt = add_generation_prompt,
                enable_thinking       = self.config.enable_thinking,
            )
        except TypeError:
            # Older tokenizer versions without enable_thinking support
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize              = False,
                add_generation_prompt = add_generation_prompt,
            )

    # -- Static helpers -------------------------------------------------------

    @staticmethod
    def strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from generated text."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def extract_board_commands(text: str) -> List[str]:
        """
        Extract board command tags from completed (non-streaming) generated text.

        Returns raw tag strings in generation order for the backend parser.

        Limitations
        -----------
        - <|wait|> is a vocabulary token intercepted at the streaming level --
          it will not appear as a string in completed text and is not matched here.
        - Nested <branch> tags are not correctly handled by regex; the streaming
          state machine (with its depth counter) is the authoritative parser for
          branches. Use this method only for post-hoc inspection of flat output.
        """
        self_closing = (
            r"(?:"
            r"<move\b[^>]*/>"
            r"|<next_move/>"
            r"|<prev_move/>"
            r"|<goto_move\b[^>]*/>"
            r"|<reset_board/>"
            r"|<set_position\b[^>]*/>"
            r"|<flip_board\b[^>]*/>"
            r"|<main_line/>"
            r"|<clear_preview/>"
            r"|<highlight\b[^>]*/>"
            r"|<highlight_zone\b[^>]*/>"
            r"|<arrow\b[^>]*/>"
            r"|<circle\b[^>]*/>"
            r"|<clear_highlight\b[^>]*/>"
            r"|<clear_highlights/>"
            r"|<clear_arrows/>"
            r"|<clear_circles/>"
            r"|<clear_annotations/>"
            r"|<hide\b[^>]*/>"
            r"|<show\b[^>]*/>"
            r"|<isolate\b[^>]*/>"
            r"|<fade\b[^>]*/>"
            r"|<unfade\b[^>]*/>"
            r")"
        )
        block = (
            r"(?:"
            r"<moves>.*?</moves>"
            r"|<preview_line\b[^>]*>.*?</preview_line>"
            r"|<ask>.*?</ask>"
            r")"
        )
        pattern = re.compile(f"(?:{self_closing}|{block})", re.DOTALL)
        return pattern.findall(text)
