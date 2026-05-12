"""HyperCLOVAX-SEED-Think reasoning parser for vLLM.

The chat_template's generation prompt depends on the ``thinking`` flag:
  thinking=true  → '<|im_start|>assistant\\n<think>\\n'
                   → model output: [reasoning]</think>\\n\\n[content]
  thinking=false → '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
                   → model output: [content]   (no </think> in output)

vLLM instantiates this parser per request, passing ``chat_template_kwargs``
to ``__init__``. We capture ``thinking`` at init time so that the streaming
callback (which does not receive the request) knows which mode to apply.
"""

from collections.abc import Sequence
from typing import Optional, Union

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

THINK_END = "</think>"
IM_END = "<|im_end|>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _partial_prefix_len(text: str, target: str) -> int:
    """Length of the longest prefix of ``target`` matching the suffix of ``text``.

    Used to detect protocol tokens split across delta boundaries. The exact-
    match case (full ``target`` as a suffix) is excluded; callers should check
    that separately with ``in`` or ``find``.
    """
    max_len = min(len(text), len(target) - 1)
    for ln in range(max_len, 0, -1):
        if text[-ln:] == target[:ln]:
            return ln
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class HyperCLOVAXSeedThinkReasoningParser(ReasoningParser):
    """Reasoning parser for the HyperCLOVAX-SEED-Think model.

    Only a single boundary token (``</think>``) is tracked, so we use
    ``str.partition`` directly rather than the multi-string mixin pattern
    used by other HyperCLOVAX parsers.
    """

    # Reasoning block delimiter (used by guided-decoding integrations).
    @property
    def reasoning_end_str(self) -> str | None:
        return THINK_END

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # Per-request init: capture the mode from chat_template_kwargs so the
        # streaming callback (which doesn't receive the request) knows what to do.
        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
        self.thinking: bool = bool(chat_template_kwargs.get("thinking", False))

        # When thinking=false the model output has no </think>; start in content mode.
        self.no_reasoning_content: bool = not self.thinking

        # add_special_tokens=False avoids a leading BOS in the comparison sequence.
        self.think_end_tokens: list[int] = tokenizer.encode(
            THINK_END, add_special_tokens=False
        )
        self.end_token_id = self.vocab.get(IM_END)

        self.buffer_string: str = ""
        # After </think> transitions us into content mode we want to drop the
        # `</think>\n\n` separator the chat_template inserts, so the very next
        # content delta lstrips leading newlines. Mirrors the lstrip in
        # ``extract_reasoning`` for the non-streaming path.
        self._strip_pending_newlines: bool = False

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        # Authoritative signal: presence of </think> in the output.
        reasoning, sep, content = model_output.partition(THINK_END)
        if sep:
            return reasoning.strip("\n") or None, content.lstrip("\n") or None

        # </think> absent and thinking=true: truncated mid-reasoning → all reasoning.
        if self.thinking:
            return model_output, None

        # Otherwise: treat as plain content (matches the chat_template else branch).
        return None, model_output

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        if not current_text:
            return None

        if self.no_reasoning_content:
            if self._strip_pending_newlines:
                delta_text = delta_text.lstrip("\n")
                if not delta_text:
                    return None
                self._strip_pending_newlines = False
            return DeltaMessage(content=delta_text)

        self.buffer_string += delta_text

        # </think> seen → end reasoning, switch to content mode.
        reasoning_part, sep, content_part = self.buffer_string.partition(THINK_END)
        if sep:
            self.no_reasoning_content = True
            self.buffer_string = ""
            content = content_part.lstrip("\n") or None
            # If the closing delta carried only `</think>` (no trailing content
            # yet), defer the lstrip to the next content delta.
            if content is None:
                self._strip_pending_newlines = True
            reasoning = reasoning_part or None
            if reasoning is None and content is None:
                return None
            return DeltaMessage(reasoning=reasoning, content=content)

        # </think> may straddle deltas; hold the buffer if its tail is a partial.
        if _partial_prefix_len(self.buffer_string, THINK_END) > 0:
            return None

        # Safe to emit as reasoning_content.
        emit = self.buffer_string
        self.buffer_string = ""
        return DeltaMessage(reasoning=emit)

    # ------------------------------------------------------------------
    # Structured output helpers
    # ------------------------------------------------------------------

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # Structured decoding asks "has reasoning ended at any point so far?",
        # so we must accept the end marker anywhere in the sequence, not only
        # at the tail. Short-circuit on the no-reasoning case and on the
        # explicit <|im_end|> stop token first.
        if self.no_reasoning_content or (
            self.end_token_id is not None and self.end_token_id in input_ids
        ):
            return True
        n = len(self.think_end_tokens)
        if n == 0 or len(input_ids) < n:
            return False
        return any(
            input_ids[i:i + n] == self.think_end_tokens
            for i in range(len(input_ids) - n + 1)
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Locate the </think> sequence and return everything after it. The
        # original basic_parsers reference uses ``end_token_id`` to denote the
        # reasoning-end token, which is ``</think>`` for this model — not
        # ``<|im_end|>``.
        n = len(self.think_end_tokens)
        if n == 0:
            return []
        # Skip the very last position so that an end-marker at the boundary
        # (no trailing content yet) returns an empty list.
        upper = len(input_ids) - n
        for i in range(upper):
            if input_ids[i:i + n] == self.think_end_tokens:
                return list(input_ids[i + n:])
        return []


