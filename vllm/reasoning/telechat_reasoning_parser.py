# SPDX-License-Identifier: Apache-2.0
# Copyright The TeleChat Authors.
#
# Reasoning parser plugin for TeleChat models.
# Usage:
#   vllm serve <model> \
#       --reasoning-parser telechat \
#       --reasoning-parser-plugin telechat_reasoning_parser.py

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike

START_THINK = "<think>"
END_THINK = "</think>"


@ReasoningParserManager.register_module("telechat")
class TeleChatReasoningParser(ReasoningParser):
    """Reasoning parser for TeleChat models.

    NOTE: the chat template injects the ``<think>`` / ``</think>`` markers into
    the *prompt* via ``add_generation_prompt``; the model itself does not emit
    the opening ``<think>``. As a result the generated text looks like:

    * thinking enabled  → ``…reasoning…</think>content`` (contains ``</think>``)
    * thinking disabled → ``content`` (NO marker at all — ``</think>`` already
      lives at the tail of the prompt)

    Therefore the only reliable signal in the generated output is whether
    ``</think>`` is present. When it is absent the whole output is content.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        if not self.model_tokenizer:
            raise ValueError(
                "TeleChatReasoningParser requires a valid tokenizer."
            )

        # Per-request flag, refreshed in ``adjust_request``. When the caller
        # explicitly disables thinking, the generated output is pure content
        # and contains no ``</think>`` marker. Defaults to ``False`` (thinking
        # on) so behaviour stays safe even if ``adjust_request`` is not called.
        self._reasoning_disabled: bool = False

        self.start_token = START_THINK
        self.end_token = END_THINK

        start_id = self.vocab.get(self.start_token)
        end_id = self.vocab.get(self.end_token)

        if start_id is None:
            raise RuntimeError(
                f"TeleChatReasoningParser: '{self.start_token}' not found "
                "in tokenizer vocabulary."
            )
        if end_id is None:
            raise RuntimeError(
                f"TeleChatReasoningParser: '{self.end_token}' not found "
                "in tokenizer vocabulary."
            )

        self.start_token_id: int = start_id
        self.end_token_id: int = end_id

    # ------------------------------------------------------------------
    # Per-request setup
    # ------------------------------------------------------------------

    def adjust_request(self, request):
        """Capture whether thinking is disabled for this request.

        Clients toggle thinking via
        ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``.
        We refresh the flag on every request so a shared parser instance can
        never leak state between requests.
        """
        enabled = True
        kwargs = getattr(request, "chat_template_kwargs", None)
        if isinstance(kwargs, dict):
            value = kwargs.get("enable_thinking", kwargs.get("thinking"))
            if value is not None:
                enabled = bool(value)
        self._reasoning_disabled = not enabled
        return request

    # ------------------------------------------------------------------
    # Properties used by structured-output engines (xgrammar, etc.)
    # ------------------------------------------------------------------

    @property
    def reasoning_start_str(self) -> str:
        return self.start_token

    @property
    def reasoning_end_str(self) -> str:
        return self.end_token

    # ------------------------------------------------------------------
    # Token-level checks for structured output engines
    # ------------------------------------------------------------------

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.start_token_id:
                return False
            if input_ids[i] == self.end_token_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Sequence[int],
    ) -> bool:
        return self.end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.end_token_id in input_ids[:-1]:
            return input_ids[input_ids.index(self.end_token_id) + 1 :]
        return []

    # ------------------------------------------------------------------
    # Non-streaming extraction
    # ------------------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        # ``<think>`` lives in the prompt, so strip it if it somehow appears.
        before, sep, after = model_output.partition(self.start_token)
        text = after if sep else before

        # No ``</think>`` in the generated output means thinking was disabled
        # (the marker already sits at the end of the prompt). The whole output
        # is regular content, not reasoning.
        if self.end_token not in text:
            return None, text or None

        reasoning, _, content = text.partition(self.end_token)
        return reasoning or None, content or None

    # ------------------------------------------------------------------
    # Streaming extraction
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Thinking disabled: the output is pure content, no markers appear.
        if self._reasoning_disabled and self.end_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text) if delta_text else None

        # Skip lone special tokens (start / end markers emitted solo).
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.start_token_id,
            self.end_token_id,
        ):
            return None

        has_start_prev = self.start_token_id in previous_token_ids
        has_start_delta = self.start_token_id in delta_token_ids
        has_end_prev = self.end_token_id in previous_token_ids
        has_end_delta = self.end_token_id in delta_token_ids

        # ---- Case 1: <think> was already seen in previous tokens ----
        if has_start_prev:
            if has_end_delta:
                end_idx = delta_text.find(self.end_token)
                reasoning = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            if has_end_prev:
                return DeltaMessage(content=delta_text)
            return DeltaMessage(reasoning=delta_text)

        # ---- Case 2: <think> appears in the current delta ----
        if has_start_delta:
            start_idx = delta_text.find(self.start_token)
            if has_end_delta:
                end_idx = delta_text.find(self.end_token)
                if start_idx == -1 or end_idx == -1:
                    return DeltaMessage(reasoning=delta_text)
                reasoning = delta_text[
                    start_idx + len(self.start_token) : end_idx
                ]
                content = delta_text[end_idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            reasoning = (
                delta_text[start_idx + len(self.start_token) :]
                if start_idx != -1
                else delta_text
            )
            return DeltaMessage(reasoning=reasoning or None)

        # ---- Case 3: no <think> at all (thinking-off / skip mode) ----
        # The template generates </think> immediately; everything after
        # that marker is regular content.
        if has_end_delta:
            end_idx = delta_text.find(self.end_token)
            reasoning = delta_text[:end_idx]
            content = delta_text[end_idx + len(self.end_token) :]
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )
        if has_end_prev:
            return DeltaMessage(content=delta_text)

        # Fallback: treat everything as reasoning (we haven't seen
        # any marker yet — shouldn't happen in normal flow).
        return DeltaMessage(reasoning=delta_text)

    # ------------------------------------------------------------------
    # Reasoning token counting
    # ------------------------------------------------------------------

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if (
            self.end_token_id in token_ids
            and self.start_token_id not in token_ids[: token_ids.index(self.end_token_id)]
        ):
            return token_ids.index(self.end_token_id)

        count = 0
        depth = 0
        for tid in token_ids:
            if tid == self.start_token_id:
                depth += 1
                continue
            if tid == self.end_token_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count
