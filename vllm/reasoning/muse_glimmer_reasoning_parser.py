# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-content parser for MuseGlimmer channel-scoped output."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.muse_glimmer_utils import (
    advance_emitted,
    current_assistant_turn,
    open_recipient,
    safe_open_body,
    visible_channels,
)


class MuseGlimmerReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._emitted_reasoning = ""
        self._emitted_content = ""
        self._initial_recipient: str | None = None

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Preserve MuseGlimmer framing so channel parsing remains possible."""
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Report an open answer or tool channel to engine-side callers."""
        try:
            text = self.model_tokenizer.decode(input_ids)
        except Exception:
            return False
        recipient = open_recipient(current_assistant_turn(text))
        return recipient not in (None, "self")

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """Continue classifying generation in the prompt's open channel."""
        try:
            text = self.model_tokenizer.decode(prompt_token_ids)
        except Exception:
            return
        self._initial_recipient = open_recipient(current_assistant_turn(text))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content-id slicing is unreliable for multi-token markers.
        return []

    def _seeded_text(self, text: str) -> str:
        if self._initial_recipient is None:
            return text
        return f"to={self._initial_recipient}<|message|>{text}"

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        """Promote any unstreamed answer body when generation is truncated."""
        content, _reasoning, _content_open, _reasoning_open = visible_channels(
            self._seeded_text(previous_text)
        )
        remainder = content[len(self._emitted_content) :]
        return remainder or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning while preserving framed text for channel consumers."""
        _content, reasoning, _content_open, _reasoning_open = visible_channels(
            model_output
        )
        return reasoning or None, model_output or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Stream clean reasoning and answer bodies from the shared segmenter."""
        content, reasoning, content_open, reasoning_open = visible_channels(
            self._seeded_text(current_text)
        )
        if content_open:
            content = safe_open_body(content)
        if reasoning_open:
            reasoning = safe_open_body(reasoning)

        reasoning_delta, self._emitted_reasoning = advance_emitted(
            self._emitted_reasoning, reasoning
        )
        content_delta, self._emitted_content = advance_emitted(
            self._emitted_content, content
        )
        if not reasoning_delta and not content_delta:
            return None

        return DeltaMessage(
            reasoning=reasoning_delta or None,
            content=content_delta or None,
        )
