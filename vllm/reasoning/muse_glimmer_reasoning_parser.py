# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-content parser for MuseGlimmer channel-scoped output."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.muse_glimmer_utils import (
    REASONING_RECIPIENT,
    advance_emitted,
    channel_seed,
    current_assistant_turn,
    flush_open_body,
    framing_start,
    has_channel_framing,
    has_complete_channel,
    open_recipient,
    safe_open_body,
    safe_unframed_tail,
    strip_frozen_tail,
    visible_channels,
)


class MuseGlimmerReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._emitted_reasoning = ""
        self._emitted_content = ""
        # None until the framed path first runs. Set at that point to the
        # whole-text content cursor (the segmenter drops pre-header text, so
        # the framed path re-anchors instead of wedging on the unframed
        # prefix); the finish-time unframed flush resumes from it.
        self._emitted_content_pre_flip: str | None = None
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
        return recipient not in (None, REASONING_RECIPIENT)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """Continue classifying generation in the prompt's open channel.

        Only the open channel's recipient is seeded, not its body text: a
        `continue_final_message` that stops mid-tool-call therefore parses
        nothing from the prefilled body (documented limitation, matching the
        Rust parser). Non-streaming `parse()` receives no prompt token ids,
        so a prefilled channel continuation additionally drops text that a
        later framed header would re-anchor (same limitation, one path over).
        """
        try:
            text = self.model_tokenizer.decode(prompt_token_ids)
        except Exception:
            return
        self._initial_recipient = open_recipient(current_assistant_turn(text))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content-id slicing is unreliable for multi-token markers.
        return []

    def _seeded_text(self, text: str) -> str:
        seed = channel_seed(self._initial_recipient)
        return text if seed is None else seed + text

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        """Promote any unstreamed answer body when generation is truncated."""
        seeded = self._seeded_text(previous_text)
        if not has_complete_channel(seeded):
            # The streaming fallback held these tails back; flush them now.
            # This also recovers text the streaming side held while a stray
            # marker (e.g. a quoted `<|start|>`) never completed a header.
            content = flush_open_body(seeded)
            cursor = self._emitted_content
            if self._emitted_content_pre_flip is not None:
                # The stream flipped unframed->framed without any channel
                # ever completing: framed content stayed empty, so the
                # whole-text flush resumes from the pre-flip cursor.
                cursor = self._emitted_content_pre_flip
            remainder, self._emitted_content = advance_emitted(cursor, content)
            return remainder or None
        content, _reasoning, _content_open, _reasoning_open = visible_channels(
            seeded, flush_growing=True
        )
        remainder, self._emitted_content = advance_emitted(
            self._emitted_content, content
        )
        return remainder or None

    def get_streaming_fallback_reasoning(self, previous_text: str) -> str | None:
        """Flush reasoning text held back while its channel was still open."""
        _content, reasoning, _content_open, _reasoning_open = visible_channels(
            self._seeded_text(previous_text), flush_growing=True
        )
        remainder, self._emitted_reasoning = advance_emitted(
            self._emitted_reasoning, reasoning
        )
        return remainder or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning while preserving framed text for channel consumers."""
        # An open (truncated) reasoning body flushes minus trailing partial
        # framing, matching the streaming path.
        _content, reasoning, _content_open, _reasoning_open = visible_channels(
            model_output, flush_growing=True
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
        seeded = self._seeded_text(current_text)
        if not has_channel_framing(seeded):
            # No channel framing anywhere (e.g. a grammar-constrained answer
            # that never opened a channel): stream the text as plain content,
            # mirroring the non-streaming unframed fallback.
            content = safe_unframed_tail(seeded)
            content_delta, self._emitted_content = advance_emitted(
                self._emitted_content, content
            )
            return DeltaMessage(content=content_delta) if content_delta else None
        content, reasoning, content_open, reasoning_open = visible_channels(
            seeded, withhold_open_untagged=True
        )
        if content_open:
            content = safe_open_body(content)
        if reasoning_open:
            reasoning = safe_open_body(reasoning)
        flip_delta = ""
        if self._emitted_content_pre_flip is None:
            # First framed delta: the segmenter drops the pre-header region.
            # The region is frozen now, so flush it verbatim minus the tail
            # the unframed path never streams (end markers, whitespace), then
            # re-anchor: framed content never continues the unframed prefix.
            pre = strip_frozen_tail(seeded[: framing_start(seeded)])
            flip_delta, self._emitted_content_pre_flip = advance_emitted(
                self._emitted_content, pre
            )
            self._emitted_content = ""

        reasoning_delta, self._emitted_reasoning = advance_emitted(
            self._emitted_reasoning, reasoning
        )
        content_delta, self._emitted_content = advance_emitted(
            self._emitted_content, content
        )
        if flip_delta:
            content_delta = flip_delta + content_delta
        if not reasoning_delta and not content_delta:
            return None

        return DeltaMessage(
            reasoning=reasoning_delta or None,
            content=content_delta or None,
        )
