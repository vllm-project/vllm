# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from .identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


# DSML tool-call start markers that the DSv4 model emits when it decides to
# invoke a tool. The reasoning parser treats these as an implicit
# end-of-reasoning when the explicit </think> token is missing, which the
# model occasionally fails to emit at long context.
_DSV4_TOOL_CALL_IMPLICIT_END_MARKERS: tuple[str, ...] = (
    "<｜DSML｜tool_calls>",
)


class DeepSeekV4ThinkingReasoningParser(DeepSeekR1ReasoningParser):
    """
    DeepSeek V4 thinking-mode reasoning parser.

    Extends :class:`DeepSeekR1ReasoningParser` with one behavior change:
    if the model emits a DSML tool-call start marker without first emitting
    ``</think>``, treat the marker as an implicit end-of-reasoning so the
    tool call is correctly handed off to the tool parser.

    Works around a model behavior observed at long context (~95k–100k
    input tokens) where DSv4-Flash sometimes skips ``</think>`` before
    opening ``<｜DSML｜tool_calls>``. Without this defensive split the
    orchestrator stays in reasoning phase, the tool parser never runs, and
    the caller sees a turn with reasoning but no tool call —
    indistinguishable on the client side from "model gave up".

    Healthy paths (explicit ``</think>``) are unchanged. The marker check
    fires only when no explicit start/end token has been seen.

    State (``_implicit_end_seen``) is per-instance and per-stream because
    a fresh ``Parser`` (and therefore a fresh reasoning parser) is
    constructed for each request.
    """

    implicit_end_markers: tuple[str, ...] = _DSV4_TOOL_CALL_IMPLICIT_END_MARKERS

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Per-stream sticky flag: once the implicit end marker is observed,
        # the rest of the stream is content and the orchestrator's
        # is_reasoning_end check must return True for every subsequent delta.
        self._implicit_end_seen: bool = False

    def _find_implicit_end_marker(self, text: str) -> tuple[str, int] | None:
        """Return ``(marker, index)`` of the earliest implicit end marker in
        ``text``, or ``None`` if none of the configured markers are present.
        """
        earliest: tuple[str, int] | None = None
        for marker in self.implicit_end_markers:
            idx = text.find(marker)
            if idx < 0:
                continue
            if earliest is None or idx < earliest[1]:
                earliest = (marker, idx)
        return earliest

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Honor the explicit </think> contract first.
        if super().is_reasoning_end(input_ids):
            return True
        # Sticky: once we've observed the marker in streaming, the rest of
        # the stream is content.
        if self._implicit_end_seen:
            return True
        # Non-streaming fallback: scan decoded text for the marker. We only
        # decode when start/end tokens are absent, which is the failure
        # mode we target.
        if not input_ids:
            return False
        if self.start_token_id in input_ids or self.end_token_id in input_ids:
            return False
        try:
            decoded = self.model_tokenizer.decode(list(input_ids))
        except Exception:
            return False
        return self._find_implicit_end_marker(decoded) is not None

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True
        if self._implicit_end_seen:
            return True
        return False

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        # Parent emitted content (explicit </think> in or before this delta).
        # Healthy path — nothing to do.
        if ret is not None and ret.content is not None:
            return ret

        # Defensive check is only meaningful in implicit-reasoning mode
        # (no explicit <think>/</think> in the stream). If any explicit
        # token is present, defer to parent.
        if (
            self.start_token_id in previous_token_ids
            or self.start_token_id in delta_token_ids
            or self.end_token_id in previous_token_ids
            or self.end_token_id in delta_token_ids
        ):
            return ret

        # Sticky: marker observed in an earlier delta — everything here is
        # content for the tool parser.
        if self._implicit_end_seen:
            return DeltaMessage(content=delta_text)

        marker_in_current = self._find_implicit_end_marker(current_text)
        if marker_in_current is None:
            # No marker anywhere; parent's classification stands.
            return ret

        # First sighting of the implicit end marker.
        self._implicit_end_seen = True
        _marker_str, marker_idx_current = marker_in_current
        # Position within delta_text where the marker begins.
        marker_idx_delta = marker_idx_current - len(previous_text)
        if marker_idx_delta < 0:
            # Marker straddles into previous_text but wasn't detected there
            # (parent path didn't hit). Treat all of delta_text as content.
            return DeltaMessage(content=delta_text)

        reasoning_part = delta_text[:marker_idx_delta] or None
        content_part = delta_text[marker_idx_delta:] or None
        if reasoning_part is None and content_part is None:
            return ret
        return DeltaMessage(reasoning=reasoning_part, content=content_part)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # If parent finds </think>, use its behavior.
        parent_result = super().extract_content_ids(input_ids)
        if parent_result:
            return parent_result
        # Fall back to text-scan for the implicit marker.
        if not input_ids:
            return []
        try:
            decoded = self.model_tokenizer.decode(list(input_ids))
        except Exception:
            return []
        marker = self._find_implicit_end_marker(decoded)
        if marker is None:
            return []
        # Without per-token offsets we can't slice input_ids exactly at the
        # marker boundary. Return everything as content tokens — the
        # orchestrator drives tool-call extraction off ``current_text`` for
        # DSML grammars, so this conservative answer is acceptable.
        return list(input_ids)


class DeepSeekV4ReasoningParser(ReasoningParser):
    """
    V4 reasoning parser that delegates to either
    :class:`DeepSeekV4ThinkingReasoningParser` (the V4-aware extension of
    R1) or :class:`IdentityReasoningParser` based on the ``thinking`` /
    ``enable_thinking`` chat-template kwargs.

    Replaces the previous arrangement where ``deepseek_v4`` reused
    :class:`DeepSeekV3ReasoningParser`, which lacked the implicit
    DSML-tool-call end-of-reasoning handling.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", False))
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        thinking = thinking or enable_thinking

        self._parser: ReasoningParser
        if thinking:
            self._parser = DeepSeekV4ThinkingReasoningParser(
                tokenizer, *args, **kwargs
            )
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    @property
    def reasoning_start_str(self) -> str | None:
        return self._parser.reasoning_start_str

    @property
    def reasoning_end_str(self) -> str | None:
        return self._parser.reasoning_end_str

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        return self._parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
