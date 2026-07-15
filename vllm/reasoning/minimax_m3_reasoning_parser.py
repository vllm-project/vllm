# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for MiniMax M3 explicit thinking blocks.

    MiniMax M3 emits reasoning as:

        <mm:think>reasoning text</mm:think>assistant content

    The M3 tokenizer exposes both markers as complete vocabulary entries, but
    generated marker text may be tokenized into smaller pieces. The streaming
    parser therefore uses text markers for extraction instead of relying on the
    single vocabulary IDs. The chat template may also prefill the start marker
    when ``thinking_mode="enabled"``, so generated text can begin directly
    inside a reasoning block without emitting ``<mm:think>`` again.
    """

    @property
    def start_token(self) -> str:
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        return "</mm:think>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._start_token_ids = self._encode_marker(self.start_token)
        self._end_token_ids = self._encode_marker(self.end_token)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._initial_in_reasoning = chat_kwargs.get("thinking_mode") == "enabled"
        self._reasoning_ended_streaming = False
        self._reasoning_active_streaming = self._initial_in_reasoning
        self._pending_marker_streaming = False
        self._last_streaming_delta_token_ids: tuple[int, ...] | None = None
        self._last_streaming_content_token_ids: list[int] | None = None

    def _encode_text(self, text: str) -> list[int]:
        try:
            return list(self.model_tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.model_tokenizer.encode(text))

    def _encode_marker(self, marker: str) -> tuple[int, ...]:
        return tuple(self._encode_text(marker))

    def _decode_text(self, token_ids: Sequence[int]) -> str:
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=False
            )
        except TypeError:
            return self.model_tokenizer.decode(list(token_ids))

    def _content_suffix_token_ids(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        content: str | None,
    ) -> list[int]:
        if content is None:
            return []
        if content == delta_text:
            return list(delta_token_ids)
        if delta_text.endswith(content):
            prefix_text = delta_text[: len(delta_text) - len(content)]
            for index in range(len(delta_token_ids) + 1):
                if self._decode_text(delta_token_ids[:index]) == prefix_text:
                    return list(delta_token_ids[index:])
        return self._encode_text(content)

    @staticmethod
    def _contains_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return False
        marker_len = len(marker_ids)
        return any(
            tuple(token_ids[i : i + marker_len]) == tuple(marker_ids)
            for i in range(len(token_ids) - marker_len + 1)
        )

    @staticmethod
    def _rfind_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> int:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return -1
        marker_len = len(marker_ids)
        for i in range(len(token_ids) - marker_len, -1, -1):
            if tuple(token_ids[i : i + marker_len]) == tuple(marker_ids):
                return i
        return -1

    @staticmethod
    def _find_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int], start: int = 0
    ) -> int:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return -1
        marker_len = len(marker_ids)
        for i in range(start, len(token_ids) - marker_len + 1):
            if tuple(token_ids[i : i + marker_len]) == tuple(marker_ids):
                return i
        return -1

    @staticmethod
    def _starts_with_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return False
        return tuple(token_ids[: len(marker_ids)]) == tuple(marker_ids)

    @staticmethod
    def _ends_with_token_sequence_prefix(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids:
            return False
        max_len = min(len(token_ids), len(marker_ids) - 1)
        for prefix_len in range(max_len, 0, -1):
            if tuple(token_ids[-prefix_len:]) == tuple(marker_ids[:prefix_len]):
                return True
        return False

    @staticmethod
    def _strip_partial_marker_suffix(text: str, marker: str) -> str:
        max_len = min(len(text), len(marker) - 1)
        for suffix_len in range(max_len, 0, -1):
            if marker.startswith(text[-suffix_len:]):
                return text[:-suffix_len]
        return text

    @staticmethod
    def _visible_delta(previous: str | None, current: str | None) -> str | None:
        if not current:
            return None
        if not previous:
            return current
        if current.startswith(previous):
            delta = current[len(previous) :]
            return delta or None
        return current

    def _visible_segments(self, text: str) -> tuple[str | None, str | None]:
        if not text:
            return None, None

        if not self._initial_in_reasoning:
            if self.end_token.startswith(text) and len(text) < len(self.end_token):
                return None, None
            if text.startswith(self.end_token):
                text = text[len(self.end_token) :]
                if not text:
                    return None, None

        if self._initial_in_reasoning:
            reasoning, end, content = text.partition(self.end_token)
            if end:
                return reasoning or None, content or None
            reasoning = self._strip_partial_marker_suffix(reasoning, self.end_token)
            return reasoning or None, None

        # Only a leading start marker opens reasoning; hold back while the text so
        # far is still a prefix of that marker. A start marker anywhere else is
        # literal content, so it never gets stripped mid-stream.
        if not text.startswith(self.start_token):
            if self.start_token.startswith(text):
                return None, None
            return None, text

        after_start = text[len(self.start_token) :]
        reasoning, end, content_after = after_start.partition(self.end_token)
        if end:
            return reasoning or None, content_after or None

        reasoning = self._strip_partial_marker_suffix(reasoning, self.end_token)
        return reasoning or None, None

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        # Markers are only meaningful at the boundary they delimit: a reasoning
        # block is opened by a leading start marker (or the prefilled template in
        # enabled mode) and closed by the first end marker. A start marker that
        # the model emits inside content is literal text, not a block opener.
        if self._initial_in_reasoning:
            reasoning, end, content = model_output.partition(self.end_token)
            if not end:
                return model_output, None
            return reasoning, content or None

        # A leading closer is a stray token; drop it. Later closers stay content.
        if model_output.startswith(self.end_token):
            content = model_output[len(self.end_token) :]
            return None, content or None

        if not model_output.startswith(self.start_token):
            return None, model_output

        after_start = model_output[len(self.start_token) :]
        reasoning, end, content_after = after_start.partition(self.end_token)
        if not end:
            return reasoning, None
        return reasoning, content_after or None

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._reasoning_ended_streaming:
            return True

        if self._reasoning_active_streaming or self._pending_marker_streaming:
            return False

        delta_ids = tuple(delta_ids)
        if self._contains_token_sequence(delta_ids, self._end_token_ids):
            return True
        if self._contains_token_sequence(input_ids, self._end_token_ids):
            return True
        if self._initial_in_reasoning:
            return False
        if self._ends_with_token_sequence_prefix(input_ids, self._start_token_ids):
            return False
        if self._ends_with_token_sequence_prefix(input_ids, self._end_token_ids):
            return False
        if not self._contains_token_sequence(input_ids, self._start_token_ids):
            return bool(input_ids)
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if (
            self._last_streaming_delta_token_ids == tuple(input_ids)
            and self._last_streaming_content_token_ids is not None
        ):
            content_ids = self._last_streaming_content_token_ids
            self._last_streaming_delta_token_ids = None
            self._last_streaming_content_token_ids = None
            return list(content_ids)

        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index >= 0:
            return input_ids[end_index + len(self._end_token_ids) :]

        # Reasoning is open only from a leading start marker (or the prefilled
        # template). A start marker elsewhere is literal content, so without a
        # closer the whole sequence is content.
        leading_reasoning = self._initial_in_reasoning or (
            self._starts_with_token_sequence(input_ids, self._start_token_ids)
        )
        if leading_reasoning:
            return []
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not delta_text:
            return None

        if not previous_text:
            self._reasoning_ended_streaming = False
            self._reasoning_active_streaming = self._initial_in_reasoning
            self._pending_marker_streaming = False
            self._last_streaming_delta_token_ids = None
            self._last_streaming_content_token_ids = None
        previous_reasoning, previous_content = self._visible_segments(previous_text)
        current_reasoning, current_content = self._visible_segments(current_text)
        if self.end_token in current_text or current_content is not None:
            self._reasoning_ended_streaming = True
            self._reasoning_active_streaming = False
            self._pending_marker_streaming = False
        else:
            self._last_streaming_delta_token_ids = None
            self._last_streaming_content_token_ids = None
            self._reasoning_active_streaming = (
                self._initial_in_reasoning
                or self.start_token in current_text
                or current_reasoning is not None
            )
            self._pending_marker_streaming = not self._reasoning_active_streaming and (
                self.start_token.startswith(current_text)
                or self.end_token.startswith(current_text)
            )
        reasoning = self._visible_delta(previous_reasoning, current_reasoning)
        content = self._visible_delta(previous_content, current_content)
        if self._reasoning_ended_streaming:
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = self._content_suffix_token_ids(
                delta_text, delta_token_ids, content
            )
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        # Reasoning spans from a leading start marker (or the prefilled template)
        # to the first closer. A non-leading start marker is literal content.
        token_ids = list(token_ids)
        if self._initial_in_reasoning:
            body_start = 0
        elif self._starts_with_token_sequence(token_ids, self._start_token_ids):
            body_start = len(self._start_token_ids)
        else:
            return 0
        end_index = self._find_token_sequence(
            token_ids, self._end_token_ids, body_start
        )
        body_end = end_index if end_index >= 0 else len(token_ids)
        return body_end - body_start

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Operates on the full prompt/history token stream to initialize
        # reasoning state, so it uses rightmost-marker (current-state) semantics
        # rather than the leading-only rule used for generated output above.
        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)
        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index < 0:
            return False
        if start_index < 0:
            return True
        return end_index > start_index
