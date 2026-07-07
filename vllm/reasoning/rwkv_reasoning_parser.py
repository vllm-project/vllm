# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class RWKVReasoningParser(ReasoningParser):
    """Reasoning parser for RWKV-style ``<think>...</think>`` text.

    RWKV tokenizers may split ``<think>`` and ``</think>`` into multiple token
    IDs, so this parser tracks the full marker sequences instead of relying on a
    single vocabulary lookup.
    """

    start_token = "<think>"
    end_token = "</think>"

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking") is True

        self.start_token_ids = self._encode_marker(self.start_token)
        self.end_token_ids = self._encode_marker(self.end_token)

    @property
    def reasoning_start_str(self) -> str:
        return self.start_token

    @property
    def reasoning_end_str(self) -> str:
        return self.end_token

    def _encode_marker(self, marker: str) -> list[int]:
        token_ids = self.model_tokenizer.encode(
            marker,
            add_special_tokens=False,
        )
        if not token_ids:
            raise RuntimeError(
                f"{self.__class__.__name__} could not tokenize {marker!r}."
            )
        return list(token_ids)

    @staticmethod
    def _find_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
        if not pattern or len(pattern) > len(values):
            return -1
        last_start = len(values) - len(pattern)
        for start in range(last_start + 1):
            if list(values[start : start + len(pattern)]) == list(pattern):
                return start
        return -1

    @staticmethod
    def _rfind_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
        if not pattern or len(pattern) > len(values):
            return -1
        for start in range(len(values) - len(pattern), -1, -1):
            if list(values[start : start + len(pattern)]) == list(pattern):
                return start
        return -1

    @classmethod
    def _has_subsequence(cls, values: Sequence[int], pattern: Sequence[int]) -> bool:
        return cls._find_subsequence(values, pattern) >= 0

    @staticmethod
    def _without_trailing_partial_marker(text: str, markers: Sequence[str]) -> str:
        for marker in markers:
            max_prefix = min(len(marker) - 1, len(text))
            for prefix_len in range(max_prefix, 0, -1):
                if text.endswith(marker[:prefix_len]):
                    return text[:-prefix_len]
        return text

    def _extract_reasoning_text(
        self,
        model_output: str,
        *,
        streaming: bool = False,
    ) -> tuple[str | None, str | None]:
        start_index = model_output.find(self.start_token)
        has_start = start_index >= 0
        if has_start:
            model_output = model_output[start_index + len(self.start_token) :]

        end_index = model_output.find(self.end_token)
        if end_index >= 0:
            reasoning = model_output[:end_index]
            content = model_output[end_index + len(self.end_token) :] or None
        elif has_start or self.thinking_enabled:
            reasoning = model_output
            content = None
        else:
            reasoning = None
            content = model_output

        if streaming:
            if reasoning is not None and content is None:
                reasoning = self._without_trailing_partial_marker(
                    reasoning,
                    (self.end_token,),
                )
            if content is not None:
                content = self._without_trailing_partial_marker(
                    content,
                    (self.start_token, self.end_token),
                )

        return reasoning or None, content or None

    @staticmethod
    def _delta_text(previous: str | None, current: str | None) -> str | None:
        previous = previous or ""
        current = current or ""
        if not current:
            return None
        delta = current[len(previous) :] if current.startswith(previous) else current
        return delta or None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        last_start = self._rfind_subsequence(input_ids, self.start_token_ids)
        last_end = self._rfind_subsequence(input_ids, self.end_token_ids)
        return last_end >= 0 and last_end > last_start

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Sequence[int],
    ) -> bool:
        if self._has_subsequence(delta_ids, self.end_token_ids):
            return True
        if not self.is_reasoning_end(input_ids):
            return False

        last_end = self._rfind_subsequence(input_ids, self.end_token_ids)
        delta_start = max(0, len(input_ids) - len(delta_ids))
        return last_end >= delta_start

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_index = self._find_subsequence(input_ids, self.end_token_ids)
        if end_index < 0:
            return []
        return input_ids[end_index + len(self.end_token_ids) :]

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        start_index = self._find_subsequence(token_ids, self.start_token_ids)
        if start_index >= 0:
            reasoning_start = start_index + len(self.start_token_ids)
        elif self.thinking_enabled:
            reasoning_start = 0
        else:
            return 0

        end_index = self._find_subsequence(token_ids, self.end_token_ids)
        reasoning_end = end_index if end_index >= 0 else len(token_ids)
        return max(0, reasoning_end - reasoning_start)

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest | None",
    ) -> tuple[str | None, str | None]:
        del request
        return self._extract_reasoning_text(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids
        previous_reasoning, previous_content = self._extract_reasoning_text(
            previous_text,
            streaming=True,
        )
        current_reasoning, current_content = self._extract_reasoning_text(
            current_text,
            streaming=True,
        )

        reasoning_delta = self._delta_text(previous_reasoning, current_reasoning)
        content_delta = self._delta_text(previous_content, current_content)
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)
