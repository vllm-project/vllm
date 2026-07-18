# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for the iQuest Coder V2 tokenizer template."""

from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike


class IquestCoderV2ReasoningParser(ReasoningParser):
    """Extract reasoning from the current iQuest Coder assistant turn."""

    _START_TOKEN = "<think>"
    _END_TOKEN = "</think>"
    _ASSISTANT_TOKEN = "<|iquestcoder_assistant|>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking")
        enable_thinking = chat_kwargs.get("enable_thinking")
        self._thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking or enable_thinking)
        )

        required_tokens = (
            self._START_TOKEN,
            self._END_TOKEN,
            self._ASSISTANT_TOKEN,
        )
        missing_tokens = [token for token in required_tokens if token not in self.vocab]
        if missing_tokens:
            raise ValueError(
                "Tokenizer is missing required iQuest Coder tokens: "
                + ", ".join(missing_tokens)
            )

        self._start_token_id = self.vocab[self._START_TOKEN]
        self._end_token_id = self.vocab[self._END_TOKEN]
        self._assistant_token_id = self.vocab[self._ASSISTANT_TOKEN]

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self._thinking_enabled:
            return True

        for token_id in reversed(input_ids):
            if token_id == self._start_token_id:
                return False
            if token_id == self._end_token_id:
                return True
            if token_id == self._assistant_token_id:
                return False
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        del input_ids
        return not self._thinking_enabled or self._end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids

        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] == self._end_token_id:
                return input_ids[index + 1 :]
            if input_ids[index] in (self._start_token_id, self._assistant_token_id):
                return []
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        del request
        if not self._thinking_enabled:
            return None, model_output

        start_index = model_output.find(self._START_TOKEN)
        if start_index != -1:
            model_output = model_output[start_index + len(self._START_TOKEN) :]

        end_index = model_output.find(self._END_TOKEN)
        if end_index == -1:
            return model_output or None, None

        reasoning = model_output[:end_index]
        content = model_output[end_index + len(self._END_TOKEN) :]
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del previous_text, current_text, current_token_ids
        if not delta_text:
            return None
        if not self._thinking_enabled:
            return DeltaMessage(content=delta_text)

        start_index = delta_text.find(self._START_TOKEN)
        if start_index != -1:
            delta_text = delta_text[start_index + len(self._START_TOKEN) :]

        end_index = delta_text.find(self._END_TOKEN)
        if end_index != -1:
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self._END_TOKEN) :]
            if not reasoning and not content:
                return None
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )

        if self._end_token_id in delta_token_ids:
            return None
        if self._end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)
        if not delta_text:
            return None
        return DeltaMessage(reasoning=delta_text)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._thinking_enabled:
            return 0

        count = 0
        in_reasoning = True
        for token_id in token_ids:
            if token_id in (self._assistant_token_id, self._start_token_id):
                in_reasoning = True
            elif token_id == self._end_token_id:
                in_reasoning = False
            elif in_reasoning:
                count += 1
        return count


__all__ = ["IquestCoderV2ReasoningParser"]
