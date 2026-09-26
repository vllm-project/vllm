# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from .identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class GigaChat35ReasoningParser(ReasoningParser):
    """
    GigaChat 3.5 parser that delegates to either DeepSeekR1ReasoningParser or
    IdentityReasoningParser based on the `reasoning` chat-template kwarg.

    With `reasoning=True` the chat template opens `<think>` in the generation
    prompt, so the model output contains only the closing `</think>`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        reasoning = bool(chat_kwargs.get("reasoning", False))

        self._parser: ReasoningParser
        if reasoning:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
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


class GigaChat35ReasoningWithThinkingParser(GigaChat35ReasoningParser):
    """
    GigaChat35ReasoningParser that defaults to reasoning mode, for
    GigaChat 3.5 Reasoning models whose chat template always opens `<think>`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        chat_kwargs = dict(kwargs.get("chat_template_kwargs", {}) or {})
        if chat_kwargs.get("reasoning") is None:
            chat_kwargs["reasoning"] = True
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, *args, **kwargs)
