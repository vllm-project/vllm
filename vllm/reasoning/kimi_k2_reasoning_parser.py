# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from .identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
else:
    ChatCompletionRequest = Any


logger = init_logger(__name__)


class KimiK2ReasoningParser(ReasoningParser):
    """
    Kimi K2 parser that delegates to either DeepSeekR1ReasoningParser or
    IdentityReasoningParser based on `thinking` and `separate_reasoning`.

    Unlike DeepSeekV3ReasoningParser which defaults to NOT thinking,
    KimiK2ReasoningParser defaults to thinking mode (uses DeepSeekR1ReasoningParser).

    This parser also filters out "(no content)" placeholder text that the
    Kimi K2 model sometimes generates when making tool calls without text.
    """

    def _clean_content(self, text: str | None) -> str | None:
        """
        Clean content by stripping "(no content)" placeholder.
        Returns None if content is empty after cleaning.
        """
        if text is None:
            return None
        cleaned = text.replace("(no content)", "")
        if not cleaned.strip():
            return None
        return cleaned

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.pop("chat_template_kwargs", {}) or {}
        # Key difference: default to True instead of False
        thinking = bool(chat_kwargs.pop("thinking", True))

        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest"
    ) -> tuple[str | None, str | None]:
        reasoning, content = self._parser.extract_reasoning(model_output, request)
        # Filter "(no content)" from content
        return reasoning, self._clean_content(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        result = self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        # Filter "(no content)" from the result
        if result is None:
            return None

        # Clean the content field if present
        cleaned_content = self._clean_content(result.content)

        # If content was the only field and it's now empty, return None
        if result.content is not None and cleaned_content is None:
            # Check if there's reasoning content to return
            if result.reasoning is not None:
                return DeltaMessage(reasoning=result.reasoning)
            # Check if there are tool calls
            if result.tool_calls is not None:
                return DeltaMessage(tool_calls=result.tool_calls)
            # Nothing left to return
            return None

        # If content was cleaned but not emptied, update it
        if result.content != cleaned_content:
            return DeltaMessage(
                content=cleaned_content,
                reasoning=result.reasoning,
                tool_calls=result.tool_calls,
            )

        return result
