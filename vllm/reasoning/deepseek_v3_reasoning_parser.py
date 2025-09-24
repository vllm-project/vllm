# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import (DeepSeekR1ReasoningParser, ReasoningParser,
                            ReasoningParserManager)

from .identity_reasoning_parser import IdentityReasoningParser

logger = init_logger(__name__)


@ReasoningParserManager.register_module("deepseek_v3")
class DeepSeekV3ReasoningParser(ReasoningParser):
    """
    V3 parser that delegates to either DeepSeekR1ReasoningParser or
    IdentityReasoningParser based on `thinking` and `separate_reasoning`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer)

        thinking = bool(kwargs.pop("thinking", False))

        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer)
        else:
            self._parser = IdentityReasoningParser(tokenizer)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        return self._parser.extract_reasoning_content(model_output, request)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        return self._parser.extract_reasoning_content_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
