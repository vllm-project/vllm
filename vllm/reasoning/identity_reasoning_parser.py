# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class IdentityReasoningParser(ReasoningParser):
    """
    Identity reasoning parser.

    This parser does not attempt to parse or strip out reasoning tokens.
    It treats the entire model output as content and ignores reasoning.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer)
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # Always return True, since we never treat reasoning specially
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Identity: return all tokens as content
        return input_ids

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        # Just wrap delta_text as content, ignore reasoning
        if delta_text:
            return DeltaMessage(content=delta_text)
        return None

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        # No reasoning separation: return None for reasoning_content,
        # and full model_output as content
        return None, model_output
