# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ResponsesRequest,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

from vllm.transformers_utils.tokenizer import AnyTokenizer

@ReasoningParserManager.register_module("minimax_m2")
class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MiniMax M2 model.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"


@ReasoningParserManager.register_module("minimax_m2_append_think")
class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
    """
    Reasoning parser for MiniMax M2 model.
    """
    is_first_token: bool

    def __init__(self, tokenizer: AnyTokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.is_first_token = True

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self.is_first_token:
            delta_text = "<think>" + delta_text
            self.is_first_token = False
        return DeltaMessage(content=delta_text)

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        return None, "</think>" + model_output
