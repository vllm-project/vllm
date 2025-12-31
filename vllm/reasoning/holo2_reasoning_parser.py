# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import (
    ReasoningParser,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class Holo2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Holo2 models which are based on Qwen3.

    The Holo2 model uses <think>...</think> tokens to denote reasoning text but <think>
    is part of the chat template. This parser extracts the reasoning content until
    </think> in the model's output.

    The model provides a switch to enable or disable reasoning
    output via the 'thinking=False' parameter.

    Chat template args:
    - thinking: Whether to enable reasoning output (default: True)


    Parsing rules on model output:
        - thinking == False
            -> Model output is treated as purely the content |content|
        - thinking == True
            -> Model output is |reasoning_content|</think>|content|
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Deepseek V3 and Holo2 are similar. However, Holo2 models think by default.
        # this parser without user specified chat template args is initiated once for
        # all requests in the structured output manager. So it is important that without
        # user specified chat template args, the default thinking is True.

        enable_thinking = bool(chat_kwargs.get("thinking", True))

        if enable_thinking:
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
        self, model_output: str, request: ChatCompletionRequest
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
    ) -> DeltaMessage | None:
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
