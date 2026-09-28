# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GptOss model.

    The GptOss model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return True

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )
