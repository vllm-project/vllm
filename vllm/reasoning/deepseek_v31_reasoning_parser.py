# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.reasoning import DeepSeekR1ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("deepseek_v31")
class DeepSeekV31ReasoningParser(DeepSeekR1ReasoningParser):

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: Union[ChatCompletionRequest, ResponsesRequest],
    ) -> Union[DeltaMessage, None]:
        if request.chat_template_kwargs is not None and \
           request.chat_template_kwargs.get("thinking", False):
            return super().extract_reasoning_content_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                request,
            )

        return DeltaMessage(content=delta_text)

    def extract_reasoning_content(
        self, model_output: str, request: Union[ChatCompletionRequest, ResponsesRequest]
    ) -> tuple[Optional[str], Optional[str]]:
        if request.chat_template_kwargs is not None and \
           request.chat_template_kwargs.get("thinking", False):
            return super().extract_reasoning_content(model_output, request)

        return None, model_output
