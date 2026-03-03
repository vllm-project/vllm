# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any

from vllm import PromptType
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
)
from vllm.inputs import ProcessorInputs
from vllm.renderers.inputs import TokPrompt


class ClassifyIOProcessor(PoolingIOProcessor):
    def pre_process_online(
        self, request: ClassificationCompletionRequest | ClassificationChatRequest
    ) -> list[TokPrompt] | None:
        if isinstance(request, ClassificationChatRequest):
            self._validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            _, engine_prompts = self._preprocess_chat_online(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=None,
            )
        elif isinstance(request, ClassificationCompletionRequest):
            engine_prompts = self._preprocess_completion_online(
                request,
                prompt_input=request.input,
                prompt_embeds=None,
            )
        else:
            raise ValueError("Invalid classification request type")
        return engine_prompts

    def pre_process_offline(
        self,
        prompts: PromptType | Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[ProcessorInputs]:
        return self._preprocess_completion_offline(
            prompts=prompts, tokenization_kwargs=tokenization_kwargs
        )
