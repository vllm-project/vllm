# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Final, TypeAlias

import numpy as np

from vllm import ClassificationOutput
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.engine.serving import ServeContext
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationData,
    ClassificationRequest,
    ClassificationResponse,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


ClassificationServeContext: TypeAlias = ServeContext[ClassificationRequest]


class ServingClassification(PoolingServing):
    request_id_prefix = "classify"

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        trust_request_chat_template: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template

    async def _preprocess(
        self,
        ctx: ClassificationServeContext,
    ) -> None:
        ctx.lora_request = self._maybe_get_adapters(ctx.request)

        if isinstance(ctx.request, ClassificationChatRequest):
            self._validate_chat_template(
                request_chat_template=ctx.request.chat_template,
                chat_template_kwargs=ctx.request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )

            _, ctx.engine_prompts = await self._preprocess_chat(
                ctx.request,
                ctx.request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=None,
            )
        elif isinstance(ctx.request, ClassificationCompletionRequest):
            ctx.engine_prompts = await self._preprocess_completion(
                ctx.request,
                prompt_input=ctx.request.input,
                prompt_embeds=None,
            )
        else:
            raise ValueError("Invalid classification request type")
        return None

    async def _build_response(
        self,
        ctx: ClassificationServeContext,
    ) -> ClassificationResponse:
        id2label = getattr(self.model_config.hf_config, "id2label", {})

        items: list[ClassificationData] = []
        num_prompt_tokens = 0

        final_res_batch_checked = ctx.final_res_batch

        for idx, final_res in enumerate(final_res_batch_checked):
            classify_res = ClassificationOutput.from_base(final_res.outputs)

            probs = classify_res.probs
            predicted_index = int(np.argmax(probs))
            label = id2label.get(predicted_index)

            item = ClassificationData(
                index=idx,
                label=label,
                probs=probs,
                num_classes=len(probs),
            )

            items.append(item)
            prompt_token_ids = final_res.prompt_token_ids
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return ClassificationResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.model_name,
            data=items,
            usage=usage,
        )
