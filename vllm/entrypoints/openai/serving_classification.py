# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from typing import cast

import jinja2
import numpy as np
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationData,
    ClassificationRequest,
    ClassificationResponse,
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_engine import (
    ClassificationServeContext,
    OpenAIServing,
    ServeContext,
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.renderer import RenderConfig
from vllm.logger import init_logger
from vllm.outputs import ClassificationOutput, PoolingRequestOutput
from vllm.pooling_params import PoolingParams

logger = init_logger(__name__)


class ClassificationMixin(OpenAIServing):
    chat_template: str | None
    chat_template_content_format: ChatTemplateContentFormatOption
    trust_request_chat_template: bool

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """
        Process classification inputs: tokenize text, resolve adapters,
        and prepare model-specific inputs.
        """
        ctx = cast(ClassificationServeContext, ctx)
        try:
            ctx.tokenizer = await self.engine_client.get_tokenizer()

            request_obj = ctx.request

            if isinstance(request_obj, ClassificationChatRequest):
                chat_request = request_obj
                messages = chat_request.messages
                trust_request_chat_template = getattr(
                    self,
                    "trust_request_chat_template",
                    False,
                )
                ret = self._validate_chat_template(
                    request_chat_template=chat_request.chat_template,
                    chat_template_kwargs=chat_request.chat_template_kwargs,
                    trust_request_chat_template=trust_request_chat_template,
                )
                if ret:
                    return ret

                (
                    _,
                    _,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    cast(ChatCompletionRequest, chat_request),
                    ctx.tokenizer,
                    messages,
                    chat_template=(
                        chat_request.chat_template
                        or getattr(self, "chat_template", None)
                    ),
                    chat_template_content_format=cast(
                        ChatTemplateContentFormatOption,
                        getattr(self, "chat_template_content_format", "auto"),
                    ),
                    add_generation_prompt=False,
                    continue_final_message=False,
                    add_special_tokens=chat_request.add_special_tokens,
                )
                ctx.engine_prompts = engine_prompts

            elif isinstance(request_obj, ClassificationCompletionRequest):
                completion_request = request_obj
                input_data = completion_request.input
                if input_data in (None, ""):
                    return self.create_error_response(
                        "Input or messages must be provided",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )
                if isinstance(input_data, list) and not input_data:
                    ctx.engine_prompts = []
                    return None

                renderer = self._get_renderer(ctx.tokenizer)
                prompt_input = cast(str | list[str], input_data)
                ctx.engine_prompts = await renderer.render_prompt(
                    prompt_or_prompts=prompt_input,
                    config=self._build_render_config(completion_request),
                )
            else:
                return self.create_error_response(
                    "Invalid classification request type",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            return None

        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    def _build_response(
        self,
        ctx: ServeContext,
    ) -> ClassificationResponse | ErrorResponse:
        """
        Convert model outputs to a formatted classification response
        with probabilities and labels.
        """
        ctx = cast(ClassificationServeContext, ctx)
        items: list[ClassificationData] = []
        num_prompt_tokens = 0

        final_res_batch_checked = cast(list[PoolingRequestOutput], ctx.final_res_batch)

        for idx, final_res in enumerate(final_res_batch_checked):
            classify_res = ClassificationOutput.from_base(final_res.outputs)

            probs = classify_res.probs
            predicted_index = int(np.argmax(probs))
            label = getattr(self.model_config.hf_config, "id2label", {}).get(
                predicted_index
            )

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

    def _build_render_config(self, request: ClassificationRequest) -> RenderConfig:
        return RenderConfig(
            max_length=self.max_model_len,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
        )


class ServingClassification(ClassificationMixin):
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
        self.chat_template_content_format = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template

    async def create_classify(
        self,
        request: ClassificationRequest,
        raw_request: Request,
    ) -> ClassificationResponse | ErrorResponse:
        model_name = self.models.model_name()
        request_id = f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"

        ctx = ClassificationServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            request_id=request_id,
        )

        return await super().handle(ctx)  # type: ignore

    def _create_pooling_params(
        self,
        ctx: ServeContext[ClassificationRequest],
    ) -> PoolingParams | ErrorResponse:
        pooling_params = super()._create_pooling_params(ctx)
        if isinstance(pooling_params, ErrorResponse):
            return pooling_params

        try:
            pooling_params.verify("classify", self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        return pooling_params
