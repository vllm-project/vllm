# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from typing import cast

import numpy as np
from fastapi import Request
from typing_extensions import override

import jinja2

from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
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
    @override
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

            messages = getattr(ctx.request, "messages", None)
            if messages:
                error_check_ret = self._validate_chat_template(
                    request_chat_template=getattr(ctx.request, "chat_template", None),
                    chat_template_kwargs=getattr(
                        ctx.request, "chat_template_kwargs", None
                    ),
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret

                (
                    _,
                    _,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    ctx.request,
                    ctx.tokenizer,
                    messages,
                    chat_template=(
                        getattr(ctx.request, "chat_template", None)
                        or self.chat_template
                    ),
                    chat_template_content_format=self.chat_template_content_format,
                    add_generation_prompt=False,
                    continue_final_message=False,
                    add_special_tokens=getattr(
                        ctx.request, "add_special_tokens", False
                    ),
                )
                ctx.engine_prompts = engine_prompts
            else:
                if ctx.request.input is None or (
                    isinstance(ctx.request.input, str) and not ctx.request.input
                ):
                    return self.create_error_response(
                        "Input or messages must be provided",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )
                if (
                    isinstance(ctx.request.input, list)
                    and len(ctx.request.input) == 0
                ):
                    return None

                renderer = self._get_renderer(ctx.tokenizer)
                ctx.engine_prompts = await renderer.render_prompt(
                    prompt_or_prompts=ctx.request.input,
                    config=self._build_render_config(ctx.request),
                )

            return None

        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    @override
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

    @override
    def _create_pooling_params(
        self,
        ctx: ClassificationServeContext,
    ) -> PoolingParams | ErrorResponse:
        pooling_params = super()._create_pooling_params(ctx)
        if isinstance(pooling_params, ErrorResponse):
            return pooling_params

        try:
            pooling_params.verify("classify", self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        return pooling_params
