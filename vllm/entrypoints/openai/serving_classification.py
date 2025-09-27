# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from typing import Optional, Union, cast

import numpy as np
from fastapi import Request
from typing_extensions import override

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ClassificationData,
                                              ClassificationRequest,
                                              ClassificationResponse,
                                              ErrorResponse, UsageInfo)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import (ClassificationServeContext,
                                                    OpenAIServing,
                                                    ServeContext)
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
    ) -> Optional[ErrorResponse]:
        """
        Process classification inputs: tokenize text, resolve adapters,
        and prepare model-specific inputs.
        """
        ctx = cast(ClassificationServeContext, ctx)
        if isinstance(ctx.request.input, str) and not ctx.request.input:
            return self.create_error_response(
                "Input cannot be empty for classification",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        if isinstance(ctx.request.input, list) and len(ctx.request.input) == 0:
            return None

        try:
            ctx.tokenizer = await self.engine_client.get_tokenizer()

            renderer = self._get_renderer(ctx.tokenizer)
            ctx.engine_prompts = await renderer.render_prompt(
                prompt_or_prompts=ctx.request.input,
                config=self._build_render_config(ctx.request))

            return None

        except (ValueError, TypeError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    @override
    def _build_response(
        self,
        ctx: ServeContext,
    ) -> Union[ClassificationResponse, ErrorResponse]:
        """
        Convert model outputs to a formatted classification response
        with probabilities and labels.
        """
        ctx = cast(ClassificationServeContext, ctx)
        items: list[ClassificationData] = []
        num_prompt_tokens = 0

        final_res_batch_checked = cast(list[PoolingRequestOutput],
                                       ctx.final_res_batch)

        for idx, final_res in enumerate(final_res_batch_checked):
            classify_res = ClassificationOutput.from_base(final_res.outputs)

            probs = classify_res.probs
            predicted_index = int(np.argmax(probs))
            label = getattr(self.model_config.hf_config, "id2label",
                            {}).get(predicted_index)

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

    def _build_render_config(self,
                             request: ClassificationRequest) -> RenderConfig:
        return RenderConfig(
            max_length=self.max_model_len,
            truncate_prompt_tokens=request.truncate_prompt_tokens)


class ServingClassification(ClassificationMixin):
    request_id_prefix = "classify"

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )

    async def create_classify(
        self,
        request: ClassificationRequest,
        raw_request: Request,
    ) -> Union[ClassificationResponse, ErrorResponse]:
        model_name = self.models.model_name()
        request_id = (f"{self.request_id_prefix}-"
                      f"{self._base_request_id(raw_request)}")

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
    ) -> Union[PoolingParams, ErrorResponse]:
        pooling_params = super()._create_pooling_params(ctx)
        if isinstance(pooling_params, ErrorResponse):
            return pooling_params

        try:
            pooling_params.verify("classify", self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        return pooling_params
