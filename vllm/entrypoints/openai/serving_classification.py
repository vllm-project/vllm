# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Optional, Union

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
from vllm.entrypoints.openai.serving_engine import OpenAIServing, ServeContext
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import ClassificationOutput

logger = init_logger(__name__)


class ServingClassification(OpenAIServing):
    _id_prefix = "classify"

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
        )

    async def create_classify(
        self,
        request: ClassificationRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ClassificationResponse, ErrorResponse]:
        response = await self.handle(request, raw_request)
        if isinstance(response, (ClassificationResponse, ErrorResponse)):
            return response

        return self.create_error_response("Unexpected response type")

    @override
    async def _preprocess(self, ctx: ServeContext):
        """
        Process classification inputs: tokenize text, resolve adapters,
        and prepare model-specific inputs.
        """
        if isinstance(ctx.request.input, str) and not ctx.request.input:
            return self.create_error_response(
                "Input cannot be empty for classification",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        if isinstance(ctx.request.input, list) and len(ctx.request.input) == 0:
            return None

        try:
            (
                ctx.lora_request,
                ctx.prompt_adapter_request,
            ) = self._maybe_get_adapters(ctx.request)

            ctx.tokenizer = await self.engine_client.get_tokenizer(
                ctx.lora_request)

            if ctx.prompt_adapter_request is not None:
                raise NotImplementedError(
                    "Prompt adapter is not supported for classification models"
                )

            (
                ctx.request_prompts,
                ctx.engine_prompts,
            ) = await self._preprocess_completion(
                ctx.request,
                ctx.tokenizer,
                ctx.request.input,
                truncate_prompt_tokens=ctx.request.truncate_prompt_tokens,
            )

            return None

        except (ValueError, TypeError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    @override
    def _build_response(self, ctx: ServeContext):
        """
        Convert model outputs to a formatted classification response
        with probabilities and labels.
        """
        items: list[ClassificationData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(ctx.final_res_batch):
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
