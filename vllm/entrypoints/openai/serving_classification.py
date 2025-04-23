# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Optional, Union, cast

import numpy as np
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ClassificationData,
                                              ClassificationRequest,
                                              ClassificationResponse,
                                              ErrorResponse, UsageInfo)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    TextTokensPrompt)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import ClassificationOutput, PoolingRequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.utils import merge_async_iterators

logger = init_logger(__name__)


class OpenAIServingClassification(OpenAIServing):

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
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        validation_error = self._validate_request(request)
        if validation_error is not None:
            return validation_error

        model_name = self._get_model_name(request.model)
        request_id = f"classify-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        if isinstance(request.input, list) and len(request.input) == 0:
            return self.request_output_to_classify_response(
                [],
                request_id,
                created_time,
                model_name,
            )

        (request_prompts, engine_prompts, lora_request,
         prompt_adapter_request) = (
             await self._prepare_classification_inputs(request))

        generators = await self._get_classification_generators(
            request_id,
            request,
            raw_request,
            engine_prompts,
            request_prompts,
            lora_request,
            prompt_adapter_request,
        )

        num_prompts = len(engine_prompts)
        final_res_batch: list[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts
        result_generator = merge_async_iterators(*generators)

        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(list[PoolingRequestOutput],
                                           final_res_batch)

            return self.request_output_to_classify_response(
                final_res_batch_checked,
                request_id,
                created_time,
                model_name,
            )

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def _validate_request(
            self, request: ClassificationRequest) -> Optional[ErrorResponse]:
        if (request.truncate_prompt_tokens is not None
                and request.truncate_prompt_tokens > self.max_model_len):
            return self.create_error_response(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please, select a smaller truncation size.")

        return None

    async def _prepare_classification_inputs(self,
                                             request: ClassificationRequest):
        """
        Process classification inputs: tokenize text, resolve adapters, 
        and prepare model-specific inputs.
        """
        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if prompt_adapter_request is not None:
                raise NotImplementedError(
                    "Prompt adapter is not supported for classification models"
                )

            request_prompts, engine_prompts = await self._preprocess_completion(
                request,
                tokenizer,
                request.input,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            return request_prompts, engine_prompts, \
                    lora_request, prompt_adapter_request

        except (ValueError, TypeError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    async def _get_classification_generators(
        self,
        request_id: str,
        request: ClassificationRequest,
        raw_request: Optional[Request],
        engine_prompts: list[TokensPrompt],
        request_prompts: list[TextTokensPrompt],
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
    ):
        """
        Create generators for classification by encoding inputs via the model.
        """
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        try:
            trace_headers = (None if raw_request is None else await
                             self._get_trace_headers(raw_request.headers))

            pooling_params = request.to_pooling_params()

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(
                    request_id_item,
                    request_prompts[i],
                    params=pooling_params,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                )

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)

            return generators

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def request_output_to_classify_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> ClassificationResponse:
        """
        Convert model outputs to a formatted classification response 
        with probabilities and labels.
        """
        items: list[ClassificationData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
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
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )
