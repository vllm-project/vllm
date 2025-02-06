# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

import torch
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse, ScoreRequest,
                                              ScoreResponse, ScoreResponseData,
                                              UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import make_async, merge_async_iterators

logger = init_logger(__name__)


def make_pairs(text_1: Union[List[str], str], text_2: Union[List[str],
                                                            str]) -> List:
    if isinstance(text_1, (str, dict)):
        # Convert a single prompt to a list.
        text_1 = [text_1]
    text_1 = [t for t in text_1]

    if isinstance(text_2, (str, dict)):
        # Convert a single prompt to a list.
        text_2 = [text_2]
    text_2 = [t for t in text_2]
    if len(text_1) > 1 and len(text_1) != len(text_2):
        raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
    if len(text_1) == 0:
        raise ValueError("At least one text element must be given")
    if len(text_2) == 0:
        raise ValueError("At least one text_pair element must be given")

    if len(text_1) == 1:
        text_1 = text_1 * len(text_2)

    return [(t1, t2) for t1, t2 in zip(text_1, text_2)]


class OpenAIServingScores(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

    async def _embedding_score(
        self,
        tokenizer: Union[AnyTokenizer],
        text_1: List[Union[List[str], str]],
        text_2: List[Union[List[str], str]],
        request: ScoreRequest,
        model_name=str,
        request_id=str,
        created_time=int,
        truncate_prompt_tokens: Optional[int] = None,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest | None] = None,
        raw_request: Optional[Request] = None,
    ) -> Union[ScoreResponse, ErrorResponse]:

        request_prompts = []
        engine_prompts = []

        try:
            input_pairs = make_pairs(text_1, text_2)
            for q, t in input_pairs:
                request_prompt = f"{q}{tokenizer.sep_token}{t}"

                tokenization_kwargs: Dict[str, Any] = {}
                if truncate_prompt_tokens is not None:
                    tokenization_kwargs["truncation"] = True
                    tokenization_kwargs["max_length"] = truncate_prompt_tokens

                tokenize_async = make_async(tokenizer.__call__,
                                            executor=self._tokenizer_executor)

                #first of the pair
                prompt_inputs_q = await tokenize_async(text=q,
                                                       **tokenization_kwargs)

                input_ids_q = prompt_inputs_q["input_ids"]

                text_token_prompt_q = \
                    self._validate_input(request, input_ids_q, q)

                engine_prompt_q = TokensPrompt(
                    prompt_token_ids=text_token_prompt_q["prompt_token_ids"],
                    token_type_ids=prompt_inputs_q.get("token_type_ids"))

                #second of the pair
                prompt_inputs_t = await tokenize_async(text=t,
                                                       **tokenization_kwargs)
                input_ids_t = prompt_inputs_t["input_ids"]

                text_token_prompt_t = \
                    self._validate_input(request, input_ids_t, t)

                engine_prompt_t = TokensPrompt(
                    prompt_token_ids=text_token_prompt_t["prompt_token_ids"],
                    token_type_ids=prompt_inputs_t.get("token_type_ids"))

                request_prompts.append(request_prompt)
                engine_prompts.append((engine_prompt_q, engine_prompt_t))

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[PoolingRequestOutput, None]] = []

        try:
            pooling_params = request.to_pooling_params()

            for i, engine_prompt in enumerate(engine_prompts):
                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                request_id_item_0 = f"{request_id}-{i}"

                self._log_inputs(request_id_item_0,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                generator_0 = self.engine_client.encode(
                    engine_prompt[0],
                    pooling_params,
                    request_id_item_0,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                request_id_item_1 = f"{request_id}-{i}.1"

                self._log_inputs(request_id_item_1,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                generator_1 = self.engine_client.encode(
                    engine_prompt[1],
                    pooling_params,
                    request_id_item_1,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator_0)
                generators.append(generator_1)

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: List[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts

        try:
            embeddings = []
            async for i, res in result_generator:
                embeddings.append(res)

            scores = []
            scorer = torch.nn.CosineSimilarity(0)

            for i in range(0, len(embeddings), 2):
                pair_score = scorer(embeddings[i].outputs.data,
                                    embeddings[i + 1].outputs.data)

                if (pad_token_id := getattr(tokenizer, "pad_token_id",
                                            None)) is not None:
                    tokens = embeddings[i].prompt_token_ids + [
                        pad_token_id
                    ] + embeddings[i + 1].prompt_token_ids
                else:
                    tokens = embeddings[i].prompt_token_ids + embeddings[
                        i + 1].prompt_token_ids

                scores.append(
                    PoolingRequestOutput(
                        request_id=
                        f"{embeddings[i].request_id}_{embeddings[i+1].request_id}",
                        outputs=pair_score,
                        prompt_token_ids=tokens,
                        finished=True))

            final_res_batch = scores
            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(List[PoolingRequestOutput],
                                           final_res_batch)

            response = self.request_output_to_score_response(
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

        return response

    async def _cross_encoding_score(
        self,
        tokenizer: Union[AnyTokenizer],
        text_1: List[Union[List[str], str]],
        text_2: List[Union[List[str], str]],
        request: ScoreRequest,
        model_name=str,
        request_id=str,
        created_time=int,
        truncate_prompt_tokens: Optional[int] = None,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest | None] = None,
        raw_request: Optional[Request] = None,
    ) -> Union[ScoreResponse, ErrorResponse]:

        request_prompts = []
        engine_prompts = []

        try:
            if isinstance(tokenizer, MistralTokenizer):
                raise ValueError(
                    "MistralTokenizer not supported for cross-encoding")

            input_pairs = make_pairs(text_1, text_2)
            for q, t in input_pairs:
                request_prompt = f"{q}{tokenizer.sep_token}{t}"

                tokenization_kwargs: Dict[str, Any] = {}
                if truncate_prompt_tokens is not None:
                    tokenization_kwargs["truncation"] = True
                    tokenization_kwargs["max_length"] = truncate_prompt_tokens

                tokenize_async = make_async(tokenizer.__call__,
                                            executor=self._tokenizer_executor)
                prompt_inputs = await tokenize_async(text=q,
                                                     text_pair=t,
                                                     **tokenization_kwargs)

                input_ids = prompt_inputs["input_ids"]
                text_token_prompt = \
                    self._validate_input(request, input_ids, request_prompt)
                engine_prompt = TokensPrompt(
                    prompt_token_ids=text_token_prompt["prompt_token_ids"],
                    token_type_ids=prompt_inputs.get("token_type_ids"))

                request_prompts.append(request_prompt)
                engine_prompts.append(engine_prompt)

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[PoolingRequestOutput, None]] = []

        try:
            pooling_params = request.to_pooling_params()

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: List[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts

        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(List[PoolingRequestOutput],
                                           final_res_batch)

            response = self.request_output_to_score_response(
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

        return response

    async def create_score(
        self,
        request: ScoreRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ScoreResponse, ErrorResponse]:
        """
        Score API similar to Sentence Transformers cross encoder

        See https://sbert.net/docs/package_reference/cross_encoder
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        model_name = request.model
        request_id = f"score-{self._base_request_id(raw_request)}"
        created_time = int(time.time())
        truncate_prompt_tokens = request.truncate_prompt_tokens

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for scoring models")

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if truncate_prompt_tokens is not None and \
                    truncate_prompt_tokens > self.max_model_len:
                raise ValueError(
                    f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
                    f"is greater than max_model_len ({self.max_model_len})."
                    f" Please, select a smaller truncation size.")

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        if self.model_config.is_cross_encoder:
            response = await self._cross_encoding_score(
                tokenizer=tokenizer,
                text_1=request.text_1,
                text_2=request.text_2,
                request=request,
                model_name=model_name,
                request_id=request_id,
                created_time=created_time,
                truncate_prompt_tokens=truncate_prompt_tokens,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                raw_request=raw_request)

        else:
            response = await self._embedding_score(
                tokenizer=tokenizer,
                text_1=request.text_1,
                text_2=request.text_2,
                request=request,
                model_name=model_name,
                request_id=request_id,
                created_time=created_time,
                truncate_prompt_tokens=truncate_prompt_tokens,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                raw_request=raw_request)

        return response

    def request_output_to_score_response(
        self,
        final_res_batch: List[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> ScoreResponse:
        items: List[ScoreResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            classify_res = ScoringRequestOutput.from_base(final_res)

            item = ScoreResponseData(
                index=idx,
                score=classify_res.outputs.score,
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return ScoreResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )
