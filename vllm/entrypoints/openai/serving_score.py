# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Union

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
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               PreTrainedTokenizer,
                                               PreTrainedTokenizerFast)
from vllm.utils import make_async, merge_async_iterators

logger = init_logger(__name__)


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
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        text_1: List[str],
        text_2: List[str],
        request: ScoreRequest,
        request_id=str,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        lora_request: Optional[Union[LoRARequest, None]] = None,
        prompt_adapter_request: Optional[Union[PromptAdapterRequest,
                                               None]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> List[PoolingRequestOutput]:

        input_texts = text_1 + text_2

        engine_prompts: List[TokensPrompt] = []
        tokenize_async = make_async(tokenizer.__call__,
                                    executor=self._tokenizer_executor)

        tokenization_kwargs = tokenization_kwargs or {}
        tokenized_prompts = [
            tokenize_async(t, **tokenization_kwargs) for t in input_texts
        ]

        for tok_result, input_text in zip(
                await asyncio.gather(*tokenized_prompts), input_texts):

            text_token_prompt = \
                self._validate_input(
                    request,
                    tok_result["input_ids"],
                    input_text)

            engine_prompts.append(
                TokensPrompt(
                    prompt_token_ids=text_token_prompt["prompt_token_ids"]))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[PoolingRequestOutput, None]] = []
        pooling_params = request.to_pooling_params()

        for i, engine_prompt in enumerate(engine_prompts):

            request_id_item = f"{request_id}-{i}"

            self._log_inputs(request_id_item,
                             input_texts[i],
                             params=pooling_params,
                             lora_request=lora_request,
                             prompt_adapter_request=prompt_adapter_request)

            generators.append(
                self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                ))

        result_generator = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: List[PoolingRequestOutput] = []

        embeddings: List[Optional[PoolingRequestOutput]] =\
              [None] * len(engine_prompts)

        async for i, res in result_generator:
            embeddings[i] = res

        emb_text_1 = embeddings[:len(text_1)]
        emb_text_2 = embeddings[len(text_1):]

        if len(emb_text_1) == 1:
            emb_text_1 = emb_text_1 * len(emb_text_2)

        scorer = torch.nn.CosineSimilarity(0)

        for emb_1, emb_2 in zip(emb_text_1, emb_text_2):
            assert emb_1 is not None
            assert emb_2 is not None
            pair_score = scorer(emb_1.outputs.data, emb_2.outputs.data)

            padding = []
            if (pad_token_id := getattr(tokenizer, "pad_token_id",
                                        None)) is not None:
                padding = [pad_token_id]

            tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

            final_res_batch.append(
                PoolingRequestOutput(
                    request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                    outputs=pair_score,
                    prompt_token_ids=tokens,
                    finished=True))

        return final_res_batch

    async def _cross_encoding_score(
        self,
        tokenizer: Union[AnyTokenizer],
        text_1: List[str],
        text_2: List[str],
        request: ScoreRequest,
        request_id=str,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        lora_request: Optional[Union[LoRARequest, None]] = None,
        prompt_adapter_request: Optional[Union[PromptAdapterRequest,
                                               None]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> List[PoolingRequestOutput]:

        request_prompts: List[str] = []
        engine_prompts: List[TokensPrompt] = []

        if len(text_1) == 1:
            text_1 = text_1 * len(text_2)

        input_pairs = [(t1, t2) for t1, t2 in zip(text_1, text_2)]

        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError(
                "MistralTokenizer not supported for cross-encoding")

        tokenize_async = make_async(tokenizer.__call__,
                                    executor=self._tokenizer_executor)

        tokenization_kwargs = tokenization_kwargs or {}
        tokenized_prompts = [
            tokenize_async(text=t1, text_pair=t2, **tokenization_kwargs)
            for t1, t2 in input_pairs
        ]

        for prompt_inputs, (t1, t2) in zip(
                await asyncio.gather(*tokenized_prompts), input_pairs):

            request_prompt = f"{t1}{tokenizer.sep_token}{t2}"

            input_ids = prompt_inputs["input_ids"]
            text_token_prompt = \
                self._validate_input(request, input_ids, request_prompt)
            engine_prompt = TokensPrompt(
                prompt_token_ids=text_token_prompt["prompt_token_ids"],
                token_type_ids=prompt_inputs.get("token_type_ids"))

            request_prompts.append(request_prompt)
            engine_prompts.append(engine_prompt)

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[PoolingRequestOutput, None]] = []

        pooling_params = request.to_pooling_params()

        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}-{i}"

            self._log_inputs(request_id_item,
                             request_prompts[i],
                             params=pooling_params,
                             lora_request=lora_request,
                             prompt_adapter_request=prompt_adapter_request)

            generator = self.engine_client.encode(
                engine_prompt,
                pooling_params,
                request_id_item,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
            )

            generators.append(generator)

        result_generator = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: List[
            Optional[PoolingRequestOutput]] = [None] * len(engine_prompts)

        async for i, res in result_generator:
            final_res_batch[i] = res

        return [out for out in final_res_batch if out is not None]

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

        tokenization_kwargs: Dict[str, Any] = {}
        if truncate_prompt_tokens is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

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

        trace_headers = (None if raw_request is None else await
                         self._get_trace_headers(raw_request.headers))

        text_1 = request.text_1
        text_2 = request.text_2

        if isinstance(text_1, str):
            text_1 = [text_1]
        if isinstance(text_2, str):
            text_2 = [text_2]
        if len(text_1) > 1 and len(text_1) != len(text_2):
            raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
        if len(text_1) == 0:
            raise ValueError("At least one text element must be given")
        if len(text_2) == 0:
            raise ValueError("At least one text_pair element must be given")

        try:
            if self.model_config.is_cross_encoder:
                final_res_batch = await self._cross_encoding_score(
                    tokenizer=tokenizer,
                    text_1=text_1,
                    text_2=text_2,
                    request=request,
                    request_id=request_id,
                    tokenization_kwargs=tokenization_kwargs,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    trace_headers=trace_headers)

            else:
                final_res_batch = await self._embedding_score(
                    tokenizer=tokenizer,
                    text_1=text_1,
                    text_2=text_2,
                    request=request,
                    request_id=request_id,
                    tokenization_kwargs=tokenization_kwargs,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    trace_headers=trace_headers)

            response = self.request_output_to_score_response(
                final_res_batch,
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
