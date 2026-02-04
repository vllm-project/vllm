# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.score.protocol import (
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    ScoreRequest,
    ScoreResponse,
    ScoreResponseData,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreData,
    ScoreInputs,
    _cosine_similarity,
    compress_token_type_ids,
    get_score_prompt,
    validate_score_input,
)
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.async_utils import make_async, merge_async_iterators

logger = init_logger(__name__)


class ServingScores(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        score_template: str | None = None,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )
        self.score_template = score_template

        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)

        self.is_cross_encoder = self.model_config.is_cross_encoder
        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.architecture = self.model_config.architecture

        if self.is_cross_encoder:
            self._score_func = self._cross_encoding_score
        else:
            self._score_func = self._embedding_score

    async def _embedding_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        lora_request: LoRARequest | None | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        input_texts: list[str] = []
        for text in data_1 + data_2:
            if not isinstance(text, str):
                raise NotImplementedError(
                    "Embedding scores currently do not support multimodal input."
                )
            input_texts.append(text)

        model_config = self.model_config
        tokenizer = self.renderer.get_tokenizer()

        encode_async = make_async(
            tokenizer.encode,
            executor=self._tokenizer_executor,
        )

        tokenization_kwargs = request.build_tok_params(model_config).get_encode_kwargs()
        tokenized_prompts = await asyncio.gather(
            *(encode_async(t, **tokenization_kwargs) for t in input_texts)
        )

        engine_prompts: list[TokensPrompt] = []
        for tok_result, input_text in zip(tokenized_prompts, input_texts):
            text_token_prompt = self._validate_input(request, tok_result, input_text)

            engine_prompts.append(
                TokensPrompt(prompt_token_ids=text_token_prompt["prompt_token_ids"])
            )

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        pooling_params = request.to_pooling_params()

        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                input_texts[i],
                params=pooling_params,
                lora_request=lora_request,
            )

            generators.append(
                self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )
            )

        result_generator = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: list[PoolingRequestOutput] = []

        embeddings: list[PoolingRequestOutput | None] = [None] * len(engine_prompts)

        async for i, res in result_generator:
            embeddings[i] = res

        emb_data_1: list[PoolingRequestOutput] = []
        emb_data_2: list[PoolingRequestOutput] = []

        for i in range(0, len(data_1)):
            assert (emb := embeddings[i]) is not None
            emb_data_1.append(emb)

        for i in range(len(data_1), len(embeddings)):
            assert (emb := embeddings[i]) is not None
            emb_data_2.append(emb)

        if len(emb_data_1) == 1:
            emb_data_1 = emb_data_1 * len(emb_data_2)

        final_res_batch = _cosine_similarity(
            tokenizer=tokenizer, embed_1=emb_data_1, embed_2=emb_data_2
        )

        return final_res_batch

    async def _cross_encoding_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        lora_request: LoRARequest | None | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        tokenizer = self.renderer.get_tokenizer()
        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError("MistralTokenizer not supported for cross-encoding")

        model_config = self.model_config

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        tok_kwargs = request.build_tok_params(model_config).get_encode_kwargs()
        input_pairs = [(t1, t2) for t1, t2 in zip(data_1, data_2)]
        preprocess_async = make_async(
            self._preprocess_score,
            executor=self._tokenizer_executor,
        )
        preprocessed_prompts = await asyncio.gather(
            *(
                preprocess_async(
                    request=request,
                    tokenizer=tokenizer,
                    tokenization_kwargs=tok_kwargs,
                    data_1=t1,
                    data_2=t2,
                )
                for t1, t2 in input_pairs
            )
        )

        request_prompts: list[str] = []
        engine_prompts: list[TokensPrompt] = []
        for full_prompt, engine_prompt in preprocessed_prompts:
            request_prompts.append(full_prompt)
            engine_prompts.append(engine_prompt)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        default_pooling_params = request.to_pooling_params()

        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                request_prompts[i],
                params=default_pooling_params,
                lora_request=lora_request,
            )

            if token_type_ids := engine_prompt.pop("token_type_ids", None):
                pooling_params = default_pooling_params.clone()
                compressed = compress_token_type_ids(token_type_ids)
                pooling_params.extra_kwargs = {"compressed_token_type_ids": compressed}
            else:
                pooling_params = default_pooling_params

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
        final_res_batch: list[PoolingRequestOutput | None] = [None] * len(
            engine_prompts
        )

        async for i, res in result_generator:
            final_res_batch[i] = res

        return [out for out in final_res_batch if out is not None]

    def _preprocess_score(
        self,
        request: RerankRequest | ScoreRequest,
        tokenizer: TokenizerLike,
        tokenization_kwargs: dict[str, Any],
        data_1: ScoreData,
        data_2: ScoreData,
    ) -> tuple[str, TokensPrompt]:
        model_config = self.model_config
        full_prompt, engine_prompt = get_score_prompt(
            model_config=model_config,
            data_1=data_1,
            data_2=data_2,
            tokenizer=tokenizer,
            tokenization_kwargs=tokenization_kwargs,
            score_template=self.score_template,
        )
        self._validate_input(request, engine_prompt["prompt_token_ids"], full_prompt)
        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        return full_prompt, engine_prompt

    async def _run_scoring(
        self,
        data_1: ScoreInputs,
        data_2: ScoreInputs,
        request: ScoreRequest | RerankRequest,
        request_id: str,
        raw_request: Request | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        lora_request = self._maybe_get_adapters(request)

        trace_headers = (
            None
            if raw_request is None
            else await self._get_trace_headers(raw_request.headers)
        )

        score_data_1, score_data_2 = validate_score_input(
            data_1,
            data_2,
            is_multimodal_model=self.is_multimodal_model,
            architecture=self.architecture,
        )

        return await self._score_func(
            data_1=score_data_1,
            data_2=score_data_2,
            request=request,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )

    async def create_score(
        self,
        request: ScoreRequest,
        raw_request: Request | None = None,
    ) -> ScoreResponse | ErrorResponse:
        """
        Score API similar to Sentence Transformers cross encoder

        See https://sbert.net/docs/package_reference/cross_encoder
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"score-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        try:
            final_res_batch = await self._run_scoring(
                request.data_1,
                request.data_2,
                request,
                request_id,
                raw_request,
            )
            if isinstance(final_res_batch, ErrorResponse):
                return final_res_batch

            return self.request_output_to_score_response(
                final_res_batch,
                request_id,
                created_time,
                self.models.model_name(),
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)

    async def do_rerank(
        self, request: RerankRequest, raw_request: Request | None = None
    ) -> RerankResponse | ErrorResponse:
        """
        Rerank API based on JinaAI's rerank API; implements the same
        API interface. Designed for compatibility with off-the-shelf
        tooling, since this is a common standard for reranking APIs

        See example client implementations at
        https://github.com/infiniflow/ragflow/blob/main/rag/llm/rerank_model.py
        numerous clients use this standard.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"rerank-{self._base_request_id(raw_request)}"
        documents = request.documents

        try:
            final_res_batch = await self._run_scoring(
                request.query,
                documents,
                request,
                request_id,
                raw_request,
            )
            if isinstance(final_res_batch, ErrorResponse):
                return final_res_batch

            top_n = request.top_n if request.top_n > 0 else len(final_res_batch)

            return self.request_output_to_rerank_response(
                final_res_batch,
                request_id,
                self.models.model_name(),
                documents,
                top_n,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)

    def request_output_to_score_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> ScoreResponse:
        items: list[ScoreResponseData] = []
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

    def request_output_to_rerank_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        model_name: str,
        documents: ScoreInputs,
        top_n: int,
    ) -> RerankResponse:
        """
        Convert the output of do_rank to a RerankResponse
        """

        if not isinstance(documents, list):
            documents = [documents]

        results: list[RerankResult] = []
        num_prompt_tokens = 0
        for idx, final_res in enumerate(final_res_batch):
            classify_res = ScoringRequestOutput.from_base(final_res)

            document = documents[idx]
            if isinstance(document, str):
                rerank_document = RerankDocument(text=document)
            else:
                rerank_document = RerankDocument(
                    multi_modal=document.get("content", [])
                )

            result = RerankResult(
                index=idx,
                document=rerank_document,
                relevance_score=classify_res.outputs.score,
            )
            results.append(result)
            prompt_token_ids = final_res.prompt_token_ids
            num_prompt_tokens += len(prompt_token_ids)

        # sort by relevance, then return the top n if set
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if top_n < len(documents):
            results = results[:top_n]

        return RerankResponse(
            id=request_id,
            model=model_name,
            results=results,
            usage=RerankUsage(
                total_tokens=num_prompt_tokens, prompt_tokens=num_prompt_tokens
            ),
        )
