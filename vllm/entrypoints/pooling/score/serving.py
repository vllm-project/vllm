# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.async_utils import merge_async_iterators

from ...chat_utils import ChatTemplateContentFormatOption
from .io_processor import (
    CrossEncoderIOProcessor,
    EmbeddingScoreIOProcessor,
    LateInteractionIOProcessor,
)
from .protocol import (
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    ScoreRequest,
    ScoreResponse,
    ScoreResponseData,
)
from .utils import (
    ScoreData,
    ScoreInputs,
    _cosine_similarity,
    compress_token_type_ids,
    compute_maxsim_score,
    validate_score_input,
)

logger = init_logger(__name__)


ScoreType = Literal["cross_encoder", "late_interaction", "embedding"]

ScoreIOProcessors: dict[ScoreType, PoolingIOProcessor] = {
    "cross_encoder": CrossEncoderIOProcessor,
    "late_interaction": LateInteractionIOProcessor,
    "embedding": EmbeddingScoreIOProcessor,
}


class ServingScores(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        score_template: str | None = None,
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

        tokenizer = self.renderer.get_tokenizer()
        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError("MistralTokenizer not supported for cross-encoding")

        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)

        self.is_cross_encoder = self.model_config.is_cross_encoder
        self.architecture = self.model_config.architecture
        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.is_late_interaction = self.model_config.is_late_interaction

        self.score_type: ScoreType
        if self.is_cross_encoder:
            self.score_type = "cross_encoder"
        elif self.is_late_interaction:
            self.score_type = "late_interaction"
        else:
            self.score_type = "embedding"

        self.io_processor = ScoreIOProcessors[self.score_type](
            model_config=models.model_config,
            renderer=models.renderer,
            chat_template=score_template,
            chat_template_content_format=chat_template_content_format,
            trust_request_chat_template=trust_request_chat_template,
        )

        if self.is_cross_encoder:
            self._score_func = self._cross_encoding_score
        elif self.is_late_interaction:
            self._score_func = self._late_interaction_score
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
        tokenizer = self.renderer.get_tokenizer()
        input_texts, engine_prompts = await self.io_processor.pre_process(
            data_1=data_1,
            data_2=data_2,
            request=request,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        pooling_params = request.to_pooling_params("embed")

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

    async def _late_interaction_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        """
        Late interaction scoring (ColBERT MaxSim).

        Encodes queries and documents into per-token embeddings, then computes
        MaxSim: sum over query tokens of max similarity to any document token.
        """

        tokenizer = self.renderer.get_tokenizer()
        input_texts, engine_prompts = await self.io_processor.pre_process(
            data_1=data_1,
            data_2=data_2,
            request=request,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        pooling_params = request.to_pooling_params("token_embed")

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

        # Collect token embeddings
        embeddings: list[PoolingRequestOutput | None] = [None] * len(engine_prompts)

        async for i, res in result_generator:
            embeddings[i] = res

        # Split into query and document embeddings
        emb_data_1: list[PoolingRequestOutput] = []
        emb_data_2: list[PoolingRequestOutput] = []

        for i in range(0, len(data_1)):
            assert (emb := embeddings[i]) is not None
            emb_data_1.append(emb)

        for i in range(len(data_1), len(embeddings)):
            assert (emb := embeddings[i]) is not None
            emb_data_2.append(emb)

        # Expand queries if 1:N scoring
        if len(emb_data_1) == 1:
            emb_data_1 = emb_data_1 * len(emb_data_2)

        # Compute MaxSim scores
        from vllm.outputs import PoolingOutput

        scores: list[PoolingRequestOutput] = []
        padding: list[int] = []
        if (pad_token_id := tokenizer.pad_token_id) is not None:
            padding = [pad_token_id]

        for emb_1, emb_2 in zip(emb_data_1, emb_data_2):
            # emb_1.outputs.data: [query_len, dim]
            # emb_2.outputs.data: [doc_len, dim]
            q_emb = emb_1.outputs.data
            d_emb = emb_2.outputs.data

            maxsim_score = compute_maxsim_score(q_emb, d_emb)

            tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

            scores.append(
                PoolingRequestOutput(
                    request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                    outputs=PoolingOutput(data=maxsim_score),
                    prompt_token_ids=tokens,
                    num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,
                    finished=True,
                )
            )

        return scores

    async def _cross_encoding_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        lora_request: LoRARequest | None | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        default_pooling_params = request.to_pooling_params("score")

        engine_prompts, request_prompts = await self.io_processor.pre_process(
            data_1=data_1,
            data_2=data_2,
            request=request,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )

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
