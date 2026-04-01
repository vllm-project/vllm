# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.renderers import BaseRenderer
from vllm.v1.pool.late_interaction import (
    build_late_interaction_doc_params,
    build_late_interaction_query_params,
)

from .io_processor import ScoringIOProcessors, ScoringServeContext
from .protocol import (
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    ScoreRequest,
    ScoreResponse,
    ScoreResponseData,
    ScoringRequest,
)
from .typing import ScoreInput

logger = init_logger(__name__)


class ServingScores(PoolingServing):
    request_id_prefix = "score"

    def __init__(
        self,
        engine_client: EngineClient,
        *args,
        enable_flash_late_interaction: bool = True,
        **kwargs,
    ):
        self.score_type = engine_client.model_config.score_type
        self.enable_flash_late_interaction = (
            self.score_type == "late-interaction" and enable_flash_late_interaction
        )

        super().__init__(engine_client, *args, **kwargs)

    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> PoolingIOProcessor:
        score_type = model_config.score_type
        if self.enable_flash_late_interaction:
            score_type = "flash-late-interaction"

        assert score_type in ScoringIOProcessors
        processor_cls = ScoringIOProcessors[score_type]
        return processor_cls(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )

    async def __call__(
        self,
        request: ScoringRequest,
        raw_request: Request | None = None,
    ) -> Response:
        if not self.enable_flash_late_interaction:
            return await super().__call__(request, raw_request)

        return await self.flash_late_interaction(request, raw_request)

    async def _build_response(
        self,
        ctx: ScoringServeContext,
    ) -> JSONResponse:
        final_res_batch = ctx.final_res_batch
        request_id = ctx.request_id
        created_time = ctx.created_time
        model_name = self.models.model_name()

        if isinstance(ctx.request, ScoreRequest):
            return self._request_output_to_score_response(
                final_res_batch,
                request_id,
                created_time,
                model_name,
            )
        elif isinstance(ctx.request, RerankRequest):
            return self._request_output_to_rerank_response(
                final_res_batch,
                request_id,
                model_name,
                ctx.request.documents,
                ctx.request.top_n if ctx.request.top_n > 0 else len(final_res_batch),
            )
        else:
            raise NotImplementedError("")

    def _request_output_to_score_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> JSONResponse:
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

        response = ScoreResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )

        return JSONResponse(content=response.model_dump())

    def _request_output_to_rerank_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        model_name: str,
        documents: ScoreInput | list[ScoreInput],
        top_n: int,
    ) -> JSONResponse:
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

        response = RerankResponse(
            id=request_id,
            model=model_name,
            results=results,
            usage=RerankUsage(
                total_tokens=num_prompt_tokens, prompt_tokens=num_prompt_tokens
            ),
        )

        return JSONResponse(content=response.model_dump())

    ###################################################################################
    ### Run pooling score MaxSim on worker side (GPU) in the API server process
    ### Can significantly improve late-interaction scoring performance.

    async def flash_late_interaction(
        self,
        request: ScoringRequest,
        raw_request: Request | None = None,
    ) -> Response:
        model_name = self.models.model_name()
        request_id = f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"

        await self._check_model(request)

        ctx = ScoringServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            request_id=request_id,
            pooling_params=request.to_pooling_params("token_embed"),
        )

        self._validate_request(ctx)
        self._maybe_get_adapters(ctx)

        await self.io_processor.pre_process_online_async(ctx)

        # stage 1: encode queries and cache token embeddings on workers.
        await self._flash_late_interaction_encode_queries(ctx)
        # stage 2: encode docs and return scalar scores from workers.
        await self._flash_late_interaction_encode_docs(ctx)

        await self.io_processor.post_process_online_async(ctx)
        return await self._build_response(ctx)

    async def _flash_late_interaction_encode_queries(self, ctx: ScoringServeContext):
        assert isinstance(ctx.n_queries, int)

        n_queries = ctx.n_queries
        n_docs = len(ctx.engine_inputs) - n_queries
        query_engine_inputs = ctx.engine_inputs[:n_queries]

        query_keys = [f"{ctx.request_id}-query-{i}" for i in range(n_queries)]
        query_uses = [n_docs if n_queries == 1 else 1] * n_queries

        query_pooling_params_list = []
        for i in range(n_queries):
            pooling_params = ctx.pooling_params.clone()
            pooling_params.late_interaction_params = (
                build_late_interaction_query_params(
                    query_key=query_keys[i],
                    query_uses=query_uses[i],
                )
            )
            query_pooling_params_list.append(pooling_params)

        query_ctx = ScoringServeContext(
            request=ctx.request,
            raw_request=ctx.raw_request,
            model_name=ctx.model_name,
            request_id=ctx.request_id,
            pooling_params=query_pooling_params_list,
            prompt_request_ids=query_keys,
            engine_inputs=query_engine_inputs,
        )

        assert (
            len(query_ctx.pooling_params)
            == len(query_ctx.engine_inputs)
            == len(query_ctx.prompt_request_ids)
            == n_queries
        )

        await self._prepare_generators(query_ctx)
        await self._collect_batch(query_ctx)

    async def _flash_late_interaction_encode_docs(self, ctx: ScoringServeContext):
        assert isinstance(ctx.n_queries, int)

        n_queries = ctx.n_queries
        n_docs = len(ctx.engine_inputs) - n_queries
        doc_engine_inputs = ctx.engine_inputs[n_queries:]

        query_keys = [f"{ctx.request_id}-query-{i}" for i in range(n_queries)]
        doc_keys = [f"{ctx.request_id}-doc-{i}" for i in range(n_docs)]

        doc_pooling_params_list = []
        for i in range(n_docs):
            query_idx = 0 if n_queries == 1 else i
            pooling_params = ctx.pooling_params.clone()
            pooling_params.late_interaction_params = build_late_interaction_doc_params(
                query_key=query_keys[query_idx]
            )
            doc_pooling_params_list.append(pooling_params)

        doc_ctx = ScoringServeContext(
            request=ctx.request,
            raw_request=ctx.raw_request,
            model_name=ctx.model_name,
            request_id=ctx.request_id,
            pooling_params=doc_pooling_params_list,
            prompt_request_ids=doc_keys,
            engine_inputs=doc_engine_inputs,
        )

        assert (
            len(doc_ctx.pooling_params)
            == len(doc_ctx.engine_inputs)
            == len(doc_ctx.prompt_request_ids)
            == n_docs
        )

        await self._prepare_generators(doc_ctx)
        await self._collect_batch(doc_ctx)

        ctx.final_res_batch = doc_ctx.final_res_batch
