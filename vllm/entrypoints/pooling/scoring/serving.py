# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias

from fastapi.responses import JSONResponse

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.renderers import BaseRenderer

from .io_processor import BiEncoderIOProcessor
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
from .typing import ScoreInputs

logger = init_logger(__name__)

ScoreServeContext: TypeAlias = PoolingServeContext[ScoreRequest | RerankRequest]


class ServingScores(PoolingServing):
    request_id_prefix = "score"

    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> BiEncoderIOProcessor:
        return BiEncoderIOProcessor(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )

    async def _build_response(
        self,
        ctx: ScoreServeContext,
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
        documents: ScoreInputs,
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
