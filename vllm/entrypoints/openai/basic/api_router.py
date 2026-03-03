# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter()

logger = init_logger(__name__)


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/responses
    # - /v1/responses/{response_id}
    # - /v1/responses/{response_id}/cancel
    # - /v1/messages
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/audio/translations
    # - /v1/embeddings
    # - /pooling
    # - /classify
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(content={"server_load": request.app.state.server_load_metrics})


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


def register_basic_api_routers(app: FastAPI):
    app.include_router(router)
