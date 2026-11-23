# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, NonNegativeInt, ValidationError

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter()

logger = init_logger(__name__)


def base(request: Request) -> ServingTokenization:
    # Reuse the existing instance
    return tokenization(request)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


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


@router.get("/kv_event_sources")
async def kv_event_sources(raw_request: Request):
    """KV-event publisher config per DP rank; empty on render-only servers."""
    client: EngineClient | None = raw_request.app.state.engine_client
    return client.get_kv_event_sources() if client is not None else {}


class ForecastRequest(BaseModel):
    """Body of `POST /forecast`.

    Every cached length costs one replay of the scheduler state on the
    engine's input thread, so both lists are bounded.
    """

    cached: list[NonNegativeInt] = Field(default_factory=list, max_length=8)
    queries: list[tuple[NonNegativeInt, NonNegativeInt]] = Field(
        default_factory=list, max_length=256
    )


async def _ttft_forecast(request: Request, body: ForecastRequest) -> JSONResponse:
    forecast = getattr(engine_client(request), "forecast_ttft", None)
    if forecast is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="This engine client has no TTFT forecast.",
        )
    return JSONResponse(content=await forecast(body.cached, body.queries or None))


@router.get("/forecast")
async def get_ttft_forecast(request: Request, cached: str = "0"):
    """Seconds to first token for a request arriving now, by new tokens.

    One curve per cached-prefix length in `cached` (comma separated). Each
    curve is a list of segments `[u0, t0, u1, t1]`: a request needing `u` new
    tokens, `u0 < u <= u1`, is forecast at `t0 + (t1 - t0) * (u - u0) / (u1 - u0)`.
    """
    try:
        body = ForecastRequest(cached=[c for c in cached.split(",") if c])
    except ValidationError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=e.errors(include_url=False)
        ) from e
    return await _ttft_forecast(request, body)


@router.post("/forecast")
async def post_ttft_forecast(request: Request, body: ForecastRequest):
    """Forecast for explicit `queries`: a list of [new tokens, cached tokens]."""
    return await _ttft_forecast(request, body)
