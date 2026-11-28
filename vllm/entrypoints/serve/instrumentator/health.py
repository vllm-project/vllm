# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.plugins.endpoint_plugins.interface import endpoint_plugins_ready
from vllm.v1.engine.exceptions import EngineDeadError, EngineUnhealthyError

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


@router.get("/ready", response_class=Response)
async def ready(raw_request: Request) -> Response:
    """Process-local engine and plugin readiness, excluding draining servers."""
    state = raw_request.app.state
    if getattr(state, "draining", False):
        return Response(status_code=503)
    client = engine_client(raw_request)
    if client is not None:
        try:
            await client.check_health()
        except EngineDeadError:
            return Response(status_code=503)
        except Exception:
            # Draining may begin during the check.
            if getattr(state, "draining", False):
                return Response(status_code=503)
            raise
    if getattr(state, "draining", False) or not endpoint_plugins_ready(state):
        return Response(status_code=503)
    return Response(status_code=200)


@router.get(
    "/health_generate",
    response_class=Response,
    responses={
        503: {
            "description": (
                "Engine is not ready to serve generation requests, or the "
                "server is draining."
            ),
            "content": {"application/json": {"schema": {"type": "object"}}},
        }
    },
)
async def health_generate(
    raw_request: Request, timeout_s: Annotated[float, Query(gt=0)] = 20.0
) -> Response:
    """Generation readiness check: 503 when an engine rank cannot run the model
    or this API server is draining.

    `timeout_s` is how long to wait for each engine rank before reporting
    `probe_timeout`.
    """
    state = raw_request.app.state
    if getattr(state, "draining", False):
        return _draining_response()
    client = engine_client(raw_request)
    if client is None:
        return Response(status_code=200)
    content: dict[str, Any] | None = None
    try:
        await client.check_health_generate(timeout_s)
    except EngineDeadError:
        content = {"status": "not_ready", "reason": "engine_dead"}
    except EngineUnhealthyError as e:
        content = {
            "status": "not_ready",
            "reason": e.reason,
            "message": str(e),
            **e.details,
        }
    except Exception:
        # Draining may begin during the probe.
        if getattr(state, "draining", False):
            return _draining_response()
        raise
    if getattr(state, "draining", False):
        return _draining_response()
    if content is None:
        return Response(status_code=200)
    return JSONResponse(status_code=503, content=content)


def _draining_response() -> JSONResponse:
    return JSONResponse(
        status_code=503, content={"status": "not_ready", "reason": "draining"}
    )
