# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

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


@router.get("/health/decode")
async def health_decode(raw_request: Request) -> JSONResponse:
    """Engine forward-progress liveness check.

    Unlike ``/health`` — which only asks "is the engine task alive?" —
    this endpoint asks "is the engine actually decoding?". It returns:

    * **200 ``{"status": "ok", ...}``** when the engine is making
      forward progress (a decoded token was observed within the stall
      threshold) OR has decoded at least one token in the past and
      currently has no Running requests (i.e. genuinely idle).
    * **200 ``{"status": "idle", "running": 0, ...}``** when no
      decoded token has ever been observed (cold start, never served a
      request) and no requests are Running. Idle is not stalled.
    * **503 ``{"status": "stalled", ...}``** when ``running > 0`` AND
      a decoded token has been observed in the past AND the time since
      the last decoded token exceeds the stall threshold (default 60s,
      override via ``VLLM_DECODE_LIVENESS_STALL_SECONDS``).

    Motivation: certain failure modes (notably NCCL P2P deadlocks
    surviving a container restart in TP>1 / PP>1 configurations,
    tracked in vllm-project/vllm#45094) leave the API server process
    alive and ``/health`` returning 200 indefinitely, even though
    generation throughput has dropped to 0 tok/s and every in-flight
    request will hang forever. Orchestrators (k8s probes, docker
    healthchecks, custom watchdogs) have no way to detect this from
    ``/health`` alone. This endpoint exposes the engine's per-step
    forward-progress timestamp so orchestrators can probe it and act
    (restart the container, page on-call, etc.) when generation has
    stalled.

    Backwards compat: this is a NEW, orthogonal endpoint. ``/health``
    semantics are unchanged. Consumers must opt in by probing
    ``/health/decode`` specifically.
    """
    threshold = envs.VLLM_DECODE_LIVENESS_STALL_SECONDS
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "running": 0,
                "last_token_age_seconds": None,
                "stall_threshold_seconds": threshold,
            },
        )

    try:
        running, last_token_age = await client.get_decode_liveness()
    except Exception as e:  # noqa: BLE001 — protocol allows any failure mode
        # If the engine can't even tell us its liveness state, treat
        # that as stalled. We do not raise — the endpoint exists to
        # give orchestrators a stable signal.
        logger.warning("get_decode_liveness() failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "stalled",
                "running": None,
                "last_token_age_seconds": None,
                "stall_threshold_seconds": threshold,
                "error": str(e),
            },
        )

    # Engine has never decoded a token. If no Running work, that's a
    # legitimate cold/idle state; otherwise we have nothing to compare
    # against, so call it idle too — first-token latency could legally
    # exceed the threshold for very long prefills.
    if last_token_age is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "idle",
                "running": running,
                "last_token_age_seconds": None,
                "stall_threshold_seconds": threshold,
            },
        )

    # Idle: nothing in flight, nothing to stall on.
    if running == 0:
        return JSONResponse(
            status_code=200,
            content={
                "status": "idle",
                "running": 0,
                "last_token_age_seconds": last_token_age,
                "stall_threshold_seconds": threshold,
            },
        )

    # Stalled: requests in flight, but no token in too long.
    if last_token_age > threshold:
        return JSONResponse(
            status_code=503,
            content={
                "status": "stalled",
                "running": running,
                "last_token_age_seconds": last_token_age,
                "stall_threshold_seconds": threshold,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "running": running,
            "last_token_age_seconds": last_token_age,
            "stall_threshold_seconds": threshold,
        },
    )
