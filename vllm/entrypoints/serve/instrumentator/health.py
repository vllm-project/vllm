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
    this endpoint asks "is the engine actually making forward progress?".

    It uses three API-process-local signals (see
    :meth:`AsyncLLM.get_decode_liveness` / ``DecodeLivenessTracker``), all of
    which advance WITHOUT EngineCore cooperation:

    * ``inflight`` — requests admitted to the API process that have not
      finished. Counted at admission (before the request is even sent to
      EngineCore) and at finish, so it is correct even when EngineCore is
      wedged.
    * ``last_progress_age`` — seconds since the engine last produced ANY
      output (a pure prefill chunk counts). ``None`` if it never has.
    * ``oldest_unprogressed_admission_age`` — seconds since the oldest
      in-flight request that has received ZERO outputs was admitted.
      ``None`` if every in-flight request has produced at least one output.

    Responses:

    * **200 ``{"status": "ok", ...}``** — work is in flight and the engine
      is producing output within the stall threshold, OR everything has
      progressed and we are simply waiting (within threshold).
    * **200 ``{"status": "idle", "inflight": 0, ...}``** — nothing in
      flight. Idle is never stalled. Covers cold start (nothing ever
      admitted) and a drained engine.
    * **503 ``{"status": "stalled", ...}``** — ``inflight > 0`` AND EITHER
      (a) the engine produced output before but nothing for longer than the
      stall threshold (mid-stream stall), OR (b) the oldest in-flight request
      has received ZERO outputs since admission for longer than the stall
      threshold (step-0 / never-progressed stall). Rule (b) is what catches
      a decode-step-0 deadlock, where ``last_progress_age`` is ``None``
      because the engine never emitted anything at all — the failure mode in
      vllm-project/vllm#45094, where the API server stays alive and
      ``/health`` returns 200 indefinitely.

    Stall threshold defaults to 60s, override via
    ``VLLM_DECODE_LIVENESS_STALL_SECONDS``.

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
                "inflight": 0,
                "last_progress_age_seconds": None,
                "oldest_unprogressed_admission_age_seconds": None,
                "stall_threshold_seconds": threshold,
            },
        )

    try:
        (
            inflight,
            last_progress_age,
            oldest_unprogressed_age,
        ) = await client.get_decode_liveness()
    except Exception as e:  # noqa: BLE001 — protocol allows any failure mode
        # If the engine can't even tell us its liveness state, treat
        # that as stalled. We do not raise — the endpoint exists to
        # give orchestrators a stable signal.
        logger.warning("get_decode_liveness() failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "stalled",
                "inflight": None,
                "last_progress_age_seconds": None,
                "oldest_unprogressed_admission_age_seconds": None,
                "stall_threshold_seconds": threshold,
                "error": str(e),
            },
        )

    body = {
        "inflight": inflight,
        "last_progress_age_seconds": last_progress_age,
        "oldest_unprogressed_admission_age_seconds": oldest_unprogressed_age,
        "stall_threshold_seconds": threshold,
    }

    # Idle: nothing in flight, nothing to stall on. Covers cold start (nothing
    # ever admitted) and a fully drained engine.
    if inflight <= 0:
        return JSONResponse(status_code=200, content={"status": "idle", **body})

    # Stalled rule (a) — mid-stream: the engine produced output in the past
    # but nothing for longer than the threshold.
    stalled_midstream = (
        last_progress_age is not None and last_progress_age > threshold
    )
    # Stalled rule (b) — step 0 / never-progressed: an in-flight request has
    # received ZERO outputs since admission for longer than the threshold.
    # This is the case last_progress_age cannot see (it may be None or recent
    # due to OTHER requests) — it is what detects the decode-step-0 deadlock.
    stalled_step0 = (
        oldest_unprogressed_age is not None and oldest_unprogressed_age > threshold
    )
    if stalled_midstream or stalled_step0:
        return JSONResponse(status_code=503, content={"status": "stalled", **body})

    # In flight and progressing (or within threshold of admission).
    return JSONResponse(status_code=200, content={"status": "ok", **body})
