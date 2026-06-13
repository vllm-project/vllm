# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import math

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

# Bound on how long an incoming /sleep or /wake_up will wait for an
# in-flight transition on the same engine to complete. If we can't
# acquire the lock within this window, the request returns 409 so the
# caller can decide whether to retry, give up, or surface the conflict.
# Real cumem_tag sleep transitions complete in well under 1s; 5s is
# generous headroom for level=2 dump + reload paths.
_TRANSITION_LOCK_TIMEOUT_S = 5.0

# Single in-process mutex that serializes the read-decide-act sequence
# across all /sleep and /wake_up handlers on this engine. Without this,
# two concurrent /sleep calls can both observe is_sleeping=False and
# both proceed to client.sleep() (TOCTOU); a /sleep racing an in-flight
# /wake_up can land on a half-awake engine and corrupt its state.
#
# The lock is stored on `app.state` (rather than a module global) so
# (a) a fresh test app gets a fresh lock and (b) the lock binds to the
# correct asyncio event loop on first use — FastAPI TestClient runs
# each request in a new loop and a module-level asyncio.Lock created
# at import time would attach to the wrong loop the first time it's
# awaited.
#
# NOTE (multi-worker scope): This lock is per-worker (per-process). If
# the dev API is run with --workers N (multi-process), cross-worker
# concurrent /sleep calls can still race at the engine layer. The dev
# API is single-process by design; production use through the standard
# vLLM server proxies through core_client IPC which serializes natively
# at the engine-core boundary. Documented here so a future operator
# who passes --workers N to the dev API doesn't assume safety this
# lock cannot provide. (If runtime detection of multi-worker mode is
# added in future — e.g. via a uvicorn lifespan hook setting
# `app.state.workers > 1` — emit a warning here.)
_LOCK_ATTR = "_dev_sleep_transition_lock"

# Hint to clients on how long to back off before retrying after a 409.
# Derived from the lock-acquisition timeout so a future change to the
# lock window cascades automatically — the invariant we need is that
# `Retry-After` is at least as long as the lock window, otherwise a
# polite client will retry while the lock is still held by the prior
# call. `math.ceil` makes the integer second-count safe for non-integer
# lock-timeout values (e.g. 5.0 -> 5, 4.1 -> 5).
#
# `max(1, ...)` floors the value at 1 second. This protects against a
# misconfiguration of `_TRANSITION_LOCK_TIMEOUT_S` to 0, a negative value,
# or a sub-second value: any of those would yield `Retry-After: 0` (or a
# negative integer that violates RFC 7231's delta-seconds production),
# which would invite a busy-retry storm from a polite client. The floor is
# defense-in-depth — current code uses 5.0 — but cheap insurance against
# someone tuning the timeout in the future without thinking about the
# downstream Retry-After contract.
_RETRY_AFTER_S = max(1, int(math.ceil(_TRANSITION_LOCK_TIMEOUT_S)))


def _get_transition_lock(request: Request) -> asyncio.Lock:
    state = request.app.state
    lock = getattr(state, _LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(state, _LOCK_ATTR, lock)
    return lock


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


def _sleeping_state_label(
    level: int | None, *, partial_wake: bool = False
) -> str:
    """Render the engine's sleep depth as a stable string for the JSON
    response body. Callers (orchestrators, dashboards) can match on
    this without re-deriving the level themselves.

    `partial_wake=True` is set after a tagged /wake_up call when the
    engine still reports is_sleeping=True (some tags came back, others
    remain offloaded). Without this signal, callers see `sleeping_l1`
    after a `?tags=weights` wake and can't tell that weights are in
    fact resident — they think the wake was a no-op. The combined
    label `sleeping_l1_partial_wake` is intentionally a strict suffix
    of `sleeping_l1` so existing prefix-matching dashboards continue
    to classify it correctly.
    """
    if level is None:
        return "awake"
    base = "sleeping_l0" if level == 0 else f"sleeping_l{level}"
    if partial_wake:
        return f"{base}_partial_wake"
    return base


async def _current_sleep_level(client: EngineClient) -> int | None:
    """Look up the current depth via the engine. Falls back to the
    is_sleeping() bool for engine implementations that haven't been
    upgraded to expose a level (treating any sleeping state as level 0
    is conservative — if a caller asks for level 1+ on such an engine
    we will (correctly) escalate rather than skip).
    """
    return await client.get_sleep_level()


@router.post("/sleep")
async def sleep(raw_request: Request):
    level = int(raw_request.query_params.get("level", "1"))
    mode = raw_request.query_params.get("mode", "abort")
    client = engine_client(raw_request)

    # Serialize against any other in-flight /sleep or /wake_up on this
    # engine. Without this, two concurrent callers can both observe
    # "awake" and both call client.sleep(); or one /sleep can land
    # mid-/wake_up and arrive at an inconsistent depth.
    lock = _get_transition_lock(raw_request)
    try:
        await asyncio.wait_for(
            lock.acquire(), timeout=_TRANSITION_LOCK_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        logger.warning(
            "/sleep could not acquire transition lock within %.1fs; "
            "another sleep/wake transition is still in flight.",
            _TRANSITION_LOCK_TIMEOUT_S,
        )
        return JSONResponse(
            status_code=409,
            headers={"Retry-After": str(_RETRY_AFTER_S)},
            content={
                "error": "transition_in_progress",
                "detail": (
                    "Another /sleep or /wake_up is currently transitioning "
                    "this engine. Retry after it completes."
                ),
            },
        )

    try:
        # Idempotency-with-escalation: only short-circuit if we're
        # already at *or below* the requested depth (i.e. as deep or
        # deeper). A pre-fix "is_sleeping" check would silently swallow
        # a level-1 escalation request when the engine was only
        # level-0 paused, leaving weights resident on GPU.
        current_level = await _current_sleep_level(client)
        if current_level is not None and current_level >= level:
            logger.warning(
                "/sleep(level=%d) called on engine already at level=%d; "
                "treating as no-op.",
                level,
                current_level,
            )
            return JSONResponse(
                status_code=200,
                content={
                    "already_sleeping": True,
                    "current_state": _sleeping_state_label(current_level),
                    "requested_level": level,
                },
            )

        await client.sleep(level, mode)
        # Re-read level after the call so the response reflects what the
        # engine actually transitioned to (e.g. partial-failure paths
        # could leave the level lower than requested in the future).
        new_level = await _current_sleep_level(client)
        return JSONResponse(
            status_code=200,
            content={
                "already_sleeping": False,
                "current_state": _sleeping_state_label(new_level),
                "requested_level": level,
            },
        )
    finally:
        lock.release()


@router.post("/wake_up")
async def wake_up(raw_request: Request):
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        # set to None to wake up all tags if no tags are provided
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    client = engine_client(raw_request)

    lock = _get_transition_lock(raw_request)
    try:
        await asyncio.wait_for(
            lock.acquire(), timeout=_TRANSITION_LOCK_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        logger.warning(
            "/wake_up could not acquire transition lock within %.1fs; "
            "another sleep/wake transition is still in flight.",
            _TRANSITION_LOCK_TIMEOUT_S,
        )
        return JSONResponse(
            status_code=409,
            headers={"Retry-After": str(_RETRY_AFTER_S)},
            content={
                "error": "transition_in_progress",
                "detail": (
                    "Another /sleep or /wake_up is currently transitioning "
                    "this engine. Retry after it completes."
                ),
            },
        )

    try:
        # Full-wake idempotency: only short-circuit when the caller
        # asked for an unscoped wake AND the engine is already fully
        # awake. Tagged/partial wakes always go through to the
        # executor, which is the authoritative source of truth for the
        # per-tag bookkeeping (a partial wake with the engine already
        # awake will surface its own no-op via that path).
        is_sleeping = await client.is_sleeping()
        if tags is None and not is_sleeping:
            logger.warning(
                "/wake_up called on an already-awake engine; treating as no-op"
            )
            return JSONResponse(
                status_code=200,
                content={
                    "already_awake": True,
                    "current_state": "awake",
                },
            )

        await client.wake_up(tags)
        new_level = await _current_sleep_level(client)
        # Detect a partial-wake outcome: the caller asked for specific
        # tags AND the engine still reports is_sleeping=True after the
        # call (some other tag is still offloaded). Without this, the
        # response would say `sleeping_l1` and callers would assume the
        # wake was a no-op even though weights have in fact come back.
        partial_wake = False
        if tags is not None and new_level is not None:
            # Re-read is_sleeping after wake to determine if any tags
            # remain sleeping (partial wake). Adds one IPC RTT inside
            # the lock, but only on the tagged-wake path; full wake
            # (tags=None) skips this read.
            still_sleeping = await client.is_sleeping()
            partial_wake = bool(still_sleeping)
        return JSONResponse(
            status_code=200,
            content={
                "already_awake": False,
                "current_state": _sleeping_state_label(
                    new_level, partial_wake=partial_wake
                ),
            },
        )
    finally:
        lock.release()


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    client = engine_client(raw_request)
    is_sleeping = await client.is_sleeping()
    current_level = await _current_sleep_level(client)
    return JSONResponse(
        content={
            "is_sleeping": is_sleeping,
            "current_state": _sleeping_state_label(current_level),
        }
    )


def attach_router(app: FastAPI):
    app.include_router(router)
