# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fast step-barrier pause endpoints for weight synchronization.

These endpoints allow callers to quickly pause the engine (without waiting
for in-flight requests to drain), retrieve a monotonic step counter, and
then enforce a cluster-wide barrier via /pause/step/barrier.

Typical usage (from an RL training loop):
    1. POST /pause/step → get step_counter
    2. POST /pause/step/barrier with target_steps = step_counter + 1
    3. (all engines are now at a known step barrier)
    4. update model weights
    5. POST /resume
"""

import asyncio
import json
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _require_async_llm(client: EngineClient) -> AsyncLLM:
    """Cast engine client to AsyncLLM or raise an error."""
    if not isinstance(client, AsyncLLM):
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="Step-barrier pause endpoints require V1 AsyncLLM engine",
        )
    return client


async def _pause_first_engine(client: AsyncLLM) -> int:
    """Broadcast pause to all engine cores, return step_counter from first reply."""
    futures = [
        client.engine_core._call_utility_async("pause", engine=engine)
        for engine in client.engine_core.core_engines
    ]
    for coro in asyncio.as_completed(futures):
        step_counter = await coro
        return step_counter
    # Should never reach here if there is at least one engine
    raise RuntimeError("No engine cores available")


@router.post("/pause/step")
async def pause_step(raw_request: Request) -> JSONResponse:
    """
    Request all engines to pause at the next step boundary (fast, non-blocking).

    Returns a JSON response containing:
      - paused: true
      - step_counter: monotonic step count at pause time
      - recommended_target_step: step_counter + 1 (for barrier)
      - message: explanation of the contract
      - status: "paused"

    This is a control-plane signal; it does NOT wait for in-flight requests.
    To enforce a cluster-wide barrier, call /pause/step/barrier.
    """
    client = _require_async_llm(engine_client(raw_request))

    logger.info("API server (/pause/step) received request")
    step_counter = await _pause_first_engine(client)

    return JSONResponse(
        content={
            "paused": True,
            "step_counter": step_counter,
            "recommended_target_step": step_counter + 1,
            "message": (
                "Pause requested. Engine will stop scheduling new work at the "
                "next step boundary. For a full barrier (all engines reach a "
                "known step), call /pause/step/barrier with "
                "target_steps=recommended_target_step."
            ),
            "status": "paused",
        },
        status_code=HTTPStatus.OK.value,
    )


@router.post("/pause/step/barrier")
async def pause_step_barrier(raw_request: Request) -> JSONResponse:
    """
    Resume engines and block until all have reached target_steps, then pause.

    Request body: {"target_steps": <int>}

    This is the barrier endpoint: it returns only when every engine has
    executed at least target_steps and is paused.
    """
    client = _require_async_llm(engine_client(raw_request))

    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e

    target_steps = body.get("target_steps")
    if target_steps is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'target_steps' in request body",
        )
    try:
        target_steps = int(target_steps)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'target_steps' must be an integer",
        ) from e

    logger.info("API server (/pause/step/barrier) target=%d", target_steps)

    # Broadcast run_until_target_step_count to all engines
    await client.engine_core.call_utility_async(
        "run_until_target_step_count", target_steps
    )

    # Poll until all engines have reached the target step (barrier)
    # We check step_counter >= target_steps rather than is_engine_paused
    # because is_engine_paused could be true from a prior pause state
    max_wait_seconds = 300  # 5 minute timeout
    poll_interval = 0.01  # 10ms
    waited = 0.0
    log_interval = 1.0  # Log progress every second
    last_log_time = 0.0
    while waited < max_wait_seconds:
        step_counters = await client.engine_core.call_utility_async(
            "get_step_counter"
        )
        # Handle both single value and list of values (multi-engine)
        if isinstance(step_counters, list):
            min_step = min(step_counters)
        else:
            min_step = step_counters
        if min_step >= target_steps:
            logger.info(
                "Barrier reached: all engines at step %d >= %d",
                min_step,
                target_steps,
            )
            return JSONResponse(
                content={"status": "ok"}, status_code=HTTPStatus.OK.value
            )
        # Log progress periodically
        if waited - last_log_time >= log_interval:
            logger.info(
                "Waiting for barrier: current step=%d, target=%d, waited=%.1fs",
                min_step,
                target_steps,
                waited,
            )
            last_log_time = waited
        await asyncio.sleep(poll_interval)
        waited += poll_interval

    # Timeout - engines didn't reach target in time
    raise HTTPException(
        status_code=HTTPStatus.REQUEST_TIMEOUT.value,
        detail=f"Timeout waiting for engines to reach step {target_steps}",
    )


def attach_router(app: FastAPI):
    app.include_router(router)
