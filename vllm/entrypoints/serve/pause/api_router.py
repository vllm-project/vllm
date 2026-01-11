# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fast step-barrier pause endpoint for weight synchronization.

This endpoint allows callers to quickly pause the engine and optionally
wait for a cluster-wide barrier where all engines reach a known step.

Typical usage (from an RL training loop):
    1. POST /pause/step  (waits for barrier by default)
    2. update model weights
    3. POST /resume

For advanced use cases:
    - POST /pause/step?no_barrier=true  → fast pause, no waiting
    - POST /pause/step?barrier=50       → wait until all engines reach step 50
"""

import asyncio
from http import HTTPStatus
from typing import cast

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient

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


async def _pause_all_engines(client: AsyncLLM) -> int:
    """Pause all engine cores, return the step counter from the first core."""
    engine_core = cast(AsyncMPClient, client.engine_core)
    step_counters = await asyncio.gather(
        *[
            engine_core._call_utility_async("pause", engine=engine)
            for engine in engine_core.core_engines
        ]
    )
    return step_counters[0]


async def _wait_for_barrier(client: AsyncLLM, target: int) -> int:
    """Wait until all engines reach the target step. Returns final step."""
    engine_core = cast(AsyncMPClient, client.engine_core)

    # Broadcast run_until_target_step_count to all engines
    await asyncio.gather(
        *[
            engine_core._call_utility_async(
                "run_until_target_step_count", target, engine=engine
            )
            for engine in engine_core.core_engines
        ]
    )

    # Poll until all engines have reached the target step
    max_wait_seconds = 300  # 5 minute timeout
    poll_interval = 0.01  # 10ms
    waited = 0.0
    log_interval = 1.0  # Log progress every second
    last_log_time = 0.0

    while waited < max_wait_seconds:
        step_counters = await asyncio.gather(
            *[
                engine_core._call_utility_async("get_step_counter", engine=engine)
                for engine in engine_core.core_engines
            ]
        )
        min_step = min(step_counters)

        if min_step >= target:
            logger.info(
                "Barrier reached: all engines at step %d >= %d",
                min_step,
                target,
            )
            return min_step

        # Log progress periodically
        if waited - last_log_time >= log_interval:
            logger.info(
                "Waiting for barrier: current step=%d, target=%d, waited=%.1fs",
                min_step,
                target,
                waited,
            )
            last_log_time = waited

        await asyncio.sleep(poll_interval)
        waited += poll_interval

    # Timeout - engines didn't reach target in time
    raise HTTPException(
        status_code=HTTPStatus.REQUEST_TIMEOUT.value,
        detail=f"Timeout waiting for engines to reach step {target}",
    )


@router.post("/pause/step")
async def pause_step(
    raw_request: Request,
    no_barrier: bool = Query(False),
    barrier: int | None = Query(None),
) -> JSONResponse:
    """
    Pause all engines at a step boundary.

    Query parameters:
        no_barrier: If true, return immediately after pause signal (fast).
                    Default is false (waits for barrier).
        barrier: Custom target step to wait for. If not specified and
                 no_barrier is false, defaults to step_counter + 1.

    Returns:
        paused: true
        step_counter: the step at which engines are paused

    Examples:
        POST /pause/step                    → pause + wait for step+1
        POST /pause/step?no_barrier=true    → fast pause, no waiting
        POST /pause/step?barrier=50         → pause + wait until step 50
    """
    logger.info(
        "API server (/pause/step) received request: no_barrier=%s, barrier=%s",
        no_barrier,
        barrier,
    )

    if no_barrier and barrier is not None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Cannot specify both no_barrier=true and barrier=<value>",
        )

    client = _require_async_llm(engine_client(raw_request))

    # 1. Pause immediately, get step_counter from first engine
    step_counter = await _pause_all_engines(client)

    # 2. Handle no_barrier case - fast return
    if no_barrier:
        return JSONResponse(
            content={
                "paused": True,
                "step_counter": step_counter,
            },
            status_code=HTTPStatus.OK.value,
        )

    # 3. Compute target: explicit barrier or step_counter + 1
    target = barrier if barrier is not None else step_counter + 1

    # 4. Wait for barrier
    final_step = await _wait_for_barrier(client, target)

    return JSONResponse(
        content={
            "paused": True,
            "step_counter": final_step,
        },
        status_code=HTTPStatus.OK.value,
    )


def attach_router(app: FastAPI):
    app.include_router(router)
