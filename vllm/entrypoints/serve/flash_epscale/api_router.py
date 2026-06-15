# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.dev.sleep.api_router import optional_tags
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_TAGS = ["shared_weights", "expert_weights", "kv_cache"]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _optional_timeout(payload: dict, key: str, default: float) -> float:
    value = payload.get(key, default)
    if type(value) not in (int, float) or value <= 0:
        raise HTTPException(status_code=400, detail=f"{key} must be a positive number")
    return float(value)


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _flash_epscale_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "flash_epscale_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.flash_epscale_lock = lock
    return lock


async def _query_ep_state(client: EngineClient) -> dict[str, Any]:
    states = await client.collective_rpc("get_ep_sleep_state")
    if not states:
        raise HTTPException(status_code=500, detail="failed to query EP sleep state")
    first = states[0]
    if any(s != first for s in states[1:]):
        raise HTTPException(
            status_code=500,
            detail=f"inconsistent EP sleep state across workers: {states}",
        )
    return first


def _parse_ep_state(state: dict[str, Any]) -> tuple[int, int, list[int]]:
    try:
        ep_world_size = int(state["ep_world_size"])
        active_ep_size = int(state["active_ep_size"])
        sleeping_ep_ranks = [int(rank) for rank in state["sleeping_ep_ranks"]]
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"malformed EP sleep state from workers: {state}",
        ) from e

    if ep_world_size <= 0:
        raise HTTPException(
            status_code=500,
            detail=f"invalid ep_world_size in EP sleep state: {ep_world_size}",
        )
    if not 1 <= active_ep_size <= ep_world_size:
        raise HTTPException(
            status_code=500,
            detail=f"invalid active_ep_size in EP sleep state: {active_ep_size}",
        )
    if len(set(sleeping_ep_ranks)) != len(sleeping_ep_ranks):
        raise HTTPException(
            status_code=500,
            detail=(
                "invalid EP sleep state: sleeping_ep_ranks contains "
                f"duplicates: {sleeping_ep_ranks}"
            ),
        )

    expected_sleeping = list(range(active_ep_size, ep_world_size))
    if sleeping_ep_ranks != expected_sleeping:
        raise HTTPException(
            status_code=500,
            detail=(
                "invalid EP sleep state: sleeping_ep_ranks must be the suffix "
                f"{expected_sleeping} for active_ep_size={active_ep_size}, got "
                f"{sleeping_ep_ranks}"
            ),
        )

    return ep_world_size, active_ep_size, sleeping_ep_ranks


@asynccontextmanager
async def _timed(timing: dict[str, float], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timing[key] = _elapsed_ms(start)


@asynccontextmanager
async def _paused(client: EngineClient, timing: dict[str, float]):
    """Stop-the-world only around the steps that require global quiescence
    (NCCL split, EPLB remap, sleep/wake RPCs). Always attempts resume on exit;
    resume failures propagate so the endpoint cannot report false success."""
    async with _timed(timing, "pause"):
        await client.pause_generation(mode="abort", clear_cache=False)
    try:
        yield
    finally:
        async with _timed(timing, "resume"):
            await client.resume_generation()


router = APIRouter()


@router.post("/flash_epscale")
async def flash_epscale(raw_request: Request):
    payload = await raw_request.json()
    target_ep_size = payload.get("ep_size")
    if type(target_ep_size) is not int:
        raise HTTPException(status_code=400, detail="ep_size must be an integer")
    tags = optional_tags(payload, "tags") or DEFAULT_TAGS
    drain_timeout = _optional_timeout(payload, "drain_timeout", 300)
    level = payload.get("level", 1)
    if level not in (1, 2):
        raise HTTPException(status_code=400, detail="level must be 1 or 2")
    client = _engine_client(raw_request)

    timing: dict[str, float] = {}
    total_start = time.perf_counter()

    async with _flash_epscale_lock(raw_request):
        async with _timed(timing, "query_state"):
            state = await _query_ep_state(client)
        ep_world_size, active_ep_size, current_sleeping = _parse_ep_state(state)

        if target_ep_size <= 0 or target_ep_size > ep_world_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"ep_size must be in [1, {ep_world_size}], got {target_ep_size}"
                ),
            )

        if target_ep_size == active_ep_size:
            client.set_active_data_parallel_size(target_ep_size)
            timing["total"] = _elapsed_ms(total_start)
            logger.info("flash_epscale noop timing_ms=%s", timing)
            return JSONResponse(
                content={
                    "ok": True,
                    "ep_world_size": ep_world_size,
                    "active_ep_size": active_ep_size,
                    "sleeping_ep_ranks": current_sleeping,
                    "changed": False,
                    "action": "noop",
                    "tags": tags,
                    "timing_ms": timing,
                }
            )

        target_sleeping = list(range(target_ep_size, ep_world_size))

        if target_ep_size < active_ep_size:
            action = "scale_down"
            await _scale_down(
                client,
                target_ep_size=target_ep_size,
                active_ep_size=active_ep_size,
                target_sleeping=target_sleeping,
                current_sleeping=current_sleeping,
                tags=tags,
                drain_timeout=drain_timeout,
                timing=timing,
                level=level,
            )
        else:
            action = "scale_up"
            await _scale_up(
                client,
                target_ep_size=target_ep_size,
                target_sleeping=target_sleeping,
                current_sleeping=current_sleeping,
                tags=tags,
                timing=timing,
                level=level,
            )

        async with _timed(timing, "final_state"):
            final = await _query_ep_state(client)
        _, final_active, final_sleeping = _parse_ep_state(final)
        if final_active != target_ep_size or final_sleeping != target_sleeping:
            timing["total"] = _elapsed_ms(total_start)
            logger.error("flash_epscale final state mismatch timing_ms=%s", timing)
            raise HTTPException(
                status_code=500,
                detail=(
                    "flash_epscale finished with unexpected EP sleep state: "
                    f"expected active_ep_size={target_ep_size}, "
                    f"sleeping_ep_ranks={target_sleeping}, got "
                    f"active_ep_size={final_active}, "
                    f"sleeping_ep_ranks={final_sleeping}"
                ),
            )

        timing["total"] = _elapsed_ms(total_start)
        logger.info("flash_epscale %s timing_ms=%s", action, timing)
        return JSONResponse(
            content={
                "ok": True,
                "ep_world_size": ep_world_size,
                "active_ep_size": final_active,
                "sleeping_ep_ranks": final_sleeping,
                "changed": True,
                "action": action,
                "tags": tags,
                "timing_ms": timing,
            }
        )


async def _scale_down(
    client: EngineClient,
    *,
    target_ep_size: int,
    active_ep_size: int,
    target_sleeping: list[int],
    current_sleeping: list[int],
    tags: list[str],
    drain_timeout: float,
    timing: dict[str, float],
    level: int = 1,
) -> None:
    """Shrink the active EP set.

    Active ranks keep serving while the soon-to-sleep ranks drain.  Only the
    NCCL split / EPLB remap / sleep RPCs run inside the pause window.
    """
    # 1. Stop routing to the ranks that will sleep; they finish in flight.
    #    Active ranks are unaffected.
    try:
        async with _timed(timing, "route_shrink"):
            client.set_active_data_parallel_size(target_ep_size)
        async with _timed(timing, "drain"):
            await client.wait_for_dp_ranks_to_drain(target_sleeping, drain_timeout)
    except Exception as e:
        # Restore routing so requests are not stranded.
        try:
            client.set_active_data_parallel_size(active_ep_size)
        except Exception:
            logger.exception("flash_epscale scale_down route restore failed")
        logger.exception("flash_epscale scale_down drain failed")
        raise HTTPException(
            status_code=500, detail=f"flash_epscale scale_down drain failed: {e}"
        ) from e

    transition_completed = False
    try:
        # 2. Stop-the-world only for steps that require global quiescence.
        async with _paused(client, timing):
            sleep_started = False
            try:
                if current_sleeping:
                    async with _timed(timing, "wake"):
                        await client.collective_rpc(
                            "wake_up_ep_ranks",
                            kwargs={
                                "sleeping_ep_ranks": current_sleeping,
                                "tags": tags,
                                "level": level,
                            },
                        )
                async with _timed(timing, "resize"):
                    await client.collective_rpc(
                        "resize_sleep_ep_ranks",
                        kwargs={"sleeping_ep_ranks": target_sleeping},
                    )
                async with _timed(timing, "sleep"):
                    sleep_started = True
                    await client.collective_rpc(
                        "sleep_ep_ranks_by_tags",
                        kwargs={
                            "sleeping_ep_ranks": target_sleeping,
                            "tags": tags,
                            "level": level,
                        },
                    )
                transition_completed = True
            except Exception:
                await _restore_scale_down_sleep_state(
                    client,
                    current_sleeping=current_sleeping,
                    target_sleeping=target_sleeping,
                    tags=tags,
                    timing=timing,
                    wake_target_sleeping=sleep_started,
                    level=level,
                )
                raise
    except Exception as e:
        if not transition_completed:
            try:
                client.set_active_data_parallel_size(active_ep_size)
            except Exception:
                logger.exception("flash_epscale scale_down route restore failed")
        logger.exception("flash_epscale scale_down transition failed")
        raise HTTPException(
            status_code=500,
            detail=f"flash_epscale scale_down transition failed: {e}",
        ) from e


async def _restore_scale_down_sleep_state(
    client: EngineClient,
    *,
    current_sleeping: list[int],
    target_sleeping: list[int],
    tags: list[str],
    timing: dict[str, float],
    wake_target_sleeping: bool,
    level: int = 1,
) -> None:
    """Best-effort rollback for failures after scale_down routing shrinks."""
    try:
        if wake_target_sleeping and target_sleeping:
            async with _timed(timing, "rollback_wake"):
                await client.collective_rpc(
                    "wake_up_ep_ranks",
                    kwargs={
                        "sleeping_ep_ranks": target_sleeping,
                        "tags": tags,
                        "level": level,
                    },
                )
        async with _timed(timing, "rollback_resize"):
            await client.collective_rpc(
                "resize_sleep_ep_ranks",
                kwargs={"sleeping_ep_ranks": current_sleeping},
            )
        if current_sleeping:
            async with _timed(timing, "rollback_sleep"):
                await client.collective_rpc(
                    "sleep_ep_ranks_by_tags",
                    kwargs={
                        "sleeping_ep_ranks": current_sleeping,
                        "tags": tags,
                        "level": level,
                    },
                )
    except Exception:
        logger.exception("flash_epscale scale_down rollback failed")


async def _scale_up(
    client: EngineClient,
    *,
    target_ep_size: int,
    target_sleeping: list[int],
    current_sleeping: list[int],
    tags: list[str],
    timing: dict[str, float],
    level: int = 1,
) -> None:
    """Grow the active EP set.

    Routing is opened to the new ranks only after the NCCL group is resized
    and they are awake, so requests never reach a rank that is not ready.
    """
    try:
        async with _paused(client, timing):
            wake_completed = False
            resize_completed = False
            sleep_started = False
            try:
                if current_sleeping:
                    async with _timed(timing, "wake"):
                        await client.collective_rpc(
                            "wake_up_ep_ranks",
                            kwargs={
                                "sleeping_ep_ranks": current_sleeping,
                                "tags": tags,
                                "level": level,
                            },
                        )
                    wake_completed = True
                async with _timed(timing, "resize"):
                    await client.collective_rpc(
                        "resize_sleep_ep_ranks",
                        kwargs={"sleeping_ep_ranks": target_sleeping},
                    )
                resize_completed = True
                if target_sleeping:
                    async with _timed(timing, "sleep"):
                        sleep_started = True
                        await client.collective_rpc(
                            "sleep_ep_ranks_by_tags",
                            kwargs={
                                "sleeping_ep_ranks": target_sleeping,
                                "tags": tags,
                                "level": level,
                            },
                        )
            except Exception:
                await _restore_scale_up_sleep_state(
                    client,
                    current_sleeping=current_sleeping,
                    target_sleeping=target_sleeping,
                    tags=tags,
                    timing=timing,
                    wake_completed=wake_completed,
                    resize_completed=resize_completed,
                    sleep_started=sleep_started,
                    level=level,
                )
                raise
    except Exception as e:
        logger.exception("flash_epscale scale_up transition failed")
        raise HTTPException(
            status_code=500,
            detail=f"flash_epscale scale_up transition failed: {e}",
        ) from e

    # Open routing to the newly active ranks only after the group is ready.
    async with _timed(timing, "route_grow"):
        client.set_active_data_parallel_size(target_ep_size)


async def _restore_scale_up_sleep_state(
    client: EngineClient,
    *,
    current_sleeping: list[int],
    target_sleeping: list[int],
    tags: list[str],
    timing: dict[str, float],
    wake_completed: bool,
    resize_completed: bool,
    sleep_started: bool,
    level: int = 1,
) -> None:
    """Best-effort rollback for failures before scale_up routing opens."""
    try:
        if sleep_started and target_sleeping:
            async with _timed(timing, "rollback_wake"):
                await client.collective_rpc(
                    "wake_up_ep_ranks",
                    kwargs={
                        "sleeping_ep_ranks": target_sleeping,
                        "tags": tags,
                        "level": level,
                    },
                )
        if resize_completed:
            async with _timed(timing, "rollback_resize"):
                await client.collective_rpc(
                    "resize_sleep_ep_ranks",
                    kwargs={"sleeping_ep_ranks": current_sleeping},
                )
        if wake_completed and current_sleeping:
            async with _timed(timing, "rollback_sleep"):
                await client.collective_rpc(
                    "sleep_ep_ranks_by_tags",
                    kwargs={
                        "sleeping_ep_ranks": current_sleeping,
                        "tags": tags,
                        "level": level,
                    },
                )
    except Exception:
        logger.exception("flash_epscale scale_up rollback failed")


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
