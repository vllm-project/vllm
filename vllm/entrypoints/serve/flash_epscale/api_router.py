# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import os
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


class _PostResizeError(RuntimeError):
    """A failure after the resize is live: the new EP topology is serving,
    only the post-resume memory release failed. Callers must not roll back
    routing to the old size."""


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


async def _rpc(
    client: EngineClient, timing: dict[str, float], key: str, method: str, **kwargs
):
    logger.info("flash_epscale: %s(%s)", method, kwargs)
    async with _timed(timing, key):
        return await client.collective_rpc(method, kwargs=kwargs)


@asynccontextmanager
async def _paused(client: EngineClient, timing: dict[str, float]):
    """Stop-the-world only around the steps that require global quiescence
    (NCCL split, EPLB remap/refill, NIXL reconnect). Always attempts resume
    on exit; resume failures propagate so the endpoint cannot report false
    success."""
    logger.info("flash_epscale: entering pause_generation(mode=wait)")
    async with _timed(timing, "pause"):
        await client.pause_generation(mode="wait", clear_cache=False)
    # Free cached-but-unused activation blocks so the weight transfers in
    # the window have headroom. Best-effort: the cache refills lazily.
    try:
        await _rpc(client, timing, "empty_cache", "empty_cache")
    except Exception:
        logger.warning("flash_epscale: empty_cache failed (ignored)", exc_info=True)
    try:
        yield
    finally:
        logger.info("flash_epscale: entering resume_generation")
        async with _timed(timing, "resume"):
            await client.resume_generation()


async def _sleep_ranks(
    client: EngineClient,
    timing: dict[str, float],
    key: str,
    ranks: list[int],
    tags: list[str],
    level: int,
) -> None:
    await _rpc(
        client,
        timing,
        key,
        "sleep_ep_ranks_by_tags",
        sleeping_ep_ranks=ranks,
        tags=tags,
        level=level,
    )


async def _transition(
    client: EngineClient,
    *,
    current_sleeping: list[int],
    target_sleeping: list[int],
    tags: list[str],
    level: int,
    timing: dict[str, float],
) -> None:
    """Move the EP sleep state from ``current_sleeping`` to
    ``target_sleeping`` (either direction).

    Phases:
      1. Pre-pause: rank-local CuMem wake of the currently sleeping ranks.
         No collectives, so active ranks keep serving; the woken ranks stay
         logically slept (and skip forwards) until the resize below.
      2. Pause window: everything that must not interleave with forward
         collectives — NIXL reconnect, L2 dense/EPLB-map refill, EPLB
         expert remap + DP NCCL split, NIXL disconnect from new sleepers.
      3. Post-resume: rank-local CuMem sleep + NIXL buffer destroy on the
         new sleeping ranks. They are masked and skipped by then, so
         active ranks serve while the memory is released.

    Failures in phase 1/2 roll back to ``current_sleeping`` best-effort.
    Phase 3 failures raise ``_PostResizeError``: the resize is live and
    must not be rolled back; only the memory release failed.
    """
    if current_sleeping:
        try:
            await _rpc(
                client,
                timing,
                "wake",
                "wake_up_ep_ranks",
                sleeping_ep_ranks=current_sleeping,
                tags=tags,
                level=level,
            )
        except Exception:
            try:
                await _sleep_ranks(
                    client, timing, "rollback_sleep", current_sleeping, tags, level
                )
            except Exception:
                logger.exception("flash_epscale wake rollback failed")
            raise

    if target_sleeping:
        # Pre-pin the host backup buffers the post-resume sleep will
        # offload into, while the soon-to-sleep ranks are already drained.
        # Best-effort: the sleep itself pins on demand if this fails.
        try:
            await _rpc(
                client,
                timing,
                "warm_backup",
                "warm_sleep_backup",
                sleeping_ep_ranks=target_sleeping,
                tags=tags,
                level=level,
            )
        except Exception:
            logger.warning(
                "flash_epscale: warm_sleep_backup failed (ignored)", exc_info=True
            )

    async with _paused(client, timing):
        try:
            if current_sleeping:
                # Restore NIXL all2all state before the refill/resize:
                # destroyed-buffer ranks rebuild, live ranks reconnect.
                await _rpc(
                    client,
                    timing,
                    "nixl_ensure",
                    "ensure_nixl_buffer",
                    sleeping_ep_ranks=current_sleeping,
                )
                if level == 2:
                    # L2 discarded weights with no CPU backup; refill dense
                    # weights and EPLB maps from active peers (collective).
                    await _rpc(
                        client,
                        timing,
                        "l2_refill",
                        "finalize_l2_wake",
                        sleeping_ep_ranks=current_sleeping,
                    )
            await _rpc(
                client,
                timing,
                "resize",
                "resize_sleep_ep_ranks",
                sleeping_ep_ranks=target_sleeping,
            )
            if target_sleeping:
                # Alive ranks must forget the sleeping peers' NIXL endpoints
                # now, or a later wake hangs in the connect handshake. Cheap
                # and touches the live agents, so it stays in the window.
                await _rpc(
                    client,
                    timing,
                    "nixl_disconnect",
                    "disconnect_nixl_from_sleeping",
                    sleeping_ep_ranks=target_sleeping,
                )
        except Exception:
            # Still inside the pause window: collectives are safe here.
            try:
                async with _timed(timing, "rollback"):
                    await client.collective_rpc(
                        "resize_sleep_ep_ranks",
                        kwargs={"sleeping_ep_ranks": current_sleeping},
                    )
                    if current_sleeping:
                        await client.collective_rpc(
                            "sleep_ep_ranks_by_tags",
                            kwargs={
                                "sleeping_ep_ranks": current_sleeping,
                                "tags": tags,
                                "level": level,
                            },
                        )
            except Exception:
                logger.exception("flash_epscale rollback failed")
            raise

    if not target_sleeping:
        return
    try:
        await _sleep_ranks(client, timing, "sleep", target_sleeping, tags, level)
        if os.environ.get("VLLM_FLASH_EPSCALE_SKIP_NIXL_TEARDOWN") == "1":
            logger.info(
                "flash_epscale: skipping NIXL buffer destroy (env override); "
                "disconnect already applied"
            )
        else:
            try:
                await _rpc(
                    client,
                    timing,
                    "nixl_destroy",
                    "destroy_nixl_buffer",
                    sleeping_ep_ranks=target_sleeping,
                )
            except Exception:
                logger.warning(
                    "flash_epscale: nixl destroy failed (ignored)", exc_info=True
                )
    except Exception as e:
        raise _PostResizeError(
            f"resize to sleeping={target_sleeping} is live, but releasing "
            f"memory on the sleeping ranks failed: {e}"
        ) from e


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

        target_sleeping = list(range(target_ep_size, ep_world_size))

        if target_ep_size == active_ep_size:
            client.set_active_data_parallel_size(target_ep_size)
            action = "noop"
        elif target_ep_size < active_ep_size:
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

        if action != "noop":
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
                "active_ep_size": target_ep_size,
                "sleeping_ep_ranks": target_sleeping
                if action != "noop"
                else current_sleeping,
                "changed": action != "noop",
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

    Routing shrinks first so active ranks keep serving while the
    soon-to-sleep ranks drain; only the collective steps run inside the
    pause window (see ``_transition``).
    """
    try:
        async with _timed(timing, "route_shrink"):
            client.set_active_data_parallel_size(target_ep_size)
        # Reroute eligible in-flight requests off the soon-to-sleep ranks
        # onto the active prefix. Best-effort: requests that cannot be
        # safely re-issued (n>1, pooling, streaming input, non-DELTA
        # output) fall through to the drain step below.
        async with _timed(timing, "reroute"):
            rerouted = await client.reroute_inflight_to_active(target_sleeping)
        timing["reroute_count"] = float(len(rerouted))
        async with _timed(timing, "drain"):
            await client.wait_for_dp_ranks_to_drain(target_sleeping, drain_timeout)
        await _transition(
            client,
            current_sleeping=current_sleeping,
            target_sleeping=target_sleeping,
            tags=tags,
            level=level,
            timing=timing,
        )
    except Exception as e:
        # Restore routing so requests are not stranded — unless the resize
        # is already live, in which case the shrunk routing is correct.
        if not isinstance(e, _PostResizeError):
            try:
                client.set_active_data_parallel_size(active_ep_size)
            except Exception:
                logger.exception("flash_epscale scale_down route restore failed")
        logger.exception("flash_epscale scale_down failed")
        raise HTTPException(
            status_code=500, detail=f"flash_epscale scale_down failed: {e}"
        ) from e


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

    Routing opens to the new ranks only after the transition completes, so
    requests never reach a rank that is not ready.
    """
    try:
        await _transition(
            client,
            current_sleeping=current_sleeping,
            target_sleeping=target_sleeping,
            tags=tags,
            level=level,
            timing=timing,
        )
    except Exception as e:
        logger.exception("flash_epscale scale_up failed")
        raise HTTPException(
            status_code=500, detail=f"flash_epscale scale_up failed: {e}"
        ) from e

    async with _timed(timing, "route_grow"):
        client.set_active_data_parallel_size(target_ep_size)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
