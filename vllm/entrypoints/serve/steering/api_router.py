# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.steering.protocol import SetSteeringRequest
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

# Serializes steering mutations (set / clear) so the two-phase
# validate-then-apply flow in /set cannot be interleaved with
# another /set or /clear request.
_steering_lock = asyncio.Lock()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/v1/steering/set")
async def set_steering(
    request: SetSteeringRequest,
    raw_request: Request,
) -> JSONResponse:
    """Set activation steering vectors on decoder layers.

    Vectors are applied to the residual stream after the MLP in each
    specified layer.  Only the listed layers are modified; other layers
    keep their current state.  Call ``POST /v1/steering/clear`` first
    to zero everything if a full replacement is intended.
    """
    engine = engine_client(raw_request)

    # Pre-multiply scales into vectors so the worker only needs to copy.
    scaled: dict[int, list[float]] = {}
    for layer_idx, vec in request.vectors.items():
        scale = 1.0
        if request.scales and layer_idx in request.scales:
            scale = request.scales[layer_idx]
        if scale != 1.0:
            scaled[layer_idx] = [v * scale for v in vec]
        else:
            scaled[layer_idx] = vec

    try:
        async with _steering_lock:
            # Phase 1 — validate on every worker without mutating
            # buffers.  This prevents pipeline-parallel workers from
            # partially applying an update when a later stage would
            # reject it.
            results = await engine.collective_rpc(
                "set_steering_vectors", args=(scaled, True)
            )
            # Each worker returns the layer indices it *would* update.
            # Union across workers (TP replicas report the same layers,
            # PP stages report disjoint layers).
            validated_layers: set[int] = set()
            for per_worker in results:
                validated_layers.update(per_worker)

            # Reject requests that reference layers not owned by any
            # worker — a typo like layer 999 must not be silently
            # ignored while the valid layers succeed.
            requested_layers = set(scaled.keys())
            missing = requested_layers - validated_layers
            if missing:
                return JSONResponse(
                    content={
                        "error": (
                            f"Layer(s) {sorted(missing)} not found in "
                            f"model. Steerable layers that matched: "
                            f"{sorted(validated_layers) or 'none'}"
                        ),
                    },
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            if not validated_layers:
                return JSONResponse(
                    content={
                        "error": (
                            "No steerable layers found.  The loaded "
                            "model may not support activation steering, "
                            "or the requested layer indices do not "
                            "exist."
                        ),
                    },
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )

            # Phase 2 — all workers validated; now apply.
            await engine.collective_rpc(
                "set_steering_vectors", args=(scaled, False)
            )

        return JSONResponse(
            content={
                "status": "ok",
                "layers_updated": sorted(validated_layers),
            },
        )
    except (ValueError, RuntimeError) as err:
        # ValueError is raised directly in single-proc mode.
        # MultiprocExecutor wraps worker exceptions as RuntimeError
        # with the original message embedded in the string.
        err_str = str(err)
        if ("expected vector of size" in err_str
                or "non-finite" in err_str):
            return JSONResponse(
                content={"error": err_str},
                status_code=HTTPStatus.BAD_REQUEST.value,
            )
        logger.exception("Failed to set steering vectors")
        return JSONResponse(
            content={"error": f"Failed to set steering vectors: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    except Exception as err:
        # AsyncMPClient wraps failures as plain Exception — check
        # for validation messages so we return 400, not 500.
        err_str = str(err)
        if ("expected vector of size" in err_str
                or "non-finite" in err_str):
            return JSONResponse(
                content={"error": err_str},
                status_code=HTTPStatus.BAD_REQUEST.value,
            )
        logger.exception("Failed to set steering vectors")
        return JSONResponse(
            content={"error": f"Failed to set steering vectors: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/v1/steering/clear")
async def clear_steering(raw_request: Request) -> JSONResponse:
    """Reset all steering vectors to zero (no-op steering)."""
    engine = engine_client(raw_request)

    try:
        async with _steering_lock:
            await engine.collective_rpc("clear_steering_vectors")
        return JSONResponse(content={"status": "ok"})
    except Exception as err:
        logger.exception("Failed to clear steering vectors")
        return JSONResponse(
            content={"error": f"Failed to clear steering vectors: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.get("/v1/steering")
async def get_steering(raw_request: Request) -> JSONResponse:
    """Return which layers currently have non-zero steering vectors."""
    engine = engine_client(raw_request)

    try:
        results = await engine.collective_rpc("get_steering_status")
        # Union across all workers so pipeline-parallel ranks
        # (which own disjoint layer ranges) are all represented.
        active: dict = {}
        for worker_result in results:
            active.update(worker_result)
        return JSONResponse(
            content={
                "active_layers": {
                    str(k): v for k, v in sorted(active.items())
                },
            },
        )
    except Exception as err:
        logger.exception("Failed to get steering status")
        return JSONResponse(
            content={"error": f"Failed to get steering status: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
