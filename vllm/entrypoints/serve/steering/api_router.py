# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.steering.protocol import SetSteeringRequest
from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)

router = APIRouter()

# Serializes steering mutations (set / clear) so the two-phase
# validate-then-apply flow in /set cannot be interleaved with
# another /set or /clear request.
_steering_lock = asyncio.Lock()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _scale_layer_vectors(
    layer_vecs: dict[int, list[float]],
    scales: dict[int, float] | None,
) -> dict[int, list[float]]:
    """Pre-multiply per-layer scale factors into vectors."""
    scaled: dict[int, list[float]] = {}
    for layer_idx, vec in layer_vecs.items():
        scale = 1.0
        if scales and layer_idx in scales:
            scale = scales[layer_idx]
        if scale != 1.0:
            scaled[layer_idx] = [v * scale for v in vec]
        else:
            scaled[layer_idx] = vec
    return scaled


@router.post("/v1/steering/set")
async def set_steering(
    request: SetSteeringRequest,
    raw_request: Request,
) -> JSONResponse:
    """Set activation steering vectors on decoder layers.

    Vectors can be provided via the ``vectors`` field (shorthand for
    the default ``post_mlp_pre_ln`` hook point) and/or via the
    ``hook_vectors`` field which maps hook point names to layer-vector
    dicts.  Both fields are merged; ``hook_vectors`` entries for
    ``post_mlp_pre_ln`` are merged with ``vectors`` (``hook_vectors``
    wins on conflict).

    When ``replace`` is ``True``, all existing vectors are cleared
    atomically before the new ones are applied.
    """
    engine = engine_client(raw_request)

    # Validate hook_vectors keys up front.
    if request.hook_vectors:
        invalid_hooks = (
            set(request.hook_vectors.keys()) - VALID_HOOK_POINT_NAMES
        )
        if invalid_hooks:
            return JSONResponse(
                content={
                    "error": (
                        f"Invalid hook point name(s): "
                        f"{sorted(invalid_hooks)}. "
                        f"Valid values: "
                        f"{sorted(VALID_HOOK_POINT_NAMES)}"
                    ),
                },
                status_code=HTTPStatus.BAD_REQUEST.value,
            )

    # Build hook-point-aware vector dict.
    all_hook_vectors: dict[str, dict[int, list[float]]] = {}

    # Legacy vectors field -> post_mlp_pre_ln
    if request.vectors:
        all_hook_vectors["post_mlp_pre_ln"] = _scale_layer_vectors(
            request.vectors, request.scales
        )

    # Explicit hook_vectors
    if request.hook_vectors:
        for hook_name, layer_vecs in request.hook_vectors.items():
            scaled_hv = _scale_layer_vectors(layer_vecs, request.scales)
            if hook_name in all_hook_vectors:
                all_hook_vectors[hook_name].update(scaled_hv)
            else:
                all_hook_vectors[hook_name] = scaled_hv

    if not all_hook_vectors:
        return JSONResponse(
            content={
                "error": (
                    "No vectors provided. Include at least one layer "
                    "index and vector via 'vectors' or 'hook_vectors'."
                ),
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

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

    # Wrap legacy layer-indexed vectors under the default hook point
    # so the worker receives the new hook-point-aware format:
    # {hook_point_str: {layer_idx: [floats]}}
    hook_vectors: dict[str, dict[int, list[float]]] = {
        DEFAULT_HOOK_POINT.value: scaled,
    }

    try:
        async with _steering_lock:
            # Phase 1 -- validate on every worker without mutating
            # buffers.  This prevents pipeline-parallel workers from
            # partially applying an update when a later stage would
            # reject it.
            results = await engine.collective_rpc(
                "set_steering_vectors", args=(hook_vectors, True)
            )
            # Each worker returns the layer indices it *would* update.
            # Union across workers (TP replicas report the same layers,
            # PP stages report disjoint layers).
            validated_layers: set[int] = set()
            for per_worker in results:
                validated_layers.update(per_worker)

            # Collect all requested layers across all hook points.
            requested_layers: set[int] = set()
            for layer_vecs in all_hook_vectors.values():
                requested_layers.update(layer_vecs.keys())

            # Reject requests that reference layers not owned by any
            # worker.
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

            # Atomic replacement: clear all vectors before applying
            # new ones, within the same lock acquisition.
            if request.replace:
                await engine.collective_rpc("clear_steering_vectors")

            # Phase 2 — all workers validated; now apply.
            await engine.collective_rpc(
                "set_steering_vectors", args=(hook_vectors, False)
            )

        return JSONResponse(
            content={
                "status": "ok",
                "hook_points": sorted(all_hook_vectors.keys()),
                "layers_updated": sorted(validated_layers),
            },
        )
    except SteeringVectorError as err:
        # Single-process: typed exception comes through directly.
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as err:
        # Multi-process: exception type is lost; match by message content.
        err_str = str(err)
        if "expected vector of size" in err_str or "non-finite" in err_str:
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
                "active_layers": {str(k): v for k, v in sorted(active.items())},
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
