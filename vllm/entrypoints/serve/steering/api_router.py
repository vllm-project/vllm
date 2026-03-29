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

    The ``vectors`` field maps hook point names to per-layer vector dicts.

    When ``replace`` is ``True``, all existing vectors are cleared
    atomically before the new ones are applied.
    """
    engine = engine_client(raw_request)

    # Validate hook point names.
    invalid_hooks = set(request.vectors.keys()) - VALID_HOOK_POINT_NAMES
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

    # Scale vectors.
    all_hook_vectors: dict[str, dict[int, list[float]]] = {}
    for hook_name, layer_vecs in request.vectors.items():
        all_hook_vectors[hook_name] = _scale_layer_vectors(layer_vecs, request.scales)

    if not all_hook_vectors:
        return JSONResponse(
            content={
                "error": (
                    "No vectors provided. Include at least one hook "
                    "point with layer vectors."
                ),
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    try:
        async with _steering_lock:
            # Phase 1 -- validate on every worker without mutating
            # buffers.
            results = await engine.collective_rpc(
                "set_steering_vectors",
                args=(all_hook_vectors, True),
            )
            validated_layers: set[int] = set()
            for per_worker in results:
                validated_layers.update(per_worker)

            # Collect all requested layers across all hook points.
            requested_layers: set[int] = set()
            for layer_vecs in all_hook_vectors.values():
                requested_layers.update(layer_vecs.keys())

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

            if request.replace:
                await engine.collective_rpc("clear_steering_vectors")

            # Phase 2 -- apply.
            await engine.collective_rpc(
                "set_steering_vectors",
                args=(all_hook_vectors, False),
            )

        return JSONResponse(
            content={
                "status": "ok",
                "hook_points": sorted(all_hook_vectors.keys()),
                "layers_updated": sorted(validated_layers),
            },
        )
    except SteeringVectorError as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as err:
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
