# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine import PauseMode
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/health")
async def admin_health(raw_request: Request) -> JSONResponse:
    """Health and readiness check for the vLLM instance."""
    engine = engine_client(raw_request)
    try:
        await engine.check_health()
        return JSONResponse(
            content={
                "status": "healthy",
                "version": VLLM_VERSION,
                "is_running": engine.is_running,
                "is_paused": await engine.is_paused(),
            },
        )
    except EngineDeadError:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "version": VLLM_VERSION,
            },
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )


@router.get("/models")
async def admin_models(raw_request: Request) -> JSONResponse:
    """List loaded models and adapters."""
    openai_serving_models = raw_request.app.state.openai_serving_models
    models_list = await openai_serving_models.show_available_models()
    return JSONResponse(content=models_list.model_dump())


@router.get("/queue")
async def admin_queue(raw_request: Request) -> JSONResponse:
    """Return queue and concurrency statistics."""
    server_load = getattr(raw_request.app.state, "server_load_metrics", None)
    load_tracking = getattr(raw_request.app.state, "enable_server_load_tracking", False)
    return JSONResponse(
        content={
            "server_load": server_load if load_tracking else None,
            "load_tracking_enabled": load_tracking,
        },
    )


@router.post("/drain")
async def admin_drain(
    raw_request: Request,
    mode: Annotated[PauseMode, Query()] = "wait",
    clear_cache: Annotated[bool, Query()] = True,
) -> JSONResponse:
    """Drain in-flight requests and pause generation.

    This endpoint pauses the engine, optionally waiting for in-flight
    requests to complete. Useful for graceful shutdown, maintenance,
    or Kubernetes pre-stop hooks.

    Args:
        mode: How to handle in-flight requests:
            - ``"abort"``: Abort all in-flight requests immediately.
            - ``"wait"``: Wait for in-flight requests to complete (default).
            - ``"keep"``: Freeze requests in queue; they resume on /resume.
        clear_cache: Whether to clear KV/prefix caches after draining.
    """
    args = raw_request.app.state.args
    if getattr(args, "admin_readonly", False):
        return JSONResponse(
            content={"error": "Admin API is in read-only mode."},
            status_code=HTTPStatus.FORBIDDEN.value,
        )

    engine = engine_client(raw_request)
    try:
        await engine.pause_generation(
            mode=mode,
            clear_cache=clear_cache,
            wait_for_inflight_requests=(mode == "wait"),
        )
        return JSONResponse(content={"status": "drained"})
    except ValueError as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception:
        logger.exception("Failed to drain")
        return JSONResponse(
            content={"error": "Failed to drain. See server logs for details."},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/resume")
async def admin_resume(raw_request: Request) -> JSONResponse:
    """Resume generation after a drain."""
    args = raw_request.app.state.args
    if getattr(args, "admin_readonly", False):
        return JSONResponse(
            content={"error": "Admin API is in read-only mode."},
            status_code=HTTPStatus.FORBIDDEN.value,
        )

    engine = engine_client(raw_request)
    try:
        await engine.resume_generation()
        return JSONResponse(content={"status": "resumed"})
    except Exception:
        logger.exception("Failed to resume generation")
        return JSONResponse(
            content={"error": "Failed to resume. See server logs for details."},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.get("/config")
async def admin_config(raw_request: Request) -> JSONResponse:
    """Return the current engine and model configuration."""
    engine = engine_client(raw_request)
    model_config = engine.model_config
    parallel_config = engine.vllm_config.parallel_config
    return JSONResponse(
        content={
            "model": model_config.model,
            "dtype": str(model_config.dtype),
            "max_model_len": model_config.max_model_len,
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
            "enable_expert_parallel": parallel_config.enable_expert_parallel,
        },
    )


@router.post("/reload_model")
async def admin_reload_model(raw_request: Request) -> JSONResponse:
    """Reload model (placeholder — not yet implemented)."""
    return JSONResponse(
        content={"error": "Model reload is not yet implemented."},
        status_code=HTTPStatus.NOT_IMPLEMENTED.value,
    )


def attach_router(app: FastAPI):
    args = getattr(app.state, "args", None)
    if args is None or not getattr(args, "enable_admin_api", False):
        return
    logger.info("Admin control plane API enabled at /v1/admin/*")
    app.include_router(router)
