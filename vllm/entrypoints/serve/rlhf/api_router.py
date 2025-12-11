# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/pause")
async def pause_generation(
    raw_request: Request,
    wait_for_inflight_requests: bool = Query(False),
    clear_cache: bool = Query(True),
) -> JSONResponse:
    """Pause generation requests to allow weight updates.

    Args:
        wait_for_inflight_requests: When ``True`` waits for in-flight
            requests to finish before pausing. When ``False`` (default),
            aborts any in-flight requests immediately.
        clear_cache: Whether to clear KV/prefix caches after draining.
    """

    engine = engine_client(raw_request)

    try:
        await engine.pause_generation(
            wait_for_inflight_requests=wait_for_inflight_requests,
            clear_cache=clear_cache,
        )
        return JSONResponse(
            content={"status": "paused"},
            status_code=HTTPStatus.OK.value,
        )

    except ValueError as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to pause generation")
        return JSONResponse(
            content={"error": f"Failed to pause generation: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/resume")
async def resume_generation(raw_request: Request) -> JSONResponse:
    """Resume generation after a pause."""

    engine = engine_client(raw_request)

    try:
        await engine.resume_generation()
        return JSONResponse(
            content={"status": "resumed"},
            status_code=HTTPStatus.OK.value,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to resume generation")
        return JSONResponse(
            content={"error": f"Failed to resume generation: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.get("/is_paused")
async def is_paused(raw_request: Request) -> JSONResponse:
    """Return the current pause status."""

    engine = engine_client(raw_request)

    try:
        paused = await engine.is_paused()
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to fetch pause status")
        return JSONResponse(
            content={"error": f"Failed to fetch pause status: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    return JSONResponse(content={"is_paused": paused})


def attach_router(app: FastAPI):
    app.include_router(router)
