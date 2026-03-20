# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HTTP endpoints for CRIU-safe GPU snapshot lifecycle.

These endpoints let an external orchestrator (e.g. Modal) drive the
suspend/resume cycle when vLLM is running as a subprocess:

  POST /snapshot/suspend  — tear down NCCL; weights stay on GPU
  POST /snapshot/resume   — rebuild NCCL with fresh loopback addresses

Enable with: VLLM_SERVER_DEV_MODE=1
"""

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter(prefix="/snapshot")


def _engine(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/suspend")
async def suspend(raw_request: Request) -> JSONResponse:
    """Suspend engine for a CRIU checkpoint.

    Tears down NCCL and distributed state while keeping model weights on GPU.
    Call this just before CRIU freezes the process; call /snapshot/resume
    after CRIU restores it.
    """
    logger.info("Suspending engine for CRIU snapshot")
    try:
        await _engine(raw_request).suspend()
        return JSONResponse(
            content={"status": "suspended"},
            status_code=HTTPStatus.OK.value,
        )
    except NotImplementedError:
        return JSONResponse(
            content={"error": "suspend() not supported by this engine"},
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        )
    except Exception as err:
        logger.exception("Failed to suspend engine")
        return JSONResponse(
            content={"error": f"Failed to suspend: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/resume")
async def resume(raw_request: Request) -> JSONResponse:
    """Resume engine after a CRIU snapshot restore.

    Rebuilds NCCL with fresh loopback addresses.  The HTTP server itself
    is already running (CRIU restored the entire process); only the engine
    internals need reinitialisation.
    """
    logger.info("Resuming engine after CRIU restore")
    try:
        await _engine(raw_request).resume()
        return JSONResponse(
            content={"status": "resumed"},
            status_code=HTTPStatus.OK.value,
        )
    except NotImplementedError:
        return JSONResponse(
            content={"error": "resume() not supported by this engine"},
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        )
    except Exception as err:
        logger.exception("Failed to resume engine")
        return JSONResponse(
            content={"error": f"Failed to resume: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


def attach_router(app: FastAPI) -> None:
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
