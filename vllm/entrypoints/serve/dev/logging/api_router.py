# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger, parse_log_level

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/set_log_level")
async def set_log_level(
    raw_request: Request,
    level: Annotated[str, Query()],
    logger_name: Annotated[str | None, Query(alias="logger")] = None,
) -> JSONResponse:
    """Set a logger level in the receiving API process and its engines/workers.

    Updates this frontend process, every EngineCore managed by this client,
    and every worker of those EngineCores. With ``--api-server-count > 1``,
    peer API-server frontends are not notified.

    Omit ``logger`` (or pass ``root`` / ``vllm``) to update the ``vllm`` root
    logger and its handlers. Pass a module logger name to update only that
    logger. This route is always registered and does not require
    ``VLLM_SERVER_DEV_MODE``.
    """
    try:
        parse_log_level(level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    applied = await engine_client(raw_request).set_log_level(level, logger_name)
    logger.info(
        "Set log level to %s for logger %s", applied["level"], applied["logger"]
    )
    return JSONResponse(
        content={
            "status": "ok",
            "level": applied["level"],
            "logger": applied["logger"],
        }
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
