# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/collective_rpc")
async def collective_rpc(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    method = body.get("method")
    if method is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'method' in request body",
        )
    # For security reason, only serialized string args/kwargs are passed.
    # User-defined `method` is responsible for deserialization if needed.
    args: list[str] = body.get("args", [])
    kwargs: dict[str, str] = body.get("kwargs", {})
    timeout: float | None = body.get("timeout")
    results = await engine_client(raw_request).collective_rpc(
        method=method, timeout=timeout, args=tuple(args), kwargs=kwargs
    )
    if results is None:
        return Response(status_code=200)
    response: list[Any] = []
    for result in results:
        if result is None or isinstance(result, dict | list):
            response.append(result)
        else:
            response.append(str(result))
    return JSONResponse(content={"results": response})


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
