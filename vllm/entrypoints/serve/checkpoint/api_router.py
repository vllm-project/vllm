# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm.engine.protocol import EngineClient
from vllm.v1.engine import CheckpointState, PauseMode


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


class CheckpointStateResponse(BaseModel):
    state: CheckpointState
    error: str | None = None


_STATE_RESPONSE = {"model": CheckpointStateResponse}


def _state_response(
    state: CheckpointState, error: str | None, status_code: HTTPStatus
) -> JSONResponse:
    content = CheckpointStateResponse(state=state, error=error)
    return JSONResponse(
        content=content.model_dump(mode="json", exclude_none=True),
        status_code=status_code.value,
    )


@router.post(
    "/checkpoint/prepare",
    responses={
        HTTPStatus.OK.value: _STATE_RESPONSE,
        HTTPStatus.BAD_REQUEST.value: _STATE_RESPONSE,
    },
)
async def checkpoint_prepare(raw_request: Request, mode: PauseMode = "wait"):
    """Drain with `mode` and release what a checkpoint must not contain.
    Returns once started; poll /checkpoint/status. Repeating is a no-op."""
    client = engine_client(raw_request)
    state, error = await client.checkpoint_state()
    if state == CheckpointState.RUNNING:
        if refusal := await client.checkpoint_prepare(mode):
            return _state_response(state, refusal, HTTPStatus.BAD_REQUEST)
        state, error = await client.checkpoint_state()
    elif state in (CheckpointState.RESTORING, CheckpointState.FAILED):
        return _state_response(state, error, HTTPStatus.BAD_REQUEST)
    return _state_response(state, error, HTTPStatus.OK)


@router.post(
    "/checkpoint/restore",
    responses={
        HTTPStatus.OK.value: _STATE_RESPONSE,
        HTTPStatus.BAD_REQUEST.value: _STATE_RESPONSE,
    },
)
async def checkpoint_restore(raw_request: Request):
    """Bring back what preparing released and serve. Orchestrators may
    redeliver, so a restore in progress or done returns 200."""
    client = engine_client(raw_request)
    state, error = await client.checkpoint_state()
    if state in (CheckpointState.PREPARING, CheckpointState.FAILED):
        return _state_response(state, error, HTTPStatus.BAD_REQUEST)
    if state == CheckpointState.PREPARED:
        await client.checkpoint_restore()
        state, error = await client.checkpoint_state()
    return _state_response(state, error, HTTPStatus.OK)


@router.get(
    "/checkpoint/status",
    responses={
        HTTPStatus.OK.value: _STATE_RESPONSE,
        HTTPStatus.INTERNAL_SERVER_ERROR.value: _STATE_RESPONSE,
        HTTPStatus.SERVICE_UNAVAILABLE.value: _STATE_RESPONSE,
    },
)
async def checkpoint_status(raw_request: Request):
    # Polled by orchestrators: 200 once prepared (safe to capture), 503 until
    # then, 500 after a failure.
    state, error = await engine_client(raw_request).checkpoint_state()
    if state == CheckpointState.PREPARED:
        status_code = HTTPStatus.OK
    elif state == CheckpointState.FAILED:
        status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    else:
        status_code = HTTPStatus.SERVICE_UNAVAILABLE
    return _state_response(state, error, status_code)


def attach_router(app: FastAPI):
    app.include_router(router)
