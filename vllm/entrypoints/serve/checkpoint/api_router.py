# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.v1.engine import CheckpointState, PauseMode


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


def _state_response(
    state: CheckpointState, error: str | None, status_code: int
) -> JSONResponse:
    content = {"state": state} if error is None else {"state": state, "error": error}
    return JSONResponse(content=content, status_code=status_code)


@router.post("/checkpoint/prepare")
async def checkpoint_prepare(raw_request: Request, mode: PauseMode = "wait"):
    """Drain with `mode` and release what a checkpoint must not contain.
    Returns once started; poll /checkpoint/status. Repeating is a no-op."""
    client = engine_client(raw_request)
    state, error = await client.checkpoint_state()
    if state == CheckpointState.RUNNING:
        if refusal := await client.checkpoint_prepare(mode):
            return _state_response(state, refusal, 400)
        state, error = await client.checkpoint_state()
    elif state in (CheckpointState.RESUMING, CheckpointState.FAILED):
        return _state_response(state, error, 400)
    return _state_response(state, error, 200)


@router.post("/checkpoint/resume")
async def checkpoint_resume(raw_request: Request):
    """Bring back what preparing released and serve. Orchestrators may
    redeliver, so a resume in progress or done returns 200."""
    client = engine_client(raw_request)
    state, error = await client.checkpoint_state()
    if state in (CheckpointState.PREPARING, CheckpointState.FAILED):
        return _state_response(state, error, 400)
    if state == CheckpointState.PREPARED:
        await client.checkpoint_restore()
        state, error = await client.checkpoint_state()
    return _state_response(state, error, 200)


@router.get("/checkpoint/status")
async def checkpoint_status(raw_request: Request):
    # Polled by orchestrators: 200 once prepared (safe to capture), 503 until
    # then, 500 after a failure.
    state, error = await engine_client(raw_request).checkpoint_state()
    if state == CheckpointState.PREPARED:
        status_code = 200
    elif state == CheckpointState.FAILED:
        status_code = 500
    else:
        status_code = 503
    return _state_response(state, error, status_code)


def attach_router(app: FastAPI):
    app.include_router(router)
