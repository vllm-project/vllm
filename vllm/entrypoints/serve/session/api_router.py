# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel

from vllm.entrypoints.serve.utils.api_utils import validate_json_request

router = APIRouter()


class CloseSessionRequest(BaseModel):
    session_id: str


@router.post("/close_session", dependencies=[Depends(validate_json_request)])
async def close_session(request: CloseSessionRequest, raw_request: Request):
    """Stop preferring a session's cached blocks for retention.

    The blocks stay in the prefix cache until evicted as usual. Unknown
    sessions are ignored, as is the call when `--session-cache-ttl` is unset.
    """
    await raw_request.app.state.engine_client.close_session(request.session_id)
    return Response(status_code=200)


def attach_router(app: FastAPI):
    app.include_router(router)
