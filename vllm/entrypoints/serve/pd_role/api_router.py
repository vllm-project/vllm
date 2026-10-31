# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .state import PDRole, PDRoleState

router = APIRouter()


class PDRoleRequest(BaseModel):
    role: PDRole
    expected_epoch: int = Field(ge=0)
    drain_timeout: float = Field(default=120, gt=0, le=3600)


def role_state(request: Request) -> PDRoleState:
    state = getattr(request.app.state, "pd_role", None)
    if state is None:
        raise HTTPException(404, "Runtime P/D role switching is not enabled")
    return state


@router.get("/v1/pd_role")
async def get_pd_role(request: Request):
    return role_state(request).status()


@router.post("/v1/pd_role")
async def switch_pd_role(body: PDRoleRequest, request: Request):
    state = role_state(request)
    try:
        state.start(body.role, body.expected_epoch, body.drain_timeout)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(state.status(), status_code=202)
