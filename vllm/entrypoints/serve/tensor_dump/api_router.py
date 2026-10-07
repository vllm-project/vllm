import json
from typing import Literal

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/dumper/{action}")
async def control_dumper(
    action: Literal["configure", "reset", "get_state"],
    request: Request,
    body: dict | None = None,
):
    results = await request.app.state.engine_client.collective_rpc(
        "dumper_control", args=(action, json.dumps(body or {}))
    )
    return JSONResponse(content=results)


def attach_router(app: FastAPI):
    app.include_router(router)
