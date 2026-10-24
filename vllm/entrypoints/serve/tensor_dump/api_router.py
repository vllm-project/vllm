from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.v1.worker.gpu.tensor_dump import TensorDumpConfig

router = APIRouter()


@router.post("/dumper/{action}")
async def control_dumper(
    action: Literal["configure", "reset", "get_state"],
    request: Request,
    body: TensorDumpConfig | None = None,
):
    if action == "configure" and body is None:
        raise HTTPException(
            status_code=422, detail="configure requires a configuration"
        )
    results = await request.app.state.engine_client.collective_rpc(
        "dumper_control", args=(action, body.model_dump() if body is not None else None)
    )
    return JSONResponse(content=results)


def attach_router(app: FastAPI):
    if envs.VLLM_TENSOR_DUMP_DIR:
        app.include_router(router)
