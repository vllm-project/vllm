from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.config.tensor_dump import TensorDumpOptions

router = APIRouter()


@router.post("/dumper/{action}")
async def control_dumper(
    action: Literal["configure", "reset", "get_state"],
    request: Request,
    body: TensorDumpOptions | None = None,
):
    if action == "configure" and body is None:
        raise HTTPException(
            status_code=422, detail="configure requires a configuration"
        )
    results = await request.app.state.engine_client.collective_rpc(
        "dumper_control", args=(action, asdict(body) if body is not None else None)
    )
    return JSONResponse(content=results)


def attach_router(app: FastAPI):
    config = getattr(app.state.args, "tensor_dump_config", None)
    if config is not None and config.output_dir is not None:
        app.include_router(router)
