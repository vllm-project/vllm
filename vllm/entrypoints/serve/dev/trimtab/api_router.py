# SPDX-License-Identifier: Apache-2.0
# trimtab (github.com/numinous-technology/trimtab) dev router.

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _core(request: Request):
    return request.app.state.engine_client.engine_core


@router.post("/trimtab/set_knobs")
async def trimtab_set_knobs(raw_request: Request):
    knobs = await raw_request.json()
    result = await _core(raw_request).call_utility_async("trimtab_set_knobs", knobs)
    return JSONResponse(content=result, status_code=200 if result["ok"] else 400)


@router.post("/trimtab/reinit")
async def trimtab_reinit(raw_request: Request):
    fields = await raw_request.json()
    result = await _core(raw_request).call_utility_async("trimtab_reinit", fields)
    return JSONResponse(content=result, status_code=200 if result["ok"] else 400)


@router.get("/trimtab/knobs")
async def trimtab_get_knobs(raw_request: Request):
    return JSONResponse(content=await _core(raw_request).call_utility_async("trimtab_get_knobs"))


def attach_router(app: FastAPI):
    app.include_router(router)
