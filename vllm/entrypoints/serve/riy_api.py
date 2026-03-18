# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RIY Admin API: Expert statistics and mask management.

Endpoints:
    GET  /riy/stats          - Export raw expert statistics
    POST /riy/stats/start    - Start stats collection
    POST /riy/stats/stop     - Stop stats collection
    POST /riy/stats/reset    - Reset all counters
    GET  /riy/mask           - Get current expert mask
    POST /riy/mask           - Set expert mask
    DELETE /riy/mask         - Clear expert mask
    POST /riy/profile/load   - Load mask from profile JSON on disk
"""

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from vllm.model_executor.layers.fused_moe.riy import get_riy_state

router = APIRouter(prefix="/riy", tags=["riy"])


def attach_router(app: FastAPI):
    app.include_router(router)


class MaskRequest(BaseModel):
    pruned_experts: list[list[int]]


class ProfileLoadRequest(BaseModel):
    path: str


@router.get("/stats")
async def get_stats():
    riy = get_riy_state()
    if not riy.enabled:
        raise HTTPException(status_code=503,
                            detail="RIY not initialized (no MoE model loaded)")
    return riy.get_stats()


@router.post("/stats/start")
async def start_collection():
    riy = get_riy_state()
    if not riy.enabled:
        raise HTTPException(status_code=503,
                            detail="RIY not initialized")
    riy.start_collection()
    return {"status": "collecting"}


@router.post("/stats/stop")
async def stop_collection():
    riy = get_riy_state()
    riy.stop_collection()
    return {"status": "stopped"}


@router.post("/stats/reset")
async def reset_stats():
    riy = get_riy_state()
    riy.reset_stats()
    return {"status": "reset"}


@router.get("/mask")
async def get_mask():
    riy = get_riy_state()
    return {
        "pruned_experts": riy.get_mask(),
        "count": len(riy.get_mask()),
    }


@router.post("/mask")
async def set_mask(req: MaskRequest):
    riy = get_riy_state()
    if not riy.enabled:
        raise HTTPException(status_code=503,
                            detail="RIY not initialized")
    experts = [tuple(x) for x in req.pruned_experts]
    riy.set_mask(experts)
    return {"status": "mask_set", "count": len(experts)}


@router.delete("/mask")
async def clear_mask():
    riy = get_riy_state()
    riy.clear_mask()
    return {"status": "mask_cleared"}


@router.post("/profile/load")
async def load_profile(req: ProfileLoadRequest):
    riy = get_riy_state()
    if not riy.enabled:
        raise HTTPException(status_code=503,
                            detail="RIY not initialized")
    try:
        riy.load_profile(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "status": "profile_loaded",
        "path": req.path,
        "count": len(riy.get_mask()),
    }
