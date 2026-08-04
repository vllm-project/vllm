# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.get("/inference_info")
async def get_inference_info(request: Request, format: Optional[str] = None):
    """
    Get inference information including vLLM config and LMCache details

    Args:
        format: Optional format parameter (currently unused, for future extension)

    Returns:
        PlainTextResponse: JSON string containing inference information
    """
    lmcache_adapter = request.app.state.lmcache_adapter

    try:
        inference_info = lmcache_adapter.get_inference_info()
        return PlainTextResponse(
            content=json.dumps(inference_info, indent=2, default=str),
            media_type="application/json",
        )
    except Exception as e:
        error_info = {"error": "Failed to get inference info", "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )


@router.get("/inference_version")
async def get_inference_version(request: Request):
    """
    Get vLLM version information

    Returns:
        PlainTextResponse: vLLM version string
    """
    lmcache_adapter = request.app.state.lmcache_adapter

    try:
        version_info = lmcache_adapter.get_inference_version()
        version_response = {"vllm_version": version_info}
        return PlainTextResponse(
            content=json.dumps(version_response, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_info = {"error": "Failed to get inference version", "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )
