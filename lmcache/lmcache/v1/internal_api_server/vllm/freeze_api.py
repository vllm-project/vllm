# SPDX-License-Identifier: Apache-2.0
# Standard
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.put("/freeze/enable")
async def enable_freeze(request: Request):
    """
    Enable freeze mode for the LMCache engine.

    When freeze mode is enabled:
    - All store operations will be skipped (no new data stored)
    - Only local_cpu backend will be used for retrieval
    - No admit/evict messages will be generated
    This protects the local_cpu hot cache from changes.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A JSON response indicating the operation status.

    Example:
        ```bash
        curl -X PUT "http://localhost:8000/freeze/enable"
        # Response: {"status": "success", "freeze": true}
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if not lmcache_engine:
            error_info = {
                "error": "/freeze/enable API is unavailable",
                "message": "LMCache engine not configured.",
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=503,  # Service Unavailable
            )

        lmcache_engine.freeze(True)
        success_info = {
            "status": "success",
            "freeze": True,
            "message": "Freeze mode enabled successfully",
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_msg = "Failed to enable freeze mode"
        error_info = {"error": error_msg, "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )


@router.put("/freeze/disable")
async def disable_freeze(request: Request):
    """
    Disable freeze mode for the LMCache engine.

    When freeze mode is disabled, store operations will proceed normally.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A JSON response indicating the operation status.

    Example:
        ```bash
        curl -X PUT "http://localhost:8000/freeze/disable"
        # Response: {"status": "success", "freeze": false}
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if not lmcache_engine:
            error_info = {
                "error": "/freeze/disable API is unavailable",
                "message": "LMCache engine not configured.",
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=503,  # Service Unavailable
            )

        lmcache_engine.freeze(False)
        success_info = {
            "status": "success",
            "freeze": False,
            "message": "Freeze mode disabled successfully",
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_msg = "Failed to disable freeze mode"
        error_info = {"error": error_msg, "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )


@router.get("/freeze/status")
async def get_freeze_status(request: Request):
    """
    Get the current freeze mode status of the LMCache engine.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: JSON response with current freeze mode status.

    Example:
        ```bash
        curl -X GET "http://localhost:8000/freeze/status"
        # Response: {"status": "success", "freeze": true}
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if not lmcache_engine:
            error_info = {
                "error": "/freeze/status API is unavailable",
                "message": "LMCache engine not configured.",
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=503,  # Service Unavailable
            )

        freeze_mode = lmcache_engine.is_frozen()
        mode_str = "enabled" if freeze_mode else "disabled"
        success_info = {
            "status": "success",
            "freeze": freeze_mode,
            "message": "Freeze mode is " + mode_str,
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_msg = "Failed to get freeze mode status"
        error_info = {"error": error_msg, "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )
