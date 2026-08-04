# SPDX-License-Identifier: Apache-2.0
# Standard
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


def _get_engine(request: Request):
    """
    Extract the LMCache engine from the request state.

    Returns:
        A tuple of (engine, error_response). If engine is None,
        error_response contains a PlainTextResponse to return.
    """
    adapter = request.app.state.lmcache_adapter
    engine = getattr(adapter, "lmcache_engine", None)
    if not engine:
        error_info = {
            "error": "hot_cache API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return None, PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=503,
        )
    return engine, None


@router.put("/hot_cache/enable")
async def enable_hot_cache(request: Request):
    """
    Enable hot cache for the LocalCPUBackend.

    When hot cache is enabled, frequently accessed KV cache
    data will be kept in CPU memory for faster retrieval.

    Args:
        request: The FastAPI request object.

    Returns:
        PlainTextResponse: JSON response with operation status.

    Example:
        ```bash
        curl -X PUT "http://localhost:8000/hot_cache/enable"
        # Response: {"status": "success", "hot_cache": true}
        ```
    """
    try:
        engine, err = _get_engine(request)
        if err:
            return err

        engine.set_hot_cache(True)
        return PlainTextResponse(
            content=json.dumps(
                {
                    "status": "success",
                    "hot_cache": True,
                    "message": "Hot cache enabled successfully",
                },
                indent=2,
            ),
            media_type="application/json",
        )
    except Exception as e:
        return PlainTextResponse(
            content=json.dumps(
                {
                    "error": "Failed to enable hot cache",
                    "message": str(e),
                },
                indent=2,
            ),
            media_type="application/json",
            status_code=500,
        )


@router.put("/hot_cache/disable")
async def disable_hot_cache(request: Request):
    """
    Disable hot cache for the LocalCPUBackend.

    When hot cache is disabled, existing hot cache entries
    will be cleared and no new data will be written.

    Args:
        request: The FastAPI request object.

    Returns:
        PlainTextResponse: JSON response with operation status.

    Example:
        ```bash
        curl -X PUT "http://localhost:8000/hot_cache/disable"
        # Response: {"status": "success", "hot_cache": false}
        ```
    """
    try:
        engine, err = _get_engine(request)
        if err:
            return err

        engine.set_hot_cache(False)
        return PlainTextResponse(
            content=json.dumps(
                {
                    "status": "success",
                    "hot_cache": False,
                    "message": ("Hot cache disabled successfully"),
                },
                indent=2,
            ),
            media_type="application/json",
        )
    except Exception as e:
        return PlainTextResponse(
            content=json.dumps(
                {
                    "error": "Failed to disable hot cache",
                    "message": str(e),
                },
                indent=2,
            ),
            media_type="application/json",
            status_code=500,
        )


@router.get("/hot_cache/status")
async def get_hot_cache_status(request: Request):
    """
    Get the current hot cache status of LocalCPUBackend.

    Args:
        request: The FastAPI request object.

    Returns:
        PlainTextResponse: JSON response with hot cache status.

    Example:
        ```bash
        curl -X GET "http://localhost:8000/hot_cache/status"
        # Response: {"status": "success", "hot_cache": true}
        ```
    """
    try:
        engine, err = _get_engine(request)
        if err:
            return err

        enabled = engine.is_hot_cache_enabled()
        mode_str = "enabled" if enabled else "disabled"
        return PlainTextResponse(
            content=json.dumps(
                {
                    "status": "success",
                    "hot_cache": enabled,
                    "message": "Hot cache is " + mode_str,
                },
                indent=2,
            ),
            media_type="application/json",
        )
    except Exception as e:
        return PlainTextResponse(
            content=json.dumps(
                {
                    "error": "Failed to get hot cache status",
                    "message": str(e),
                },
                indent=2,
            ),
            media_type="application/json",
            status_code=500,
        )
