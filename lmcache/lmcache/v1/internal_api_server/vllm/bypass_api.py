# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

router = APIRouter()


def _get_storage_manager(request: Request):
    """
    Get the StorageManager from the request's app state.

    Returns:
        Tuple of (storage_manager, error_response).
        If storage_manager is None, error_response is set.
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
    if not lmcache_engine:
        error_info = {
            "error": "Bypass API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return None, JSONResponse(
            content=error_info,
            status_code=503,
        )
    sm = lmcache_engine.storage_manager
    if sm is None:
        error_info = {
            "error": "Bypass API is unavailable",
            "message": "StorageManager not initialized.",
        }
        return None, JSONResponse(
            content=error_info,
            status_code=503,
        )
    return sm, None


@router.get("/bypass/list")
async def list_bypassed_backends(request: Request):
    """
    List all currently bypassed backends and all available
    backend names.

    Args:
        request: The FastAPI request object.

    Returns:
        JSONResponse: JSON with bypassed and all backends.
    """
    try:
        sm, err = _get_storage_manager(request)
        if err:
            return err

        result = {
            "status": "success",
            "bypassed_backends": sm.get_bypassed_backends(),
            "all_backends": sm.get_all_backend_names(),
        }
        return JSONResponse(content=result)
    except Exception as e:
        error_info = {
            "error": "Failed to list bypassed backends",
            "message": str(e),
        }
        return JSONResponse(
            content=error_info,
            status_code=500,
        )


@router.put("/bypass/add")
async def add_bypass_backend(request: Request):
    """
    Add a backend to the bypass list.

    Query Parameters:
        backend_name: The name of the backend to bypass.

    Args:
        request: The FastAPI request object.

    Returns:
        JSONResponse: JSON with operation result.
    """
    try:
        sm, err = _get_storage_manager(request)
        if err:
            return err

        backend_name = request.query_params.get("backend_name", "")
        if not backend_name:
            error_info = {
                "error": "Missing parameter",
                "message": "backend_name is required",
            }
            return JSONResponse(
                content=error_info,
                status_code=400,
            )

        all_names = sm.get_all_backend_names()
        if backend_name not in all_names:
            error_info = {
                "error": "Unknown backend",
                "message": "Backend '%s' not found. "
                "Available: %s" % (backend_name, all_names),
            }
            return JSONResponse(
                content=error_info,
                status_code=400,
            )

        already = sm.is_backend_bypassed(backend_name)
        if not already:
            sm.set_backend_bypass(backend_name, True)

        result = {
            "status": "success",
            "backend_name": backend_name,
            "bypassed": True,
            "was_already_bypassed": already,
            "bypassed_backends": sm.get_bypassed_backends(),
        }
        return JSONResponse(content=result)
    except Exception as e:
        error_info = {
            "error": "Failed to add bypass backend",
            "message": str(e),
        }
        return JSONResponse(
            content=error_info,
            status_code=500,
        )


@router.put("/bypass/remove")
async def remove_bypass_backend(request: Request):
    """
    Remove a backend from the bypass list, restoring it
    to normal operation.

    Query Parameters:
        backend_name: The backend name to restore.

    Args:
        request: The FastAPI request object.

    Returns:
        JSONResponse: JSON with operation result.
    """
    try:
        sm, err = _get_storage_manager(request)
        if err:
            return err

        backend_name = request.query_params.get("backend_name", "")
        if not backend_name:
            error_info = {
                "error": "Missing parameter",
                "message": "backend_name is required",
            }
            return JSONResponse(
                content=error_info,
                status_code=400,
            )

        all_names = sm.get_all_backend_names()
        if backend_name not in all_names:
            error_info = {
                "error": "Unknown backend",
                "message": "Backend '%s' not found. "
                "Available: %s" % (backend_name, all_names),
            }
            return JSONResponse(
                content=error_info,
                status_code=400,
            )

        was_bypassed = sm.is_backend_bypassed(backend_name)
        if was_bypassed:
            sm.set_backend_bypass(backend_name, False)

        result = {
            "status": "success",
            "backend_name": backend_name,
            "bypassed": False,
            "was_bypassed": was_bypassed,
            "bypassed_backends": sm.get_bypassed_backends(),
        }
        return JSONResponse(content=result)
    except Exception as e:
        error_info = {
            "error": "Failed to remove bypass backend",
            "message": str(e),
        }
        return JSONResponse(
            content=error_info,
            status_code=500,
        )
