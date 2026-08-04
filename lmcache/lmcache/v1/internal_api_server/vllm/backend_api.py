# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _get_storage_manager(request: Request):
    """Extract storage_manager from the request app state.

    Returns:
        The StorageManager instance, or None if unavailable.
    """
    engine = getattr(
        request.app.state.lmcache_adapter,
        "lmcache_engine",
        None,
    )
    if engine is None:
        return None
    return getattr(engine, "storage_manager", None)


def _unavailable_response(endpoint: str) -> JSONResponse:
    """Return a 503 response when the engine is unavailable."""
    return JSONResponse(
        content={
            "error": "%s API is unavailable" % endpoint,
            "message": "LMCache engine not configured.",
        },
        status_code=503,
    )


@router.get("/backends")
async def list_backends(request: Request):
    """List all active storage backends.

    Returns:
        JSONResponse: JSON dict mapping backend name
        to class name.

    Example:
        .. code-block:: bash

            curl http://localhost:7000/backends
    """
    sm = _get_storage_manager(request)
    if sm is None:
        return _unavailable_response("/backends")

    try:
        backends = sm.list_backends()
        return JSONResponse(content=backends)
    except Exception as e:
        logger.exception("Failed to list backends")
        return JSONResponse(
            content={
                "error": "Failed to list backends",
                "message": str(e),
            },
            status_code=500,
        )


@router.delete("/backends/{backend_name}")
async def close_backend(request: Request, backend_name: str):
    """Close and remove a specific storage backend.

    The backend will be closed and removed from the internal
    dict so that no stale references remain.

    Args:
        backend_name: Name of the backend to close
            (e.g. ``RemoteBackend``, ``LocalDiskBackend``).

    Returns:
        JSONResponse: Status of the close operation.

    Example:
        .. code-block:: bash

            curl -X DELETE \\
                http://localhost:7000/backends/RemoteBackend
    """
    sm = _get_storage_manager(request)
    if sm is None:
        return _unavailable_response("/backends")

    try:
        success = sm.close_backend(backend_name)
        if success:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Backend %s closed" % backend_name,
                    "backends": sm.list_backends(),
                },
            )
        return JSONResponse(
            content={
                "status": "not_found",
                "message": "Backend %s not found" % backend_name,
                "backends": sm.list_backends(),
            },
            status_code=404,
        )
    except Exception as e:
        logger.exception("Failed to close backend: %s", backend_name)
        return JSONResponse(
            content={
                "error": "Failed to close backend",
                "message": str(e),
            },
            status_code=500,
        )


@router.post("/backends")
async def create_backends(request: Request):
    """Create new storage backends from current config.

    Backends that are already present will be skipped.
    This allows a workflow of:

    1. ``DELETE /backends/RemoteBackend`` — close old backend
    2. ``POST /conf`` — update config (e.g. ``remote_url``)
    3. ``POST /backends`` — create new backends from config

    Returns:
        JSONResponse: Dict of newly created backends and the
        full list of active backends.

    Example:
        .. code-block:: bash

            curl -X POST http://localhost:7000/backends
    """
    sm = _get_storage_manager(request)
    if sm is None:
        return _unavailable_response("/backends")

    try:
        created = sm.create_backends()
        return JSONResponse(
            content={
                "status": "success",
                "created": created,
                "backends": sm.list_backends(),
            },
        )
    except Exception as e:
        logger.exception("Failed to create backends")
        return JSONResponse(
            content={
                "error": "Failed to create backends",
                "message": str(e),
            },
            status_code=500,
        )


@router.post("/backends/{backend_name}/recreate")
async def recreate_backend(request: Request, backend_name: str):
    """Recreate a specific storage backend atomically.

    The backend will be closed and recreated from the current
    config in a single step.

    Args:
        backend_name: Name of the backend to recreate
            (e.g. ``RemoteBackend``).

    Returns:
        JSONResponse: Dict of recreated backends and the
        full list of active backends.

    Example:
        .. code-block:: bash

            curl -X POST \\
                http://localhost:7000/backends/RemoteBackend/recreate
    """
    sm = _get_storage_manager(request)
    if sm is None:
        return _unavailable_response("/backends")

    try:
        created = sm.recreate_backend(backend_name)
        return JSONResponse(
            content={
                "status": "success",
                "recreated": created,
                "backends": sm.list_backends(),
            },
        )
    except KeyError:
        return JSONResponse(
            content={
                "status": "not_found",
                "message": ("Backend %s not found" % backend_name),
                "backends": sm.list_backends(),
            },
            status_code=404,
        )
    except Exception as e:
        logger.exception("Failed to recreate backend: %s", backend_name)
        return JSONResponse(
            content={
                "error": "Failed to recreate backend",
                "message": str(e),
            },
            status_code=500,
        )
