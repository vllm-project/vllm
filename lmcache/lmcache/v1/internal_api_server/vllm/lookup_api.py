# SPDX-License-Identifier: Apache-2.0
# Standard
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


def _json_response(data: dict, status_code: int = 200) -> PlainTextResponse:
    """Helper to create JSON response."""
    return PlainTextResponse(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        status_code=status_code,
    )


def _get_role(adapter) -> str:
    """Get the role from the manager's engine metadata.

    Returns "scheduler", "worker", or "unknown".
    """
    metadata = getattr(adapter, "lmcache_engine_metadata", None)
    if metadata is not None:
        role = getattr(metadata, "role", None)
        if role is not None:
            return role
    return "unknown"


@router.get("/lookup/info")
async def get_lookup_info(request: Request):
    """
    Get information about the current lookup client and server.

    Example:
        curl http://localhost:6999/lookup/info
    """
    adapter = request.app.state.lmcache_adapter
    if not hasattr(adapter, "get_lookup_info"):
        return _json_response({"error": "API unavailable"}, 503)
    return _json_response(adapter.get_lookup_info())


@router.post("/lookup/close")
async def close_lookup(request: Request):
    """
    Close the current lookup client (scheduler) or server (worker).

    Example:
        curl -X POST http://localhost:6999/lookup/close
    """
    adapter = request.app.state.lmcache_adapter
    role = _get_role(adapter)

    if role == "scheduler":
        if not hasattr(adapter, "close_lookup_client"):
            return _json_response({"error": "API unavailable"}, 503)
        result = adapter.close_lookup_client()
    elif role == "worker":
        if not hasattr(adapter, "close_lookup_server"):
            return _json_response({"error": "API unavailable"}, 503)
        result = adapter.close_lookup_server()
    else:
        return _json_response({"error": "Unknown role"}, 400)

    result["role"] = role
    return _json_response(result)


@router.post("/lookup/create")
async def create_lookup(request: Request, dryrun: bool = False):
    """
    Create a new lookup client (scheduler) or server (worker).

    Args:
        dryrun: If true, only show what would be created without creating it.

    Example:
        # Actually create
        curl -X POST http://localhost:6999/lookup/create

        # Dryrun - show what would be created
        curl -X POST "http://localhost:6999/lookup/create?dryrun=true"
    """
    adapter = request.app.state.lmcache_adapter
    role = _get_role(adapter)

    if role == "scheduler":
        if not hasattr(adapter, "create_lookup_client"):
            return _json_response({"error": "API unavailable"}, 503)
        result = adapter.create_lookup_client(dryrun=dryrun)
    elif role == "worker":
        if not hasattr(adapter, "create_lookup_server"):
            return _json_response({"error": "API unavailable"}, 503)
        result = adapter.create_lookup_server(dryrun=dryrun)
    else:
        return _json_response({"error": "Unknown role"}, 400)

    result["role"] = role
    if "error" in result:
        return _json_response(result, 400)
    return _json_response(result)


@router.post("/lookup/recreate")
async def recreate_lookup(request: Request):
    """
    Recreate the lookup client (scheduler) or server (worker).

    This is equivalent to calling /lookup/close + /lookup/create.

    IMPORTANT: Update configuration via /conf API before calling this.

    Example:
        # Step 1: Update config
        curl -X PUT http://localhost:6999/conf \\
        curl -X POST http://localhost:6999/conf \\
            -d '{"enable_scheduler_bypass_lookup": true}'

        # Step 2: Recreate
        curl -X POST http://localhost:6999/lookup/recreate
    """
    adapter = request.app.state.lmcache_adapter
    role = _get_role(adapter)

    if role == "scheduler":
        if not hasattr(adapter, "recreate_lookup_client"):
            return _json_response({"error": "API unavailable"}, 503)
        result = adapter.recreate_lookup_client()
    elif role == "worker":
        if not hasattr(adapter, "recreate_lookup_server"):
            return _json_response({"error": "API unavailable"}, 503)
        result = adapter.recreate_lookup_server()
    else:
        return _json_response({"error": "Unknown role"}, 400)

    result["role"] = role
    if "error" in result:
        return _json_response(result, 400)
    return _json_response(result)
