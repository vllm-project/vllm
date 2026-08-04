# SPDX-License-Identifier: Apache-2.0
"""
Per-``cache_salt`` quota management endpoints.

The adapter's ``IsolatedLRU`` eviction policy uses these quotas to
scope per-bucket eviction decisions. Quotas are soft: setting a limit
does not reject writes — any over-budget bucket is evicted at the
next eviction cycle (~1s). Removing or never setting a quota leaves
the bucket at an effective limit of ``0`` bytes, so its data is
cleared next cycle (allowlist semantics).
"""

# Standard
from typing import Any
import math

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


# ``cache_salt=""`` (un-salted / anonymous traffic) cannot appear in a
# URL path parameter, so the API accepts this sentinel in its place.
#
# COLLISION CAVEAT: a user who legitimately stores data with
# ``cache_salt="_default"`` cannot be managed via this HTTP API
# distinctly from anonymous traffic — both map to the same path
# parameter. Deployments that want to manage such a user should set
# ``cache_salt`` to any other value (e.g. ``"default"`` without the
# underscore, or a tenant prefix).
_DEFAULT_SALT_SENTINEL = "_default"


def _unescape_salt(path_salt: str) -> str:
    """Translate the URL sentinel back to the empty-string salt."""
    return "" if path_salt == _DEFAULT_SALT_SENTINEL else path_salt


def _escape_salt(salt: str) -> str:
    """Translate the empty-string salt to the URL sentinel."""
    return _DEFAULT_SALT_SENTINEL if salt == "" else salt


def _gb(n_bytes: int) -> float:
    """Convert bytes → GB for the JSON payload."""
    return n_bytes / (1024**3)


def _get_storage_manager(request: Request) -> Any:
    """Resolve the shared ``StorageManager`` or return a 503 response.

    Returns either the ``StorageManager`` instance or a
    ``JSONResponse`` (503) ready to be returned directly from the
    endpoint — keeps the endpoint bodies short.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return engine.storage_manager


@router.put("/quota/{cache_salt}")
async def set_quota(cache_salt: str, request: Request) -> Any:
    """Create or update a quota for the given ``cache_salt``.

    Body: ``{"limit_gb": <float>}`` (required, non-negative).
    """
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
    if not isinstance(body, dict) or "limit_gb" not in body:
        return JSONResponse(
            status_code=400,
            content={"error": "body must be {'limit_gb': <float>}"},
        )
    try:
        limit_gb = float(body["limit_gb"])
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"error": "limit_gb must be numeric"},
        )
    if not math.isfinite(limit_gb):
        # ``nan`` / ``inf`` would propagate to an int() cast below that
        # raises OverflowError / ValueError (reported as a 500). Reject
        # here so the error surfaces as a clean 400.
        return JSONResponse(
            status_code=400,
            content={"error": "limit_gb must be finite"},
        )
    if limit_gb < 0:
        return JSONResponse(
            status_code=400,
            content={"error": "limit_gb must be non-negative"},
        )

    salt = _unescape_salt(cache_salt)
    try:
        sm.quota_manager.set_quota(salt, int(limit_gb * (1024**3)))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {
        "cache_salt": _escape_salt(salt),
        "limit_gb": limit_gb,
        "status": "ok",
    }


@router.get("/quota/{cache_salt}")
async def get_quota(cache_salt: str, request: Request) -> Any:
    """Read the current quota + live usage for a single salt."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm

    salt = _unescape_salt(cache_salt)
    exists = sm.quota_manager.has_quota(salt)
    limit_bytes = sm.quota_manager.get_limit_bytes(salt)
    usage_bytes = sm.get_usage_bytes_by_cache_salt().get(salt, 0)
    return {
        "cache_salt": _escape_salt(salt),
        "limit_gb": _gb(limit_bytes),
        "current_usage_gb": _gb(usage_bytes),
        "exists": exists,
    }


@router.delete("/quota/{cache_salt}")
async def delete_quota(cache_salt: str, request: Request) -> Any:
    """Remove a salt's quota entry.

    Any bytes still cached under this salt become over-budget on the
    next eviction cycle (effective limit drops to 0) and will be
    evicted.
    """
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm

    salt = _unescape_salt(cache_salt)
    removed = sm.quota_manager.delete_quota(salt)
    return {
        "cache_salt": _escape_salt(salt),
        "status": "removed" if removed else "not_found",
    }


@router.get("/quota")
async def list_quotas(request: Request) -> Any:
    """List every registered quota alongside its live usage."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm

    usage_map = sm.get_usage_bytes_by_cache_salt()
    users: dict[str, dict[str, float]] = {}
    for entry in sm.quota_manager.list_quotas():
        used_bytes = usage_map.get(entry.cache_salt, 0)
        users[_escape_salt(entry.cache_salt)] = {
            "limit_gb": _gb(entry.limit_bytes),
            "current_usage_gb": _gb(used_bytes),
        }
    return {"users": users}
