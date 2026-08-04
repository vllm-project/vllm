# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import is_dataclass
from typing import Any
import json

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.multiprocess.http_apis.dependencies import get_context
from lmcache.v1.utils.json_utils import make_json_safe, safe_asdict

router = APIRouter()


class _IndentedJSONResponse(JSONResponse):
    """JSONResponse with indented output for readability."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode("utf-8")


@router.get("/config")
async def config(request: Request) -> Any:
    """
    Return all server configurations (mp, storage_manager,
    observability) as a single JSON object.

    Args:
        request (Request): The incoming HTTP request; its
            ``app.state.configs`` mapping is serialized.

    Returns:
        Any: A JSON response whose body is a dict keyed by
        config name. Returns HTTP 503 if ``configs`` is not
        initialized yet.

    Exceptions:
        None.
    """
    configs = getattr(request.app.state, "configs", None)
    if configs is None:
        return JSONResponse(
            status_code=503,
            content={"error": "configs not initialized"},
        )
    result = {}
    for name, cfg in configs.items():
        if is_dataclass(cfg) and not isinstance(cfg, type):
            result[name] = safe_asdict(cfg)
        else:
            result[name] = make_json_safe(cfg)
    return _IndentedJSONResponse(content=result)


@router.get("/config/adapters", response_model=None)
async def list_adapters(request: Request) -> dict[str, object]:
    """Enumerate the live cache adapters.

    Listing which storage adapters the engine loaded is a configuration
    inspection concern, so it lives under the ``/config`` group rather than
    ``/cache``. This is the single live adapter listing -- it supersedes the
    removed ``/reconfigure/backends`` (the reconfigurable backends are the
    ``type_name`` values whose ``reconfigurable`` flag is ``true``). It is a thin
    pass-through to :meth:`ObjectService.list_adapters`.

    Args:
        request (Request): The incoming HTTP request; its per-app
            :class:`MPHTTPContext` is resolved via :func:`get_context`.

    Returns:
        dict[str, object]: ``{"adapters": [{"index", "type_name", "tier",
        "primary", "reconfigurable"}, ...]}``. Pass a ``reconfigurable``
        adapter's ``type_name`` as the ``{backend}`` path parameter to
        ``GET /reconfigure/{backend}/status`` and the reconfigure operations.

    Raises:
        HTTPException: ``503`` if the engine context is not initialized yet.
    """
    return get_context(request).object_service.list_adapters()
