# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Optional, Union
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import _CONFIG_DEFINITIONS, LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value

logger = init_logger(__name__)

router = APIRouter()


def _is_mutable_config(key: str) -> bool:
    """Check if a config key is mutable at runtime.

    NOTE: This is currently experimental. All configs default to
    mutable=True unless explicitly set to False. Once the feature
    is stabilized, the default will be changed to mutable=False.
    """
    return _CONFIG_DEFINITIONS.get(key, {}).get("mutable", True)


def _get_config_dict(
    config: LMCacheEngineConfig,
    keys: Union[list[str], Optional[dict[str, dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Get the config dict filtered by keys"""
    if keys is None:
        keys = _CONFIG_DEFINITIONS
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}


@router.get("/conf")
async def get_config(request: Request, names: Optional[str] = None):
    config = request.app.state.lmcache_adapter.config
    # Parse query parameter names (comma-separated list of config names)
    keys = names.split(",") if names else None
    config_dict = _get_config_dict(config, keys)
    return PlainTextResponse(
        content=json.dumps(config_dict, indent=2), media_type="text/plain"
    )


@router.get("/meta")
async def get_metadata(request: Request, names: Optional[str] = None):
    """
    Get metadata of the cache engine
    """
    metadata = request.app.state.lmcache_adapter.lmcache_engine_metadata

    if names:
        attr_list = names.split(",")
        metadata_dict = {}
        for attr in attr_list:
            if (
                hasattr(metadata, attr)
                and not attr.startswith("__")
                and not callable(getattr(metadata, attr))
            ):
                value = getattr(metadata, attr)
                if isinstance(value, (torch.dtype, torch.Size)):
                    value = str(value)
                metadata_dict[attr] = value
    else:
        metadata_dict = {
            attr: getattr(metadata, attr)
            for attr in dir(metadata)
            if not attr.startswith("__") and not callable(getattr(metadata, attr))
        }
        for key, value in metadata_dict.items():
            if isinstance(value, (torch.dtype, torch.Size)):
                metadata_dict[key] = str(value)

    return PlainTextResponse(
        content=json.dumps(metadata_dict, indent=2, default=str),
        media_type="text/plain",
    )


@router.post("/conf")
async def set_config(request: Request):
    """Set config values dynamically.

    NOTE: Currently experimental — all configs are mutable at runtime
    by default unless explicitly set "mutable": False in
    _CONFIG_DEFINITIONS. The default will change to immutable once
    the feature is stabilized.

    Request body should be JSON with config name-value pairs:
        {"min_retrieve_tokens": 512, "save_decode_cache": true}

    Returns:
        PlainTextResponse: JSON response with updated config values
    """
    config = request.app.state.lmcache_adapter.config

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            content={"error": "Invalid JSON body", "message": str(e)},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            content={"error": "Request body must be a JSON object"},
            status_code=400,
        )

    updated: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for key, value in body.items():
        if not _is_mutable_config(key):
            errors[key] = "Config is not mutable at runtime"
            continue

        if key not in _CONFIG_DEFINITIONS:
            errors[key] = "Unknown config"
            continue

        # Use shared validation and setting function
        if validate_and_set_config_value(config, key, value):
            updated[key] = getattr(config, key)
            logger.info("Config %s updated to %s", key, updated[key])
        else:
            errors[key] = "Failed to set config value"

    result: Dict[str, Any] = {"updated": updated}
    if errors:
        result["errors"] = errors

    status_code = 400 if errors and not updated else 200
    return JSONResponse(
        content=result,
        status_code=status_code,
    )
