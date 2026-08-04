# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple, Union
import json
import time

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)

router = APIRouter()


def _create_json_response(data: dict, status_code: int = 200) -> PlainTextResponse:
    return PlainTextResponse(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        status_code=status_code,
    )


def _get_lookup_client(
    request: Request,
) -> Tuple[
    Optional[Union[LookupClientInterface, ChunkStatisticsLookupClient]],
    Optional[PlainTextResponse],
]:
    lookup_client = getattr(request.app.state.lmcache_adapter, "lookup_client", None)
    if not lookup_client:
        return None, _create_json_response(
            {"error": "API unavailable", "message": "Lookup client not configured."},
            status_code=503,
        )
    return lookup_client, None


def _get_statistics_client(
    request: Request,
) -> Tuple[Optional[ChunkStatisticsLookupClient], Optional[PlainTextResponse]]:
    lookup_client, error_response = _get_lookup_client(request)
    if error_response:
        return None, error_response
    if not isinstance(lookup_client, ChunkStatisticsLookupClient):
        return None, _create_json_response(
            {
                "error": "Not available",
                "message": "Client does not support statistics.",
            },
            status_code=400,
        )
    return lookup_client, None


def _handle_exception(operation: str, error: Exception) -> PlainTextResponse:
    return _create_json_response(
        {"error": f"Failed to {operation}", "message": str(error)}, status_code=500
    )


@router.post("/chunk_statistics/start")
async def start_chunk_statistics(request: Request):
    try:
        lookup_client, error_response = _get_lookup_client(request)
        if error_response:
            return error_response
        assert lookup_client is not None
        if isinstance(lookup_client, ChunkStatisticsLookupClient):
            lookup_client.start_statistics()
            return _create_json_response({"status": "success", "message": "Started"})
        return _create_json_response(
            {
                "error": "Not available",
                "message": "Client does not support statistics.",
            },
            status_code=400,
        )
    except Exception as e:
        return _handle_exception("start", e)


@router.post("/chunk_statistics/stop")
async def stop_chunk_statistics(request: Request):
    try:
        stats_client, error_response = _get_statistics_client(request)
        if error_response:
            return error_response
        assert stats_client is not None
        stats_client.stop_statistics()
        return _create_json_response({"status": "success", "message": "Stopped"})
    except Exception as e:
        return _handle_exception("stop", e)


@router.post("/chunk_statistics/reset")
async def reset_chunk_statistics(request: Request):
    try:
        stats_client, error_response = _get_statistics_client(request)
        if error_response:
            return error_response
        assert stats_client is not None
        stats_client.reset_statistics()
        return _create_json_response({"status": "success", "message": "Reset"})
    except Exception as e:
        return _handle_exception("reset", e)


@router.get("/chunk_statistics/status")
async def get_chunk_statistics_status(request: Request):
    try:
        stats_client, error_response = _get_statistics_client(request)
        if error_response:
            return error_response
        assert stats_client is not None
        config = request.app.state.lmcache_adapter.config
        stats = stats_client.get_statistics()
        stats.update(
            {
                "timestamp": time.time(),
                "auto_exit_enabled": (
                    config.chunk_statistics_auto_exit_timeout_hours > 0.0
                    or config.chunk_statistics_auto_exit_target_unique_chunks
                    is not None
                ),
                "auto_exit_timeout_hours": (
                    config.chunk_statistics_auto_exit_timeout_hours
                ),
                "auto_exit_target_unique_chunks": (
                    config.chunk_statistics_auto_exit_target_unique_chunks
                ),
            }
        )
        return _create_json_response(stats)
    except Exception as e:
        return _handle_exception("get status", e)
