# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.get("/admin/kv_cache_budget")
async def get_kv_cache_budget(raw_request: Request):
    """
    Get current KV cache budget settings.
    """
    client =  engine_client(raw_request)
    if hasattr(client, "get_kv_cache_budget"):
        current_blocks, current_bytes = await client.get_kv_cache_budget()
    elif hasattr(client, "engine") and hasattr(client.engine, "get_kv_cache_budget"):
        current_blocks, current_bytes = client.engine.get_kv_cache_budget()
    else:
        current_blocks, current_bytes = None, None
    return JSONResponse(
        content={
            "ok": True,
            "budget": {
                "blocks": current_blocks,
                "bytes": current_bytes,
            },
        }
    )

@router.post("/admin/kv_cache_budget")
async def set_kv_cache_budget(raw_request: Request):
    """
    Set runtime KV cache budget for dynamic memory allocation.

    This allows adjusting KV cache size at runtime to enable multiple
    vLLM instances to share GPU memory effectively.

    Query parameters:
        blocks: Target number of KV cache blocks (optional)
        bytes: Target KV cache size in bytes (optional)

    Returns:
        JSON with current budget settings
    """
    blocks_str = raw_request.query_params.get("blocks")
    bytes_str = raw_request.query_params.get("bytes")

    blocks = int(blocks_str) if blocks_str is not None else None
    bytes_val = int(bytes_str) if bytes_str is not None else None

    if blocks is None and bytes_val is None:
        # Try to get from JSON body
        try:
            body = await raw_request.json()
            blocks = body.get("blocks")
            bytes_val = body.get("bytes")
        except Exception:
            pass

    if blocks is None and bytes_val is None:
        raise HTTPException(
            status_code=400,
            detail="Must specify either 'blocks' or 'bytes' parameter",
        )

    logger.info(f"Setting KV cache budget: blocks={blocks}, bytes={bytes_val}")
    client = engine_client(raw_request)
    # Call the engine method
    if hasattr(client, "set_kv_cache_budget"):
        await client.set_kv_cache_budget(blocks=blocks, bytes=bytes_val)
    elif hasattr(client, "engine") and hasattr(client.engine, "set_kv_cache_budget"):
        client.engine.set_kv_cache_budget(blocks=blocks, bytes=bytes_val)
    else:
        raise HTTPException(
            status_code=501,
            detail="KV cache budget control not supported by this engine",
        )

    # Get current budget
    if hasattr(client, "get_kv_cache_budget"):
        current_blocks, current_bytes = await client.get_kv_cache_budget()
    elif hasattr(client, "engine") and hasattr(client.engine, "get_kv_cache_budget"):
        current_blocks, current_bytes = client.engine.get_kv_cache_budget()
    else:
        current_blocks, current_bytes = blocks, bytes_val

    return JSONResponse(
        content={
            "ok": True,
            "budget": {
                "blocks": current_blocks,
                "bytes": current_bytes,
            },
        }
    )

def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
