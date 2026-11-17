# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter()


async def _call_every_rank(raw_request: Request, method: str, *args) -> list:
    """Call method on every engine; one result per data-parallel rank."""
    engine_core = raw_request.app.state.engine_client.engine_core
    call_all = getattr(engine_core, "call_utility_all_async", None)
    if call_all is not None:
        return list(await call_all(method, *args))
    return [await engine_core.call_utility_async(method, *args)]


@router.post("/v1/kv_offload/import_from_peer")
async def import_from_peer(
    raw_request: Request,
    host: str = Query(...),
    port: int = Query(...),
    max_chunks: int | None = Query(default=None),
    timeout_s: float = Query(default=600.0),
):
    """Copy the CPU-tier KV cache of the P2P peer at host, one import per rank.

    port is the peer's base p2p port; each data-parallel rank imports from the
    peer rank with its index. With multi-port external DP load balancing, call
    every rank's API server. Example:
       POST /v1/kv_offload/import_from_peer?host=10.0.0.5&port=7777
    """
    try:
        ranks = await _call_every_rank(
            raw_request, "start_kv_peer_import", host, port, max_chunks, timeout_s
        )
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(content={"ranks": ranks})


@router.get("/v1/kv_offload/import_status")
async def import_status(
    raw_request: Request,
    import_id: str | None = Query(default=None),
):
    """Progress of one import, or of every import still remembered, per rank."""
    try:
        ranks = await _call_every_rank(raw_request, "kv_peer_import_status", import_id)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(content={"ranks": ranks})


def attach_router(app: FastAPI):
    app.include_router(router)
