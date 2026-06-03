# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.get("/mm_processor_stats")
async def get_mm_processor_stats(request: Request):
    """Return per-request multimodal processor timing stats.

    Collects preprocessing stats from the renderer's timing registry and
    encoder stats from GPU workers via collective_rpc, then merges them
    by request_id.

    This endpoint is only available when ``--enable-mm-processor-stats``
    is set on the server. Both registries use clear-on-read semantics,
    so calling this endpoint drains accumulated stats.
    """
    # 1. Preprocessing stats (API process, clear-on-read)
    renderer = request.app.state.openai_serving_render.renderer
    preprocess_stats = renderer._mm_timing_registry.stat()

    # 2. Encoder stats (worker process, via collective_rpc, clear-on-read)
    engine_client = request.app.state.engine_client
    encoder_results = await engine_client.collective_rpc("get_encoder_timing_stats")

    # 3. Merge preprocessing + encoder stats
    # Import here to avoid circular imports at module level
    from vllm.benchmarks.mm_processor import merge_timing_stats

    merged = merge_timing_stats(preprocess_stats, encoder_results)

    return JSONResponse(content={"mm_processor_stats": merged})


def attach_router(app: FastAPI):
    if getattr(app.state.args, "enable_mm_processor_stats", False):
        app.include_router(router)
