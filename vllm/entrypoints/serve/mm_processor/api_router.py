# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _merge_timing_stats(
    preprocess_stats: dict[str, dict[str, float]],
    encoder_results: list[dict[str, dict[str, float | int]] | None],
) -> dict[str, dict[str, float]]:
    """Merge preprocessing stats with encoder stats collected from workers.

    Args:
        preprocess_stats: Per-request preprocessing stats from
            ``renderer._mm_timing_registry.stat()``.
        encoder_results: List of per-worker results from
            ``collective_rpc("get_encoder_timing_stats")``.

    Returns:
        Dictionary mapping request_id to merged stats dict.
    """
    # Aggregate encoder stats across workers (take max)
    encoder_stats: dict[str, dict[str, float]] = {}
    for worker_stats in encoder_results:
        if not worker_stats:
            continue

        for request_id, stats_dict in worker_stats.items():
            if request_id not in encoder_stats:
                encoder_stats[request_id] = dict(stats_dict)
            else:
                current_time = encoder_stats[request_id].get(
                    "encoder_forward_secs", 0.0
                )
                new_time = stats_dict.get("encoder_forward_secs", 0.0)
                encoder_stats[request_id]["encoder_forward_secs"] = max(
                    current_time, new_time
                )

                current_calls = encoder_stats[request_id].get("num_encoder_calls", 0)
                new_calls = stats_dict.get("num_encoder_calls", 0)
                encoder_stats[request_id]["num_encoder_calls"] = max(
                    current_calls, new_calls
                )

    # Merge preprocessing + encoder by request_id
    merged_stats: dict[str, dict[str, float]] = {}

    for request_id, prep_dict in preprocess_stats.items():
        merged_stats[request_id] = dict(prep_dict)

    for request_id, enc_dict in encoder_stats.items():
        if request_id in merged_stats:
            merged_stats[request_id].update(enc_dict)
        else:
            # V1 engine may append worker suffix to request_id
            # (e.g., "req-123" -> "req-123-0"), try suffix-stripping
            matched = False
            for existing_id in list(merged_stats.keys()):
                if existing_id.startswith(request_id):
                    merged_stats[existing_id].update(enc_dict)
                    matched = True
                    break
            if not matched:
                merged_stats[request_id] = dict(enc_dict)

    return merged_stats


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
    merged = _merge_timing_stats(preprocess_stats, encoder_results)

    return JSONResponse(content={"mm_processor_stats": merged})


def attach_router(app: FastAPI):
    if getattr(app.state.args, "enable_mm_processor_stats", False):
        app.include_router(router)
