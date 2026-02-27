# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug endpoints for benchmarking and profiling.

These endpoints support the vllm bench iterations command for precise
prefill/decode measurement using sleep(level=0) to pause scheduling.
"""

import glob
import os

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/debug", tags=["debug"])


def engine_client(request: Request):
    """Get the engine client from app state."""
    return request.app.state.engine_client


@router.post("/sleep")
async def debug_sleep(request: Request, level: int = 0):
    """Pause scheduling. level=0 pauses without freeing GPU memory."""
    await engine_client(request).sleep(level)
    return JSONResponse(content={"status": "sleeping", "level": level})


@router.post("/wake_up")
async def debug_wake_up(request: Request):
    """Resume scheduling."""
    await engine_client(request).wake_up()
    return JSONResponse(content={"status": "awake"})


@router.get("/batch_info")
async def debug_batch_info(request: Request):
    """Return current scheduler batch composition.

    Used by vllm bench iterations to detect when all requests have
    finished prefill and entered decode (num_waiting == 0), so that
    profiling captures only steady-state decode iterations.

    For DP>1, sums request counts across all DP engine cores.
    """
    client = engine_client(request)
    all_counts = await client.engine_core.call_utility_all_async(
        "get_request_counts")
    total_running = sum(c[0] for c in all_counts)
    total_waiting = sum(c[1] for c in all_counts)
    return JSONResponse(content={
        "num_running": total_running,
        "num_waiting": total_waiting,
    })


@router.post("/prefill_only")
async def debug_prefill_only(request: Request, enabled: bool = True):
    """Set prefill-only scheduling mode.

    When enabled, the scheduler skips decode tokens for running requests,
    giving the full token budget to prefilling waiting requests.  This
    ensures all requests finish prefill before any decode begins.

    Use enabled=false to resume normal (prefill + decode) scheduling.
    """
    client = engine_client(request)
    await client.engine_core.call_utility_async("set_prefill_only", enabled)
    return JSONResponse(content={
        "status": "prefill_only" if enabled else "normal",
    })


@router.post("/profile/start")
async def debug_profile_start(request: Request, prefix: str = "benchmark",
                              delay: int = 0, max_steps: int = 0):
    """Start profiling with optional trace prefix and delay.

    Args:
        prefix: Subdirectory name for trace files.
        delay: Number of engine steps to skip before starting the trace.
               Use delay=1 for decode benchmarks to exclude the prefill step.
        max_steps: Stop tracing automatically after this many engine steps.
                   Use max_steps=1 for prefill benchmarks to exclude the
                   decode step.  0 means no auto-stop (manual stop required).
    """
    await engine_client(request).start_profile(prefix, delay=delay,
                                                max_steps=max_steps)
    return JSONResponse(content={"status": "profiling", "prefix": prefix,
                                 "delay": delay, "max_steps": max_steps})


@router.post("/profile/stop")
async def debug_profile_stop(request: Request):
    """Stop profiling and save trace. Returns decode-only elapsed if delay was used."""
    elapsed_ms = await engine_client(request).stop_profile()
    response = {"status": "stopped"}
    if elapsed_ms is not None:
        response["elapsed_ms"] = elapsed_ms
    return JSONResponse(content=response)


@router.get("/traces")
async def debug_list_traces(request: Request):
    """List available trace files.

    Searches recursively under the profiler directory for trace files.
    Supports both torch profiler (.json, .json.gz) and JAX profiler
    (.trace.json.gz, .xplane.pb) output formats.
    Returns relative paths from the profiler directory.
    """
    vllm_config = request.app.state.vllm_config
    profiler_dir = vllm_config.profiler_config.torch_profiler_dir
    if not profiler_dir:
        return JSONResponse(
            content={"traces": [], "error": "No profiler directory configured"}
        )

    trace_files: list[str] = []
    for pattern in ["**/*.json", "**/*.json.gz", "**/*.xplane.pb"]:
        for f in glob.glob(os.path.join(profiler_dir, pattern), recursive=True):
            # Return path relative to profiler_dir so client can download
            rel_path = os.path.relpath(f, profiler_dir)
            trace_files.append(rel_path)
    return JSONResponse(
        content={"traces": sorted(trace_files), "directory": profiler_dir}
    )


@router.get("/config")
async def debug_config(request: Request):
    """Return server parallelism configuration for benchmarking.

    Reports both vLLM-visible parallelism and platform-specific config.
    On TPU, the real DP size is in sharding_config (vLLM resets DP to 1
    since TPU handles DP internally via DPScheduler).
    """
    vllm_config = request.app.state.vllm_config
    parallel = vllm_config.parallel_config

    config = {
        "data_parallel_size": parallel.data_parallel_size,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "data_parallel_size_local": parallel.data_parallel_size_local,
        "world_size": parallel.world_size,
    }

    # Expose real DP/TP from TPU sharding config if available
    sharding_config = getattr(vllm_config, "sharding_config", None)
    if sharding_config is not None:
        strategy = getattr(sharding_config, "sharding_strategy", None)
        if strategy is not None:
            config["real_data_parallel_size"] = getattr(
                strategy, "data_parallelism", parallel.data_parallel_size
            )
            config["real_tensor_parallel_size"] = getattr(
                strategy, "tensor_parallelism", parallel.tensor_parallel_size
            )
            config["expert_parallelism"] = getattr(
                strategy, "expert_parallelism", 1
            )

    return JSONResponse(content=config)


@router.get("/traces/{filename:path}")
async def debug_get_trace(request: Request, filename: str):
    """Download a trace file.

    The filename can be a relative path from the profiler directory,
    e.g. "prefill_ctx1024/plugins/profile/2026_02_13/trace.json.gz".
    """
    vllm_config = request.app.state.vllm_config
    profiler_dir = vllm_config.profiler_config.torch_profiler_dir
    if not profiler_dir:
        raise HTTPException(status_code=404, detail="No profiler directory configured")

    # Resolve the path and ensure it stays within profiler_dir (security)
    trace_path = os.path.normpath(os.path.join(profiler_dir, filename))
    if not trace_path.startswith(os.path.normpath(profiler_dir)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(trace_path):
        raise HTTPException(status_code=404, detail=f"Trace file not found: {filename}")

    if filename.endswith(".gz"):
        media_type = "application/gzip"
    elif filename.endswith(".pb"):
        media_type = "application/octet-stream"
    else:
        media_type = "application/json"
    return FileResponse(
        trace_path, media_type=media_type, filename=os.path.basename(filename)
    )


def register_debug_api_router(app: FastAPI) -> None:
    """Register the debug API router with the app."""
    app.include_router(router)
