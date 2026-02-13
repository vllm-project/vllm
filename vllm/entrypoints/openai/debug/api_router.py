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


@router.post("/profile/start")
async def debug_profile_start(request: Request, prefix: str = "benchmark"):
    """Start GPU profiling with optional trace prefix."""
    await engine_client(request).start_profile(prefix)
    return JSONResponse(content={"status": "profiling", "prefix": prefix})


@router.post("/profile/stop")
async def debug_profile_stop(request: Request):
    """Stop profiling and save trace."""
    await engine_client(request).stop_profile()
    return JSONResponse(content={"status": "stopped"})


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
    """Return server parallelism configuration for benchmarking."""
    vllm_config = request.app.state.vllm_config
    parallel = vllm_config.parallel_config
    return JSONResponse(
        content={
            "data_parallel_size": parallel.data_parallel_size,
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "data_parallel_size_local": parallel.data_parallel_size_local,
            "world_size": parallel.world_size,
        }
    )


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
