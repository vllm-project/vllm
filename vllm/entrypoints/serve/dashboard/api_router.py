# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dashboard router for vLLM web UI.

Provides endpoints for:
- /dashboard - Main dashboard HTML page
- /dashboard/api/info - Server information JSON (includes full config & env)
- /dashboard/api/metrics - Metrics JSON
- /dashboard/api/collect-env - Collect environment info for debugging
"""

import pathlib

import pydantic
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

router = APIRouter()
PydanticVllmConfig = pydantic.TypeAdapter(VllmConfig)


def _get_vllm_env_vars() -> tuple[dict, set]:
    """Get all VLLM environment variables and which ones are explicitly set."""
    import os

    from vllm.config.utils import normalize_value

    vllm_envs = {}
    explicit_envs = set()

    for key in dir(envs):
        if key.startswith("VLLM_") and "KEY" not in key:
            value = getattr(envs, key, None)
            if value is not None:
                value = normalize_value(value)
                vllm_envs[key] = value
                # Check if explicitly set in environment
                if key in os.environ:
                    explicit_envs.add(key)

    return vllm_envs, explicit_envs


def _get_explicit_cli_args(args) -> set:
    """Get CLI args that were explicitly set by user (non-default values)."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.cli_args import FrontendArgs

    explicit_args: set[str] = set()
    if args is None:
        return explicit_args

    # Get default values from dataclasses
    try:
        frontend_defaults = FrontendArgs()
        engine_defaults = AsyncEngineArgs(model="")

        for key, default_val in vars(frontend_defaults).items():
            if hasattr(args, key):
                current_val = getattr(args, key)
                if current_val != default_val:
                    explicit_args.add(key)

        for key, default_val in vars(engine_defaults).items():
            if hasattr(args, key) and key != "model":
                current_val = getattr(args, key)
                if current_val != default_val:
                    explicit_args.add(key)

        # model is always explicit if set
        if hasattr(args, "model") and args.model:
            explicit_args.add("model")
        if hasattr(args, "served_model_name") and args.served_model_name:
            explicit_args.add("served_model_name")

    except Exception as e:
        logger.debug("Failed to get explicit CLI args: %s", e)

    return explicit_args


def _get_dashboard_html() -> str:
    """Load the dashboard HTML from static file."""
    static_dir = pathlib.Path(__file__).parent / "static"
    html_file = static_dir / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<html><body><h1>Dashboard HTML not found</h1></body></html>"


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_index() -> HTMLResponse:
    """Serve the main dashboard page."""
    html_content = _get_dashboard_html()
    return HTMLResponse(content=html_content)


@router.get("/dashboard/api/info")
async def dashboard_info(request: Request) -> JSONResponse:
    """Get server information for dashboard display."""
    info: dict = {
        "version": VLLM_VERSION,
        "status": "running",
    }

    # Get model information
    try:
        serving_models = request.app.state.openai_serving_models
        if serving_models is not None:
            models_response = await serving_models.show_available_models()
            info["models"] = [
                {"id": model.id, "root": model.root} for model in models_response.data
            ]
    except Exception as e:
        logger.warning("Failed to get model info for dashboard: %s", e)
        info["models"] = []

    # Get full engine config
    try:
        vllm_config = getattr(request.app.state, "vllm_config", None)
        if vllm_config is not None:
            info["vllm_config"] = PydanticVllmConfig.dump_python(
                vllm_config, mode="json", fallback=str
            )
    except Exception as e:
        logger.warning("Failed to get vllm_config for dashboard: %s", e)

    # Get environment variables (with explicit markers)
    try:
        vllm_env, explicit_envs = _get_vllm_env_vars()
        info["vllm_env"] = vllm_env
        info["explicit_envs"] = list(explicit_envs)
    except Exception as e:
        logger.warning("Failed to get vllm_env for dashboard: %s", e)

    # Get explicit CLI args
    try:
        args = getattr(request.app.state, "args", None)
        explicit_args = _get_explicit_cli_args(args)
        info["explicit_args"] = list(explicit_args)
    except Exception as e:
        logger.warning("Failed to get explicit_args for dashboard: %s", e)

    return JSONResponse(content=info)


@router.get("/dashboard/api/metrics")
async def dashboard_metrics(request: Request) -> JSONResponse:
    """Get metrics for dashboard display."""
    metrics: dict = {}

    # Try to get metrics from prometheus registry
    try:
        from vllm.v1.metrics.reader import (
            Counter,
            Gauge,
            Histogram,
            get_metrics_snapshot,
        )

        snapshot = get_metrics_snapshot()
        for metric in snapshot:
            name = metric.name
            if isinstance(metric, (Counter, Gauge)):
                metrics[name] = {
                    "value": metric.value,
                    "labels": metric.labels,
                }
            elif isinstance(metric, Histogram):
                metrics[name] = {
                    "count": metric.count,
                    "sum": metric.sum,
                    "labels": metric.labels,
                }
    except ImportError:
        logger.debug("Metrics reader not available")
    except Exception as e:
        logger.warning("Failed to get metrics for dashboard: %s", e)

    # Get server load if tracking is enabled
    try:
        server_load = getattr(request.app.state, "server_load_metrics", None)
        if server_load is not None:
            metrics["server_load"] = {"value": server_load, "labels": {}}
    except Exception:
        pass

    return JSONResponse(content=metrics)


@router.get("/dashboard/api/collect-env")
async def dashboard_collect_env() -> JSONResponse:
    """Collect environment information for debugging.

    This runs the same collection as `vllm collect-env` CLI command.
    Useful for users to copy environment info when reporting issues.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        from vllm.collect_env import get_pretty_env_info

        # Run in thread pool since it executes subprocess commands
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            env_info = await loop.run_in_executor(executor, get_pretty_env_info)

        return JSONResponse(content={"output": env_info, "status": "success"})
    except Exception as e:
        logger.warning("Failed to collect environment info: %s", e)
        return JSONResponse(
            content={"output": str(e), "status": "error"}, status_code=500
        )


def attach_router(app: FastAPI) -> None:
    """Attach dashboard router if enabled via args."""
    args = getattr(app.state, "args", None)
    if args is None or not getattr(args, "enable_dashboard", False):
        return

    logger.info("Enabling vLLM dashboard at /dashboard")
    app.include_router(router)
