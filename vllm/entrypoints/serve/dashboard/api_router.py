# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dashboard router for vLLM web UI.

Provides endpoints for:
- /dashboard - Main dashboard HTML page
- /dashboard/api/info - Server information JSON (config, env, load, status)
- /dashboard/api/metrics - Metrics JSON
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
    from vllm.v1.engine.exceptions import EngineDeadError

    info: dict = {
        "version": VLLM_VERSION,
        "status": "running",
    }

    # Check engine health
    try:
        engine_client = getattr(request.app.state, "engine_client", None)
        if engine_client is not None:
            await engine_client.check_health()
            info["status"] = "running"
    except EngineDeadError:
        info["status"] = "unhealthy"
    except Exception as e:
        logger.debug("Failed to check engine health: %s", e)

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

    # Get server load metrics
    try:
        server_load = getattr(request.app.state, "server_load_metrics", None)
        if server_load is not None:
            info["server_load"] = server_load
    except Exception as e:
        logger.debug("Failed to get server_load for dashboard: %s", e)

    # Check if engine is sleeping (only available in dev mode)
    try:
        engine_client = getattr(request.app.state, "engine_client", None)
        if engine_client is not None:
            is_sleeping_method = getattr(engine_client, "is_sleeping", None)
            if is_sleeping_method is not None:
                info["is_sleeping"] = await is_sleeping_method()
    except Exception as e:
        logger.debug("Failed to check is_sleeping for dashboard: %s", e)

    # Check if engine is paused (for RLHF workflows)
    try:
        engine_client = getattr(request.app.state, "engine_client", None)
        if engine_client is not None:
            is_paused_method = getattr(engine_client, "is_paused", None)
            if is_paused_method is not None:
                info["is_paused"] = await is_paused_method()
    except Exception as e:
        logger.debug("Failed to check is_paused for dashboard: %s", e)

    return JSONResponse(content=info)


@router.get("/dashboard/api/metrics")
async def dashboard_metrics(request: Request) -> JSONResponse:
    """Get metrics for dashboard display.

    Returns both Prometheus metrics and internal engine stats that are only
    accessible in-process (not available via external /metrics endpoint).
    """
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
            # For labeled metrics, create unique keys to avoid overwriting
            # e.g. vllm:request_success with finished_reason label
            if metric.labels:
                # Filter out common labels (model_name, engine) for key
                key_labels = {
                    k: v
                    for k, v in metric.labels.items()
                    if k not in ("model_name", "engine")
                }
                if key_labels:
                    # Create key like "vllm:request_success:stop"
                    label_suffix = ":".join(str(v) for v in key_labels.values())
                    name = f"{metric.name}:{label_suffix}"
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


def attach_router(app: FastAPI) -> None:
    """Attach dashboard router if enabled via args."""
    args = getattr(app.state, "args", None)
    if args is None or not getattr(args, "enable_dashboard", False):
        return

    logger.info("Enabling vLLM dashboard at /dashboard")
    app.include_router(router)
