# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dashboard router for vLLM web UI.

Provides endpoints for:
- /dashboard - Main dashboard HTML page
- /dashboard/api/info - Server information JSON (config, env, load, status)
- /dashboard/api/metrics - Metrics JSON
"""

import asyncio
import functools
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


def _get_explicit_cli_args(args) -> tuple[set, dict]:
    """Get CLI args that were explicitly set by user (non-default values).

    Returns a tuple of (set of explicit arg names, dict of arg name -> value).
    """
    from vllm.config.utils import normalize_value
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.cli_args import FrontendArgs

    explicit_args: set[str] = set()
    non_default_args: dict = {}
    if args is None:
        return explicit_args, non_default_args

    # Get default values from dataclasses
    try:
        frontend_defaults = FrontendArgs()
        engine_defaults = AsyncEngineArgs(model="")

        for key, default_val in vars(frontend_defaults).items():
            if hasattr(args, key):
                current_val = getattr(args, key)
                if current_val != default_val:
                    explicit_args.add(key)
                    non_default_args[key] = normalize_value(current_val)

        for key, default_val in vars(engine_defaults).items():
            if hasattr(args, key) and key != "model":
                current_val = getattr(args, key)
                if current_val != default_val:
                    explicit_args.add(key)
                    non_default_args[key] = normalize_value(current_val)

        # model is always explicit if set
        if hasattr(args, "model") and args.model:
            explicit_args.add("model")
            non_default_args["model"] = args.model
        if hasattr(args, "served_model_name") and args.served_model_name:
            explicit_args.add("served_model_name")
            non_default_args["served_model_name"] = normalize_value(
                args.served_model_name
            )

    except Exception as e:
        logger.debug("Failed to get explicit CLI args: %s", e)

    return explicit_args, non_default_args


_resolved_attn_backend: str | None = None


def _resolve_attn_backend_name(vllm_config, configured_backend) -> str | None:
    """Best-effort resolution of the actual attention backend.

    When the user leaves the backend on auto (None), we call the platform's
    selection logic with the model's head_size / dtype so the dashboard can
    show the concrete backend (e.g. FLASH_ATTN) instead of "auto".
    Returns the resolved backend name string or None on failure.
    """
    global _resolved_attn_backend
    if _resolved_attn_backend is not None:
        return _resolved_attn_backend

    try:
        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.v1.attention.selector import AttentionSelectorConfig

        mc = vllm_config.model_config
        cc = vllm_config.cache_config

        selector_cfg = AttentionSelectorConfig(
            head_size=mc.get_head_size(),
            dtype=mc.dtype,
            kv_cache_dtype=cc.cache_dtype if cc.cache_dtype != "auto" else None,
            block_size=getattr(cc, "block_size", None),
            use_mla=mc.use_mla,
        )
        cls_path = current_platform.get_attn_backend_cls(
            configured_backend,
            attn_selector_config=selector_cfg,
        )
        if not cls_path:
            return None
        for member in AttentionBackendEnum:
            if member.value and member.value == cls_path:
                _resolved_attn_backend = member.name
                return _resolved_attn_backend
        _resolved_attn_backend = cls_path.rsplit(".", 1)[-1].replace(
            "Backend", ""
        )
        return _resolved_attn_backend
    except Exception as e:
        logger.debug("Failed to resolve attention backend: %s", e)
        return None


def _get_startup_info(vllm_config) -> dict:
    """Derive startup info from vllm_config available in the API server.

    Exposes values that are computed properties or enum-typed in the config
    and would otherwise be missing or opaque in the JSON-serialized output.
    Also computes KV cache capacity and max concurrency from cache_config
    (num_gpu_blocks is set via the engine READY handshake).
    """
    info: dict = {}
    if vllm_config is None:
        return info

    # Architecture is a @property, not serialized by Pydantic
    try:
        mc = vllm_config.model_config
        info["architecture"] = mc.architecture
    except Exception:
        pass

    # Attention backend — resolve the actual backend even when config is auto
    try:
        ac = vllm_config.attention_config
        configured = getattr(ac, "backend", None)
        resolved_name = _resolve_attn_backend_name(vllm_config, configured)
        if resolved_name:
            info["attention_backend"] = resolved_name
        elif configured is not None:
            info["attention_backend"] = configured.name
    except Exception:
        pass

    # Optimization level
    try:
        opt = vllm_config.optimization_level
        if opt is not None:
            info["optimization_level"] = f"O{int(opt)}"
    except Exception:
        pass

    # Compilation config: mode, cudagraph, inductor partition, fusion passes
    try:
        cpc = vllm_config.compilation_config
        mode = getattr(cpc, "mode", None)
        if mode is not None:
            info["compilation_mode"] = mode.name
        cgm = getattr(cpc, "cudagraph_mode", None)
        if cgm is not None:
            info["cudagraph_mode"] = cgm.name
        igp = getattr(cpc, "use_inductor_graph_partition", None)
        if igp is not None:
            info["inductor_graph_partition"] = igp

        pc = getattr(cpc, "pass_config", None)
        if pc is not None:
            fusions = {}
            for attr in (
                "fuse_norm_quant",
                "fuse_act_quant",
                "fuse_attn_quant",
                "enable_sp",
                "fuse_gemm_comms",
                "fuse_allreduce_rms",
                "enable_qk_norm_rope_fusion",
            ):
                val = getattr(pc, attr, None)
                if val is True:
                    fusions[attr] = True
            if fusions:
                info["pass_config_fusions"] = fusions
    except Exception:
        pass

    # Kernel config: flashinfer autotune, MoE backend
    try:
        kc = vllm_config.kernel_config
        fi = getattr(kc, "enable_flashinfer_autotune", None)
        if fi is not None:
            info["flashinfer_autotune"] = fi
        moe = getattr(kc, "moe_backend", None)
        if moe is not None:
            info["moe_backend"] = moe
    except Exception:
        pass

    # KV cache capacity and max concurrency
    try:
        cc = vllm_config.cache_config
        mc = vllm_config.model_config

        num_gpu_blocks = getattr(cc, "num_gpu_blocks", None)
        block_size = getattr(cc, "block_size", None)
        if num_gpu_blocks and block_size:
            kv_cache_tokens = num_gpu_blocks * block_size
            info["kv_cache_tokens"] = kv_cache_tokens

            max_model_len = getattr(mc, "max_model_len", None)
            if max_model_len and max_model_len > 0:
                info["max_concurrency"] = round(
                    kv_cache_tokens / max_model_len, 2
                )
    except Exception:
        pass

    return info


@functools.lru_cache(maxsize=1)
def _get_system_env_info_cached() -> dict:
    """Get system environment info (GPU, CUDA, torch, OS, etc.).

    Cached since this info never changes during the lifetime of the server.
    """
    from vllm.collect_env import get_env_info

    return get_env_info()._asdict()


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

    # Get full engine config and derive startup info
    vllm_config = getattr(request.app.state, "vllm_config", None)
    try:
        if vllm_config is not None:
            info["vllm_config"] = PydanticVllmConfig.dump_python(
                vllm_config, mode="json", fallback=str
            )
    except Exception as e:
        logger.warning("Failed to get vllm_config for dashboard: %s", e)

    try:
        startup_info = _get_startup_info(vllm_config)
        if startup_info:
            info["startup_info"] = startup_info
    except Exception as e:
        logger.debug("Failed to get startup_info for dashboard: %s", e)

    # Get environment variables (with explicit markers)
    try:
        vllm_env, explicit_envs = _get_vllm_env_vars()
        info["vllm_env"] = vllm_env
        info["explicit_envs"] = list(explicit_envs)
    except Exception as e:
        logger.warning("Failed to get vllm_env for dashboard: %s", e)

    # Get system environment info (GPU, CUDA, torch, OS, etc.)
    try:
        info["system_env"] = await asyncio.to_thread(
            _get_system_env_info_cached
        )
    except Exception as e:
        logger.debug("Failed to get system_env for dashboard: %s", e)

    # Get explicit CLI args and non-default args dict
    try:
        args = getattr(request.app.state, "args", None)
        explicit_args, non_default_args = _get_explicit_cli_args(args)
        info["explicit_args"] = list(explicit_args)
        info["non_default_args"] = non_default_args
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
