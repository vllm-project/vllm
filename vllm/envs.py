# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import logging
import os
import tempfile
import uuid
import warnings
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Private helpers (module-level)
# ----------------------------------------------------------------------------


def _xdg_cache_home() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def _xdg_config_home() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def _default_dbo_comm_sms() -> int:
    try:
        import torch

        if getattr(torch.version, "hip", None) is not None:
            return 64
    except ImportError:
        pass
    return 20


def _tpu_pathways_default() -> bool:
    return "proxy" in os.getenv("JAX_PLATFORMS", "").lower()


def _warn_deprecated_env(name: str, removal_version: str, replacement: str) -> None:
    """Emit a FutureWarning if an env var is explicitly set."""
    if _env_set(name):
        warnings.warn(
            f"{name} is deprecated and will be removed in "
            f"{removal_version}. {replacement}",
            FutureWarning,
            stacklevel=2,
        )


def _env_set(name: str) -> bool:
    """Case-insensitive ``name in os.environ`` check.

    Sub-models use ``case_sensitive=False`` so an env var set under any
    casing is honored by pydantic at parse time. Cross-field defaults
    that gate on "is the env var explicitly set" must mirror that, or
    explicit lowercase overrides get silently overwritten.
    """
    needle = name.lower()
    return any(k.lower() == needle for k in os.environ)


# Sentinel `validation_alias` used to keep VLLM_TPU_USING_PATHWAYS off the
# pydantic-settings env-loading path. The field's value is computed by a
# default_factory from JAX_PLATFORMS, not from VLLM_TPU_USING_PATHWAYS itself.
# A string is required because pydantic-settings only accepts str-typed
# validation_alias values.
_TPU_PATHWAYS_SENTINEL = "__VLLM_TPU_USING_PATHWAYS_UNSET_SENTINEL__"


# ----------------------------------------------------------------------------
# Shared config for sub-models
# ----------------------------------------------------------------------------

_SUB_CONFIG = SettingsConfigDict(
    env_prefix="VLLM_",
    extra="ignore",
    case_sensitive=False,
    populate_by_name=True,
)


# ----------------------------------------------------------------------------
# Sub-models
# ----------------------------------------------------------------------------


class BuildSettings(BaseSettings):
    model_config = _SUB_CONFIG

    target_device: str = Field(
        default="cuda",
        description="Target device of vLLM, supporting [cuda (by default), rocm, cpu].",
    )
    main_cuda_version: str = Field(
        default="13.0",
        description=(
            "Main CUDA version of vLLM. This follows PyTorch but can be overridden."
        ),
    )
    max_jobs: str | None = Field(
        default=None,
        alias="MAX_JOBS",
        description=(
            "Maximum number of compilation jobs to run in parallel. "
            "By default this is the number of CPUs."
        ),
    )
    nvcc_threads: str | None = Field(
        default=None,
        alias="NVCC_THREADS",
        description=(
            "Number of threads to use for nvcc. By default this is 1. "
            "If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU."
        ),
    )
    use_precompiled: bool = Field(
        default=False,
        description=(
            "Use precompiled binaries (.so) instead of building from source. "
            "Implicitly enabled when VLLM_PRECOMPILED_WHEEL_LOCATION is set."
        ),
    )
    use_precompiled_rust: bool = Field(
        default=False,
        description=(
            "If set, vllm will use the precompiled Rust frontend binary (vllm-rs)."
        ),
    )
    skip_precompiled_version_suffix: bool = Field(
        default=False,
        description="If set, skip adding +precompiled suffix to version string.",
    )
    docker_build_context: bool = Field(
        default=False,
        description=(
            "Used to mark that setup.py is running in a Docker build context, "
            "in order to force the use of precompiled binaries."
        ),
    )
    cmake_build_type: Literal["Debug", "Release", "RelWithDebInfo"] | None = Field(
        default=None,
        alias="CMAKE_BUILD_TYPE",
        description=(
            'CMake build type. If not set, defaults to "Debug" or "RelWithDebInfo". '
            'Available options: "Debug", "Release", "RelWithDebInfo".'
        ),
    )
    verbose: bool = Field(
        default=False,
        alias="VERBOSE",
        description="If set, vllm will print verbose logs during installation.",
    )

    @field_validator("target_device", mode="before")
    @classmethod
    def _lower_target_device(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("main_cuda_version", mode="before")
    @classmethod
    def _normalize_main_cuda_version(cls, v: Any) -> Any:
        if v is None:
            return "13.0"
        if isinstance(v, str):
            lowered = v.lower()
            return lowered or "13.0"
        return v

    @model_validator(mode="after")
    def _force_use_precompiled_when_wheel_set(self) -> "BuildSettings":
        # If VLLM_PRECOMPILED_WHEEL_LOCATION is set, force VLLM_USE_PRECOMPILED True.
        if not self.use_precompiled and _env_set("VLLM_PRECOMPILED_WHEEL_LOCATION"):
            self.use_precompiled = True
        return self


class PathSettings(BaseSettings):
    model_config = _SUB_CONFIG

    config_root: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_config_home(), "vllm")
        ),
        description=(
            "Root directory for vLLM configuration files. Defaults to "
            "`~/.config/vllm` unless `XDG_CONFIG_HOME` is set. Note that this not "
            "only affects how vllm finds its configuration files during runtime, "
            "but also affects how vllm installs its configuration files during "
            "**installation**."
        ),
    )
    cache_root: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_cache_home(), "vllm")
        ),
        description=(
            "Root directory for vLLM cache files. Defaults to `~/.cache/vllm` "
            "unless `XDG_CACHE_HOME` is set."
        ),
    )
    assets_cache: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_cache_home(), "vllm", "assets")
        ),
        description="Path to the cache for storing downloaded assets.",
    )
    xla_cache_path: str = Field(
        default_factory=lambda: os.path.expanduser(
            os.path.join(_xdg_cache_home(), "vllm", "xla_cache")
        ),
        description=(
            "Path to the XLA persistent cache directory. Only used for XLA "
            "devices such as TPUs."
        ),
    )
    rpc_base_path: str = Field(
        default_factory=tempfile.gettempdir,
        description=(
            "Path used for IPC when the frontend api server is running in "
            "multi-processing mode to communicate with the backend engine process."
        ),
    )
    tuned_config_folder: str | None = Field(
        default=None,
        description=(
            "User override folder for tuned Triton-kernel configs. Shared by "
            "MoE, Mamba SSU, and LoRA. Filenames are distinct so one folder "
            "can hold all. Each component first checks this folder, then the "
            "configs shipped with vLLM (if any). If no JSON matches, it uses "
            "a hard-coded heuristic."
        ),
    )
    model_redirect_path: str | None = Field(
        default=None,
        description=(
            "Use model_redirect to redirect the model name to a local folder. "
            "`model_redirect` can be a json file mapping the model between "
            'repo_id and local folder: {"meta-llama/Llama-3.2-1B": '
            '"/tmp/Llama-3.2-1B"} or a space separated values table file: '
            "meta-llama/Llama-3.2-1B /tmp/Llama-3.2-1B."
        ),
    )
    lora_resolver_cache_dir: str | None = Field(
        default=None,
        description=(
            "A local directory to look in for unrecognized LoRA adapters. Only "
            "works if plugins are enabled and VLLM_ALLOW_RUNTIME_LORA_UPDATING "
            "is enabled."
        ),
    )
    lora_resolver_hf_repo_list: str | None = Field(
        default=None,
        description=(
            "A remote HF repo(s) containing one or more LoRA adapters, which may "
            "be downloaded and leveraged as needed. Only works if plugins are "
            "enabled and VLLM_ALLOW_RUNTIME_LORA_UPDATING is enabled. Values "
            "should be comma separated."
        ),
    )
    cudart_so_path: str | None = Field(
        default=None,
        description=(
            "In some systems, find_loaded_library() may not work. So we allow "
            "users to specify the path through the environment variable "
            "VLLM_CUDART_SO_PATH."
        ),
    )
    nccl_so_path: str | None = Field(
        default=None,
        description=(
            "Path to the NCCL library file. It is needed because nccl>=2.19 "
            "brought by PyTorch contains a bug: "
            "https://github.com/NVIDIA/nccl/issues/1234"
        ),
    )
    nccl_include_path: str | None = Field(
        default=None,
        description="NCCL header path.",
    )
    ld_library_path: str | None = Field(
        default=None,
        alias="LD_LIBRARY_PATH",
        description=(
            "When `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the "
            "NCCL library file in the locations specified by `LD_LIBRARY_PATH`."
        ),
    )
    cuda_home: str | None = Field(
        default=None,
        alias="CUDA_HOME",
        description=(
            "Path to the cudatoolkit home directory, under which should be bin, "
            "include, and lib directories."
        ),
    )
    cuda_compatibility_path: str | None = Field(
        default=None,
        description=(
            "Path to the CUDA compatibility libraries when CUDA compatibility "
            "is enabled."
        ),
    )
    enable_cuda_compatibility: bool = Field(
        default=False,
        description=(
            "Enable CUDA compatibility mode for datacenter GPUs with older "
            "driver versions than the CUDA toolkit major version of vLLM."
        ),
    )
    logging_config_path: str | None = Field(
        default=None,
        description=(
            "Path to a JSON file with custom logging configuration. Used when "
            "VLLM_CONFIGURE_LOGGING is enabled."
        ),
    )
    debug_dump_path: str | None = Field(
        default=None,
        description=(
            "Dump fx graphs to the given directory. It will override "
            "CompilationConfig.debug_dump_path if set."
        ),
    )
    pattern_match_debug: str | None = Field(
        default=None,
        description=(
            "Debug pattern matching inside custom passes. Should be set to the "
            "fx.Node name (e.g. 'getitem_34' or 'scaled_mm_3')."
        ),
    )
    gc_debug: str = Field(
        default="",
        description=(
            "GC debug config. "
            "VLLM_GC_DEBUG=0: disable GC debugger. "
            "VLLM_GC_DEBUG=1: enable GC debugger with gc.collect elapsed times. "
            "VLLM_GC_DEBUG='{\"top_objects\":5}': enable GC debugger with top 5 "
            "collected objects."
        ),
    )
    system_start_date: str | None = Field(
        default=None,
        description=(
            "Pin the conversation start date injected into the Harmony system "
            "message. When unset the current date is used, which introduces "
            "non-determinism (different tokens -> different model behaviour at "
            'temperature=0). Set to an ISO date string, e.g. "2023-09-12", for '
            "reproducible inference or testing."
        ),
    )

    @field_validator(
        "config_root", "cache_root", "assets_cache", "xla_cache_path", mode="after"
    )
    @classmethod
    def _expanduser(cls, v: str) -> str:
        return os.path.expanduser(v)


class ServerSettings(BaseSettings):
    model_config = _SUB_CONFIG

    host_ip: str = Field(
        default="",
        description=(
            "Used in distributed environment to determine the IP address of the "
            "current node, when the node has multiple network interfaces. If you "
            "are using multi-node inference, you should set this differently on "
            "each node."
        ),
    )
    port: int | None = Field(
        default=None,
        description=(
            "Used in distributed environment to manually set the communication "
            "port. Note: if VLLM_PORT is set, and some code asks for multiple "
            "ports, the VLLM_PORT will be used as the first port, and the rest "
            "will be generated by incrementing the VLLM_PORT value."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description="API key for vLLM API server.",
    )
    debug_log_api_server_response: bool = Field(
        default=False,
        description="Whether to log responses from API Server for debugging.",
    )
    rpc_timeout: int = Field(
        default=10000,
        description=(
            "Time in ms for the zmq client to wait for a response from the "
            "backend server for simple data operations."
        ),
    )
    http_timeout_keep_alive: int = Field(
        default=5,
        description=(
            "Timeout in seconds for keeping HTTP connections alive in API server."
        ),
    )
    max_n_sequences: int = Field(
        default=16384,
        description=(
            "Maximum allowed value for the `n` sampling parameter (number of "
            "output sequences per request). Limits resource consumption to "
            "prevent denial-of-service via excessively large fan-out. "
            "Default: 16384."
        ),
    )
    engine_iteration_timeout_s: int = Field(
        default=60,
        description="Timeout for each iteration in the engine.",
    )
    engine_ready_timeout_s: int = Field(
        default=600,
        description=(
            "Timeout in seconds for waiting for engine cores to become ready "
            "during startup. Default is 600 seconds (10 minutes)."
        ),
    )
    execute_model_timeout_seconds: int = Field(
        default=300,
        description=(
            "Timeout in seconds for execute_model RPC calls in multiprocessing "
            "executor (only applies when TP > 1)."
        ),
    )
    keep_alive_on_engine_death: bool = Field(
        default=False,
        description=(
            "If set, the OpenAI API server will stay alive even after the "
            "underlying AsyncLLMEngine errors and stops serving requests."
        ),
    )
    server_dev_mode: bool = Field(
        default=False,
        description=(
            "If set, vllm will run in development mode, which will enable some "
            "additional endpoints for developing and debugging, e.g. "
            "`/reset_prefix_cache`."
        ),
    )
    use_rust_frontend: bool = Field(
        default=False,
        description=(
            "If set, use the Rust frontend binary instead of the Python API "
            "server process(es)."
        ),
    )
    rust_frontend_path: str | None = Field(
        default=None,
        description=(
            "Path to the Rust frontend binary. Defaults to None unless "
            "VLLM_USE_RUST_FRONTEND=1, in which case the value is resolved "
            'from the env var (default "auto", which discovers the binary '
            "installed with the vllm package). Only used when "
            "VLLM_USE_RUST_FRONTEND=1."
        ),
    )
    allow_long_max_model_len: bool = Field(
        default=False,
        description=(
            "If the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN is set, it allows the "
            "user to specify a max sequence length greater than the max length "
            "derived from the model's config.json. To enable this, set "
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1."
        ),
    )
    enable_responses_api_store: bool = Field(
        default=False,
        description=(
            'Enables support for the "store" option in the OpenAI Responses '
            "API. When set to 1, vLLM's OpenAI server will retain the input "
            "and output messages for those requests in memory. By default, "
            'this is disabled (0), and the "store" option is ignored. '
            "NOTE/WARNING: 1. Messages are kept in memory only (not persisted "
            "to disk) and will be lost when the vLLM server shuts down. "
            "2. Enabling this option will cause a memory leak, as stored "
            "messages are never removed from memory until the server terminates."
        ),
    )
    allow_chunked_local_attn_with_hybrid_kv_cache: bool = Field(
        default=True,
        description=(
            "Allow chunked local attention with hybrid kv cache manager. "
            "Currently using the Hybrid KV cache manager with chunked local "
            "attention in the Llama4 models (the only models currently using "
            "chunked local attn) causes a latency regression. For this reason, "
            "we disable it by default. This flag is used to allow users to "
            "enable it if they want to (to save on kv-cache memory usage and "
            "enable longer contexts). "
            "TODO(lucas): Remove this flag once latency regression is resolved."
        ),
    )
    process_name_prefix: str = Field(
        default="VLLM",
        description=(
            "Used to set the process name prefix for vLLM processes. This is "
            'useful for debugging and monitoring purposes. The default value is "VLLM".'
        ),
    )
    loopback_ip: str = Field(
        default="",
        description="Used to force set up loopback IP.",
    )
    skip_model_name_validation: bool = Field(
        default=False,
        description=(
            "Skip model name validation in OpenAI API requests. When set to 1, "
            "any model name will be accepted in the 'model' field of API "
            "requests. This is useful for proxy/gateway scenarios where the "
            "actual model is served but different names may be used in requests."
        ),
    )
    allow_insecure_serialization: bool = Field(
        default=False,
        description=(
            "If set, allow insecure serialization using pickle. This is useful "
            "for environments where it is deemed safe to use the insecure "
            "method and it is needed for some reason."
        ),
    )
    disable_log_logo: bool = Field(
        default=False,
        description="Disable logging of vLLM logo at server startup time.",
    )
    tool_parse_regex_timeout_seconds: int = Field(
        default=1,
        description="Regex timeout for use by the vLLM tool parsing plugins.",
    )
    tool_json_error_automatic_retry: bool = Field(
        default=False,
        description=(
            "Enable automatic retry when tool call JSON parsing fails. "
            "If enabled, returns an error message to the model to retry. "
            "If disabled (default), raises an exception and fails the request."
        ),
    )
    enforce_strict_tool_calling: bool = Field(
        default=False,
        description=(
            "When 1, the model structural tags will be used to enforce the "
            "model output conforming to the model's tool-calling format and "
            "schema. Default 0 (off)."
        ),
    )
    custom_scopes_for_profiling: bool = Field(
        default=False,
        description=(
            "Add optional custom scopes for profiling, disable to avoid overheads."
        ),
    )
    nvtx_scopes_for_profiling: bool = Field(
        default=False,
        description=(
            "Add optional nvtx scopes for profiling, disable to avoid overheads."
        ),
    )
    mq_max_chunk_bytes_mb: int = Field(
        default=16,
        description=(
            "Control the max chunk bytes (in MB) for the rpc message queue. "
            "Object larger than this threshold will be broadcast to worker "
            "processes via zmq."
        ),
    )

    @model_validator(mode="after")
    def _resolve_rust_frontend_path(self) -> "ServerSettings":
        # Mirrors the legacy `_resolve_rust_frontend_path` behavior: the
        # path is only meaningful when the Rust frontend is enabled. When
        # disabled, ignore (and warn about) any explicitly-set path. When
        # enabled, an unset path or one of "auto"/"1"/"true" resolves to
        # the bundled `vllm-rs` binary.
        raw = self.rust_frontend_path
        if not self.use_rust_frontend:
            if _env_set("VLLM_RUST_FRONTEND_PATH"):
                logger.warning(
                    "VLLM_RUST_FRONTEND_PATH is set but VLLM_USE_RUST_FRONTEND "
                    "is not enabled. The Rust frontend will not be used. "
                    "Set VLLM_USE_RUST_FRONTEND=1 to enable it."
                )
            self.rust_frontend_path = None
            return self

        # Rust frontend enabled: if path env var is unset, default to "auto".
        if raw is None:
            raw = "auto"

        if raw.lower() in ("auto", "1", "true"):
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(pkg_dir, "vllm-rs")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                self.rust_frontend_path = candidate
                return self

            raise FileNotFoundError(
                "VLLM_RUST_FRONTEND_PATH=auto but the vllm-rs binary was "
                f"not found at {candidate}. "
                "Build with setuptools-rust or set the path explicitly."
            )
        self.rust_frontend_path = raw
        return self

    @field_validator("port", mode="before")
    @classmethod
    def _parse_port(cls, v: Any) -> Any:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError) as err:
            from urllib3.util import parse_url

            sval = str(v)
            try:
                parsed = parse_url(sval)
            except Exception:
                parsed = None
            if parsed is not None and parsed.scheme:
                raise ValueError(
                    f"VLLM_PORT '{sval}' appears to be a URI. "
                    "This may be caused by a Kubernetes service discovery issue,"
                    "check the warning in: https://docs.vllm.ai/en/stable/configuration/env_vars/?h=vllm_port#environment-variables"
                ) from None
            raise ValueError(f"VLLM_PORT '{sval}' must be a valid integer") from err


class LoggingSettings(BaseSettings):
    model_config = _SUB_CONFIG

    configure_logging: bool = Field(
        default=True,
        description=(
            "Logging configuration. "
            "If set to 0, vllm will not configure logging. "
            "If set to 1, vllm will configure logging using the default "
            "configuration or the configuration file specified by "
            "VLLM_LOGGING_CONFIG_PATH."
        ),
    )
    logging_level: str = Field(
        default="INFO",
        description="Used for configuring the default logging level.",
    )
    logging_prefix: str = Field(
        default="",
        description=(
            "If set, VLLM_LOGGING_PREFIX will be prepended to all log messages."
        ),
    )
    logging_stream: str = Field(
        default="ext://sys.stdout",
        description="Used for configuring the default logging stream.",
    )
    logging_color: str = Field(
        default="auto",
        description=(
            'Controls colored logging output. Options: "auto" (default, colors '
            'when terminal), "1" (always use colors), "0" (never use colors).'
        ),
    )
    no_color: bool = Field(
        default=False,
        alias="NO_COLOR",
        description="Standard unix flag for disabling ANSI color codes.",
    )
    log_stats_interval: float = Field(
        default=10.0,
        description=(
            "If set, vllm will log stats at this interval in seconds. "
            "If not set, vllm will log stats every 10 seconds."
        ),
    )
    log_batchsize_interval: float = Field(
        default=-1.0,
        description=(
            "If set to a positive value, vllm will log batch size statistics "
            "at this interval in seconds. Negative values disable batch-size "
            "logging."
        ),
    )
    trace_function: int = Field(
        default=0,
        description=(
            "Trace function calls. If set to 1, vllm will trace function "
            "calls. Useful for debugging."
        ),
    )
    ringbuffer_warning_interval: int = Field(
        default=60,
        description=(
            "Interval in seconds to log a warning message when the ring buffer is full."
        ),
    )
    debug_workspace: bool = Field(
        default=False,
        description=(
            "Debug workspace allocations. Logging of workspace resize operations."
        ),
    )
    debug_mfu_metrics: bool = Field(
        default=False,
        description="Debug logging for --enable-mfu-metrics.",
    )
    log_model_inspection: bool = Field(
        default=False,
        description=(
            "Log model inspection after loading. If enabled, logs a "
            "transformers-style hierarchical view of the model with "
            "quantization methods and attention backends."
        ),
    )

    @field_validator("logging_level", mode="after")
    @classmethod
    def _upper_logging_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("log_stats_interval", mode="after")
    @classmethod
    def _clamp_log_stats_interval(cls, v: float) -> float:
        if v <= 0.0:
            return 10.0
        return v

    @field_validator("no_color", mode="before")
    @classmethod
    def _parse_no_color(cls, v: Any) -> Any:
        # Old logic: os.getenv("NO_COLOR", "0") != "0" — any non-"0" is True.
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v != "0"
        return bool(v)


class DistributedSettings(BaseSettings):
    model_config = _SUB_CONFIG

    dp_rank: int = Field(
        default=0,
        description="Rank of the process in the data parallel setting.",
    )
    dp_rank_local: int = Field(
        default=-1,
        description=(
            "Local data-parallel rank. Defaults to VLLM_DP_RANK when this "
            "variable is not explicitly set."
        ),
    )
    dp_size: int = Field(
        default=1,
        description="World size of the data parallel setting.",
    )
    dp_master_ip: str = Field(
        default="127.0.0.1",
        description="IP address of the master node in the data parallel setting.",
    )
    dp_master_port: int = Field(
        default=0,
        description="Port of the master node in the data parallel setting.",
    )
    randomize_dp_dummy_inputs: bool = Field(
        default=False,
        description="Randomize inputs during dummy runs when using Data Parallel.",
    )
    ray_dp_pack_strategy: Literal["strict", "fill", "span"] = Field(
        default="strict",
        description=(
            "Strategy to pack the data parallel ranks for Ray. Available "
            'options: "fill" - for DP master node, allocate exactly '
            "data-parallel-size-local DP ranks; for non-master nodes, allocate "
            'as many DP ranks as can fit. "strict" - allocate exactly '
            'data-parallel-size-local DP ranks to each picked node. "span" - '
            "should be used only when a single DP rank requires multiple "
            "nodes; allocate one DP rank over as many nodes as required for "
            "set world_size. This environment variable is ignored if "
            "data-parallel-backend is not Ray."
        ),
    )
    ray_extra_env_var_prefixes_to_copy: str = Field(
        default="",
        description=(
            "Comma-separated *additional* prefixes of env vars to copy from "
            "the driver to Ray workers. These are merged with the built-in "
            "defaults defined in `vllm.ray.ray_env` (VLLM_, etc.). "
            'Example: "MYLIB_,OTHER_".'
        ),
    )
    ray_extra_env_vars_to_copy: str = Field(
        default="",
        description=(
            "Comma-separated *additional* individual env var names to copy "
            "from the driver to Ray workers. Merged with the built-in "
            "defaults defined in `vllm.ray.ray_env` (PYTHONHASHSEED). "
            'Example: "MY_SECRET,MY_FLAG".'
        ),
    )
    ray_per_worker_gpus: float = Field(
        default=1.0,
        description=(
            "Number of GPUs per worker in Ray; if it is set to be a fraction, "
            "it allows ray to schedule multiple actors on a single GPU, so "
            "that users can colocate other actors on the same GPUs as vLLM."
        ),
    )
    ray_bundle_indices: str = Field(
        default="",
        description=(
            "Bundle indices for Ray; if set, it can control precisely which "
            "indices are used for the Ray bundle, for every worker. Format: "
            'comma-separated list of integers, e.g. "0,1,2,3".'
        ),
    )
    use_ray_compiled_dag_channel_type: Literal["auto", "nccl", "shm"] = Field(
        default="auto",
        description=(
            "If the env var is set, Ray Compiled Graph uses the specified "
            "channel type to communicate between workers belonging to "
            "different pipeline-parallel stages. Available options: "
            '"auto" - use the default channel type; '
            '"nccl" - use NCCL for communication; '
            '"shm" - use shared memory and gRPC for communication.'
        ),
    )
    use_ray_compiled_dag_overlap_comm: bool = Field(
        default=False,
        description=(
            "If the env var is set, it enables GPU communication overlap "
            "(experimental feature) in Ray's Compiled Graph."
        ),
    )
    use_ray_wrapped_pp_comm: bool = Field(
        default=True,
        description=(
            "If the env var is set, it uses a Ray Communicator wrapping "
            "vLLM's pipeline parallelism communicator to interact with Ray's "
            "Compiled Graph. Otherwise, it uses Ray's NCCL communicator."
        ),
    )
    use_ray_v2_executor_backend: bool = Field(
        default=True,
        description=(
            'When True and distributed_executor_backend="ray", use '
            "RayExecutorV2 (MQ-based) instead of RayDistributedExecutor "
            "(compiled-graph backend)."
        ),
    )
    worker_multiproc_method: Literal["spawn", "fork"] = Field(
        default="fork",
        description=(
            "Use dedicated multiprocess context for workers. Both spawn and fork work."
        ),
    )
    enable_v1_multiprocessing: bool = Field(
        default=True,
        description="If set, enable multiprocessing in LLM for the V1 code path.",
    )
    local_rank: int = Field(
        default=0,
        alias="LOCAL_RANK",
        description=(
            "Local rank of the process in the distributed setting, used to "
            "determine the GPU device id."
        ),
    )
    cuda_visible_devices: str | None = Field(
        default=None,
        alias="CUDA_VISIBLE_DEVICES",
        description="Used to control the visible devices in the distributed setting.",
    )
    disable_pynccl: bool = Field(
        default=False,
        description="Disable pynccl (using torch.distributed instead).",
    )
    skip_p2p_check: bool = Field(
        default=True,
        description=(
            "We assume drivers can report p2p status correctly. If the "
            "program hangs when using custom allreduce, potentially caused "
            "by a bug in the driver (535 series), it might be helpful to "
            "set VLLM_SKIP_P2P_CHECK=0 so that vLLM can verify if p2p is "
            "actually working."
        ),
    )
    allreduce_use_symm_mem: bool = Field(
        default=True,
        description="Whether to use pytorch symmetric memory for allreduce.",
    )
    allreduce_use_flashinfer: bool = Field(
        default=False,
        description="Whether to use FlashInfer allreduce.",
    )
    use_nccl_symm_mem: bool = Field(
        default=False,
        description="Flag to enable NCCL symmetric memory allocation and registration.",
    )
    msgpack_zero_copy_threshold: int = Field(
        default=256,
        description=(
            "Control the threshold for msgspec to use 'zero copy' for "
            "serialization/deserialization of tensors. Tensors below this "
            "limit will be encoded into the msgpack buffer, and tensors "
            "above will instead be sent via a separate message. While the "
            "sending side still actually copies the tensor in all cases, on "
            "the receiving side, tensors above this limit will actually be "
            "zero-copy decoded."
        ),
    )
    use_spinloop_ext: bool = Field(
        default=False,
        description=(
            "If set to 1, use Python spinloop extension to poll in a more "
            "efficient way when using the mp backend."
        ),
    )

    @model_validator(mode="after")
    def _default_dp_rank_local_to_dp_rank(self) -> "DistributedSettings":
        # If VLLM_DP_RANK_LOCAL is not explicitly set, fall back to VLLM_DP_RANK.
        if not _env_set("VLLM_DP_RANK_LOCAL"):
            self.dp_rank_local = self.dp_rank
        return self


class CompilationSettings(BaseSettings):
    model_config = _SUB_CONFIG

    use_aot_compile: bool = Field(
        default=False,
        description=(
            "Feature flag to enable/disable AOT compilation. This will ensure "
            "compilation is done in warmup phase and the compilation will be "
            "reused in subsequent calls."
        ),
    )
    force_aot_load: bool = Field(
        default=False,
        description=(
            "Force vllm to always load AOT compiled models from disk. Failure "
            "to load will result in a hard error when this is enabled. Will "
            "be ignored when VLLM_USE_AOT_COMPILE is disabled."
        ),
    )
    use_mega_aot_artifact: bool = Field(
        default=False,
        description=(
            "Enable loading compiled models directly from cached standalone "
            "compile artifacts without re-splitting graph modules. This "
            "reduces overhead during model loading by using "
            "reconstruct_serializable_fn_from_mega_artifact."
        ),
    )
    use_bytecode_hook: bool = Field(
        default=True,
        description=(
            "Feature flag to enable/disable bytecode in "
            "TorchCompileWithNoGuardsWrapper."
        ),
    )
    use_standalone_compile: bool = Field(
        default=True,
        description=(
            "Feature flag to enable/disable Inductor standalone compile. In "
            "torch <= 2.7 we ignore this flag; in torch >= 2.9 this is "
            "enabled by default."
        ),
    )
    enable_pregrad_passes: bool = Field(
        default=True,
        description=(
            "Inductor's pre-grad passes don't do anything for vLLM. The "
            "pre-grad passes get run even on cache-hit and negatively impact "
            "vllm cold compile times by O(1s). Can remove this after the "
            "following issue gets fixed. "
            "TODO(luka): maybe_inplace requires this. "
            "https://github.com/pytorch/pytorch/issues/174502"
        ),
    )
    use_breakable_cudagraph: bool = Field(
        default=False,
        description=(
            "Experimental: use breakable cudagraph capture/replay that does "
            "not rely on torch.compile."
        ),
    )
    enable_inductor_max_autotune: bool = Field(
        default=True,
        description=(
            "Enable max_autotune & coordinate_descent_tuning in inductor_config "
            "to compile static shapes passed from compile_sizes in "
            "compilation_config. If set to 1, enable max_autotune. By default, "
            "this is enabled (1)."
        ),
    )
    enable_inductor_coordinate_descent_tuning: bool = Field(
        default=True,
        description=(
            "If set to 1, enable coordinate_descent_tuning. By default, this "
            "is enabled (1)."
        ),
    )
    disable_compile_cache: bool = Field(
        default=False,
        description="Disable the torch.compile cache.",
    )
    compile_cache_save_format: Literal["binary", "unpacked"] = Field(
        default="binary",
        description=(
            "Format for saving torch.compile cache artifacts. "
            '"binary" saves as a binary file (safe for multiple vllm serve '
            "processes accessing the same torch compile cache). "
            '"unpacked" saves as a directory structure (for '
            "inspection/debugging). NOT multiprocess safe -- race conditions "
            "may occur with multiple processes. Allows viewing and setting "
            "breakpoints in Inductor's code output files."
        ),
    )
    use_layername: bool = Field(
        default=True,
        description=(
            'If set to "0", disable LayerName opaque type for layer_name '
            "parameters in custom ops. Defaults to enabled on torch >= 2.11."
        ),
    )
    use_v2_model_runner: bool | None = Field(
        default=None,
        description=(
            "Flag to control the v2 model runner. Tri-state: ``1`` forces "
            "the v2 runner on, ``0`` forces it off, and unset (default) "
            "lets the config decide based on model and platform support."
        ),
    )

    @field_validator("use_v2_model_runner", mode="before")
    @classmethod
    def _parse_use_v2_model_runner(cls, v: Any) -> Any:
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, str):
            return {"1": True, "0": False}.get(v.strip())
        return v

    @model_validator(mode="after")
    def _apply_aot_compile_defaults(self) -> "CompilationSettings":
        # VLLM_USE_AOT_COMPILE: dynamic default based on torch version and
        # disable_compile_cache.
        if not _env_set("VLLM_USE_AOT_COMPILE"):
            try:
                from vllm.utils.torch_utils import is_torch_equal_or_newer

                self.use_aot_compile = (
                    is_torch_equal_or_newer("2.10.0") and not self.disable_compile_cache
                )
            except ImportError:
                pass

        # VLLM_USE_MEGA_AOT_ARTIFACT: depends on torch version AND use_aot_compile.
        if not _env_set("VLLM_USE_MEGA_AOT_ARTIFACT"):
            try:
                from vllm.utils.torch_utils import is_torch_equal_or_newer

                self.use_mega_aot_artifact = (
                    is_torch_equal_or_newer("2.12.0.dev") and self.use_aot_compile
                )
            except ImportError:
                pass

        return self


class MediaSettings(BaseSettings):
    model_config = _SUB_CONFIG

    image_fetch_timeout: int = Field(
        default=5,
        description=(
            "Timeout for fetching images when serving multimodal models. "
            "Default is 5 seconds."
        ),
    )
    video_fetch_timeout: int = Field(
        default=30,
        description=(
            "Timeout for fetching videos when serving multimodal models. "
            "Default is 30 seconds."
        ),
    )
    audio_fetch_timeout: int = Field(
        default=10,
        description=(
            "Timeout for fetching audio when serving multimodal models. "
            "Default is 10 seconds."
        ),
    )
    media_cache: str = Field(
        default="",
        description=(
            "Directory for caching media downloads (images, video, audio "
            "fetched from URLs during inference). Empty string disables "
            "caching."
        ),
    )
    media_cache_max_size_mb: int = Field(
        default=5120,
        description=(
            "Maximum cache size in MB. When exceeded, least-recently-used "
            "entries are evicted. Default is 5120 (5 GB)."
        ),
    )
    media_cache_ttl_hours: float = Field(
        default=24,
        description=(
            "Time-to-live in hours for cached media files. Entries older "
            "than this are evicted regardless of cache size. Default is "
            "24 hours."
        ),
    )
    media_fetch_max_retries: int = Field(
        default=3,
        description=(
            "Maximum number of retries for fetching media (images, audio, "
            "video) from URLs. Each retry quadruples the timeout. "
            "Default is 3."
        ),
    )
    media_url_allow_redirects: bool = Field(
        default=True,
        description=(
            "Whether to allow HTTP redirects when fetching from media URLs. Defaults "
            "to True."
        ),
    )
    media_loading_thread_count: int = Field(
        default=8,
        description=(
            "Max number of workers for the thread pool handling media bytes "
            "loading. Set to 1 to disable parallel processing. Default is 8."
        ),
    )
    max_audio_clip_filesize_mb: int = Field(
        default=25,
        description=(
            "Maximum filesize in MB for a single audio file when processing "
            "speech-to-text requests. Files larger than this will be "
            "rejected. Default is 25 MB."
        ),
    )
    video_loader_backend: str = Field(
        default="opencv",
        description=(
            "Backend for Video IO -- selects the frame-sampling algorithm. "
            '"opencv": uniform sampling. '
            '"opencv_dynamic": duration-aware dynamic sampling. '
            '"identity": returns raw video bytes for model processor to '
            "handle. Custom backend implementations can be registered via "
            '`@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and '
            "imported at runtime. If a non-existing backend is used, an "
            "AssertionError will be thrown."
        ),
    )
    media_connector: str = Field(
        default="http",
        description=(
            "Media connector implementation. "
            '"http": Default connector that supports fetching media via '
            "HTTP. Custom implementations can be registered via "
            '`@MEDIA_CONNECTOR_REGISTRY.register("my_custom_media_connector")` '
            "and imported at runtime. If a non-existing backend is used, an "
            "AssertionError will be thrown."
        ),
    )
    mm_hasher_algorithm: Literal["blake3", "sha256", "sha512"] = Field(
        default="blake3",
        description=(
            "Hash algorithm for multimodal content hashing. "
            '"blake3": Default, fast cryptographic hash (not FIPS 140-3 '
            "compliant). "
            '"sha256": FIPS 140-3 compliant, widely supported. '
            '"sha512": FIPS 140-3 compliant, faster on 64-bit systems. '
            "Use sha256 or sha512 for FIPS compliance in government/"
            "enterprise deployments."
        ),
    )
    object_storage_shm_buffer_name: str | None = Field(
        default=None,
        description=(
            "Name of the POSIX shared-memory segment used for multimodal "
            "object storage. When unset, vLLM auto-generates a UUID-suffixed "
            "name and writes it back to the environment."
        ),
    )
    assets_cache_model_clean: bool = Field(
        default=False,
        description=(
            "If the env var is set, we will clean model file in this path "
            "$VLLM_ASSETS_CACHE/model_streamer/$model_name."
        ),
    )

    @field_validator("mm_hasher_algorithm", mode="before")
    @classmethod
    def _lower_mm_hasher(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _autogen_object_storage_shm_buffer_name(self) -> "MediaSettings":
        # If unset, auto-generate a UUID-suffixed name and write it back to
        # os.environ so subprocesses inherit the same value.
        if self.object_storage_shm_buffer_name is None:
            env_val = os.environ.get("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME")
            if env_val is not None:
                self.object_storage_shm_buffer_name = env_val
            else:
                new_name = f"VLLM_OBJECT_STORAGE_SHM_BUFFER_{uuid.uuid4().hex}"
                os.environ["VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"] = new_name
                self.object_storage_shm_buffer_name = new_name
        return self


class CpuSettings(BaseSettings):
    model_config = _SUB_CONFIG

    cpu_kvcache_space: int | None = Field(
        default=None,
        description=(
            "(CPU backend only) CPU key-value cache space. Default is None "
            "and will be set as 4 GB."
        ),
    )
    cpu_omp_threads_bind: str = Field(
        default="auto",
        description=(
            "(CPU backend only) CPU core ids bound by OpenMP threads, e.g., "
            '"0-31", "0,1,2", "0-31,33". CPU cores of different ranks are '
            "separated by '|'."
        ),
    )
    cpu_num_of_reserved_cpu: int | None = Field(
        default=None,
        description=(
            "(CPU backend only) CPU cores not used by OMP threads. Those CPU "
            "cores will not be used by OMP threads of a rank."
        ),
    )
    cpu_sgl_kernel: bool = Field(
        default=False,
        description=(
            "(CPU backend only) Whether to use SGL kernels, optimized for small batch."
        ),
    )
    cpu_attn_split_kv: bool = Field(
        default=True,
        description="(CPU backend only) Whether to enable attention split KV.",
    )
    cpu_int4_w4a8: bool = Field(
        default=True,
        description=(
            "(CPU backend only) Whether to use SGLang INT4 W4A8 kernels for AWQ."
        ),
    )
    zentorch_weight_prepack: bool = Field(
        default=True,
        description=(
            "(Zen CPU backend) Eagerly prepack weights into ZenDNN blocked "
            "layout at model load time. Eliminates per-inference layout "
            "conversion overhead."
        ),
    )


class RocmSettings(BaseSettings):
    model_config = _SUB_CONFIG

    rocm_sleep_mem_chunk_size: int = Field(
        default=256,
        description=(
            "Flag to control the chunk size (in MB) for sleeping memory "
            "allocations under ROCm."
        ),
    )
    rocm_use_aiter: bool = Field(
        default=False,
        description=(
            "Disable aiter ops unless specifically enabled. Acts as a parent "
            "switch to enable the rest of the other operations."
        ),
    )
    rocm_use_aiter_paged_attn: bool = Field(
        default=False,
        description="Whether to use aiter paged attention. By default is disabled.",
    )
    rocm_use_aiter_linear: bool = Field(
        default=True,
        description=(
            "Use aiter linear op if aiter ops are enabled. The following "
            "list of related ops -- scaled_mm (per-tensor / rowwise) -- use "
            "aiter tuned gemms for unquantized gemms."
        ),
    )
    rocm_use_aiter_moe: bool = Field(
        default=True,
        description="Whether to use aiter moe ops. By default is enabled.",
    )
    rocm_use_aiter_rmsnorm: bool = Field(
        default=True,
        description="Use aiter rms norm op if aiter ops are enabled.",
    )
    rocm_use_aiter_mla: bool = Field(
        default=True,
        description="Whether to use aiter mla ops. By default is enabled.",
    )
    rocm_use_aiter_mha: bool = Field(
        default=True,
        description="Whether to use aiter mha ops. By default is enabled.",
    )
    rocm_use_aiter_fp4_asm_gemm: bool = Field(
        default=False,
        description="Whether to use aiter fp4 gemm asm. By default is disabled.",
    )
    rocm_use_aiter_triton_rope: bool = Field(
        default=False,
        description="Whether to use aiter rope. By default is disabled.",
    )
    rocm_use_aiter_fp8bmm: bool = Field(
        default=True,
        description=(
            "Whether to use aiter triton fp8 bmm kernel. By default is enabled."
        ),
    )
    rocm_use_aiter_fp4bmm: bool = Field(
        default=True,
        description=(
            "Whether to use aiter triton fp4 bmm kernel. By default is enabled."
        ),
    )
    rocm_use_aiter_unified_attention: bool = Field(
        default=False,
        description="Use AITER triton unified attention for V1 attention.",
    )
    rocm_use_aiter_fusion_shared_experts: bool = Field(
        default=False,
        description=(
            "Whether to use aiter fusion shared experts ops. By default is disabled."
        ),
    )
    rocm_use_aiter_triton_gemm: bool = Field(
        default=True,
        description=(
            "Whether to use aiter triton kernels for gemm ops. By default is enabled."
        ),
    )
    rocm_use_skinny_gemm: bool = Field(
        default=True,
        description="Use rocm skinny gemms.",
    )
    rocm_fp8_padding: bool = Field(
        default=True,
        description="Pad the fp8 weights to 256 bytes for ROCm.",
    )
    rocm_moe_padding: bool = Field(
        default=True,
        description="Pad the weights for the moe kernel.",
    )
    rocm_shuffle_kv_cache_layout: bool = Field(
        default=False,
        description="Whether to use the shuffled kv cache layout.",
    )
    rocm_quick_reduce_quantization: Literal["FP", "INT8", "INT6", "INT4", "NONE"] = (
        Field(
            default="NONE",
            description=(
                "Custom quick allreduce kernel for MI3* cards. Choice of "
                "quantization level: FP, INT8, INT6, INT4 or NONE. "
                "Recommended for large models to get allreduce."
            ),
        )
    )
    rocm_quick_reduce_cast_bf16_to_fp16: bool = Field(
        default=True,
        description=(
            "Custom quick allreduce kernel for MI3* cards. Due to the lack "
            "of the bfloat16 asm instruction, bfloat16 kernels are slower "
            "than fp16. If the environment variable is set to 1, the input "
            "is converted to fp16."
        ),
    )
    rocm_quick_reduce_max_size_bytes_mb: int | None = Field(
        default=None,
        description=(
            "Custom quick allreduce kernel for MI3* cards. Controls the "
            "maximum allowed number of data bytes (MB) for custom quick "
            "allreduce communication. Default: 2048 MB. Data exceeding "
            "this size will use either custom allreduce or RCCL "
            "communication."
        ),
    )
    rocm_quick_reduce_min_size_bytes_mb: int | None = Field(
        default=None,
        description=(
            "Custom quick allreduce kernel for MI3* cards. Controls the "
            "minimum allowed number of data bytes (MB) required to use "
            "custom quick allreduce communication. If unset, use the "
            "built-in threshold table."
        ),
    )
    rocm_quick_reduce_quantization_min_size_kb: int | None = Field(
        default=None,
        description=(
            "Controls the minimum tensor size (KB, where 1 KB = 1024 bytes) "
            "required to use the configured QuickReduce codec. Smaller "
            "tensors use FP QuickReduce. This does not affect QuickReduce "
            "eligibility."
        ),
    )
    rocm_fp8_mfma_page_attn: bool = Field(
        default=False,
        description="If set, use the fp8 mfma in rocm paged attention.",
    )


class TpuXpuSettings(BaseSettings):
    model_config = _SUB_CONFIG

    xla_check_recompilation: bool = Field(
        default=False,
        description="If set, assert on XLA recompilation after each execution step.",
    )
    xla_use_spmd: bool = Field(
        default=False,
        description="Enable SPMD mode for TPU backend.",
    )
    tpu_bucket_padding_gap: int = Field(
        default=0,
        description=(
            "Gap between padding buckets for the forward pass. So we have 8, "
            "we will run forward pass with [16, 24, 32, ...]."
        ),
    )
    tpu_most_model_len: int | None = Field(
        default=None,
        description=(
            "The 'most' model length to optimize for on TPU. If set, the TPU "
            "backend pre-compiles graphs targeted at this length and falls "
            "back gracefully for sequences that exceed it."
        ),
    )
    tpu_using_pathways: bool = Field(
        default_factory=_tpu_pathways_default,
        validation_alias=_TPU_PATHWAYS_SENTINEL,
        description="Whether using Pathways.",
    )
    xpu_enable_xpu_graph: bool = Field(
        default=False,
        description="Whether enable XPU graph on Intel GPU.",
    )
    xpu_use_sampler_kernel: bool = Field(
        default=True,
        description="Whether use xpu specific sample kernel.",
    )
    sparse_indexer_max_logits_mb: int = Field(
        default=512,
        description=(
            "Maximum size (in MB) for logits tensor in sparse MLA indexer "
            "prefill chunks. Bounds the [M, N] float32 logits tensor to "
            "prevent CUDA OOM. Default: 512 MB."
        ),
    )


class FlashInferSettings(BaseSettings):
    model_config = _SUB_CONFIG

    use_flashinfer_sampler: bool = Field(
        default=True,
        description=(
            "Whether to use the FlashInfer top-k / top-p sampler on CUDA. "
            "Enabled by default when the hardware supports it -- set to 0 "
            "to opt out explicitly, which forces the PyTorch-native "
            "(Triton for bs>=8) path."
        ),
    )
    use_flashinfer_moe_fp16: bool = Field(
        default=False,
        description="Allow use of FlashInfer BF16 MoE kernels for fused moe ops.",
    )
    use_flashinfer_moe_fp8: bool = Field(
        default=False,
        description="Allow use of FlashInfer FP8 MoE kernels for fused moe ops.",
    )
    use_flashinfer_moe_fp4: bool = Field(
        default=False,
        description="Allow use of FlashInfer NVFP4 MoE kernels for fused moe ops.",
    )
    use_flashinfer_moe_int4: bool = Field(
        default=False,
        description="Allow use of FlashInfer MxInt4 MoE kernels for fused moe ops.",
    )
    use_flashinfer_moe_mxfp4_mxfp8: bool = Field(
        default=False,
        description=(
            "If set to 1, use the FlashInfer MXFP8 (activation) x MXFP4 "
            "(weight) MoE backend."
        ),
    )
    use_flashinfer_moe_mxfp4_mxfp8_cutlass: bool = Field(
        default=False,
        description=(
            "If set to 1, use the FlashInfer CUTLASS backend for MXFP8 "
            "(activation) x MXFP4 (weight) MoE. This is separate from the "
            "TRTLLMGEN path controlled by VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8."
        ),
    )
    use_flashinfer_moe_mxfp4_bf16: bool = Field(
        default=False,
        description=(
            "If set to 1, use the FlashInfer BF16 (activation) x MXFP4 "
            "(weight) MoE backend."
        ),
    )
    flashinfer_moe_backend: Literal["throughput", "latency", "masked_gemm"] = Field(
        default="latency",
        description=(
            "Flashinfer MoE backend for vLLM's fused Mixture-of-Experts "
            "support. Both require compute capability 10.0 or above. "
            'Available options: "throughput" -- [default] Uses CUTLASS '
            "kernels optimized for high-throughput batch inference. "
            '"latency" -- Uses TensorRT-LLM kernels optimized for low-'
            "latency inference."
        ),
    )
    flashinfer_autotune_cache_dir: str | None = Field(
        default=None,
        description=(
            "Override the directory for the FlashInfer autotune config cache."
        ),
    )
    flashinfer_allreduce_backend: Literal["auto", "trtllm", "mnnvl"] = Field(
        default="auto",
        description="Flashinfer fused allreduce backend.",
    )
    flashinfer_workspace_buffer_size: int = Field(
        default=394 * 1024 * 1024,
        description="Control the workspace buffer size for the FlashInfer backend.",
    )
    flashinfer_allreduce_fusion_thresholds_mb: Annotated[dict, NoDecode] = Field(
        default_factory=dict,
        description=(
            "Specifies the thresholds of the communicated tensor sizes "
            "under which vllm should use flashinfer fused allreduce. The "
            "variable should be a JSON with the following format: "
            "{ <world size>: <max size in mb> }. Unspecified world sizes "
            "will fall back to { 2: 64, 4: 1, <everything else>: 0.5 }."
        ),
    )
    blockscale_fp8_gemm_flashinfer: bool = Field(
        default=True,
        description=(
            "Allow use of FlashInfer FP8 block-scale GEMM for linear "
            "layers. This uses TensorRT-LLM kernels and requires SM90+ "
            "(Hopper)."
        ),
    )
    has_flashinfer_cubin: bool = Field(
        default=False,
        description=(
            "If set, it means we pre-downloaded cubin files and flashinfer "
            "will read the cubin files directly."
        ),
    )
    max_tokens_per_expert_fp4_moe: int = Field(
        default=163840,
        description=(
            "Control the maximum number of tokens per expert supported by "
            "the NVFP4 MoE CUTLASS Kernel. This value is used to create a "
            "buffer for the blockscale tensor of activations NVFP4 "
            "Quantization. This is used to prevent the kernel from running "
            "out of memory."
        ),
    )

    @field_validator("flashinfer_allreduce_fusion_thresholds_mb", mode="before")
    @classmethod
    def _parse_json_thresholds(cls, v: Any) -> Any:
        if v is None or v == "":
            return {}
        if isinstance(v, str):
            return json.loads(v)
        return v

    @model_validator(mode="after")
    def _warn_deprecated_moe_backend_envs(self) -> "FlashInferSettings":
        moe_backend_msg = (
            "Use --moe-backend (e.g. flashinfer_trtllm, flashinfer_cutlass)."
        )
        for var in (
            "VLLM_USE_FLASHINFER_MOE_FP16",
            "VLLM_USE_FLASHINFER_MOE_FP8",
            "VLLM_USE_FLASHINFER_MOE_FP4",
            "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
        ):
            _warn_deprecated_env(var, "v0.23", moe_backend_msg)
        _warn_deprecated_env(
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
            "v0.23",
            "Use --moe-backend flashinfer_trtllm with "
            "--quantization_config.moe.activation mxfp8.",
        )
        _warn_deprecated_env(
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
            "v0.23",
            "Use --moe-backend flashinfer_cutlass with "
            "--quantization_config.moe.activation mxfp8.",
        )
        _warn_deprecated_env(
            "VLLM_FLASHINFER_MOE_BACKEND",
            "v0.23",
            "Use --moe-backend flashinfer_trtllm, flashinfer_cutlass, or "
            "flashinfer_cutedsl.",
        )
        return self


class QuantSettings(BaseSettings):
    model_config = _SUB_CONFIG

    marlin_use_atomic_add: bool = Field(
        default=False,
        description="Whether to use atomicAdd reduce in gptq/awq marlin kernel.",
    )
    marlin_input_dtype: Literal["int8", "fp8"] | None = Field(
        default=None,
        description="The activation dtype for marlin kernel.",
    )
    humming_online_quant_config: Annotated[dict[str, Any] | None, NoDecode] = Field(
        default=None,
        description="The online quantization dtype for humming kernel.",
    )
    humming_input_quant_config: Annotated[dict[str, Any] | None, NoDecode] = Field(
        default=None,
        description="The activation dtype config for humming kernel.",
    )
    humming_use_f16_accum: bool | None = Field(
        default=False,
        description="Whether to use fp16 accumulator mma.",
    )
    humming_moe_gemm_type: Literal["indexed", "grouped", "auto"] | None = Field(
        default=None,
        description=(
            "Whether to use indexed gemm for humming moe. If 1, force use "
            "indexed gemm. If 0, force use grouped gemm. If None, choose "
            "better gemm type automatically."
        ),
    )
    mxfp4_use_marlin: bool | None = Field(
        default=None,
        description="Whether to use marlin kernel in mxfp4 quantization method.",
    )
    deepepll_nvfp4_dispatch: bool = Field(
        default=False,
        description=(
            "Whether to use DeepEPLL kernels for NVFP4 quantization and "
            "dispatch method. Only supported on Blackwell GPUs and with "
            "https://github.com/deepseek-ai/DeepEP/pull/341."
        ),
    )
    use_deep_gemm: bool = Field(
        default=True,
        description="Allow use of DeepGemm kernels for fused moe ops.",
    )
    moe_use_deep_gemm: bool = Field(
        default=True,
        description=(
            "Allow use of DeepGemm specifically for MoE fused ops (overrides only MoE)."
        ),
    )
    use_deep_gemm_e8m0: bool = Field(
        default=True,
        description=(
            "Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs."
        ),
    )
    use_deep_gemm_tma_aligned_scales: bool = Field(
        default=True,
        description="Whether to create TMA-aligned scale tensor when DeepGEMM is used.",
    )
    deep_gemm_warmup: Literal["skip", "full", "relax"] = Field(
        default="relax",
        description=(
            "DeepGemm JITs the kernels on-demand. The warmup attempts to "
            "make DeepGemm JIT all the required kernels before model "
            "execution so there is no JIT'ing in the hot-path. However, "
            "this warmup increases the engine startup time by a couple of "
            'minutes. Available options: "skip": skip warmup. "full": '
            "warmup deepgemm by running all possible gemm shapes the "
            'engine could encounter. "relax": select gemm shapes to run '
            "based on some heuristics. The heuristic aims to have the same "
            "effect as running all possible gemm shapes, but provides no "
            "guarantees."
        ),
    )
    use_fused_moe_grouped_topk: bool = Field(
        default=True,
        description="Whether to use fused grouped_topk used for MoE expert selection.",
    )
    deepep_buffer_size_mb: int = Field(
        default=1024,
        description="The size in MB of the buffers (NVL and RDMA) used by DeepEP.",
    )
    deepep_high_throughput_force_intra_node: bool = Field(
        default=False,
        description=(
            "Force DeepEP to use intranode kernel for inter-node "
            "communication in high throughput mode. This is useful to "
            "achieve higher prefill throughput on systems that support "
            "multi-node nvlink (e.g. GB200)."
        ),
    )
    deepep_low_latency_use_mnnvl: bool = Field(
        default=False,
        description=(
            "Allow DeepEP to use MNNVL (multi-node nvlink) for internode_ll "
            "kernel; turn this on for better latency on GB200-like systems."
        ),
    )
    dbo_comm_sms: int = Field(
        default_factory=_default_dbo_comm_sms,
        description=(
            "The number of SMs/CUs to allocate for communication kernels "
            "when running DBO; the rest will be allocated to compute. "
            "Default: 20 on CUDA (SMs), 64 on ROCm (CUs)."
        ),
    )
    multi_stream_gemm_token_threshold: int = Field(
        default=1024,
        description=(
            "Token-count cutoff for multi-stream overlap of the attention "
            "input GEMM with auxiliary GEMMs (e.g. fused_wqa_wkv overlapped "
            "with indexer weights / kv-score projections in DeepSeek-V4). "
            "At or below this many tokens the FP8 main GEMM has idle SMs "
            "to share with the bf16 aux GEMMs and overlap is a 5-45% win; "
            "above it the FP8 GEMM saturates the device and the "
            "cross-stream sync becomes pure overhead. Set to 0 to disable "
            "the multi-stream path entirely. See #PR 41526 for the "
            "empirical result for the default value of 1024 tokens."
        ),
    )
    shared_experts_stream_token_threshold: int = Field(
        default=256,
        description=(
            "Limits when we run shared_experts in a separate stream. We "
            "found out that for large batch sizes, the separate stream "
            "execution is not beneficial (most likely because of the "
            "input clone). "
            "TODO(alexm-redhat): Tune to be more dynamic based on GPU type."
        ),
    )
    disable_shared_experts_stream: bool = Field(
        default=False,
        description=(
            "Disables parallel execution of shared_experts via separate cuda stream."
        ),
    )
    moe_routing_simulation_strategy: str = Field(
        default="",
        description=(
            "MoE routing strategy selector. See "
            "`RoutingSimulator.get_available_strategies()` for available "
            "strategies. Custom routing strategies can be registered by "
            "RoutingSimulator.register_strategy(). Note: custom strategies "
            "may not produce correct model outputs."
        ),
    )
    nvfp4_gemm_backend: (
        Literal[
            "flashinfer-b12x",
            "flashinfer-cudnn",
            "flashinfer-trtllm",
            "flashinfer-cutlass",
            "cutlass",
            "marlin",
            "emulation",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            'Supported options: "flashinfer-b12x" -- use flashinfer b12x '
            'GEMM backend (SM120/121); "flashinfer-cudnn" -- use flashinfer '
            'cudnn GEMM backend; "flashinfer-trtllm" -- use flashinfer '
            'trtllm GEMM backend; "flashinfer-cutlass" -- use flashinfer '
            'cutlass GEMM backend; "cutlass" -- use cutlass GEMM backend; '
            '"marlin" -- use marlin GEMM backend (for GPUs without native '
            'FP4 support); "emulation" -- use BF16/FP16 GEMM, dequantizing '
            "weights and running QDQ on activations (only meant for "
            "research purposes to run on devices where NVFP4 GEMM kernels "
            "are not available); <none> -- automatically pick an available "
            "backend."
        ),
    )
    use_nvfp4_ct_emulations: bool = Field(
        default=False,
        description=(
            "Controls whether or not emulations are used for NVFP4 "
            "generations on machines < 100 for compressed-tensors models."
        ),
    )
    q_scale_constant: int = Field(
        default=200,
        alias="Q_SCALE_CONSTANT",
        description=(
            "Divisor for dynamic query scale factor calculation for FP8 KV Cache."
        ),
    )
    k_scale_constant: int = Field(
        default=200,
        alias="K_SCALE_CONSTANT",
        description=(
            "Divisor for dynamic key scale factor calculation for FP8 KV Cache."
        ),
    )
    v_scale_constant: int = Field(
        default=100,
        alias="V_SCALE_CONSTANT",
        description=(
            "Divisor for dynamic value scale factor calculation for FP8 KV Cache."
        ),
    )
    kv_cache_layout: Literal["NHD", "HND"] | None = Field(
        default=None,
        description=(
            "KV Cache layout used throughout vllm. Some common values are: "
            "NHD, HND, where N=num_blocks, H=num_heads and D=head_size. "
            "The default value will leave the layout choice to the "
            "backend. Mind that backends may only implement and support a "
            "subset of all possible layouts."
        ),
    )
    ssm_conv_state_layout: Literal["SD", "DS"] | None = Field(
        default=None,
        description=(
            "SSM conv state layout used for Mamba models. "
            "SD: (state_len, dim) -- dim contiguous (default). "
            "DS: (dim, state_len) -- TP-sharded dim on dim1, consistent "
            "with SSM temporal state and HND KV cache layout."
        ),
    )
    mla_disable: bool = Field(
        default=False,
        description="If set, vLLM will disable the MLA attention optimizations.",
    )
    triton_attn_use_td: bool | None = Field(
        default=None,
        description=(
            "Use tensor descriptors for Q/K/V loads and output stores in the "
            "Triton unified-attention kernel. Enables HW 2D block reads on "
            "Intel Xe2/Xe3; the non-TD branch is dead-code-eliminated at "
            "Triton compile time so other platforms see no overhead. "
            "Tri-state override: unset (default) lets the `triton_attn` "
            "backend auto-select per platform (currently auto-enabled on "
            "XPU only); ``1`` forces TD on; ``0`` forces TD off. Useful "
            "for A/B benchmarking the TD path."
        ),
    )
    compute_nans_in_logits: bool = Field(
        default=False,
        description=(
            "Enable checking whether the generated logits contain NaNs, "
            "indicating corrupted output. Useful for debugging low level "
            "bugs or bad hardware but it may add compute overhead."
        ),
    )
    use_fbgemm: bool = Field(
        default=False,
        description="Flag to enable FBGemm kernels on model execution.",
    )
    use_oink_ops: bool = Field(
        default=False,
        description=(
            "Optional: enable external Oink custom ops (e.g., Blackwell "
            "RMSNorm). Disabled by default."
        ),
    )
    batch_invariant: bool = Field(
        default=False,
        description=(
            "Enable batch-invariant mode: deterministic results regardless "
            "of batch composition. Requires NVIDIA GPU with compute "
            "capability >= 9.0."
        ),
    )
    float32_matmul_precision: Literal["highest", "high", "medium"] = Field(
        default="highest",
        description=(
            "Controls PyTorch float32 matmul precision mode within vLLM "
            "workers. Valid options mirror torch.set_float32_matmul_precision."
        ),
    )
    use_triton_awq: bool = Field(
        default=False,
        description="If set, vLLM will use Triton implementations of AWQ.",
    )

    @field_validator("float32_matmul_precision", mode="before")
    @classmethod
    def _lower_float32(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator("moe_routing_simulation_strategy", mode="after")
    @classmethod
    def _lower_moe_routing(cls, v: str) -> str:
        return v.lower()

    @field_validator(
        "humming_online_quant_config", "humming_input_quant_config", mode="before"
    )
    @classmethod
    def _parse_humming_json(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if os.path.exists(v):
                with open(v) as f:
                    return json.load(f)
            return json.loads(v)
        return v

    @field_validator("humming_use_f16_accum", mode="before")
    @classmethod
    def _parse_humming_f16(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return bool(int(v))
        return v

    @field_validator("mxfp4_use_marlin", mode="before")
    @classmethod
    def _parse_mxfp4(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return bool(int(v))
        return v

    @field_validator("triton_attn_use_td", mode="before")
    @classmethod
    def _parse_triton_attn_use_td(cls, v: Any) -> Any:
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, str):
            return {"1": True, "0": False}.get(v.strip())
        return v

    @model_validator(mode="after")
    def _warn_deprecated_backend_envs(self) -> "QuantSettings":
        _warn_deprecated_env(
            "VLLM_MXFP4_USE_MARLIN",
            "v0.23",
            "Use --moe-backend marlin or --linear-backend marlin.",
        )
        _warn_deprecated_env(
            "VLLM_USE_NVFP4_CT_EMULATIONS",
            "v0.23",
            "Use --linear-backend emulation.",
        )
        _warn_deprecated_env(
            "VLLM_NVFP4_GEMM_BACKEND",
            "v0.23",
            "Use --linear-backend.",
        )
        _warn_deprecated_env(
            "VLLM_USE_FBGEMM",
            "v0.23",
            "Use --linear-backend fbgemm.",
        )
        return self


class ConnectorSettings(BaseSettings):
    model_config = _SUB_CONFIG

    nixl_side_channel_host: str = Field(
        default="localhost",
        description="IP address used for NIXL handshake between remote agents.",
    )
    nixl_side_channel_port: int = Field(
        default=5600,
        description="Port used for NIXL handshake between remote agents.",
    )
    nixl_ep_max_num_ranks: int = Field(
        default=32,
        description="NIXL EP max number of ranks.",
    )
    mooncake_bootstrap_port: int = Field(
        default=8998,
        description="Port used for Mooncake handshake between remote agents.",
    )
    mooncake_abort_request_timeout: int = Field(
        default=480,
        description=(
            "Timeout (in seconds) for MooncakeConnector in PD disaggregated setup."
        ),
    )
    mooncake_store_tier_log: bool = Field(
        default=False,
        description=(
            "Log per-batch memory/disk tier breakdown on external GETs in "
            "the Mooncake store connector."
        ),
    )
    mooncake_disk_staging_usable_ratio: float = Field(
        default=0.9,
        description=(
            "Fraction of the owner's DirectIO staging buffer to fill per "
            "GET batch in the Mooncake store connector."
        ),
    )
    preferred_segment: str | None = Field(
        default=None,
        alias="MOONCAKE_PREFERRED_SEGMENT",
        description=(
            'Pin this rank to a specific Mooncake owner segment ("host:port").'
        ),
    )
    requester_local_hostname: str | None = Field(
        default=None,
        alias="MOONCAKE_REQUESTER_LOCAL_HOSTNAME",
        description=(
            "Override the hostname the rank registers as a Mooncake requester."
        ),
    )
    kv_events_use_int_block_hashes: bool = Field(
        default=True,
        description=(
            "Represent block hashes in KV cache events as 64-bit integers "
            "instead of raw bytes. Defaults to True for backward "
            "compatibility."
        ),
    )
    disable_request_id_randomization: bool = Field(
        default=False,
        description=(
            "Temporary: skip adding random suffix to internal request IDs. "
            "May be needed for KV connectors that match request IDs across "
            "instances."
        ),
    )
    elastic_ep_scale_up_launch: bool = Field(
        default=False,
        description=(
            "Whether it is a scale up launch engine for elastic EP. Should "
            "only be set by EngineCoreClient."
        ),
    )
    elastic_ep_drain_requests: bool = Field(
        default=False,
        description=(
            "Whether to wait for all requests to drain before sending the "
            "scaling command in elastic EP."
        ),
    )
    use_simple_kv_offload: bool = Field(
        default=False,
        description="Enable simple KV offload.",
    )
    weight_offloading_disable_pin_memory: bool = Field(
        default=False,
        description="Disable using pytorch's pin memory for CPU offloading.",
    )
    weight_offloading_disable_uva: bool = Field(
        default=False,
        description=(
            "Disable using UVA (Unified Virtual Addressing) for CPU offloading."
        ),
    )
    enable_cudagraph_gc: bool = Field(
        default=False,
        description=(
            "Controls garbage collection during CUDA graph capture. If set "
            "to 0 (default), enables GC freezing to speed up capture time. "
            "If set to 1, allows GC to run during capture."
        ),
    )
    memory_profiler_estimate_cudagraphs: bool = Field(
        default=True,
        description=(
            "If set to 1, enable CUDA graph memory estimation during "
            "memory profiling. This profiles CUDA graph memory usage to "
            "provide more accurate KV cache memory allocation. Enabled by "
            "default as of v0.21.0."
        ),
    )
    v1_output_proc_chunk_size: int = Field(
        default=128,
        description=(
            "Controls the maximum number of requests to handle in a single "
            "asyncio task when processing per-token outputs in the V1 "
            "AsyncLLM interface. It is applicable when handling a high "
            "concurrency of streaming requests. Setting this too high can "
            "result in a higher variance of inter-message latencies."
        ),
    )
    v1_use_outlines_cache: bool = Field(
        default=False,
        description=(
            "Whether to turn on the outlines cache for V1. This cache is "
            "unbounded and on disk, so it's not safe to use in an "
            "environment with potentially malicious users."
        ),
    )
    xgrammar_cache_mb: int = Field(
        default=512,
        description=(
            "Control the cache size used by the xgrammar compiler. The "
            "default of 512 MB should be enough for roughly 1000 JSON "
            "schemas. It can be changed with this variable if needed for "
            "some reason."
        ),
    )


class UsageSettings(BaseSettings):
    model_config = _SUB_CONFIG

    usage_stats_server: str = Field(
        default="https://stats.vllm.ai",
        description="URL of the server used for usage stats collection.",
    )
    no_usage_stats: bool = Field(
        default=False,
        description=(
            "If set, disable sending usage statistics to the vLLM usage stats server."
        ),
    )
    do_not_track: bool = Field(
        default=False,
        validation_alias=AliasChoices("VLLM_DO_NOT_TRACK", "DO_NOT_TRACK"),
        description=(
            "Disable usage stats reporting. Also accepts the legacy "
            "DO_NOT_TRACK environment variable."
        ),
    )
    usage_source: str = Field(
        default="production",
        description="Label identifying the deployment context reported in usage stats.",
    )
    ci_use_s3: bool = Field(
        default=False,
        description=(
            "Whether to use S3 path for model loading in CI via RunAI Streamer."
        ),
    )
    test_force_fp8_marlin: bool = Field(
        default=False,
        description=(
            "If set, forces FP8 Marlin to be used for FP8 quantization "
            "regardless of the hardware support for FP8 compute."
        ),
    )
    test_force_load_format: str = Field(
        default="dummy",
        description=(
            "Test-only: forces the model loader to use the given load "
            'format (e.g. "dummy") regardless of the format detected from '
            "the checkpoint."
        ),
    )
    use_modelscope: bool = Field(
        default=False,
        description=(
            "If true, will load models from ModelScope instead of Hugging "
            "Face Hub. Note that the value is true or false, not numbers."
        ),
    )
    use_fastokens: bool = Field(
        default=False,
        description=(
            "If true, replace the Rust BPE backend that powers HF fast "
            "tokenizers with the `fastokens` "
            "(https://github.com/crusoecloud/fastokens) shim. Applies to any "
            "tokenizer mode that loads an HF fast tokenizer (`hf`, "
            "`deepseek_v32`, `deepseek_v4`, `qwen_vl`, ...). The `fastokens` "
            "Python package must be installed."
        ),
    )
    s3_access_key_id: str | None = Field(
        default=None,
        alias="S3_ACCESS_KEY_ID",
        description="S3 access key id, used for tensorizer to load model from S3.",
    )
    s3_secret_access_key: str | None = Field(
        default=None,
        alias="S3_SECRET_ACCESS_KEY",
        description="S3 secret access key, used for tensorizer to load model from S3.",
    )
    s3_endpoint_url: str | None = Field(
        default=None,
        alias="S3_ENDPOINT_URL",
        description="S3 endpoint URL, used for tensorizer to load model from S3.",
    )
    plugins: Annotated[list[str] | None, NoDecode] = Field(
        default=None,
        description=(
            "A list of plugin names to load, separated by commas. If this "
            "is not set, it means all plugins will be loaded. If this is "
            "set to an empty string, no plugins will be loaded."
        ),
    )
    disabled_kernels: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "List of quantization kernels that should be disabled, used "
            "for testing and performance comparisons. Currently only "
            "affects MPLinearKernel selection (kernels: MacheteLinearKernel, "
            "MarlinLinearKernel, ExllamaLinearKernel)."
        ),
    )
    allow_runtime_lora_updating: bool = Field(
        default=False,
        description="If set, allow loading or unloading lora adapters in runtime.",
    )
    gpt_oss_system_tool_mcp_labels: Annotated[
        set[Literal["container", "code_interpreter", "web_search_preview"]],
        NoDecode,
    ] = Field(
        default_factory=set,
        description=(
            "Valid values are container, code_interpreter, web_search_preview. "
            "Example: VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=container,code_interpreter. "
            "If the server_label of your mcp tool is not in this list it will be "
            "completely ignored."
        ),
    )
    gpt_oss_harmony_system_instructions: bool = Field(
        default=False,
        description="Allows harmony instructions to be injected on system messages.",
    )
    use_experimental_parser_context: bool = Field(
        default=False,
        description=(
            "Experimental: use this to enable MCP tool calling for non harmony models."
        ),
    )
    lora_disable_pdl: bool = Field(
        default=False,
        description=(
            "Disable PDL for LoRA, as enabling PDL with LoRA on SM100 "
            "causes Triton compilation to fail."
        ),
    )
    lora_enable_dual_stream: bool = Field(
        default=False,
        description=(
            "Whether to enable dual cuda streams for LoRA computation (used "
            "by both BaseLinearLayerWithLoRA and FusedMoEWithLoRA to overlap "
            "the base layer compute with the LoRA fast path)."
        ),
    )
    enable_fla_packed_recurrent_decode: bool = Field(
        default=True,
        description=(
            "Whether to enable FLA's packed recurrent decode path for "
            "linear-attention models. Default: enabled."
        ),
    )
    pp_layer_partition: str | None = Field(
        default=None,
        description="Pipeline stage partition strategy.",
    )

    @field_validator("plugins", mode="before")
    @classmethod
    def _parse_plugins(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return v.split(",")
        return v

    @field_validator("disabled_kernels", mode="before")
    @classmethod
    def _parse_disabled_kernels(cls, v: Any) -> Any:
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return v.split(",")
        return v

    @field_validator("gpt_oss_system_tool_mcp_labels", mode="before")
    @classmethod
    def _parse_gpt_oss_labels(cls, v: Any) -> Any:
        if v is None or v == "":
            return set()
        if isinstance(v, (set, list, tuple)):
            return {str(x).strip() for x in v if str(x).strip()}
        if isinstance(v, str):
            return {p.strip() for p in v.split(",") if p.strip()}
        return v


# ----------------------------------------------------------------------------
# Root settings
# ----------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True)

    build: BuildSettings = Field(default_factory=BuildSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging_: LoggingSettings = Field(default_factory=LoggingSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)
    compilation: CompilationSettings = Field(default_factory=CompilationSettings)
    media: MediaSettings = Field(default_factory=MediaSettings)
    cpu: CpuSettings = Field(default_factory=CpuSettings)
    rocm: RocmSettings = Field(default_factory=RocmSettings)
    tpu_xpu: TpuXpuSettings = Field(default_factory=TpuXpuSettings)
    flashinfer: FlashInferSettings = Field(default_factory=FlashInferSettings)
    quant: QuantSettings = Field(default_factory=QuantSettings)
    connector: ConnectorSettings = Field(default_factory=ConnectorSettings)
    usage: UsageSettings = Field(default_factory=UsageSettings)


# ----------------------------------------------------------------------------
# Registry: env var name -> (sub_attr, field_name)
# ----------------------------------------------------------------------------


def resolve_env_name(info: FieldInfo, field_name: str, prefix: str) -> str:
    """Resolve the canonical environment variable name for a pydantic field.

    Used by the registry builder and the docs generator
    (docs/mkdocs/plugins/gen_env_vars.py).

    - explicit ``alias=`` wins (used for non-VLLM_ env vars)
    - ``AliasChoices`` -> first choice (e.g., DO_NOT_TRACK fallback)
    - sentinel for VLLM_TPU_USING_PATHWAYS -> the canonical name
    - otherwise: ``prefix + field_name.upper()``
    """
    if info.alias is not None:
        return info.alias
    va = info.validation_alias
    if va is None:
        return prefix + field_name.upper()
    if isinstance(va, AliasChoices):
        first = va.choices[0]
        return first if isinstance(first, str) else str(first)
    if isinstance(va, str):
        if va == _TPU_PATHWAYS_SENTINEL:
            return "VLLM_TPU_USING_PATHWAYS"
        return va
    return prefix + field_name.upper()


def _build_registry() -> dict[str, tuple[str, str]]:
    registry: dict[str, tuple[str, str]] = {}
    for sub_attr, sub_field in Settings.model_fields.items():
        annotation = sub_field.annotation
        if not (isinstance(annotation, type) and issubclass(annotation, BaseSettings)):
            continue
        sub_cls: type[BaseSettings] = annotation
        prefix = sub_cls.model_config.get("env_prefix", "") or ""
        for field_name, info in sub_cls.model_fields.items():
            env_name = resolve_env_name(info, field_name, prefix)
            if env_name in registry:
                raise RuntimeError(
                    f"Env var {env_name!r} registered by multiple fields: "
                    f"{registry[env_name]} and {(sub_attr, field_name)}"
                )
            registry[env_name] = (sub_attr, field_name)
    return registry


_VAR_TO_PATH: dict[str, tuple[str, str]] = _build_registry()


def _build_sub_classes() -> dict[str, type[BaseSettings]]:
    out: dict[str, type[BaseSettings]] = {}
    for sub_attr, sub_field in Settings.model_fields.items():
        annotation = sub_field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseSettings):
            out[sub_attr] = annotation
    return out


_SUB_CLASSES: dict[str, type[BaseSettings]] = _build_sub_classes()


# ----------------------------------------------------------------------------
# Lazy settings accessor
# ----------------------------------------------------------------------------


_settings: Settings | None = None


def _get_settings() -> Settings:
    """Return the cached Settings singleton (lazily constructed)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_attr(name: str) -> Any:
    sub_attr, field_name = _VAR_TO_PATH[name]
    if _is_envs_cache_enabled():
        # Cached path: read off the singleton.
        sub = getattr(_get_settings(), sub_attr)
    else:
        # Construct only the requested sub-model so an invalid value in an
        # unrelated env var (validated by a different sub-model) doesn't
        # poison reads of this one. Matches the pre-refactor per-getter
        # behavior where only the var being read could fail.
        sub = _SUB_CLASSES[sub_attr]()
    return getattr(sub, field_name)


# ----------------------------------------------------------------------------
# Back-compat shim: environment_variables dict
# ----------------------------------------------------------------------------


def _make_env_getter(name: str) -> Callable[[], Any]:
    return lambda: _get_attr(name)


environment_variables: dict[str, Callable[[], Any]] = {
    name: _make_env_getter(name) for name in _VAR_TO_PATH
}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def __getattr__(name: str):
    """Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in _VAR_TO_PATH:
        return _get_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_VAR_TO_PATH.keys())


def is_set(name: str) -> bool:
    """Check if an environment variable is explicitly set."""
    if name in _VAR_TO_PATH:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_environ(hard_fail: bool) -> None:
    for env in os.environ:
        if env.startswith("VLLM_") and env not in _VAR_TO_PATH:
            if hard_fail:
                raise ValueError(f"Unknown vLLM environment variable detected: {env}")
            else:
                logger.warning("Unknown vLLM environment variable detected: %s", env)


def _is_envs_cache_enabled() -> bool:
    """Checked if __getattr__ is wrapped with functools.cache"""
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """Enable caching of environment variables.

    NOTE: Currently invoked after service initialization to reduce runtime
    overhead. This also means environment variables should NOT be updated
    after the service is initialized.
    """
    if _is_envs_cache_enabled():
        return
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    for key in _VAR_TO_PATH:
        __getattr__(key)


def disable_envs_cache() -> None:
    """Reset the environment variables cache.

    Useful for isolating environments between unit tests.
    """
    global __getattr__, _settings
    if _is_envs_cache_enabled():
        assert hasattr(__getattr__, "__wrapped__")
        __getattr__ = __getattr__.__wrapped__
    _settings = None


def get_vllm_port() -> int | None:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggests k8s service discovery issue.
    """
    if "VLLM_PORT" not in os.environ:
        return None
    port = os.getenv("VLLM_PORT", "0")
    try:
        return int(port)
    except ValueError as err:
        from urllib3.util import parse_url

        parsed = parse_url(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://docs.vllm.ai/en/stable/configuration/env_vars/?h=vllm_port#environment-variables"
            ) from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err


def compile_factors() -> dict[str, object]:
    """Return env vars used for torch.compile cache keys.

    Start with every known vLLM env var; drop entries in ``ignored_factors``;
    hash everything else. This keeps the cache key aligned across workers."""

    ignored_factors: set[str] = {
        "MAX_JOBS",
        "VLLM_RPC_BASE_PATH",
        "VLLM_USE_MODELSCOPE",
        "VLLM_RINGBUFFER_WARNING_INTERVAL",
        "VLLM_DEBUG_DUMP_PATH",
        "VLLM_PORT",
        "VLLM_CACHE_ROOT",
        "LD_LIBRARY_PATH",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_USE_RUST_FRONTEND",
        "VLLM_RUST_FRONTEND_PATH",
        "VLLM_USE_PRECOMPILED_RUST",
        "VLLM_USE_FASTOKENS",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
        "VLLM_CI_USE_S3",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_HOST_IP",
        "VLLM_FORCE_AOT_LOAD",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        "VLLM_USAGE_STATS_SERVER",
        "VLLM_NO_USAGE_STATS",
        "VLLM_DO_NOT_TRACK",
        "VLLM_LOGGING_LEVEL",
        "VLLM_LOGGING_PREFIX",
        "VLLM_LOGGING_STREAM",
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_LOGGING_COLOR",
        "VLLM_LOG_STATS_INTERVAL",
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",
        "VLLM_TUNED_CONFIG_FOLDER",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
        "VLLM_IMAGE_FETCH_TIMEOUT",
        "VLLM_VIDEO_FETCH_TIMEOUT",
        "VLLM_AUDIO_FETCH_TIMEOUT",
        "VLLM_MEDIA_CACHE",
        "VLLM_MEDIA_CACHE_MAX_SIZE_MB",
        "VLLM_MEDIA_CACHE_TTL_HOURS",
        "VLLM_MEDIA_FETCH_MAX_RETRIES",
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS",
        "VLLM_MEDIA_LOADING_THREAD_COUNT",
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",
        "VLLM_VIDEO_LOADER_BACKEND",
        "VLLM_MEDIA_CONNECTOR",
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
        "VLLM_ASSETS_CACHE",
        "VLLM_ASSETS_CACHE_MODEL_CLEAN",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",
        "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_CPU_MOE_PREPACK",
        "VLLM_ZENTORCH_WEIGHT_PREPACK",
        "VLLM_TEST_FORCE_LOAD_FORMAT",
        "VLLM_ENABLE_CUDA_COMPATIBILITY",
        "VLLM_CUDA_COMPATIBILITY_PATH",
        "VLLM_SKIP_MODEL_NAME_VALIDATION",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NO_COLOR",
    }

    from vllm.config.utils import normalize_value

    # Build Settings once; reading hundreds of factors via _get_attr would
    # otherwise reparse pydantic settings on every iteration when the
    # module-level cache is disabled.
    settings = _get_settings() if _is_envs_cache_enabled() else Settings()

    factors: dict[str, object] = {}
    for factor, (sub_attr, field_name) in _VAR_TO_PATH.items():
        if factor in ignored_factors:
            continue
        try:
            raw = getattr(getattr(settings, sub_attr), field_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping environment variable %s while hashing compile factors: %s",
                factor,
                exc,
            )
            continue
        factors[factor] = normalize_value(raw)

    ray_noset_env_vars = [
        # Refer to
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/nvidia_gpu.py#L11
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/amd_gpu.py#L11
        # https://github.com/ray-project/ray/blob/b97d21dab233c2bd8ed7db749a82a1e594222b5c/python/ray/_private/accelerators/amd_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/npu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/hpu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/neuron.py#L14
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/tpu.py#L38
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/intel_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/rbln.py#L10
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
        "RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES",
    ]

    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))

    return factors
