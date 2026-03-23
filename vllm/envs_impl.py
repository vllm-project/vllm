# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM environment variable definitions and lazy accessor.

Usage::

    import vllm.envs as envs

    print(envs.VLLM_PORT)
    print(envs.VLLM_HOST_IP)

``vllm.envs`` is backed by a singleton :class:`Envs` instance. Values are
read from ``os.environ`` lazily on first access (or on every access when
caching is disabled).
"""

import json
import logging
import os
import tempfile
from collections.abc import Callable
from typing import Any, ClassVar, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities (module-level, used by the _env_vars getters)
# ---------------------------------------------------------------------------


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def disable_compile_cache() -> bool:
    return bool(int(os.getenv("VLLM_DISABLE_COMPILE_CACHE", "0")))


def use_aot_compile() -> bool:
    from vllm.model_executor.layers.batch_invariant import (
        vllm_is_batch_invariant,
    )
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    default_value = (
        "1"
        if is_torch_equal_or_newer("2.10.0.dev")
        and not disable_compile_cache()
        # Disabling AOT_COMPILE for CPU
        # See: https://github.com/vllm-project/vllm/issues/32033
        and not current_platform.is_cpu()
        else "0"
    )

    return (
        not vllm_is_batch_invariant()
        and os.environ.get("VLLM_USE_AOT_COMPILE", default_value) == "1"
    )


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """Create a getter that validates env var against allowed choices."""

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}. "
                f"Valid options: {actual_choices}."
            )

        return value

    return _get_validated_env


def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]:
    """Create a getter that returns a validated comma-separated list."""

    def _get_validated_env_list() -> list[str]:
        value = os.getenv(env_name)
        if value is None:
            return default

        values = [v.strip() for v in value.split(",") if v.strip()]

        if not values:
            return default

        actual_choices = choices() if callable(choices) else choices

        for val in values:
            if not case_sensitive:
                check_value = val.lower()
                check_choices = [choice.lower() for choice in actual_choices]
            else:
                check_value = val
                check_choices = actual_choices

            if check_value not in check_choices:
                raise ValueError(
                    f"Invalid value '{val}' in {env_name}. "
                    f"Valid options: {actual_choices}."
                )

        return values

    return _get_validated_env_list


def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]:
    """Create a getter that returns a validated set from comma-separated env."""

    def _get_validated_env_set() -> set[str]:
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())

    return _get_validated_env_set


def get_vllm_port() -> int | None:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.
    """
    if "VLLM_PORT" not in os.environ:
        return None

    port = os.getenv("VLLM_PORT", "0")

    try:
        return int(port)
    except ValueError as err:
        from urllib.parse import urlparse

        parsed = urlparse(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue, "
                "check the warning in: "
                "https://docs.vllm.ai/en/stable/serving/env_vars.html"
            ) from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err


# ---------------------------------------------------------------------------
# Envs class
# ---------------------------------------------------------------------------


class Envs:
    """Lazy accessor for vLLM environment variables.

    Attribute access triggers ``os.environ`` lookup on every call (unless
    caching is enabled via :meth:`enable_envs_cache`).
    """

    # ------------------------------------------------------------------
    # Type annotations for all env vars (annotation-only so that instance
    # __getattr__ fires on every access rather than returning a stale class
    # attribute).  Attribute-docstrings after each annotation provide IDE
    # hover documentation.
    # ------------------------------------------------------------------

    # ================== Installation Time Env Vars ==================

    VLLM_TARGET_DEVICE: str
    """Target device for vLLM. Supported: ``cuda`` (default), ``rocm``, ``cpu``."""

    VLLM_MAIN_CUDA_VERSION: str
    """Main CUDA version used by vLLM. Follows PyTorch but can be overridden."""

    VLLM_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"]
    """PyTorch float32 matmul precision mode within vLLM workers."""

    MAX_JOBS: str | None
    """Maximum number of compilation jobs to run in parallel."""

    NVCC_THREADS: str | None
    """Number of threads to use for nvcc. If set, ``MAX_JOBS`` is reduced."""

    VLLM_USE_PRECOMPILED: bool
    """If set, vllm will use precompiled binaries (``*.so``)."""

    VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX: bool
    """If set, skip adding ``+precompiled`` suffix to version string."""

    VLLM_DOCKER_BUILD_CONTEXT: bool
    """Marks that ``setup.py`` is running in a Docker build context."""

    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None
    """CMake build type. Defaults to ``Debug`` or ``RelWithDebInfo``."""

    VERBOSE: bool
    """If set, vllm will print verbose logs during installation."""

    VLLM_CONFIG_ROOT: str
    """Root directory for vLLM configuration files."""

    # ================== Runtime Env Vars ==================

    VLLM_CACHE_ROOT: str
    """Root directory for vLLM cache files."""

    VLLM_HOST_IP: str
    """IP address of the current node in distributed environments."""

    VLLM_PORT: int | None
    """Communication port. If set, incremented ports use this as base."""

    VLLM_RPC_BASE_PATH: str
    """IPC path for frontend/backend communication in multiprocessing mode."""

    VLLM_USE_MODELSCOPE: bool
    """If true, load models from ModelScope instead of Hugging Face Hub."""

    VLLM_RINGBUFFER_WARNING_INTERVAL: int
    """Interval in seconds to log a warning when the ring buffer is full."""

    CUDA_HOME: str | None
    """Path to cudatoolkit home directory."""

    VLLM_NCCL_SO_PATH: str | None
    """Path to the NCCL library file."""

    LD_LIBRARY_PATH: str | None
    """Locations vllm searches for the NCCL library."""

    VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE: int
    """Chunk size (MB) for sleeping memory allocations under ROCm."""

    VLLM_V1_USE_PREFILL_DECODE_ATTENTION: bool
    """Use separate prefill/decode kernels for V1 attention instead of unified triton kernel."""  # noqa: E501

    VLLM_FLASH_ATTN_VERSION: int | None
    """Force vllm to use a specific flash-attention version (2 or 3)."""

    VLLM_USE_STANDALONE_COMPILE: bool
    """Feature flag to enable/disable Inductor standalone compile."""

    VLLM_PATTERN_MATCH_DEBUG: str | None
    """Debug pattern matching inside custom passes."""

    VLLM_DEBUG_DUMP_PATH: str | None
    """Dump fx graphs to the given directory."""

    VLLM_USE_AOT_COMPILE: bool
    """Feature flag to enable/disable AOT compilation."""

    VLLM_USE_BYTECODE_HOOK: bool
    """Feature flag to enable/disable bytecode in ``TorchCompileWithNoGuardsWrapper``."""  # noqa: E501

    VLLM_FORCE_AOT_LOAD: bool
    """Force vllm to always load AOT compiled models from disk."""

    LOCAL_RANK: int
    """Local rank of the process in the distributed setting."""

    CUDA_VISIBLE_DEVICES: str | None
    """Visible devices in the distributed setting."""

    VLLM_ENGINE_ITERATION_TIMEOUT_S: int
    """Timeout in seconds for each iteration in the engine."""

    VLLM_ENGINE_READY_TIMEOUT_S: int
    """Timeout in seconds for waiting for engine cores to become ready during startup."""  # noqa: E501

    VLLM_API_KEY: str | None
    """API key for vLLM API server."""

    VLLM_DEBUG_LOG_API_SERVER_RESPONSE: bool
    """Whether to log responses from API Server for debugging."""

    S3_ACCESS_KEY_ID: str | None
    """S3 access key ID, used for tensorizer to load model from S3."""

    S3_SECRET_ACCESS_KEY: str | None
    """S3 secret access key, used for tensorizer to load model from S3."""

    S3_ENDPOINT_URL: str | None
    """S3 endpoint URL, used for tensorizer to load model from S3."""

    VLLM_USAGE_STATS_SERVER: str
    """Usage stats collection server URL."""

    VLLM_NO_USAGE_STATS: bool
    """If set, disable usage stats collection."""

    VLLM_DISABLE_FLASHINFER_PREFILL: bool
    """If set, disable flashinfer prefill."""

    VLLM_DO_NOT_TRACK: bool
    """If set, disable usage tracking."""

    VLLM_USAGE_SOURCE: str
    """Source identifier for usage stats."""

    VLLM_CONFIGURE_LOGGING: bool
    """If set to 0, vllm will not configure logging."""

    VLLM_LOGGING_CONFIG_PATH: str | None
    """Path to a logging configuration file."""

    VLLM_LOGGING_LEVEL: str
    """Default logging level."""

    VLLM_LOGGING_STREAM: str
    """Default logging stream."""

    VLLM_LOGGING_PREFIX: str
    """Prefix prepended to all log messages."""

    VLLM_LOGGING_COLOR: str
    """Colored logging output. Options: ``auto``, ``1``, ``0``."""

    NO_COLOR: bool
    """Standard unix flag for disabling ANSI color codes."""

    VLLM_LOG_STATS_INTERVAL: float
    """Interval in seconds to log stats. Minimum value is 0."""

    VLLM_TRACE_FUNCTION: int
    """If set to 1, trace function calls."""

    VLLM_ATTENTION_BACKEND: str | None
    """Backend for attention computation."""

    VLLM_USE_FLASHINFER_SAMPLER: bool | None
    """If set, use flashinfer sampler."""

    VLLM_PP_LAYER_PARTITION: str | None
    """Pipeline stage partition strategy."""

    VLLM_CPU_KVCACHE_SPACE: int | None
    """(CPU backend only) CPU key-value cache space in GB."""

    VLLM_CPU_OMP_THREADS_BIND: str
    """(CPU backend only) CPU core ids bound by OpenMP threads."""

    VLLM_CPU_NUM_OF_RESERVED_CPU: int | None
    """(CPU backend only) CPU cores not used by OMP threads."""

    VLLM_CPU_SGL_KERNEL: bool
    """(CPU backend only) Whether to use SGL kernels."""

    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: Literal["auto", "nccl", "shm"]
    """Ray Compiled Graph channel type for pipeline-parallel stages."""

    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool
    """Enable GPU communication overlap in Ray's Compiled Graph."""

    VLLM_USE_RAY_WRAPPED_PP_COMM: bool
    """Use a Ray Communicator wrapping vLLM's pipeline parallelism communicator."""

    VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"]
    """Multiprocess context method for workers."""

    VLLM_ASSETS_CACHE: str
    """Path to the cache for storing downloaded assets."""

    VLLM_ASSETS_CACHE_MODEL_CLEAN: bool
    """If set, clean model files in assets cache."""

    VLLM_IMAGE_FETCH_TIMEOUT: int
    """Timeout in seconds for fetching images when serving multimodal models."""

    VLLM_VIDEO_FETCH_TIMEOUT: int
    """Timeout in seconds for fetching videos when serving multimodal models."""

    VLLM_AUDIO_FETCH_TIMEOUT: int
    """Timeout in seconds for fetching audio when serving multimodal models."""

    VLLM_MEDIA_URL_ALLOW_REDIRECTS: bool
    """Whether to allow HTTP redirects when fetching from media URLs."""

    VLLM_MEDIA_LOADING_THREAD_COUNT: int
    """Max number of workers for the thread pool handling media bytes loading."""

    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: int
    """Maximum filesize in MB for a single audio file."""

    VLLM_VIDEO_LOADER_BACKEND: str
    """Backend for Video IO."""

    VLLM_MEDIA_CONNECTOR: str
    """Media connector implementation."""

    VLLM_XLA_CACHE_PATH: str
    """Path to the XLA persistent cache directory."""

    VLLM_XLA_CHECK_RECOMPILATION: bool
    """If set, assert on XLA recompilation after each execution step."""

    VLLM_XLA_USE_SPMD: bool
    """Enable SPMD mode for TPU backend."""

    VLLM_FUSED_MOE_CHUNK_SIZE: int
    """Chunk size for fused MoE operations."""

    VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING: bool
    """Control whether to use fused MoE activation chunking."""

    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool
    """If set, the OpenAI API server will stay alive even after engine errors."""

    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool
    """If set, allow max sequence length greater than model's config.json limit."""

    VLLM_TEST_FORCE_FP8_MARLIN: bool
    """If set, force FP8 Marlin for FP8 quantization regardless of hardware support."""

    VLLM_TEST_FORCE_LOAD_FORMAT: str
    """Force a specific load format for testing."""

    VLLM_RPC_TIMEOUT: int
    """Time in ms for the zmq client to wait for a response from the backend."""

    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int
    """Timeout in seconds for keeping HTTP connections alive in API server."""

    VLLM_PLUGINS: list[str] | None
    """List of plugin names to load, separated by commas."""

    VLLM_LORA_RESOLVER_CACHE_DIR: str | None
    """Local directory to look in for unrecognized LoRA adapters."""

    # Deprecated profiling env vars
    VLLM_TORCH_CUDA_PROFILE: str | None
    """Deprecated. Enables torch CUDA profiling if set to 1."""

    VLLM_TORCH_PROFILER_DIR: str | None
    """Deprecated. Enables torch profiler if set."""

    VLLM_TORCH_PROFILER_RECORD_SHAPES: str | None
    """Deprecated. Enable torch profiler to record shapes if set to 1."""

    VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY: str | None
    """Deprecated. Enable torch profiler to profile memory if set to 1."""

    VLLM_TORCH_PROFILER_WITH_STACK: str | None
    """Deprecated. Enable torch profiler to profile stack if set to 1."""

    VLLM_TORCH_PROFILER_WITH_FLOPS: str | None
    """Deprecated. Enable torch profiler to profile flops if set to 1."""

    VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM: str | None
    """Deprecated. Disable torch profiling of the AsyncLLMEngine process if set to 1."""

    VLLM_PROFILER_DELAY_ITERS: str | None
    """Deprecated. Delay iterations before starting profiling."""

    VLLM_PROFILER_MAX_ITERS: str | None
    """Deprecated. Maximum number of iterations to profile."""

    VLLM_TORCH_PROFILER_USE_GZIP: str | None
    """Deprecated. Control whether torch profiler gzip-compresses profiling files."""

    VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL: str | None
    """Deprecated. Control whether torch profiler dumps the self_cuda_time_total table."""  # noqa: E501

    VLLM_USE_TRITON_AWQ: bool
    """If set, vllm will use Triton implementations of AWQ."""

    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool
    """If set, allow loading or unloading LoRA adapters at runtime."""

    VLLM_SKIP_P2P_CHECK: bool
    """Skip P2P check for custom allreduce."""

    VLLM_DISABLED_KERNELS: list[str]
    """List of quantization kernels that should be disabled."""

    VLLM_DISABLE_PYNCCL: bool
    """Disable pynccl (using torch.distributed instead)."""

    VLLM_ROCM_USE_AITER: bool
    """Disable aiter ops unless specifically enabled."""

    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool
    """Whether to use aiter paged attention."""

    VLLM_ROCM_USE_AITER_LINEAR: bool
    """Whether to use aiter linear op if aiter ops are enabled."""

    VLLM_ROCM_USE_AITER_MOE: bool
    """Whether to use aiter MoE ops."""

    VLLM_ROCM_USE_AITER_RMSNORM: bool
    """Whether to use aiter RMS norm op."""

    VLLM_ROCM_USE_AITER_MLA: bool
    """Whether to use aiter MLA ops."""

    VLLM_ROCM_USE_AITER_MHA: bool
    """Whether to use aiter MHA ops."""

    VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: bool
    """Whether to use aiter FP4 GEMM ASM."""

    VLLM_ROCM_USE_AITER_TRITON_ROPE: bool
    """Whether to use aiter triton rope."""

    VLLM_ROCM_USE_AITER_FP8BMM: bool
    """Whether to use aiter triton FP8 BMM kernel."""

    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool
    """Use AITER triton unified attention for V1 attention."""

    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool
    """Whether to use aiter fusion shared experts ops."""

    VLLM_ROCM_USE_AITER_TRITON_GEMM: bool
    """Whether to use aiter triton kernels for GEMM ops."""

    VLLM_ROCM_USE_SKINNY_GEMM: bool
    """Use ROCm skinny GEMMs."""

    VLLM_ROCM_FP8_PADDING: bool
    """Pad the FP8 weights to 256 bytes for ROCm."""

    VLLM_ROCM_MOE_PADDING: bool
    """Pad the weights for the MoE kernel."""

    VLLM_ROCM_CUSTOM_PAGED_ATTN: bool
    """Custom paged attention kernel for MI3* cards."""

    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: Literal["FP", "INT8", "INT6", "INT4", "NONE"]
    """Custom quick allreduce kernel quantization level for MI3* cards."""

    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool
    """Convert BF16 to FP16 for quick allreduce on MI3* cards."""

    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None
    """Maximum allowed data bytes (MB) for custom quick allreduce communication."""

    Q_SCALE_CONSTANT: int
    """Divisor for dynamic query scale factor calculation for FP8 KV Cache."""

    K_SCALE_CONSTANT: int
    """Divisor for dynamic key scale factor calculation for FP8 KV Cache."""

    V_SCALE_CONSTANT: int
    """Divisor for dynamic value scale factor calculation for FP8 KV Cache."""

    VLLM_ENABLE_V1_MULTIPROCESSING: bool
    """If set, enable multiprocessing in LLM for the V1 code path."""

    VLLM_LOG_BATCHSIZE_INTERVAL: float
    """Interval for logging batch sizes."""

    VLLM_DISABLE_COMPILE_CACHE: bool
    """If set, disable the torch.compile cache."""

    VLLM_SERVER_DEV_MODE: bool
    """If set, run in development mode with additional endpoints."""

    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int
    """Maximum number of requests per asyncio task in V1 AsyncLLM output processing."""

    VLLM_MLA_DISABLE: bool
    """If set, disable MLA attention optimizations."""

    VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH: int
    """Flash Attention MLA max number of splits for CUDA graph decode."""

    VLLM_RAY_PER_WORKER_GPUS: float
    """Number of GPUs per worker in Ray."""

    VLLM_RAY_BUNDLE_INDICES: str
    """Bundle indices for Ray."""

    VLLM_CUDART_SO_PATH: str | None
    """Path to the CUDA runtime shared library."""

    VLLM_DP_RANK: int
    """Rank of the process in the data parallel setting."""

    VLLM_DP_RANK_LOCAL: int
    """Local rank of the process in the data parallel setting. Defaults to ``VLLM_DP_RANK``."""  # noqa: E501

    VLLM_DP_SIZE: int
    """World size of the data parallel setting."""

    VLLM_DP_MASTER_IP: str
    """IP address of the master node in the data parallel setting."""

    VLLM_DP_MASTER_PORT: int
    """Port of the master node in the data parallel setting."""

    VLLM_MOE_DP_CHUNK_SIZE: int
    """Token dispatch quantum for MoE Data-Parallel."""

    VLLM_ENABLE_MOE_DP_CHUNK: bool
    """Enable chunked dispatch for MoE Data-Parallel."""

    VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool
    """Randomize inputs during dummy runs when using Data Parallel."""

    VLLM_RAY_DP_PACK_STRATEGY: Literal["strict", "fill", "span"]
    """Strategy to pack data parallel ranks for Ray."""

    VLLM_CI_USE_S3: bool
    """Whether to use S3 path for model loading in CI via RunAI Streamer."""

    VLLM_MODEL_REDIRECT_PATH: str | None
    """Use model_redirect to redirect the model name to a local folder."""

    VLLM_MARLIN_USE_ATOMIC_ADD: bool
    """Whether to use atomicAdd reduce in gptq/awq marlin kernel."""

    VLLM_MXFP4_USE_MARLIN: bool | None
    """Whether to use marlin kernel in mxfp4 quantization method."""

    VLLM_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None
    """The activation dtype for marlin kernel."""

    VLLM_DEEPEPLL_NVFP4_DISPATCH: bool
    """Whether to use DeepEPLL kernels for NVFP4 quantization and dispatch method."""

    VLLM_V1_USE_OUTLINES_CACHE: bool
    """Whether to turn on the outlines cache for V1."""

    VLLM_TPU_BUCKET_PADDING_GAP: int
    """Gap between padding buckets for the forward pass."""

    VLLM_TPU_MOST_MODEL_LEN: int | None
    """Most model length for TPU."""

    VLLM_TPU_USING_PATHWAYS: bool
    """Whether using Pathways."""

    VLLM_USE_DEEP_GEMM: bool
    """Allow use of DeepGemm kernels for fused MoE ops."""

    VLLM_MOE_USE_DEEP_GEMM: bool
    """Allow use of DeepGemm specifically for MoE fused ops."""

    VLLM_USE_DEEP_GEMM_E8M0: bool
    """Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs."""

    VLLM_DEEP_GEMM_WARMUP: Literal["skip", "full", "relax"]
    """DeepGemm JIT warmup strategy."""

    VLLM_USE_FUSED_MOE_GROUPED_TOPK: bool
    """Whether to use fused grouped_topk for MoE expert selection."""

    VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool
    """Allow use of FlashInfer FP8 block-scale GEMM for linear layers."""

    VLLM_USE_FLASHINFER_MOE_FP16: bool
    """Allow use of FlashInfer MoE kernels for fused MoE ops (FP16)."""

    VLLM_USE_FLASHINFER_MOE_FP8: bool
    """Allow use of FlashInfer MoE kernels for fused MoE ops (FP8)."""

    VLLM_USE_FLASHINFER_MOE_FP4: bool
    """Allow use of FlashInfer CUTLASS kernels for fused MoE ops (FP4)."""

    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: bool
    """Use the FlashInfer MXFP8 x MXFP4 MoE backend."""

    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: bool
    """Use the FlashInfer CUTLASS backend for MXFP8 x MXFP4 MoE."""

    VLLM_USE_FLASHINFER_MOE_MXFP4_BF16: bool
    """Use the FlashInfer BF16 x MXFP4 MoE backend."""

    VLLM_XGRAMMAR_CACHE_MB: int
    """Cache size in MB used by the xgrammar compiler."""

    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int
    """Threshold for msgspec zero-copy serialization of tensors."""

    VLLM_ALLOW_INSECURE_SERIALIZATION: bool
    """If set, allow insecure serialization using pickle."""

    VLLM_NIXL_SIDE_CHANNEL_HOST: str
    """IP address used for NIXL handshake between remote agents."""

    VLLM_NIXL_SIDE_CHANNEL_PORT: int
    """Port used for NIXL handshake between remote agents."""

    VLLM_MOONCAKE_BOOTSTRAP_PORT: int
    """Port used for Mooncake handshake between remote agents."""

    VLLM_ALL2ALL_BACKEND: Literal[
        "naive",
        "pplx",
        "deepep_high_throughput",
        "deepep_low_latency",
        "allgather_reducescatter",
        "flashinfer_all2allv",
    ]
    """All2all backend for expert parallel communication."""

    VLLM_FLASHINFER_MOE_BACKEND: Literal["throughput", "latency", "masked_gemm"]
    """FlashInfer MoE backend for vLLM's fused Mixture-of-Experts support."""

    VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: int
    """Workspace buffer size for the FlashInfer backend."""

    VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int
    """Maximum number of tokens per expert supported by the NVFP4 MoE CUTLASS Kernel."""

    VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB: dict[str, float]
    """Tensor size thresholds (MB) for using FlashInfer fused allreduce."""

    VLLM_MOE_ROUTING_SIMULATION_STRATEGY: str
    """MoE routing strategy selector."""

    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int
    """Regex timeout for use by the vLLM tool parsing plugins."""

    VLLM_SLEEP_WHEN_IDLE: bool
    """Reduce CPU usage when vLLM is idle."""

    VLLM_MQ_MAX_CHUNK_BYTES_MB: int
    """Max chunk bytes (MB) for the RPC message queue."""

    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int
    """Timeout in seconds for execute_model RPC calls."""

    VLLM_KV_CACHE_LAYOUT: Literal["NHD", "HND"] | None
    """KV cache layout used throughout vllm."""

    VLLM_COMPUTE_NANS_IN_LOGITS: bool
    """Enable checking whether the generated logits contain NaNs."""

    VLLM_USE_NVFP4_CT_EMULATIONS: bool
    """Controls whether emulations are used for NVFP4 generations on machines < 100."""

    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int
    """Timeout in seconds before KV cache is cleared on producer side (NIXL)."""

    VLLM_MORIIO_CONNECTOR_READ_MODE: bool
    """Controls the read mode for the Mori-IO connector."""

    VLLM_MORIIO_QP_PER_TRANSFER: int
    """Controls the QP per transfer for the Mori-IO connector."""

    VLLM_MORIIO_POST_BATCH_SIZE: int
    """Controls the post-processing batch size for the Mori-IO connector."""

    VLLM_MORIIO_NUM_WORKERS: int
    """Controls the number of workers for Mori operations."""

    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT: int
    """Timeout in seconds for MooncakeConnector in PD disaggregated setup."""

    VLLM_USE_CUDNN_PREFILL: bool
    """Controls whether or not to use cudnn prefill."""

    VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL: bool
    """Controls whether to use TRT-LLM ragged DeepSeek prefill."""

    VLLM_USE_TRTLLM_ATTENTION: str | None
    """Use the TRTLLM attention backend in flashinfer."""

    VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION: bool
    """If set, do not quantize Q to fp8 when using FP8 KV."""

    VLLM_HAS_FLASHINFER_CUBIN: bool
    """If set, pre-downloaded cubin files exist and flashinfer will read them."""

    VLLM_NVFP4_GEMM_BACKEND: str | None
    """NVFP4 GEMM backend selector."""

    VLLM_ENABLE_CUDAGRAPH_GC: bool
    """Controls garbage collection during CUDA graph capture."""

    VLLM_LOOPBACK_IP: str
    """Used to force set up loopback IP."""

    VLLM_PROCESS_NAME_PREFIX: str
    """Prefix for vLLM process names."""

    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool
    """Allow chunked local attention with hybrid KV cache manager."""

    VLLM_ENABLE_RESPONSES_API_STORE: bool
    """Enable support for the store option in the OpenAI Responses API."""

    VLLM_ROCM_FP8_MFMA_PAGE_ATTN: bool
    """If set, use the FP8 MFMA in ROCm paged attention."""

    VLLM_ALLREDUCE_USE_SYMM_MEM: bool
    """Whether to use PyTorch symmetric memory for allreduce."""

    VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT: bool
    """Experimental: enable MCP tool calling for non-harmony models."""

    VLLM_TUNED_CONFIG_FOLDER: str | None
    """Allows vllm to find tuned config under customized folder."""

    VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: set[str]
    """Valid MCP server labels for GPT OSS system tools."""

    VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: bool
    """Allows harmony instructions to be injected on system messages."""

    VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY: bool
    """Enable automatic retry when tool call JSON parsing fails."""

    VLLM_CUSTOM_SCOPES_FOR_PROFILING: bool
    """Add optional custom scopes for profiling."""

    VLLM_NVTX_SCOPES_FOR_PROFILING: bool
    """Add optional NVTX scopes for profiling."""

    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES: bool
    """Represent block hashes in KV cache events as 64-bit integers."""

    VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME: str
    """Name of the shared memory buffer used for object storage."""

    VLLM_DEEPEP_BUFFER_SIZE_MB: int
    """Size in MB of the buffers (NVL and RDMA) used by DeepEP."""

    VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE: bool
    """Force DeepEP to use intranode kernel for high-throughput inter-node comms."""

    VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL: bool
    """Allow DeepEP to use MNNVL for internode_ll kernel."""

    VLLM_DBO_COMM_SMS: int
    """Number of SMs to allocate for communication kernels when running DBO."""

    VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool
    """Enable max_autotune in inductor_config."""

    VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool
    """Enable coordinate_descent_tuning in inductor_config."""

    VLLM_USE_NCCL_SYMM_MEM: bool
    """Flag to enable NCCL symmetric memory allocation and registration."""

    VLLM_NCCL_INCLUDE_PATH: str | None
    """NCCL header path."""

    VLLM_USE_FBGEMM: bool
    """Flag to enable FBGemm kernels on model execution."""

    VLLM_GC_DEBUG: str
    """GC debug config."""

    VLLM_DEBUG_WORKSPACE: bool
    """Debug workspace allocations."""

    VLLM_DISABLE_SHARED_EXPERTS_STREAM: bool
    """Disables parallel execution of shared_experts via separate CUDA stream."""

    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: int
    """Limits when shared_experts run in a separate stream."""

    VLLM_COMPILE_CACHE_SAVE_FORMAT: Literal["binary", "unpacked"]
    """Format for saving torch.compile cache artifacts."""

    VLLM_USE_V2_MODEL_RUNNER: bool
    """Flag to enable v2 model runner."""

    VLLM_LOG_MODEL_INSPECTION: bool
    """Log model inspection after loading."""

    VLLM_DEBUG_MFU_METRICS: bool
    """Debug logging for --enable-mfu-metrics."""

    VLLM_USE_MEGA_AOT_ARTIFACT: bool
    """Use a mega AOT artifact combining AOT Autograd and Inductor artifacts."""

    VLLM_LORA_DISABLE_PDL: bool
    """Disable Persistent Data Loader (PDL) for LoRA Triton ops."""

    VLLM_ZENTORCH_WEIGHT_PREPACK: bool
    """Enable zentorch weight prepack for linear ops (default: enabled)."""

    VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY: bool
    """Disable pinned memory when offloading weights to CPU."""

    VLLM_WEIGHT_OFFLOADING_DISABLE_UVA: bool
    """Disable Unified Virtual Addressing for CPU weight offloading."""

    VLLM_USE_FLASHINFER_MOE_INT4: bool
    """Use FlashInfer trtllm fused MoE int4 kernels."""

    VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES: bool
    """Use TMA-aligned scales with DeepGEMM FP8 kernels."""

    VLLM_MEDIA_FETCH_MAX_RETRIES: int
    """Maximum number of retries for fetching media (images, video, audio)."""

    VLLM_MM_HASHER_ALGORITHM: str
    """Hash algorithm for multimodal content hashing (blake3, sha256, sha512)."""

    VLLM_LORA_RESOLVER_HF_REPO_LIST: str | None
    """Comma-separated list of Hugging Face repos to search for LoRA adapters."""

    VLLM_ROCM_USE_AITER_FP4BMM: bool
    """Enable ROCm aiter FP4 batch matrix multiplication kernel."""

    VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT: bool
    """Enable ROCm KV cache layout shuffling for aiter attention."""

    VLLM_ENABLE_PREGRAD_PASSES: bool
    """Enable inductor pregrad passes during compilation."""

    VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY: str
    """Comma-separated env var prefixes to copy to Ray actors (additive)."""

    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: str
    """Comma-separated env var names to copy to Ray actors (additive)."""

    VLLM_FLASHINFER_ALLREDUCE_BACKEND: str
    """Backend for FlashInfer all-reduce (auto, trtllm, or mnnvl)."""

    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION: bool
    """Disable randomization of request IDs (deprecated)."""

    VLLM_ALLREDUCE_USE_FLASHINFER: bool
    """Use FlashInfer backend for all-reduce operations."""

    VLLM_SYSTEM_START_DATE: str | None
    """System start date string (YYYY-MM-DD) for harmony system instructions."""

    VLLM_DISABLE_LOG_LOGO: bool
    """Suppress the vLLM ASCII logo on startup."""

    VLLM_ELASTIC_EP_SCALE_UP_LAUNCH: bool
    """Enable elastic expert-parallel scale-up launch mode."""

    VLLM_ELASTIC_EP_DRAIN_REQUESTS: bool
    """Drain in-flight requests before scaling in elastic EP mode."""

    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool
    """Profile CUDA graph memory during memory profiling for more accurate
      KV cache sizing."""

    VLLM_NIXL_EP_MAX_NUM_RANKS: int
    """Maximum number of EP ranks for NIXL all-to-all buffers."""

    # ------------------------------------------------------------------
    # Internal class variable: the dict of lazy getters
    # ------------------------------------------------------------------

    # The start-* and end-* markers below are used by the documentation
    # generator to extract the env var list.

    # --8<-- [start:env-vars-definition]
    _env_vars: ClassVar[dict[str, Callable[[], Any]]] = {
        # ================== Installation Time Env Vars ==================
        "VLLM_TARGET_DEVICE": lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),
        "VLLM_MAIN_CUDA_VERSION": (
            lambda: os.getenv("VLLM_MAIN_CUDA_VERSION", "").lower() or "12.9"
        ),
        "VLLM_FLOAT32_MATMUL_PRECISION": env_with_choices(
            "VLLM_FLOAT32_MATMUL_PRECISION",
            "highest",
            ["highest", "high", "medium"],
            case_sensitive=False,
        ),
        "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
        "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
        "VLLM_USE_PRECOMPILED": lambda: os.environ.get("VLLM_USE_PRECOMPILED", "")
        .strip()
        .lower()
        in ("1", "true")
        or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),
        "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": lambda: bool(
            int(os.environ.get("VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX", "0"))
        ),
        "VLLM_DOCKER_BUILD_CONTEXT": lambda: os.environ.get(
            "VLLM_DOCKER_BUILD_CONTEXT", ""
        )
        .strip()
        .lower()
        in ("1", "true"),
        "CMAKE_BUILD_TYPE": env_with_choices(
            "CMAKE_BUILD_TYPE", None, ["Debug", "Release", "RelWithDebInfo"]
        ),
        "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
        "VLLM_CONFIG_ROOT": lambda: os.path.expanduser(
            os.getenv(
                "VLLM_CONFIG_ROOT",
                os.path.join(get_default_config_root(), "vllm"),
            )
        ),
        # ================== Runtime Env Vars ==================
        "VLLM_CACHE_ROOT": lambda: os.path.expanduser(
            os.getenv(
                "VLLM_CACHE_ROOT",
                os.path.join(get_default_cache_root(), "vllm"),
            )
        ),
        "VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", ""),
        "VLLM_PORT": get_vllm_port,
        "VLLM_RPC_BASE_PATH": lambda: os.getenv(
            "VLLM_RPC_BASE_PATH", tempfile.gettempdir()
        ),
        "VLLM_USE_MODELSCOPE": lambda: os.environ.get(
            "VLLM_USE_MODELSCOPE", "False"
        ).lower()
        == "true",
        "VLLM_RINGBUFFER_WARNING_INTERVAL": lambda: int(
            os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")
        ),
        "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
        "VLLM_NCCL_SO_PATH": lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),
        "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
        "VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE": lambda: int(
            os.environ.get("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE", "256")
        ),
        "VLLM_V1_USE_PREFILL_DECODE_ATTENTION": lambda: (
            os.getenv("VLLM_V1_USE_PREFILL_DECODE_ATTENTION", "False").lower()
            in ("true", "1")
        ),
        "VLLM_FLASH_ATTN_VERSION": lambda: maybe_convert_int(
            os.environ.get("VLLM_FLASH_ATTN_VERSION", None)
        ),
        "VLLM_USE_STANDALONE_COMPILE": lambda: os.environ.get(
            "VLLM_USE_STANDALONE_COMPILE", "1"
        )
        == "1",
        "VLLM_PATTERN_MATCH_DEBUG": lambda: os.environ.get(
            "VLLM_PATTERN_MATCH_DEBUG", None
        ),
        "VLLM_DEBUG_DUMP_PATH": lambda: os.environ.get("VLLM_DEBUG_DUMP_PATH", None),
        "VLLM_USE_AOT_COMPILE": use_aot_compile,
        "VLLM_USE_BYTECODE_HOOK": lambda: bool(
            int(os.environ.get("VLLM_USE_BYTECODE_HOOK", "1"))
        ),
        "VLLM_FORCE_AOT_LOAD": lambda: os.environ.get("VLLM_FORCE_AOT_LOAD", "0")
        == "1",
        "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
        "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
        "VLLM_ENGINE_ITERATION_TIMEOUT_S": lambda: int(
            os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")
        ),
        "VLLM_ENGINE_READY_TIMEOUT_S": lambda: int(
            os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S", "600")
        ),
        "VLLM_API_KEY": lambda: os.environ.get("VLLM_API_KEY", None),
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": lambda: os.environ.get(
            "VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "False"
        ).lower()
        == "true",
        "S3_ACCESS_KEY_ID": lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
        "S3_SECRET_ACCESS_KEY": lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
        "S3_ENDPOINT_URL": lambda: os.environ.get("S3_ENDPOINT_URL", None),
        "VLLM_USAGE_STATS_SERVER": lambda: os.environ.get(
            "VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"
        ),
        "VLLM_NO_USAGE_STATS": lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0")
        == "1",
        "VLLM_DISABLE_FLASHINFER_PREFILL": lambda: os.environ.get(
            "VLLM_DISABLE_FLASHINFER_PREFILL", "0"
        )
        == "1",
        "VLLM_DO_NOT_TRACK": lambda: (
            os.environ.get("VLLM_DO_NOT_TRACK", None)
            or os.environ.get("DO_NOT_TRACK", None)
            or "0"
        )
        == "1",
        "VLLM_USAGE_SOURCE": lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),
        "VLLM_CONFIGURE_LOGGING": lambda: bool(
            int(os.getenv("VLLM_CONFIGURE_LOGGING", "1"))
        ),
        "VLLM_LOGGING_CONFIG_PATH": lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),
        "VLLM_LOGGING_LEVEL": lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),
        "VLLM_LOGGING_STREAM": lambda: os.getenv(
            "VLLM_LOGGING_STREAM", "ext://sys.stdout"
        ),
        "VLLM_LOGGING_PREFIX": lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),
        "VLLM_LOGGING_COLOR": lambda: os.getenv("VLLM_LOGGING_COLOR", "auto"),
        "NO_COLOR": lambda: os.getenv("NO_COLOR", "0") != "0",
        "VLLM_LOG_STATS_INTERVAL": lambda: val
        if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0
        else 10.0,
        "VLLM_TRACE_FUNCTION": lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),
        "VLLM_ATTENTION_BACKEND": env_with_choices(
            "VLLM_ATTENTION_BACKEND",
            None,
            lambda: list(
                __import__(
                    "vllm.v1.attention.backends.registry",
                    fromlist=["AttentionBackendEnum"],
                ).AttentionBackendEnum.__members__.keys()
            ),
        ),
        "VLLM_USE_FLASHINFER_SAMPLER": lambda: bool(
            int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"])
        )
        if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ
        else None,
        "VLLM_PP_LAYER_PARTITION": lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),
        "VLLM_CPU_KVCACHE_SPACE": lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0"))
        if "VLLM_CPU_KVCACHE_SPACE" in os.environ
        else None,
        "VLLM_CPU_OMP_THREADS_BIND": lambda: os.getenv(
            "VLLM_CPU_OMP_THREADS_BIND", "auto"
        ),
        "VLLM_CPU_NUM_OF_RESERVED_CPU": lambda: int(
            os.getenv("VLLM_CPU_NUM_OF_RESERVED_CPU", "0")
        )
        if "VLLM_CPU_NUM_OF_RESERVED_CPU" in os.environ
        else None,
        "VLLM_CPU_SGL_KERNEL": lambda: bool(int(os.getenv("VLLM_CPU_SGL_KERNEL", "0"))),
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": env_with_choices(
            "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto", ["auto", "nccl", "shm"]
        ),
        "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM": lambda: bool(
            int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
        ),
        "VLLM_USE_RAY_WRAPPED_PP_COMM": lambda: bool(
            int(os.getenv("VLLM_USE_RAY_WRAPPED_PP_COMM", "1"))
        ),
        "VLLM_WORKER_MULTIPROC_METHOD": env_with_choices(
            "VLLM_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork"]
        ),
        "VLLM_ASSETS_CACHE": lambda: os.path.expanduser(
            os.getenv(
                "VLLM_ASSETS_CACHE",
                os.path.join(get_default_cache_root(), "vllm", "assets"),
            )
        ),
        "VLLM_ASSETS_CACHE_MODEL_CLEAN": lambda: bool(
            int(os.getenv("VLLM_ASSETS_CACHE_MODEL_CLEAN", "0"))
        ),
        "VLLM_IMAGE_FETCH_TIMEOUT": lambda: int(
            os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")
        ),
        "VLLM_VIDEO_FETCH_TIMEOUT": lambda: int(
            os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "30")
        ),
        "VLLM_AUDIO_FETCH_TIMEOUT": lambda: int(
            os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "10")
        ),
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS": lambda: bool(
            int(os.getenv("VLLM_MEDIA_URL_ALLOW_REDIRECTS", "1"))
        ),
        "VLLM_MEDIA_LOADING_THREAD_COUNT": lambda: int(
            os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8")
        ),
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": lambda: int(
            os.getenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "25")
        ),
        "VLLM_VIDEO_LOADER_BACKEND": lambda: os.getenv(
            "VLLM_VIDEO_LOADER_BACKEND", "opencv"
        ),
        "VLLM_MEDIA_CONNECTOR": lambda: os.getenv("VLLM_MEDIA_CONNECTOR", "http"),
        "VLLM_XLA_CACHE_PATH": lambda: os.path.expanduser(
            os.getenv(
                "VLLM_XLA_CACHE_PATH",
                os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
            )
        ),
        "VLLM_XLA_CHECK_RECOMPILATION": lambda: bool(
            int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION", "0"))
        ),
        "VLLM_XLA_USE_SPMD": lambda: bool(int(os.getenv("VLLM_XLA_USE_SPMD", "0"))),
        "VLLM_FUSED_MOE_CHUNK_SIZE": lambda: int(
            os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE", str(16 * 1024))
        ),
        "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING": lambda: bool(
            int(os.getenv("VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING", "1"))
        ),
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH": lambda: bool(
            int(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "0"))
        ),
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": lambda: (
            os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower()
            in ("1", "true")
        ),
        "VLLM_TEST_FORCE_FP8_MARLIN": lambda: (
            os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower()
            in ("1", "true")
        ),
        "VLLM_TEST_FORCE_LOAD_FORMAT": lambda: os.getenv(
            "VLLM_TEST_FORCE_LOAD_FORMAT", "dummy"
        ),
        "VLLM_RPC_TIMEOUT": lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(
            os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")
        ),
        "VLLM_PLUGINS": lambda: None
        if "VLLM_PLUGINS" not in os.environ
        else os.environ["VLLM_PLUGINS"].split(","),
        "VLLM_LORA_RESOLVER_CACHE_DIR": lambda: os.getenv(
            "VLLM_LORA_RESOLVER_CACHE_DIR", None
        ),
        "VLLM_TORCH_CUDA_PROFILE": lambda: os.getenv("VLLM_TORCH_CUDA_PROFILE"),
        "VLLM_TORCH_PROFILER_DIR": lambda: os.getenv("VLLM_TORCH_PROFILER_DIR"),
        "VLLM_TORCH_PROFILER_RECORD_SHAPES": lambda: (
            os.getenv("VLLM_TORCH_PROFILER_RECORD_SHAPES")
        ),
        "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY": lambda: (
            os.getenv("VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY")
        ),
        "VLLM_TORCH_PROFILER_WITH_STACK": lambda: (
            os.getenv("VLLM_TORCH_PROFILER_WITH_STACK")
        ),
        "VLLM_TORCH_PROFILER_WITH_FLOPS": lambda: (
            os.getenv("VLLM_TORCH_PROFILER_WITH_FLOPS")
        ),
        "VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM": lambda: (
            os.getenv("VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM")
        ),
        "VLLM_PROFILER_DELAY_ITERS": lambda: (os.getenv("VLLM_PROFILER_DELAY_ITERS")),
        "VLLM_PROFILER_MAX_ITERS": lambda: os.getenv("VLLM_PROFILER_MAX_ITERS"),
        "VLLM_TORCH_PROFILER_USE_GZIP": lambda: os.getenv(
            "VLLM_TORCH_PROFILER_USE_GZIP"
        ),
        "VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL": lambda: (
            os.getenv("VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL")
        ),
        "VLLM_USE_TRITON_AWQ": lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": lambda: (
            os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower()
            in ("1", "true")
        ),
        "VLLM_SKIP_P2P_CHECK": lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "1") == "1",
        "VLLM_DISABLED_KERNELS": lambda: []
        if "VLLM_DISABLED_KERNELS" not in os.environ
        else os.environ["VLLM_DISABLED_KERNELS"].split(","),
        "VLLM_DISABLE_PYNCCL": lambda: (
            os.getenv("VLLM_DISABLE_PYNCCL", "False").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER", "False").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_PAGED_ATTN": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_PAGED_ATTN", "False").lower()
            in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_LINEAR": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_LINEAR", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_MOE": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_MOE", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_RMSNORM": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_RMSNORM", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_MLA": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_MLA", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_MHA": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_MHA", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_FP4_ASM_GEMM", "False").lower()
            in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_TRITON_ROPE": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "False").lower()
            in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_FP8BMM": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_FP8BMM", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower()
            in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "False").lower()
            in ("true", "1")
        ),
        "VLLM_ROCM_USE_AITER_TRITON_GEMM": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", "True").lower()
            in ("true", "1")
        ),
        "VLLM_ROCM_USE_SKINNY_GEMM": lambda: (
            os.getenv("VLLM_ROCM_USE_SKINNY_GEMM", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_FP8_PADDING": lambda: bool(
            int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))
        ),
        "VLLM_ROCM_MOE_PADDING": lambda: bool(
            int(os.getenv("VLLM_ROCM_MOE_PADDING", "1"))
        ),
        "VLLM_ROCM_CUSTOM_PAGED_ATTN": lambda: (
            os.getenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": env_with_choices(
            "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
            "NONE",
            ["FP", "INT8", "INT6", "INT4", "NONE"],
        ),
        "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": lambda: (
            os.getenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower()
            in ("true", "1")
        ),
        "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": lambda: maybe_convert_int(
            os.environ.get("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
        ),
        "Q_SCALE_CONSTANT": lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),
        "K_SCALE_CONSTANT": lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),
        "V_SCALE_CONSTANT": lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),
        "VLLM_ENABLE_V1_MULTIPROCESSING": lambda: bool(
            int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1"))
        ),
        "VLLM_LOG_BATCHSIZE_INTERVAL": lambda: float(
            os.getenv("VLLM_LOG_BATCHSIZE_INTERVAL", "-1")
        ),
        "VLLM_DISABLE_COMPILE_CACHE": disable_compile_cache,
        "VLLM_SERVER_DEV_MODE": lambda: bool(
            int(os.getenv("VLLM_SERVER_DEV_MODE", "0"))
        ),
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE": lambda: int(
            os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")
        ),
        "VLLM_MLA_DISABLE": lambda: bool(int(os.getenv("VLLM_MLA_DISABLE", "0"))),
        "VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH": lambda: int(
            os.getenv("VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH", "32")
        ),
        "VLLM_RAY_PER_WORKER_GPUS": lambda: float(
            os.getenv("VLLM_RAY_PER_WORKER_GPUS", "1.0")
        ),
        "VLLM_RAY_BUNDLE_INDICES": lambda: os.getenv("VLLM_RAY_BUNDLE_INDICES", ""),
        "VLLM_CUDART_SO_PATH": lambda: os.getenv("VLLM_CUDART_SO_PATH", None),
        "VLLM_DP_RANK": lambda: int(os.getenv("VLLM_DP_RANK", "0")),
        # Default to VLLM_DP_RANK when not explicitly set
        "VLLM_DP_RANK_LOCAL": lambda: int(
            os.getenv("VLLM_DP_RANK_LOCAL") or os.getenv("VLLM_DP_RANK") or "0"
        ),
        "VLLM_DP_SIZE": lambda: int(os.getenv("VLLM_DP_SIZE", "1")),
        "VLLM_DP_MASTER_IP": lambda: os.getenv("VLLM_DP_MASTER_IP", "127.0.0.1"),
        "VLLM_DP_MASTER_PORT": lambda: int(os.getenv("VLLM_DP_MASTER_PORT", "0")),
        "VLLM_MOE_DP_CHUNK_SIZE": lambda: int(
            os.getenv("VLLM_MOE_DP_CHUNK_SIZE", "256")
        ),
        "VLLM_ENABLE_MOE_DP_CHUNK": lambda: bool(
            int(os.getenv("VLLM_ENABLE_MOE_DP_CHUNK", "1"))
        ),
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": lambda: os.environ.get(
            "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", "0"
        )
        == "1",
        "VLLM_RAY_DP_PACK_STRATEGY": lambda: os.getenv(
            "VLLM_RAY_DP_PACK_STRATEGY", "strict"
        ),
        "VLLM_CI_USE_S3": lambda: os.environ.get("VLLM_CI_USE_S3", "0") == "1",
        "VLLM_MODEL_REDIRECT_PATH": lambda: os.environ.get(
            "VLLM_MODEL_REDIRECT_PATH", None
        ),
        "VLLM_MARLIN_USE_ATOMIC_ADD": lambda: os.environ.get(
            "VLLM_MARLIN_USE_ATOMIC_ADD", "0"
        )
        == "1",
        "VLLM_MXFP4_USE_MARLIN": lambda: maybe_convert_bool(
            os.environ.get("VLLM_MXFP4_USE_MARLIN", None)
        ),
        "VLLM_MARLIN_INPUT_DTYPE": env_with_choices(
            "VLLM_MARLIN_INPUT_DTYPE", None, ["int8", "fp8"]
        ),
        "VLLM_DEEPEPLL_NVFP4_DISPATCH": lambda: bool(
            int(os.getenv("VLLM_DEEPEPLL_NVFP4_DISPATCH", "0"))
        ),
        "VLLM_V1_USE_OUTLINES_CACHE": lambda: os.environ.get(
            "VLLM_V1_USE_OUTLINES_CACHE", "0"
        )
        == "1",
        "VLLM_TPU_BUCKET_PADDING_GAP": lambda: int(
            os.environ["VLLM_TPU_BUCKET_PADDING_GAP"]
        )
        if "VLLM_TPU_BUCKET_PADDING_GAP" in os.environ
        else 0,
        "VLLM_TPU_MOST_MODEL_LEN": lambda: maybe_convert_int(
            os.environ.get("VLLM_TPU_MOST_MODEL_LEN", None)
        ),
        "VLLM_TPU_USING_PATHWAYS": lambda: bool(
            "proxy" in os.getenv("JAX_PLATFORMS", "").lower()
        ),
        "VLLM_USE_DEEP_GEMM": lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "1"))),
        "VLLM_MOE_USE_DEEP_GEMM": lambda: bool(
            int(os.getenv("VLLM_MOE_USE_DEEP_GEMM", "1"))
        ),
        "VLLM_USE_DEEP_GEMM_E8M0": lambda: bool(
            int(os.getenv("VLLM_USE_DEEP_GEMM_E8M0", "1"))
        ),
        "VLLM_DEEP_GEMM_WARMUP": env_with_choices(
            "VLLM_DEEP_GEMM_WARMUP",
            "relax",
            ["skip", "full", "relax"],
        ),
        "VLLM_USE_FUSED_MOE_GROUPED_TOPK": lambda: bool(
            int(os.getenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1"))
        ),
        "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": lambda: bool(
            int(os.getenv("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0"))
        ),
        "VLLM_USE_FLASHINFER_MOE_FP16": lambda: bool(
            int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP16", "0"))
        ),
        "VLLM_USE_FLASHINFER_MOE_FP8": lambda: bool(
            int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP8", "0"))
        ),
        "VLLM_USE_FLASHINFER_MOE_FP4": lambda: bool(
            int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP4", "0"))
        ),
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": lambda: bool(
            int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "0"))
        ),
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS": lambda: bool(
            int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS", "0"))
        ),
        "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": lambda: bool(
            int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16", "0"))
        ),
        "VLLM_XGRAMMAR_CACHE_MB": lambda: int(
            os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")
        ),
        "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": lambda: int(
            os.getenv("VLLM_MSGPACK_ZERO_COPY_THRESHOLD", "256")
        ),
        "VLLM_ALLOW_INSECURE_SERIALIZATION": lambda: bool(
            int(os.getenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0"))
        ),
        "VLLM_NIXL_SIDE_CHANNEL_HOST": lambda: os.getenv(
            "VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"
        ),
        "VLLM_NIXL_SIDE_CHANNEL_PORT": lambda: int(
            os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")
        ),
        "VLLM_MOONCAKE_BOOTSTRAP_PORT": lambda: int(
            os.getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "8998")
        ),
        "VLLM_ALL2ALL_BACKEND": env_with_choices(
            "VLLM_ALL2ALL_BACKEND",
            None,
            [
                "naive",
                "pplx",
                "deepep_high_throughput",
                "deepep_low_latency",
                "allgather_reducescatter",
                "flashinfer_all2allv",
            ],
        ),
        "VLLM_FLASHINFER_MOE_BACKEND": env_with_choices(
            "VLLM_FLASHINFER_MOE_BACKEND",
            "latency",
            ["throughput", "latency", "masked_gemm"],
        ),
        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": lambda: int(
            os.getenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(394 * 1024 * 1024))
        ),
        "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE": lambda: int(
            os.getenv("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", "163840")
        ),
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB": lambda: json.loads(
            os.getenv("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB", "{}")
        ),
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": lambda: os.environ.get(
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""
        ).lower(),
        "VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(
            os.getenv("VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")
        ),
        "VLLM_SLEEP_WHEN_IDLE": lambda: bool(
            int(os.getenv("VLLM_SLEEP_WHEN_IDLE", "0"))
        ),
        "VLLM_MQ_MAX_CHUNK_BYTES_MB": lambda: int(
            os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "16")
        ),
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": lambda: int(
            os.getenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")
        ),
        "VLLM_KV_CACHE_LAYOUT": env_with_choices(
            "VLLM_KV_CACHE_LAYOUT", None, ["NHD", "HND"]
        ),
        "VLLM_COMPUTE_NANS_IN_LOGITS": lambda: bool(
            int(os.getenv("VLLM_COMPUTE_NANS_IN_LOGITS", "0"))
        ),
        "VLLM_USE_NVFP4_CT_EMULATIONS": lambda: bool(
            int(os.getenv("VLLM_USE_NVFP4_CT_EMULATIONS", "0"))
        ),
        "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": lambda: int(
            os.getenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "480")
        ),
        "VLLM_MORIIO_CONNECTOR_READ_MODE": lambda: (
            os.getenv("VLLM_MORIIO_CONNECTOR_READ_MODE", "False").lower()
            in ("true", "1")
        ),
        "VLLM_MORIIO_QP_PER_TRANSFER": lambda: int(
            os.getenv("VLLM_MORIIO_QP_PER_TRANSFER", "1")
        ),
        "VLLM_MORIIO_POST_BATCH_SIZE": lambda: int(
            os.getenv("VLLM_MORIIO_POST_BATCH_SIZE", "-1")
        ),
        "VLLM_MORIIO_NUM_WORKERS": lambda: int(
            os.getenv("VLLM_MORIIO_NUM_WORKERS", "1")
        ),
        "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT": lambda: int(
            os.getenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "480")
        ),
        "VLLM_USE_CUDNN_PREFILL": lambda: bool(
            int(os.getenv("VLLM_USE_CUDNN_PREFILL", "0"))
        ),
        "VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL": lambda: bool(
            int(os.getenv("VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL", "0"))
        ),
        "VLLM_USE_TRTLLM_ATTENTION": lambda: (
            None
            if "VLLM_USE_TRTLLM_ATTENTION" not in os.environ
            else os.environ["VLLM_USE_TRTLLM_ATTENTION"].lower() in ("1", "true")
        ),
        "VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION": lambda: bool(
            int(os.getenv("VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION", "0"))
        ),
        "VLLM_HAS_FLASHINFER_CUBIN": lambda: bool(
            int(os.getenv("VLLM_HAS_FLASHINFER_CUBIN", "0"))
        ),
        "VLLM_NVFP4_GEMM_BACKEND": env_with_choices(
            "VLLM_NVFP4_GEMM_BACKEND",
            None,
            [
                "flashinfer-cudnn",
                "flashinfer-trtllm",
                "flashinfer-cutlass",
                "cutlass",
            ],
        ),
        "VLLM_ENABLE_CUDAGRAPH_GC": lambda: bool(
            int(os.getenv("VLLM_ENABLE_CUDAGRAPH_GC", "0"))
        ),
        "VLLM_LOOPBACK_IP": lambda: os.getenv("VLLM_LOOPBACK_IP", ""),
        "VLLM_PROCESS_NAME_PREFIX": lambda: os.getenv(
            "VLLM_PROCESS_NAME_PREFIX", "VLLM"
        ),
        "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE": lambda: bool(
            int(os.getenv("VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "1"))
        ),
        "VLLM_ENABLE_RESPONSES_API_STORE": lambda: bool(
            int(os.getenv("VLLM_ENABLE_RESPONSES_API_STORE", "0"))
        ),
        "VLLM_ROCM_FP8_MFMA_PAGE_ATTN": lambda: bool(
            int(os.getenv("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "0"))
        ),
        "VLLM_ALLREDUCE_USE_SYMM_MEM": lambda: bool(
            int(os.getenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1"))
        ),
        "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT": lambda: bool(
            int(os.getenv("VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", "0"))
        ),
        "VLLM_TUNED_CONFIG_FOLDER": lambda: os.getenv("VLLM_TUNED_CONFIG_FOLDER", None),
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": env_set_with_choices(
            "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
            default=[],
            choices=["container", "code_interpreter", "web_search_preview"],
        ),
        "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": lambda: bool(
            int(os.getenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "0"))
        ),
        "VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY": lambda: bool(
            int(os.getenv("VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY", "0"))
        ),
        "VLLM_CUSTOM_SCOPES_FOR_PROFILING": lambda: bool(
            int(os.getenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "0"))
        ),
        "VLLM_NVTX_SCOPES_FOR_PROFILING": lambda: bool(
            int(os.getenv("VLLM_NVTX_SCOPES_FOR_PROFILING", "0"))
        ),
        "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": lambda: bool(
            int(os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1"))
        ),
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": lambda: os.getenv(
            "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", "VLLM_OBJECT_STORAGE_SHM_BUFFER"
        ),
        "VLLM_DEEPEP_BUFFER_SIZE_MB": lambda: int(
            os.getenv("VLLM_DEEPEP_BUFFER_SIZE_MB", "1024")
        ),
        "VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE": lambda: bool(
            int(os.getenv("VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE", "0"))
        ),
        "VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL": lambda: bool(
            int(os.getenv("VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL", "0"))
        ),
        "VLLM_DBO_COMM_SMS": lambda: int(os.getenv("VLLM_DBO_COMM_SMS", "20")),
        "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE": lambda: bool(
            int(os.getenv("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"))
        ),
        "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING": lambda: bool(
            int(os.getenv("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "1"))
        ),
        "VLLM_USE_NCCL_SYMM_MEM": lambda: bool(
            int(os.getenv("VLLM_USE_NCCL_SYMM_MEM", "0"))
        ),
        "VLLM_NCCL_INCLUDE_PATH": lambda: os.environ.get(
            "VLLM_NCCL_INCLUDE_PATH", None
        ),
        "VLLM_USE_FBGEMM": lambda: bool(int(os.getenv("VLLM_USE_FBGEMM", "0"))),
        "VLLM_GC_DEBUG": lambda: os.getenv("VLLM_GC_DEBUG", ""),
        "VLLM_DEBUG_WORKSPACE": lambda: bool(
            int(os.getenv("VLLM_DEBUG_WORKSPACE", "0"))
        ),
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(
            int(os.getenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0"))
        ),
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": lambda: int(
            int(os.getenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 256))
        ),
        "VLLM_COMPILE_CACHE_SAVE_FORMAT": env_with_choices(
            "VLLM_COMPILE_CACHE_SAVE_FORMAT", "binary", ["binary", "unpacked"]
        ),
        "VLLM_USE_V2_MODEL_RUNNER": lambda: bool(
            int(os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0"))
        ),
        "VLLM_LOG_MODEL_INSPECTION": lambda: bool(
            int(os.getenv("VLLM_LOG_MODEL_INSPECTION", "0"))
        ),
        "VLLM_DEBUG_MFU_METRICS": lambda: bool(
            int(os.getenv("VLLM_DEBUG_MFU_METRICS", "0"))
        ),
        "VLLM_USE_MEGA_AOT_ARTIFACT": lambda: bool(
            int(os.getenv("VLLM_USE_MEGA_AOT_ARTIFACT", "0"))
        ),
        "VLLM_LORA_DISABLE_PDL": lambda: bool(
            int(os.getenv("VLLM_LORA_DISABLE_PDL", "0"))
        ),
        "VLLM_ZENTORCH_WEIGHT_PREPACK": lambda: bool(
            int(os.getenv("VLLM_ZENTORCH_WEIGHT_PREPACK", "1"))
        ),
        "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": lambda: bool(
            int(os.getenv("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "0"))
        ),
        "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": lambda: bool(
            int(os.getenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "0"))
        ),
        "VLLM_USE_FLASHINFER_MOE_INT4": lambda: bool(
            int(os.getenv("VLLM_USE_FLASHINFER_MOE_INT4", "0"))
        ),
        "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": lambda: bool(
            int(os.getenv("VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", "0"))
        ),
        "VLLM_MEDIA_FETCH_MAX_RETRIES": lambda: int(
            os.getenv("VLLM_MEDIA_FETCH_MAX_RETRIES", "3")
        ),
        "VLLM_MM_HASHER_ALGORITHM": env_with_choices(
            "VLLM_MM_HASHER_ALGORITHM",
            "blake3",
            ["blake3", "sha256", "sha512"],
            case_sensitive=False,
        ),
        "VLLM_LORA_RESOLVER_HF_REPO_LIST": lambda: os.getenv(
            "VLLM_LORA_RESOLVER_HF_REPO_LIST", None
        ),
        "VLLM_ROCM_USE_AITER_FP4BMM": lambda: (
            os.getenv("VLLM_ROCM_USE_AITER_FP4BMM", "True").lower() in ("true", "1")
        ),
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": lambda: (
            os.getenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "False").lower()
            in ("true", "1")
        ),
        "VLLM_ENABLE_PREGRAD_PASSES": lambda: bool(
            int(os.environ.get("VLLM_ENABLE_PREGRAD_PASSES", "0"))
        ),
        "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": lambda: os.getenv(
            "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY", ""
        ),
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": lambda: os.getenv(
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", ""
        ),
        "VLLM_FLASHINFER_ALLREDUCE_BACKEND": env_with_choices(
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND",
            "trtllm",
            ["auto", "trtllm", "mnnvl"],
        ),
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": lambda: bool(
            int(os.getenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0"))
        ),
        "VLLM_ALLREDUCE_USE_FLASHINFER": lambda: bool(
            int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0"))
        ),
        "VLLM_SYSTEM_START_DATE": lambda: os.getenv("VLLM_SYSTEM_START_DATE", None),
        "VLLM_DISABLE_LOG_LOGO": lambda: bool(
            int(os.getenv("VLLM_DISABLE_LOG_LOGO", "0"))
        ),
        "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": lambda: bool(
            int(os.getenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0"))
        ),
        "VLLM_ELASTIC_EP_DRAIN_REQUESTS": lambda: bool(
            int(os.getenv("VLLM_ELASTIC_EP_DRAIN_REQUESTS", "0"))
        ),
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": lambda: bool(
            int(os.getenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0"))
        ),
        "VLLM_NIXL_EP_MAX_NUM_RANKS": lambda: int(
            os.getenv("VLLM_NIXL_EP_MAX_NUM_RANKS", "32")
        ),
    }
    # --8<-- [end:env-vars-definition]

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._use_cache: bool = False

    def __getattr__(self, name: str) -> Any:
        """Lazily evaluate env var ``name`` from ``os.environ``.

        When caching is enabled (after :meth:`enable_envs_cache`), computed
        values are stored and returned on subsequent accesses.
        """
        if name in self._env_vars:
            if self._use_cache:
                if name not in self._cache:
                    self._cache[name] = self._env_vars[name]()
                return self._cache[name]
            return self._env_vars[name]()
        raise AttributeError(f"'Envs' object has no attribute {name!r}")

    def __contains__(self, name: str) -> bool:
        """Return ``True`` if env var ``name`` is explicitly set in the environment."""
        return name in os.environ

    def __dir__(self) -> list[str]:
        return list(self._env_vars.keys())

    def is_set(self, name: str) -> bool:
        """Return ``True`` if env var ``name`` is explicitly set in ``os.environ``."""
        if name in self._env_vars:
            return name in os.environ
        raise AttributeError(f"'Envs' object has no attribute {name!r}")

    def enable_envs_cache(self) -> None:
        """Enable caching of environment variables.

        After calling this, all env vars are evaluated once and cached.
        Subsequent accesses return the cached value without re-reading
        ``os.environ``.  Invoke after service initialization to reduce
        runtime overhead.
        """
        if self._use_cache:
            return
        self._use_cache = True
        # Pre-populate cache
        for key in self._env_vars:
            if key not in self._cache:
                self._cache[key] = self._env_vars[key]()

    def disable_envs_cache(self) -> None:
        """Reset the environment variables cache.

        After calling this, env vars are re-evaluated from ``os.environ`` on
        every access.  Useful to isolate environments between unit tests.
        """
        self._use_cache = False
        self._cache.clear()

    def _is_envs_cache_enabled(self) -> bool:
        """Return ``True`` if caching is currently enabled."""
        return self._use_cache

    def compile_factors(self) -> dict[str, object]:
        """Return env vars used for torch.compile cache keys.

        Start with every known vLLM env var; drop entries in ``ignored_factors``;
        hash everything else. This keeps the cache key aligned across workers.
        """
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
            "VLLM_DP_MASTER_IP",
            "VLLM_DP_MASTER_PORT",
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
            "VLLM_ENGINE_ITERATION_TIMEOUT_S",
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
            "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
            "VLLM_SLEEP_WHEN_IDLE",
            "VLLM_IMAGE_FETCH_TIMEOUT",
            "VLLM_VIDEO_FETCH_TIMEOUT",
            "VLLM_AUDIO_FETCH_TIMEOUT",
            "VLLM_MEDIA_URL_ALLOW_REDIRECTS",
            "VLLM_MEDIA_LOADING_THREAD_COUNT",
            "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",
            "VLLM_VIDEO_LOADER_BACKEND",
            "VLLM_MEDIA_CONNECTOR",
            "VLLM_ASSETS_CACHE",
            "VLLM_ASSETS_CACHE_MODEL_CLEAN",
            "VLLM_WORKER_MULTIPROC_METHOD",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
            "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",
            "VLLM_CPU_KVCACHE_SPACE",
            "VLLM_CPU_OMP_THREADS_BIND",
            "VLLM_CPU_NUM_OF_RESERVED_CPU",
            "VLLM_CPU_MOE_PREPACK",
            "VLLM_CPU_SGL_KERNEL",
            "VLLM_TEST_FORCE_LOAD_FORMAT",
            "LOCAL_RANK",
            "CUDA_VISIBLE_DEVICES",
            "NO_COLOR",
        }

        from vllm.config.utils import normalize_value

        factors: dict[str, object] = {}
        for factor, getter in self._env_vars.items():
            if factor in ignored_factors:
                continue

            try:
                raw = getter()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Skipping environment variable %s while hashing compile "
                    "factors: %s",
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

    def validate_environ(self, hard_fail: bool) -> None:
        """Check for unknown ``VLLM_*`` environment variables.

        Iterates over ``os.environ`` and warns (or raises) for any
        ``VLLM_*`` variable that is not registered in :attr:`_env_vars`.

        Args:
            hard_fail: If ``True``, raise :exc:`ValueError` on the first
                unknown variable; otherwise emit a warning and continue.
        """
        for env in os.environ:
            if env.startswith("VLLM_") and env not in self._env_vars:
                if hard_fail:
                    raise ValueError(
                        f"Unknown vLLM environment variable detected: {env}"
                    )
                else:
                    logger.warning(
                        "Unknown vLLM environment variable detected: %s", env
                    )


# ---------------------------------------------------------------------------
# Singleton instance - ``import vllm.envs as envs`` gives this instance
# ---------------------------------------------------------------------------

envs = Envs()

# Expose the env var keys list under the legacy name for backward compatibility
# (tests and tools that use ``environment_variables`` dict can import it here)
environment_variables: dict[str, Callable[[], Any]] = Envs._env_vars


# ---------------------------------------------------------------------------
# Module-level convenience wrappers for the singleton's caching methods.
# Importable as ``from vllm.envs_impl import enable_envs_cache``.
# ---------------------------------------------------------------------------


def enable_envs_cache() -> None:
    """Enable caching for the ``envs`` singleton. See :meth:`Envs.enable_envs_cache`."""
    envs.enable_envs_cache()


def disable_envs_cache() -> None:
    """Disable caching for the ``envs`` singleton. See :meth:`Envs.disable_envs_cache`."""  # noqa: E501
    envs.disable_envs_cache()
