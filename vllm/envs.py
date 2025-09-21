# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

if TYPE_CHECKING:
    VLLM_HOST_IP: str = ""
    VLLM_PORT: Optional[int] = None
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    VLLM_USE_MODELSCOPE: bool = False
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
    VLLM_NCCL_SO_PATH: Optional[str] = None
    LD_LIBRARY_PATH: Optional[str] = None
    VLLM_USE_TRITON_FLASH_ATTN: bool = True
    VLLM_V1_USE_PREFILL_DECODE_ATTENTION: bool = False
    VLLM_USE_AITER_UNIFIED_ATTENTION: bool = False
    VLLM_FLASH_ATTN_VERSION: Optional[int] = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 60
    VLLM_API_KEY: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    VLLM_MODEL_REDIRECT_PATH: Optional[str] = None
    VLLM_CACHE_ROOT: str = os.path.expanduser("~/.cache/vllm")
    VLLM_CONFIG_ROOT: str = os.path.expanduser("~/.config/vllm")
    VLLM_USAGE_STATS_SERVER: str = "https://stats.vllm.ai"
    VLLM_NO_USAGE_STATS: bool = False
    VLLM_DISABLE_FLASHINFER_PREFILL: bool = False
    VLLM_DO_NOT_TRACK: bool = False
    VLLM_USAGE_SOURCE: str = ""
    VLLM_CONFIGURE_LOGGING: int = 1
    VLLM_LOGGING_LEVEL: str = "INFO"
    VLLM_LOGGING_PREFIX: str = ""
    VLLM_LOGGING_STREAM: str = "ext://sys.stdout"
    VLLM_LOGGING_CONFIG_PATH: Optional[str] = None
    VLLM_LOGITS_PROCESSOR_THREADS: Optional[int] = None
    VLLM_LOG_STATS_INTERVAL: float = 10.
    VLLM_TRACE_FUNCTION: int = 0
    VLLM_ATTENTION_BACKEND: Optional[str] = None
    VLLM_USE_FLASHINFER_SAMPLER: Optional[bool] = None
    VLLM_PP_LAYER_PARTITION: Optional[str] = None
    VLLM_CPU_KVCACHE_SPACE: Optional[int] = 0
    VLLM_CPU_OMP_THREADS_BIND: str = ""
    VLLM_CPU_NUM_OF_RESERVED_CPU: Optional[int] = None
    VLLM_CPU_MOE_PREPACK: bool = True
    VLLM_CPU_SGL_KERNEL: bool = False
    VLLM_XLA_CACHE_PATH: str = os.path.join(VLLM_CACHE_ROOT, "xla_cache")
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    VLLM_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024
    VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING: bool = True
    VLLM_USE_RAY_SPMD_WORKER: bool = False
    VLLM_USE_RAY_COMPILED_DAG: bool = False
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: Literal["auto", "nccl",
                                                    "shm"] = "auto"
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    VLLM_USE_RAY_WRAPPED_PP_COMM: bool = True
    VLLM_XLA_USE_SPMD: bool = False
    VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"
    VLLM_ASSETS_CACHE: str = os.path.join(VLLM_CACHE_ROOT, "assets")
    VLLM_IMAGE_FETCH_TIMEOUT: int = 5
    VLLM_VIDEO_FETCH_TIMEOUT: int = 30
    VLLM_AUDIO_FETCH_TIMEOUT: int = 10
    VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
    VLLM_VIDEO_LOADER_BACKEND: str = "opencv"
    VLLM_MM_INPUT_CACHE_GIB: int = 4
    VLLM_TARGET_DEVICE: str = "cuda"
    VLLM_MAIN_CUDA_VERSION: str = "12.8"
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_DOCKER_BUILD_CONTEXT: bool = False
    VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Optional[Literal["Debug", "Release",
                                       "RelWithDebInfo"]] = None
    VERBOSE: bool = False
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    VLLM_RPC_TIMEOUT: int = 10000  # ms
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds
    VLLM_PLUGINS: Optional[list[str]] = None
    VLLM_LORA_RESOLVER_CACHE_DIR: Optional[str] = None
    VLLM_TORCH_PROFILER_DIR: Optional[str] = None
    VLLM_TORCH_PROFILER_RECORD_SHAPES: bool = False
    VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY: bool = False
    VLLM_TORCH_PROFILER_WITH_STACK: bool = True
    VLLM_TORCH_PROFILER_WITH_FLOPS: bool = False
    VLLM_USE_TRITON_AWQ: bool = False
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    VLLM_SKIP_P2P_CHECK: bool = False
    VLLM_DISABLED_KERNELS: list[str] = []
    VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION: bool = False
    VLLM_USE_V1: bool = True
    VLLM_ROCM_USE_AITER: bool = False
    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
    VLLM_ROCM_USE_AITER_LINEAR: bool = True
    VLLM_ROCM_USE_AITER_MOE: bool = True
    VLLM_ROCM_USE_AITER_RMSNORM: bool = True
    VLLM_ROCM_USE_AITER_MLA: bool = True
    VLLM_ROCM_USE_AITER_MHA: bool = True
    VLLM_ROCM_USE_AITER_FP8BMM: bool = True
    VLLM_ROCM_USE_SKINNY_GEMM: bool = True
    VLLM_ROCM_FP8_PADDING: bool = True
    VLLM_ROCM_MOE_PADDING: bool = True
    VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    VLLM_LOG_BATCHSIZE_INTERVAL: float = -1
    VLLM_DISABLE_COMPILE_CACHE: bool = False
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    VLLM_SERVER_DEV_MODE: bool = False
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    VLLM_MLA_DISABLE: bool = False
    VLLM_RAY_PER_WORKER_GPUS: float = 1.0
    VLLM_RAY_BUNDLE_INDICES: str = ""
    VLLM_CUDART_SO_PATH: Optional[str] = None
    VLLM_DP_RANK: int = 0
    VLLM_DP_RANK_LOCAL: int = -1
    VLLM_DP_SIZE: int = 1
    VLLM_DP_MASTER_IP: str = ""
    VLLM_DP_MASTER_PORT: int = 0
    VLLM_MOE_DP_CHUNK_SIZE: int = 256
    VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_MXFP4_USE_MARLIN: Optional[bool] = None
    VLLM_V0_USE_OUTLINES_CACHE: bool = False
    VLLM_V1_USE_OUTLINES_CACHE: bool = False
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_TPU_MOST_MODEL_LEN: Optional[int] = None
    VLLM_TPU_USING_PATHWAYS: bool = False
    VLLM_USE_DEEP_GEMM: bool = True
    VLLM_USE_DEEP_GEMM_E8M0: bool = True
    VLLM_USE_DEEP_GEMM_E8M0_HOPPER: bool = False
    VLLM_SKIP_DEEP_GEMM_WARMUP: bool = False
    VLLM_USE_FUSED_MOE_GROUPED_TOPK: bool = True
    VLLM_USE_FLASHINFER_MOE_FP8: bool = False
    VLLM_USE_FLASHINFER_MOE_FP4: bool = False
    VLLM_FLASHINFER_MOE_BACKEND: Literal["throughput",
                                         "latency"] = "throughput"
    VLLM_XGRAMMAR_CACHE_MB: int = 0
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5557
    VLLM_ALL2ALL_BACKEND: Literal["naive", "pplx",
                                  "deepep_high_throughput",
                                  "deepep_low_latency",
                                  "allgather_reducescatter"] = \
                                  "allgather_reducescatter"
    VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    VLLM_SLEEP_WHEN_IDLE: bool = False
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    VLLM_KV_CACHE_LAYOUT: Optional[Literal["NHD", "HND"]] = None
    VLLM_COMPUTE_NANS_IN_LOGITS: bool = False
    VLLM_USE_NVFP4_CT_EMULATIONS: bool = False
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: Literal["FP", "INT8", "INT6", "INT4",
                                                 "NONE"] = "NONE"
    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: Optional[int] = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int = 120
    VLLM_USE_CUDNN_PREFILL: bool = False
    VLLM_ENABLE_CUDAGRAPH_GC: bool = False
    VLLM_LOOPBACK_IP: str = ""
    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = False
    VLLM_ENABLE_RESPONSES_API_STORE: bool = False
    VLLM_USE_TRTLLM_ATTENTION: Optional[str] = None
    VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION: bool = False
    VLLM_HAS_FLASHINFER_CUBIN: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_BF16: bool = False
    VLLM_ROCM_FP8_MFMA_PAGE_ATTN: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: bool = False
    VLLM_ALLREDUCE_USE_SYMM_MEM: bool = False
    VLLM_TUNED_CONFIG_FOLDER: Optional[str] = None
    VLLM_DISABLE_PAD_FOR_CUDAGRAPH: bool = False
    VLLM_GPT_OSS_USE_CONTAINER_TOOL: bool = False
    VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: bool = False
    VLLM_CUSTOM_SCOPES_FOR_PROFILING: bool = False
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = True
    VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME: str = "VLLM_OBJECT_STORAGE_SHM_BUFFER"


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def maybe_convert_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return bool(int(value))


def env_with_choices(
        env_name: str,
        default: Optional[str],
        choices: Union[list[str], Callable[[], list[str]]],
        case_sensitive: bool = True) -> Callable[[], Optional[str]]:
    """
    Create a lambda that validates environment variable against allowed choices
    
    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive
        
    Returns:
        Lambda function for environment_variables dict
    """

    def _get_validated_env() -> Optional[str]:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(f"Invalid value '{value}' for {env_name}. "
                             f"Valid options: {actual_choices}.")

        return value

    return _get_validated_env


def get_vllm_port() -> Optional[int]:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.
    """
    if 'VLLM_PORT' not in os.environ:
        return None

    port = os.getenv('VLLM_PORT', '0')

    try:
        return int(port)
    except ValueError as err:
        from urllib.parse import urlparse
        parsed = urlparse(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://docs.vllm.ai/en/stable/serving/env_vars.html"
            ) from None
        raise ValueError(
            f"VLLM_PORT '{port}' must be a valid integer") from err


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# --8<-- [start:env-vars-definition]

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of vLLM, supporting [cuda (by default),
    # rocm, cpu]
    "VLLM_TARGET_DEVICE":
    lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),

    # Main CUDA version of vLLM, supporting [12.6, 12.8, 12.9],
    # 12.8 is the default. This follows PyTorch but can be overridden.
    "VLLM_MAIN_CUDA_VERSION":
    lambda: os.getenv("VLLM_MAIN_CUDA_VERSION", "").lower() or "12.8",

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, vllm will use precompiled binaries (*.so)
    "VLLM_USE_PRECOMPILED":
    lambda: os.environ.get("VLLM_USE_PRECOMPILED", "").strip().lower() in
    ("1", "true") or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),

    # Used to mark that setup.py is running in a Docker build context,
    # in order to force the use of precompiled binaries.
    "VLLM_DOCKER_BUILD_CONTEXT":
    lambda: os.environ.get("VLLM_DOCKER_BUILD_CONTEXT", "").strip().lower() in
    ("1", "true"),

    # Whether to force using nightly wheel in python build.
    # This is used for testing the nightly wheel in python build.
    "VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL":
    lambda: bool(int(os.getenv("VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL", "0"))
                 ),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    env_with_choices("CMAKE_BUILD_TYPE", None,
        ["Debug", "Release", "RelWithDebInfo"]),

    # If set, vllm will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for vLLM configuration files
    # Defaults to `~/.config/vllm` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how vllm finds its configuration files
    # during runtime, but also affects how vllm installs its configuration
    # files during **installation**.
    "VLLM_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "vllm"),
        )),

    # ================== Runtime Env Vars ==================

    # Root directory for vLLM cache files
    # Defaults to `~/.cache/vllm` unless `XDG_CACHE_HOME` is set
    "VLLM_CACHE_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "vllm"),
        )),

    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    'VLLM_HOST_IP':
    lambda: os.getenv('VLLM_HOST_IP', ""),

    # used in distributed environment to manually set the communication port
    # Note: if VLLM_PORT is set, and some code asks for multiple ports, the
    # VLLM_PORT will be used as the first port, and the rest will be generated
    # by incrementing the VLLM_PORT value.
    'VLLM_PORT':
    get_vllm_port,

    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    'VLLM_RPC_BASE_PATH':
    lambda: os.getenv('VLLM_RPC_BASE_PATH', tempfile.gettempdir()),

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "VLLM_USE_MODELSCOPE":
    lambda: os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true",

    # Interval in seconds to log a warning message when the ring buffer is full
    "VLLM_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")),

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH":
    lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),

    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # flag to control if vllm should use triton flash attention
    "VLLM_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_USE_TRITON_FLASH_ATTN", "True").lower() in
             ("true", "1")),

    # Use separate prefill and decode kernels for V1 attention instead of
    # the unified triton kernel.
    "VLLM_V1_USE_PREFILL_DECODE_ATTENTION":
    lambda:
    (os.getenv("VLLM_V1_USE_PREFILL_DECODE_ATTENTION", "False").lower() in
     ("true", "1")),

    # Use AITER triton unified attention for V1 attention
    "VLLM_USE_AITER_UNIFIED_ATTENTION":
    lambda:
    (os.getenv("VLLM_USE_AITER_UNIFIED_ATTENTION", "False").lower() in
     ("true", "1")),

    # Force vllm to use a specific flash-attention version (2 or 3), only valid
    # when using the flash-attention backend.
    "VLLM_FLASH_ATTN_VERSION":
    lambda: maybe_convert_int(os.environ.get("VLLM_FLASH_ATTN_VERSION", None)),

    # Feature flag to enable/disable Inductor standalone compile.
    # In torch <= 2.7 we ignore this flag; in torch >= 2.8 this is
    # enabled by default.
    "VLLM_USE_STANDALONE_COMPILE":
    lambda: os.environ.get("VLLM_USE_STANDALONE_COMPILE", "1") == "1",

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),

    # timeout for each iteration in the engine
    "VLLM_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")),

    # API key for vLLM API server
    "VLLM_API_KEY":
    lambda: os.environ.get("VLLM_API_KEY", None),

    # Whether to log responses from API Server for debugging
    "VLLM_DEBUG_LOG_API_SERVER_RESPONSE":
    lambda: os.environ.get("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "False"
                           ).lower() == "true",

    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),

    # Usage stats collection
    "VLLM_USAGE_STATS_SERVER":
    lambda: os.environ.get("VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"),
    "VLLM_NO_USAGE_STATS":
    lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0") == "1",
    "VLLM_DISABLE_FLASHINFER_PREFILL":
    lambda: os.environ.get("VLLM_DISABLE_FLASHINFER_PREFILL", "0") == "1",
    "VLLM_DO_NOT_TRACK":
    lambda: (os.environ.get("VLLM_DO_NOT_TRACK", None) or os.environ.get(
        "DO_NOT_TRACK", None) or "0") == "1",
    "VLLM_USAGE_SOURCE":
    lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),

    # Logging configuration
    # If set to 0, vllm will not configure logging
    # If set to 1, vllm will configure logging using the default configuration
    #    or the configuration file specified by VLLM_LOGGING_CONFIG_PATH
    "VLLM_CONFIGURE_LOGGING":
    lambda: int(os.getenv("VLLM_CONFIGURE_LOGGING", "1")),
    "VLLM_LOGGING_CONFIG_PATH":
    lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "VLLM_LOGGING_LEVEL":
    lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),

    # this is used for configuring the default logging stream
    "VLLM_LOGGING_STREAM":
    lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout"),

    # if set, VLLM_LOGGING_PREFIX will be prepended to all log messages
    "VLLM_LOGGING_PREFIX":
    lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),

    # if set, vllm will call logits processors in a thread pool with this many
    # threads. This is useful when using custom logits processors that either
    # (a) launch additional CUDA kernels or (b) do significant CPU-bound work
    # while not holding the python GIL, or both.
    "VLLM_LOGITS_PROCESSOR_THREADS":
    lambda: int(os.getenv("VLLM_LOGITS_PROCESSOR_THREADS", "0"))
    if "VLLM_LOGITS_PROCESSOR_THREADS" in os.environ else None,

    # If set, vllm will log stats at this interval in seconds
    # If not set, vllm will log stats every 10 seconds.
    "VLLM_LOG_STATS_INTERVAL":
    lambda: val if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10.")))
        > 0. else 10.,

    # Trace function calls
    # If set to 1, vllm will trace function calls
    # Useful for debugging
    "VLLM_TRACE_FUNCTION":
    lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),

    # Backend for attention computation
    # Example options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "XFORMERS": use XFormers
    # - "FLASHINFER": use flashinfer
    # - "FLASHMLA": use FlashMLA
    # - "FLASH_ATTN_MLA": use FlashAttention for MLA
    # - "FLASHINFER_MLA": use FlashInfer for MLA
    # - "CUTLASS_MLA": use CUTLASS for MLA
    # All possible options loaded dynamically from _Backend enum
    "VLLM_ATTENTION_BACKEND":
    env_with_choices("VLLM_ATTENTION_BACKEND", None,
                     lambda: list(__import__('vllm.platforms.interface', \
                        fromlist=['_Backend'])._Backend.__members__.keys())),

    # If set, vllm will use flashinfer sampler
    "VLLM_USE_FLASHINFER_SAMPLER":
    lambda: bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ else None,

    # Pipeline stage partition strategy
    "VLLM_PP_LAYER_PARTITION":
    lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),

    # (CPU backend only) CPU key-value cache space.
    # default is None and will be set as 4 GB
    "VLLM_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0"))
    if "VLLM_CPU_KVCACHE_SPACE" in os.environ else None,

    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "VLLM_CPU_OMP_THREADS_BIND":
    lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "auto"),

    # (CPU backend only) CPU cores not used by OMP threads .
    # Those CPU cores will not be used by OMP threads of a rank.
    "VLLM_CPU_NUM_OF_RESERVED_CPU":
    lambda: int(os.getenv("VLLM_CPU_NUM_OF_RESERVED_CPU", "0"))
    if "VLLM_CPU_NUM_OF_RESERVED_CPU" in os.environ else None,

    # (CPU backend only) whether to use prepack for MoE layer. This will be
    # passed to ipex.llm.modules.GatedMLPMOE. On unsupported CPUs, you might
    # need to set this to "0" (False).
    "VLLM_CPU_MOE_PREPACK":
    lambda: bool(int(os.getenv("VLLM_CPU_MOE_PREPACK", "1"))),

    # (CPU backend only) whether to use SGL kernels, optimized for small batch.
    "VLLM_CPU_SGL_KERNEL":
    lambda: bool(int(os.getenv("VLLM_CPU_SGL_KERNEL", "0"))),

    # If the env var is set, then all workers will execute as separate
    # processes from the engine, and we use the same mechanism to trigger
    # execution on all workers.
    # Run vLLM with VLLM_USE_RAY_SPMD_WORKER=1 to enable it.
    "VLLM_USE_RAY_SPMD_WORKER":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_SPMD_WORKER", "0"))),

    # If the env var is set, it uses the Ray's Compiled Graph
    # (previously known as ADAG) API which optimizes the
    # control plane overhead.
    # Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
    # Note that this variable is set to 1 in V1 by default
    # when ray distributed executor is used.
    "VLLM_USE_RAY_COMPILED_DAG":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG", "0"))),

    # If the env var is set, Ray Compiled Graph uses the specified
    # channel type to communicate between workers belonging to
    # different pipeline-parallel stages.
    # Available options:
    # - "auto": use the default channel type
    # - "nccl": use NCCL for communication
    # - "shm": use shared memory and gRPC for communication
    # This flag is ignored if VLLM_USE_RAY_COMPILED_DAG is not set.
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":
    env_with_choices("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto",
        ["auto", "nccl", "shm"]),

    # If the env var is set, it enables GPU communication overlap
    # (experimental feature) in Ray's Compiled Graph. This flag is ignored if
    # VLLM_USE_RAY_COMPILED_DAG is not set.
    "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
                 ),

    # If the env var is set, it uses a Ray Communicator wrapping
    # vLLM's pipeline parallelism communicator to interact with Ray's
    # Compiled Graph. Otherwise, it uses Ray's NCCL communicator.
    # This flag is ignored if VLLM_USE_RAY_COMPILED_DAG is not set.
    "VLLM_USE_RAY_WRAPPED_PP_COMM":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_WRAPPED_PP_COMM", "1"))),

    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "VLLM_WORKER_MULTIPROC_METHOD":
    env_with_choices("VLLM_WORKER_MULTIPROC_METHOD", "fork",
       ["spawn", "fork"]),

    # Path to the cache for storing downloaded assets
    "VLLM_ASSETS_CACHE":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "vllm", "assets"),
        )),

    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "VLLM_IMAGE_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")),

    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "VLLM_VIDEO_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "30")),

    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "VLLM_AUDIO_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "10")),

    # Max number of workers for the thread pool handling
    # media bytes loading. Set to 1 to disable parallel processing.
    # Default is 8
    "VLLM_MEDIA_LOADING_THREAD_COUNT":
    lambda: int(os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8")),

    # Maximum filesize in MB for a single audio file when processing
    # speech-to-text requests. Files larger than this will be rejected.
    # Default is 25 MB
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB":
    lambda: int(os.getenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "25")),

    # Backend for Video IO
    # - "opencv": Default backend that uses OpenCV stream buffered backend.
    #
    # Custom backend implementations can be registered
    # via `@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "VLLM_VIDEO_LOADER_BACKEND":
    lambda: os.getenv("VLLM_VIDEO_LOADER_BACKEND", "opencv"),

    # [DEPRECATED] Cache size (in GiB per process) for multimodal input cache
    # Default is 4 GiB per API process + 4 GiB per engine core process
    "VLLM_MM_INPUT_CACHE_GIB":
    lambda: int(os.getenv("VLLM_MM_INPUT_CACHE_GIB", "4")),

    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "VLLM_XLA_CACHE_PATH":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
        )),

    # If set, assert on XLA recompilation after each execution step.
    "VLLM_XLA_CHECK_RECOMPILATION":
    lambda: bool(int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION", "0"))),

    # Enable SPMD mode for TPU backend.
    "VLLM_XLA_USE_SPMD":
    lambda: bool(int(os.getenv("VLLM_XLA_USE_SPMD", "0"))),
    "VLLM_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE", "32768")),
    # Control whether to use fused MoE activation chunking. Current chunking
    # logic is incompatible with torch.compile and causes IMA. See issue
    # https://github.com/vllm-project/vllm/issues/19631.
    "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING":
    lambda: bool(
        int(os.getenv("VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING", "1"))),

    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH":
    lambda: bool(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", 0)),

    # If the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN":
    lambda:
    (os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower() in
     ("1", "true")),

    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "VLLM_TEST_FORCE_FP8_MARLIN":
    lambda:
    (os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower() in
     ("1", "true")),
    "VLLM_TEST_FORCE_LOAD_FORMAT":
    lambda: os.getenv("VLLM_TEST_FORCE_LOAD_FORMAT", "dummy"),

    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    "VLLM_RPC_TIMEOUT":
    lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),

    # Timeout in seconds for keeping HTTP connections alive in API server
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE":
    lambda: int(os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")),

    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "VLLM_PLUGINS":
    lambda: None if "VLLM_PLUGINS" not in os.environ else os.environ[
        "VLLM_PLUGINS"].split(","),

    # a local directory to look in for unrecognized LoRA adapters.
    # only works if plugins are enabled and
    # VLLM_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    "VLLM_LORA_RESOLVER_CACHE_DIR":
    lambda: os.getenv("VLLM_LORA_RESOLVER_CACHE_DIR", None),

    # Enables torch profiler if set.
    # Both AsyncLLM's CPU traces as well as workers'
    # traces (CPU & GPU) will be saved under this directory.
    # Note that it must be an absolute path.
    "VLLM_TORCH_PROFILER_DIR":
    lambda: (None if os.getenv("VLLM_TORCH_PROFILER_DIR", None) is None else os
             .path.abspath(os.path.expanduser(os.getenv(
        "VLLM_TORCH_PROFILER_DIR", ".")))),

    # Enable torch profiler to record shapes if set
    # VLLM_TORCH_PROFILER_RECORD_SHAPES=1. If not set, torch profiler will
    # not record shapes.
    "VLLM_TORCH_PROFILER_RECORD_SHAPES":
    lambda: bool(os.getenv("VLLM_TORCH_PROFILER_RECORD_SHAPES", "0") != "0"),

    # Enable torch profiler to profile memory if set
    # VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1. If not set, torch profiler
    # will not profile memory.
    "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY":
    lambda: bool(
        os.getenv("VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY", "0") != "0"),

    # Enable torch profiler to profile stack if set
    # VLLM_TORCH_PROFILER_WITH_STACK=1. If not set, torch profiler WILL
    # profile stack by default.
    "VLLM_TORCH_PROFILER_WITH_STACK":
    lambda: bool(os.getenv("VLLM_TORCH_PROFILER_WITH_STACK", "1") != "0"),

    # Enable torch profiler to profile flops if set
    # VLLM_TORCH_PROFILER_WITH_FLOPS=1. If not set, torch profiler will
    # not profile flops.
    "VLLM_TORCH_PROFILER_WITH_FLOPS":
    lambda: bool(os.getenv("VLLM_TORCH_PROFILER_WITH_FLOPS", "0") != "0"),

    # If set, vLLM will use Triton implementations of AWQ.
    "VLLM_USE_TRITON_AWQ":
    lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),

    # If set, allow loading or unloading lora adapters in runtime,
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING":
    lambda:
    (os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower() in
     ("1", "true")),

    # We assume drivers can report p2p status correctly.
    # If the program hangs when using custom allreduce,
    # potantially caused by a bug in the driver (535 series),
    # if might be helpful to set VLLM_SKIP_P2P_CHECK=0
    # so that vLLM can verify if p2p is actually working.
    # See https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    "VLLM_SKIP_P2P_CHECK":
    lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "1") == "1",

    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "VLLM_DISABLED_KERNELS":
    lambda: [] if "VLLM_DISABLED_KERNELS" not in os.environ else os.environ[
        "VLLM_DISABLED_KERNELS"].split(","),

    # Swaps the all reduce backend that we use to coordinate the DP padding
    # information from NCCL to gloo.
    "VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION":
    lambda:
    (os.getenv("VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION", "False").lower() in
             ("true", "1")),

    # If set, use the V1 code path.
    "VLLM_USE_V1":
    lambda: bool(int(os.getenv("VLLM_USE_V1", "1"))),

    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    "VLLM_ROCM_USE_AITER":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER", "False").lower() in
             ("true", "1")),

    # Whether to use aiter paged attention.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_PAGED_ATTN":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_PAGED_ATTN", "False").lower() in
             ("true", "1")),

    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    "VLLM_ROCM_USE_AITER_LINEAR":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_LINEAR", "True").lower() in
             ("true", "1")),

    # Whether to use aiter moe ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MOE":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_MOE", "True").lower() in
             ("true", "1")),

    # use aiter rms norm op if aiter ops are enabled.
    "VLLM_ROCM_USE_AITER_RMSNORM":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_RMSNORM", "True").lower() in
             ("true", "1")),

    # Whether to use aiter mla ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MLA":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_MLA", "True").lower() in
             ("true", "1")),

    # Whether to use aiter mha ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MHA":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_MHA", "True").lower() in
             ("true", "1")),

    # Whether to use aiter triton fp8 bmm kernel
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_FP8BMM":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_FP8BMM", "True").lower() in
             ("true", "1")),

    # use rocm skinny gemms
    "VLLM_ROCM_USE_SKINNY_GEMM":
    lambda: (os.getenv("VLLM_ROCM_USE_SKINNY_GEMM", "True").lower() in
             ("true", "1")),

    # Pad the fp8 weights to 256 bytes for ROCm
    "VLLM_ROCM_FP8_PADDING":
    lambda: bool(int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))),

    # Pad the weights for the moe kernel
    "VLLM_ROCM_MOE_PADDING":
    lambda: bool(int(os.getenv("VLLM_ROCM_MOE_PADDING", "1"))),

    # custom paged attention kernel for MI3* cards
    "VLLM_ROCM_CUSTOM_PAGED_ATTN":
    lambda: (os.getenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "True").lower() in
             ("true", "1")),

    # Custom quick allreduce kernel for MI3* cards
    # Choice of quantization level: FP, INT8, INT6, INT4 or NONE
    # Recommended for large models to get allreduce
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION":
    env_with_choices("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "NONE",
                            ["FP", "INT8", "INT6", "INT4", "NONE"]),

    # Custom quick allreduce kernel for MI3* cards
    # Due to the lack of the bfloat16 asm instruction, bfloat16
    # kernels are slower than fp16,
    # If environment variable is set to 1, the input is converted to fp16
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16":
    lambda:
    (os.getenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower() in
     ("true", "1")),

    # Custom quick allreduce kernel for MI3* cards.
    # Controls the maximum allowed number of data bytes(MB) for custom quick
    # allreduce communication.
    # Default: 2048 MB.
    # Data exceeding this size will use either custom allreduce or RCCL
    # communication.
    "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB":
    lambda: maybe_convert_int(
        os.environ.get("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)),

    # Divisor for dynamic query scale factor calculation for FP8 KV Cache
    "Q_SCALE_CONSTANT":
    lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),
    # Divisor for dynamic key scale factor calculation for FP8 KV Cache
    "K_SCALE_CONSTANT":
    lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),
    # Divisor for dynamic value scale factor calculation for FP8 KV Cache
    "V_SCALE_CONSTANT":
    lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),

    # If set, enable multiprocessing in LLM for the V1 code path.
    "VLLM_ENABLE_V1_MULTIPROCESSING":
    lambda: bool(int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1"))),
    "VLLM_LOG_BATCHSIZE_INTERVAL":
    lambda: float(os.getenv("VLLM_LOG_BATCHSIZE_INTERVAL", "-1")),
    "VLLM_DISABLE_COMPILE_CACHE":
    lambda: bool(int(os.getenv("VLLM_DISABLE_COMPILE_CACHE", "0"))),

    # If set, vllm will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "VLLM_SERVER_DEV_MODE":
    lambda: bool(int(os.getenv("VLLM_SERVER_DEV_MODE", "0"))),

    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")),

    # If set, vLLM will disable the MLA attention optimizations.
    "VLLM_MLA_DISABLE":
    lambda: bool(int(os.getenv("VLLM_MLA_DISABLE", "0"))),

    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as vLLM.
    "VLLM_RAY_PER_WORKER_GPUS":
    lambda: float(os.getenv("VLLM_RAY_PER_WORKER_GPUS", "1.0")),

    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "VLLM_RAY_BUNDLE_INDICES":
    lambda: os.getenv("VLLM_RAY_BUNDLE_INDICES", ""),

    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable VLLM_CUDART_SO_PATH.
    "VLLM_CUDART_SO_PATH":
    lambda: os.getenv("VLLM_CUDART_SO_PATH", None),

    # Rank of the process in the data parallel setting
    "VLLM_DP_RANK":
    lambda: int(os.getenv("VLLM_DP_RANK", "0")),

    # Rank of the process in the data parallel setting.
    # Defaults to VLLM_DP_RANK when not set.
    "VLLM_DP_RANK_LOCAL":
    lambda: int(
        os.getenv("VLLM_DP_RANK_LOCAL", sys.modules[__name__].VLLM_DP_RANK)),

    # World size of the data parallel setting
    "VLLM_DP_SIZE":
    lambda: int(os.getenv("VLLM_DP_SIZE", "1")),

    # IP address of the master node in the data parallel setting
    "VLLM_DP_MASTER_IP":
    lambda: os.getenv("VLLM_DP_MASTER_IP", "127.0.0.1"),

    # Port of the master node in the data parallel setting
    "VLLM_DP_MASTER_PORT":
    lambda: int(os.getenv("VLLM_DP_MASTER_PORT", "0")),

    # In the context of executing MoE models with Data-Parallel, Expert-Parallel
    # and Batched All-to-All dispatch/combine kernels, VLLM_MOE_DP_CHUNK_SIZE
    # dictates the quantum of tokens that can be dispatched from a DP
    # rank. All DP ranks process the activations in VLLM_MOE_DP_CHUNK_SIZE
    # units.
    "VLLM_MOE_DP_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_MOE_DP_CHUNK_SIZE", "256")),

    # Randomize inputs during dummy runs when using Data Parallel
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS":
    lambda: os.environ.get("VLLM_RANDOMIZE_DP_DUMMY_INPUTS", "0") == "1",

    # Whether to use S3 path for model loading in CI via RunAI Streamer
    "VLLM_CI_USE_S3":
    lambda: os.environ.get("VLLM_CI_USE_S3", "0") == "1",

    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "VLLM_MODEL_REDIRECT_PATH":
    lambda: os.environ.get("VLLM_MODEL_REDIRECT_PATH", None),

    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "VLLM_MARLIN_USE_ATOMIC_ADD":
    lambda: os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0") == "1",

    # Whether to use marlin kernel in mxfp4 quantization method
    "VLLM_MXFP4_USE_MARLIN":
    lambda: maybe_convert_bool(os.environ.get("VLLM_MXFP4_USE_MARLIN", None)),

    # Whether to turn on the outlines cache for V0
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "VLLM_V0_USE_OUTLINES_CACHE":
    lambda: os.environ.get("VLLM_V0_USE_OUTLINES_CACHE", "0") == "1",

    # Whether to turn on the outlines cache for V1
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "VLLM_V1_USE_OUTLINES_CACHE":
    lambda: os.environ.get("VLLM_V1_USE_OUTLINES_CACHE", "0") == "1",

    # Gap between padding buckets for the forward pass. So we have
    # 8, we will run forward pass with [16, 24, 32, ...].
    "VLLM_TPU_BUCKET_PADDING_GAP":
    lambda: int(os.environ["VLLM_TPU_BUCKET_PADDING_GAP"])
    if "VLLM_TPU_BUCKET_PADDING_GAP" in os.environ else 0,
    "VLLM_TPU_MOST_MODEL_LEN":
    lambda: maybe_convert_int(os.environ.get("VLLM_TPU_MOST_MODEL_LEN", None)),

    # Whether using Pathways
    "VLLM_TPU_USING_PATHWAYS":
    lambda: bool("proxy" in os.getenv("JAX_PLATFORMS", "").lower()),

    # Allow use of DeepGemm kernels for fused moe ops.
    "VLLM_USE_DEEP_GEMM":
    lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "1"))),

    # Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs.
    "VLLM_USE_DEEP_GEMM_E8M0":
    lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM_E8M0", "1"))),
    # TODO(wentao): unify the two E8M0 flags after verifying the correctness.
    # Whether to use E8M0 scaling when DeepGEMM is used on Hopper GPUs.
    "VLLM_USE_DEEP_GEMM_E8M0_HOPPER":
    lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM_E8M0_HOPPER", "0"))),
    # DeepGemm JITs the kernels on-demand. The warmup attempts to make DeepGemm
    # JIT all the required kernels before model execution so there is no
    # JIT'ing in the hot-path. However, this warmup increases the engine
    # startup time by a couple of minutes.
    # Set `VLLM_SKIP_DEEP_GEMM_WARMUP` to disable the warmup.
    "VLLM_SKIP_DEEP_GEMM_WARMUP":
    lambda: bool(int(os.getenv("VLLM_SKIP_DEEP_GEMM_WARMUP", "0"))),

    # Whether to use fused grouped_topk used for MoE expert selection.
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK":
    lambda: bool(int(os.getenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1"))),

    # Allow use of FlashInfer MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP8":
    lambda: bool(int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP8", "0"))),

    # Allow use of FlashInfer CUTLASS kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP4":
    lambda: bool(int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP4", "0"))),

    # If set to 1, use the FlashInfer
    # MXFP8 (activation) x MXFP4 (weight) MoE backend.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8":
    lambda: bool(int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "0"))),

    # If set to 1, use the FlashInfer CUTLASS backend for
    # MXFP8 (activation) x MXFP4 (weight) MoE.
    # This is separate from the TRTLLMGEN path controlled by
    # VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS":
    lambda: bool(int(
        os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS", "0")
        )),

    # If set to 1, use the FlashInfer
    # BF16 (activation) x MXFP4 (weight) MoE backend.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16":
    lambda: bool(int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16", "0"))),

    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "VLLM_XGRAMMAR_CACHE_MB":
    lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),

    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded.
    "VLLM_MSGPACK_ZERO_COPY_THRESHOLD":
    lambda: int(os.getenv("VLLM_MSGPACK_ZERO_COPY_THRESHOLD", "256")),

    # If set, allow insecure serialization using pickle.
    # This is useful for environments where it is deemed safe to use the
    # insecure method and it is needed for some reason.
    "VLLM_ALLOW_INSECURE_SERIALIZATION":
    lambda: bool(int(os.getenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0"))),

    # IP address used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_HOST":
    lambda: os.getenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"),

    # Port used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_PORT":
    lambda: int(os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5557")),

    # all2all backend for vllm's expert parallel communication
    # Available options:
    # - "naive": naive all2all implementation using broadcasts
    # - "allgather_reducescatter": all2all implementation based on allgather and
    #  reducescatter
    # - "pplx": use pplx kernels
    # - "deepep_high_throughput", use deepep high-throughput kernels
    # - "deepep_low_latency", use deepep low-latency kernels
    "VLLM_ALL2ALL_BACKEND":
    env_with_choices("VLLM_ALL2ALL_BACKEND", "allgather_reducescatter",
                     ["naive", "pplx",
                     "deepep_high_throughput",
                     "deepep_low_latency",
                     "allgather_reducescatter"]),

    # Flashinfer MoE backend for vLLM's fused Mixture-of-Experts support.
    # Both require compute capability 10.0 or above.
    # Available options:
    # - "throughput":  [default]
    #     Uses CUTLASS kernels optimized for high-throughput batch inference.
    # - "latency":
    #     Uses TensorRT-LLM kernels optimized for low-latency inference.
    "VLLM_FLASHINFER_MOE_BACKEND":
    env_with_choices("VLLM_FLASHINFER_MOE_BACKEND", "throughput",
    ["throughput", "latency"]),

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE CUTLASS Kernel. This value is used to create a buffer for
    # the blockscale tensor of activations NVFP4 Quantization.
    # This is used to prevent the kernel from running out of memory.
    "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE":
    lambda: int(os.getenv("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", "163840")),

    # Specifies the thresholds of the communicated tensor sizes under which
    # vllm should use flashinfer fused allreduce. The variable should be a
    # JSON with the following format:
    #     { <world size>: <max size in mb> }
    # Unspecified world sizes will fall back to
    #     { 2: 64, 4: 1, <everything else>: 0.5 }
    "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB":
    lambda: json.loads(os.getenv(
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB", "{}")),

    # MoE routing strategy selector.
    # See `RoutingSimulator.get_available_strategies()` # for available
    # strategies.
    # Cutstom routing strategies can be registered by
    # RoutingSimulator.register_strategy()
    # Note: custom strategies may not produce correct model outputs
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY":
    lambda: os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "").lower(),

    # Regex timeout for use by the vLLM tool parsing plugins.
    "VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS":
    lambda: int(os.getenv("VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")),

    # Reduce CPU usage when vLLM is idle. Enabling this will incur small
    # latency penalty when a request eventually comes.
    "VLLM_SLEEP_WHEN_IDLE":
    lambda: bool(int(os.getenv("VLLM_SLEEP_WHEN_IDLE", "0"))),

    # Control the max chunk bytes (in MB) for the rpc message queue.
    # Object larger than this threshold will be broadcast to worker
    # processes via zmq.
    "VLLM_MQ_MAX_CHUNK_BYTES_MB":
    lambda: int(os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "16")),

    # Timeout in seconds for execute_model RPC calls in multiprocessing
    # executor (only applies when TP > 1).
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS":
    lambda: int(os.getenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")),

    # KV Cache layout used throughout vllm.
    # Some common values are:
    # - NHD
    # - HND
    # Where N=num_blocks, H=num_heads and D=head_size. The default value will
    # leave the layout choice to the backend. Mind that backends may only
    # implement and support a subset of all possible layouts.
    "VLLM_KV_CACHE_LAYOUT":
    env_with_choices("VLLM_KV_CACHE_LAYOUT", None, ["NHD", "HND"]),

    # Enable checking whether the generated logits contain NaNs,
    # indicating corrupted output. Useful for debugging low level bugs
    # or bad hardware but it may add compute overhead.
    "VLLM_COMPUTE_NANS_IN_LOGITS":
    lambda: bool(int(os.getenv("VLLM_COMPUTE_NANS_IN_LOGITS", "0"))),

    # Controls whether or not emulations are used for NVFP4
    # generations on machines < 100 for compressed-tensors
    # models
    "VLLM_USE_NVFP4_CT_EMULATIONS":
    lambda: bool(int(os.getenv("VLLM_USE_NVFP4_CT_EMULATIONS", "0"))),

    # Time (in seconds) after which the KV cache on the producer side is
    # automatically cleared if no READ notification is received from the
    # consumer. This is only applicable when using NixlConnector in a
    # disaggregated decode-prefill setup.
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT":
    lambda: int(os.getenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "120")),

    # Controls whether or not to use cudnn prefill
    "VLLM_USE_CUDNN_PREFILL":
    lambda: bool(int(os.getenv("VLLM_USE_CUDNN_PREFILL", "0"))),

    # If set to 1/True, use the TRTLLM attention backend in flashinfer.
    # If set to 0/False, use the default attention backend in flashinfer.
    # If not set, auto-detect the attention backend in flashinfer.
    "VLLM_USE_TRTLLM_ATTENTION":
    lambda: (None if "VLLM_USE_TRTLLM_ATTENTION" not in os.environ else
             os.environ["VLLM_USE_TRTLLM_ATTENTION"].lower() in ("1", "true")),

    # If set to 1, when we use fp8 kv, we do not quantize Q to fp8
    "VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION":
    lambda: bool(int(os.getenv("VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION", "0"))),

    # If set, it means we pre-downloaded cubin files and flashinfer will
    # read the cubin files directly.
    "VLLM_HAS_FLASHINFER_CUBIN":
    lambda: os.getenv("VLLM_HAS_FLASHINFER_CUBIN", False),

    # If set to 1, force the use of TRTLLM FP4 GEMM backend in flashinfer.
    # Otherwise, uses the first available of: flashinfer cutlass GEMM,
    # vllm cutlass GEMM, marlin GEMM.
    "VLLM_USE_TRTLLM_FP4_GEMM":
    lambda: bool(int(os.getenv("VLLM_USE_TRTLLM_FP4_GEMM", "0"))),

    # Controls garbage collection during CUDA graph capture.
    # If set to 0 (default), enables GC freezing to speed up capture time.
    # If set to 1, allows GC to run during capture.
    "VLLM_ENABLE_CUDAGRAPH_GC":
    lambda: bool(int(os.getenv("VLLM_ENABLE_CUDAGRAPH_GC", "0"))),

    # Disable padding to CUDA graph capture batch sizes.
    # TODO(wentao): https://github.com/vllm-project/vllm/issues/23378
    # After the issue is fixed, we can remove this flag.
    "VLLM_DISABLE_PAD_FOR_CUDAGRAPH":
    lambda: bool(int(os.getenv("VLLM_DISABLE_PAD_FOR_CUDAGRAPH", "0"))),

    # Used to force set up loopback IP
    "VLLM_LOOPBACK_IP":
    lambda: os.getenv("VLLM_LOOPBACK_IP", ""),

    # Used to set the process name prefix for vLLM processes.
    # This is useful for debugging and monitoring purposes.
    # The default value is "VLLM".
    "VLLM_PROCESS_NAME_PREFIX":
    lambda: os.getenv("VLLM_PROCESS_NAME_PREFIX", "VLLM"),

    # Allow chunked local attention with hybrid kv cache manager.
    # Currently using the Hybrid KV cache manager with chunked local attention
    # in the Llama4 models (the only models currently using chunked local attn)
    # causes a latency regression. For this reason, we disable it by default.
    # This flag is used to allow users to enable it if they want to (to save on
    # kv-cache memory usage and enable longer contexts)
    # TODO(lucas): Remove this flag once latency regression is resolved.
    "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE":
    lambda: bool(int(os.getenv(\
            "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "0"))),

    # Enables support for the "store" option in the OpenAI Responses API.
    # When set to 1, vLLM's OpenAI server will retain the input and output
    # messages for those requests in memory. By default, this is disabled (0),
    # and the "store" option is ignored.
    # NOTE/WARNING:
    # 1. Messages are kept in memory only (not persisted to disk) and will be
    #    lost when the vLLM server shuts down.
    # 2. Enabling this option will cause a memory leak, as stored messages are
    #    never removed from memory until the server terminates.
    "VLLM_ENABLE_RESPONSES_API_STORE":
    lambda: bool(int(os.getenv("VLLM_ENABLE_RESPONSES_API_STORE", "0"))),

    # If set, use the fp8 mfma in rocm paged attention.
    "VLLM_ROCM_FP8_MFMA_PAGE_ATTN":
    lambda: bool(int(os.getenv("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "0"))),

    # Whether to use pytorch symmetric memory for allreduce
    "VLLM_ALLREDUCE_USE_SYMM_MEM":
    lambda: bool(int(os.getenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "0"))),

    # Allows vllm to find tuned config under customized folder
    "VLLM_TUNED_CONFIG_FOLDER":
    lambda: os.getenv("VLLM_TUNED_CONFIG_FOLDER", None),

    # Allows vllm use container tool
    "VLLM_GPT_OSS_USE_CONTAINER_TOOL":
    lambda: bool(int(os.getenv("VLLM_GPT_OSS_USE_CONTAINER_TOOL", "0"))),

    # Allows harmony instructions to be injected on system messages
    "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS":
    lambda: bool(
        int(os.getenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "0"))),

    # Add optional custom scopes for profiling, disable to avoid overheads
    "VLLM_CUSTOM_SCOPES_FOR_PROFILING":
    lambda: bool(int(os.getenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "0"))),

    # Represent block hashes in KV cache events as 64-bit integers instead of
    # raw bytes. Defaults to True for backward compatibility.
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES":
    lambda: bool(int(os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1"))),

    # Name of the shared memory buffer used for object storage.
    # Only effective when mm_config.mm_processor_cache_type == "shm".
    "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME":
    lambda: os.getenv("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
                      "VLLM_OBJECT_STORAGE_SHM_BUFFER"),
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_vllm_use_v1(use_v1: bool):
    if is_set("VLLM_USE_V1"):
        raise ValueError(
            "Should not call set_vllm_use_v1() if VLLM_USE_V1 is set "
            "explicitly by the user. Please raise this as a Github "
            "Issue and explicitly set VLLM_USE_V1=0 or 1.")
    os.environ["VLLM_USE_V1"] = "1" if use_v1 else "0"


def compute_hash() -> str:
    """
    WARNING: Whenever a new key is added to this environment
    variables, ensure that it is included in the factors list if
    it affects the computation graph. For example, different values
    of VLLM_PP_LAYER_PARTITION will generate different computation
    graphs, so it is included in the factors list. The env vars that
    affect the choice of different kernels or attention backends should
    also be included in the factors list.
    """

    # The values of envs may affects the computation graph.
    # TODO(DefTruth): hash all environment variables?
    # for key in environment_variables:
    #     factorize(key)
    environment_variables_to_hash = [
        "VLLM_PP_LAYER_PARTITION",
        "VLLM_MLA_DISABLE",
        "VLLM_USE_TRITON_FLASH_ATTN",
        "VLLM_USE_TRITON_AWQ",
        "VLLM_DP_RANK",
        "VLLM_DP_SIZE",
        "VLLM_USE_STANDALONE_COMPILE",
        "VLLM_FUSED_MOE_CHUNK_SIZE",
        "VLLM_FLASHINFER_MOE_BACKEND",
        "VLLM_V1_USE_PREFILL_DECODE_ATTENTION",
        "VLLM_USE_AITER_UNIFIED_ATTENTION",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_DISABLED_KERNELS",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_USE_DEEP_GEMM_E8M0",
        "VLLM_USE_DEEP_GEMM_E8M0_HOPPER",
        "VLLM_USE_TRTLLM_FP4_GEMM",
        "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
        "VLLM_USE_FLASHINFER_MOE_FP8",
        "VLLM_USE_FLASHINFER_MOE_FP4",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
        "VLLM_USE_CUDNN_PREFILL",
        "VLLM_USE_TRTLLM_ATTENTION",
        "VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION",
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_PAGED_ATTN",
        "VLLM_ROCM_USE_AITER_LINEAR",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_RMSNORM",
        "VLLM_ROCM_USE_AITER_MLA",
        "VLLM_ROCM_USE_AITER_MHA",
        "VLLM_ROCM_USE_AITER_FP8BMM",
        "VLLM_ROCM_USE_SKINNY_GEMM",
        "VLLM_ROCM_FP8_PADDING",
        "VLLM_ROCM_MOE_PADDING",
        "VLLM_ROCM_CUSTOM_PAGED_ATTN",
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
        "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16",
        "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
        "VLLM_ROCM_FP8_MFMA_PAGE_ATTN",
    ]
    for key in environment_variables_to_hash:
        # if this goes out of sync with environment_variables,
        # it's not a user error, it's a bug
        assert key in environment_variables, \
            "Please update environment_variables_to_hash in envs.py"

    factors = [
        environment_variables[key]() for key in environment_variables_to_hash
    ]

    hash_str = hashlib.md5(str(factors).encode(),
                           usedforsecurity=False).hexdigest()

    return hash_str
