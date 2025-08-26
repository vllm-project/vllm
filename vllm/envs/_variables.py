# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Environment variable definitions with type annotations and default values.

This module defines all environment variables used by vLLM with their expected
data types and default values. The supported data types are:
- str: String values
- int: Integer values  
- float: Floating point values
- bool: Boolean values (typically parsed from "0"/"1" or "true"/"false")
- Optional[T]: Optional values that can be None
- list[str]: Lists of strings (typically comma-separated)

Each variable is defined with its type annotation and default value.
The actual environment variable lookup and conversion is handled by the
parent module (__init__.py).
"""

import os
import tempfile
from typing import Optional


# Environment variable definitions with type annotations and defaults
# These match the TYPE_CHECKING section from the original envs.py

# Installation Time Environment Variables
VLLM_TARGET_DEVICE: str = "cuda"
MAX_JOBS: Optional[str] = None
NVCC_THREADS: Optional[str] = None
VLLM_USE_PRECOMPILED: bool = False
VLLM_DOCKER_BUILD_CONTEXT: bool = False
VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
CMAKE_BUILD_TYPE: Optional[str] = None
VERBOSE: bool = False

# Configuration and cache paths
VLLM_CONFIG_ROOT: str = os.path.expanduser("~/.config/vllm")
VLLM_CACHE_ROOT: str = os.path.expanduser("~/.cache/vllm")

# Runtime Environment Variables
VLLM_HOST_IP: str = ""
VLLM_PORT: Optional[int] = None
VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
VLLM_USE_MODELSCOPE: bool = False
VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
CUDA_HOME: Optional[str] = None
VLLM_NCCL_SO_PATH: Optional[str] = None
LD_LIBRARY_PATH: Optional[str] = None

# Attention and kernel settings
VLLM_USE_TRITON_FLASH_ATTN: bool = True
VLLM_V1_USE_PREFILL_DECODE_ATTENTION: bool = False
VLLM_USE_AITER_UNIFIED_ATTENTION: bool = False
VLLM_FLASH_ATTN_VERSION: Optional[int] = None
VLLM_ATTENTION_BACKEND: Optional[str] = None
VLLM_USE_FLASHINFER_SAMPLER: Optional[bool] = None

# Testing and debugging
VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE: bool = True
VLLM_USE_STANDALONE_COMPILE: bool = True

# Distributed computing
LOCAL_RANK: int = 0
CUDA_VISIBLE_DEVICES: Optional[str] = None
VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 60

# API and security
VLLM_API_KEY: Optional[str] = None
VLLM_DEBUG_LOG_API_SERVER_RESPONSE: bool = False

# S3 configuration
S3_ACCESS_KEY_ID: Optional[str] = None
S3_SECRET_ACCESS_KEY: Optional[str] = None
S3_ENDPOINT_URL: Optional[str] = None

# Usage statistics
VLLM_USAGE_STATS_SERVER: str = "https://stats.vllm.ai"
VLLM_NO_USAGE_STATS: bool = False
VLLM_DO_NOT_TRACK: bool = False
VLLM_USAGE_SOURCE: str = "production"

# Logging configuration
VLLM_CONFIGURE_LOGGING: int = 1
VLLM_LOGGING_CONFIG_PATH: Optional[str] = None
VLLM_LOGGING_LEVEL: str = "INFO"
VLLM_LOGGING_PREFIX: str = ""
VLLM_LOGITS_PROCESSOR_THREADS: Optional[int] = None
VLLM_LOG_STATS_INTERVAL: float = 10.0
VLLM_TRACE_FUNCTION: int = 0

# Pipeline and partitioning
VLLM_PP_LAYER_PARTITION: Optional[str] = None

# CPU backend settings
VLLM_CPU_KVCACHE_SPACE: Optional[int] = 0
VLLM_CPU_OMP_THREADS_BIND: str = ""
VLLM_CPU_NUM_OF_RESERVED_CPU: Optional[int] = None
VLLM_CPU_MOE_PREPACK: bool = True
VLLM_CPU_SGL_KERNEL: bool = False

# XLA settings
VLLM_XLA_CACHE_PATH: str = os.path.join(os.path.expanduser("~/.cache/vllm"), "xla_cache")
VLLM_XLA_CHECK_RECOMPILATION: bool = False
VLLM_XLA_USE_SPMD: bool = False

# MoE (Mixture of Experts) settings
VLLM_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024
VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING: bool = True

# Ray distributed computing
VLLM_USE_RAY_SPMD_WORKER: bool = False
VLLM_USE_RAY_COMPILED_DAG: bool = False
VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "auto"
VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
VLLM_USE_RAY_WRAPPED_PP_COMM: bool = True
VLLM_WORKER_MULTIPROC_METHOD: str = "fork"

# Multimodal settings
VLLM_ASSETS_CACHE: str = os.path.join(os.path.expanduser("~/.cache/vllm"), "assets")
VLLM_IMAGE_FETCH_TIMEOUT: int = 5
VLLM_VIDEO_FETCH_TIMEOUT: int = 30
VLLM_AUDIO_FETCH_TIMEOUT: int = 10
VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8
VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
VLLM_VIDEO_LOADER_BACKEND: str = "opencv"
VLLM_MM_INPUT_CACHE_GIB: int = 4

# Engine and model settings
VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = False
VLLM_TEST_FORCE_FP8_MARLIN: bool = False
VLLM_TEST_FORCE_LOAD_FORMAT: str = "dummy"

# Network and communication
VLLM_RPC_TIMEOUT: int = 10000  # ms
VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds

# Plugin system
VLLM_PLUGINS: Optional[list[str]] = None
VLLM_LORA_RESOLVER_CACHE_DIR: Optional[str] = None

# Profiling
VLLM_TORCH_PROFILER_DIR: Optional[str] = None
VLLM_TORCH_PROFILER_RECORD_SHAPES: bool = False
VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY: bool = False
VLLM_TORCH_PROFILER_WITH_STACK: bool = True
VLLM_TORCH_PROFILER_WITH_FLOPS: bool = False

# Quantization and kernels
VLLM_USE_TRITON_AWQ: bool = False
VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False
VLLM_SKIP_P2P_CHECK: bool = False
VLLM_DISABLED_KERNELS: list[str] = []

# Version control
VLLM_USE_V1: bool = True

# ROCm specific settings
VLLM_ROCM_USE_AITER: bool = False
VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
VLLM_ROCM_USE_AITER_LINEAR: bool = True
VLLM_ROCM_USE_AITER_MOE: bool = True
VLLM_ROCM_USE_AITER_RMSNORM: bool = True
VLLM_ROCM_USE_AITER_MLA: bool = True
VLLM_ROCM_USE_AITER_MHA: bool = True
VLLM_ROCM_USE_SKINNY_GEMM: bool = True
VLLM_ROCM_FP8_PADDING: bool = True
VLLM_ROCM_MOE_PADDING: bool = True
VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: str = "NONE"
VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: Optional[int] = None

# V1 specific settings
VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
VLLM_LOG_BATCHSIZE_INTERVAL: float = -1
VLLM_DISABLE_COMPILE_CACHE: bool = False

# Scale constants for FP8 KV Cache
Q_SCALE_CONSTANT: int = 200
K_SCALE_CONSTANT: int = 200
V_SCALE_CONSTANT: int = 100

# Development and debugging
VLLM_SERVER_DEV_MODE: bool = False
VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
VLLM_MLA_DISABLE: bool = False

# Ray settings continued
VLLM_RAY_PER_WORKER_GPUS: float = 1.0
VLLM_RAY_BUNDLE_INDICES: str = ""

# CUDA settings
VLLM_CUDART_SO_PATH: Optional[str] = None

# Data parallel settings  
VLLM_DP_RANK: int = 0
VLLM_DP_RANK_LOCAL: int = -1
VLLM_DP_SIZE: int = 1
VLLM_DP_MASTER_IP: str = ""
VLLM_DP_MASTER_PORT: int = 0
VLLM_MOE_DP_CHUNK_SIZE: int = 256
VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool = False

# CI and testing
VLLM_CI_USE_S3: bool = False

# Model redirection and quantization
VLLM_MODEL_REDIRECT_PATH: Optional[str] = None
VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
VLLM_MXFP4_USE_MARLIN: Optional[bool] = None

# Cache settings
VLLM_V0_USE_OUTLINES_CACHE: bool = False
VLLM_V1_USE_OUTLINES_CACHE: bool = False

# TPU settings
VLLM_TPU_BUCKET_PADDING_GAP: int = 0
VLLM_TPU_MOST_MODEL_LEN: Optional[int] = None
VLLM_TPU_USING_PATHWAYS: bool = False

# DeepGemm settings
VLLM_USE_DEEP_GEMM: bool = False
VLLM_USE_DEEP_GEMM_E8M0: bool = True
VLLM_SKIP_DEEP_GEMM_WARMUP: bool = False

# FlashInfer settings
VLLM_USE_FUSED_MOE_GROUPED_TOPK: bool = True
VLLM_USE_FLASHINFER_MOE_FP8: bool = False
VLLM_USE_FLASHINFER_MOE_FP4: bool = False
VLLM_FLASHINFER_MOE_BACKEND: str = "throughput"
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: bool = False
VLLM_USE_FLASHINFER_MOE_MXFP4_BF16: bool = False

# Additional settings
VLLM_XGRAMMAR_CACHE_MB: int = 0
VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False

# NIXL settings
VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5557
VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int = 120

# Communication backends
VLLM_ALL2ALL_BACKEND: str = "naive"

# Expert parallel settings
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
VLLM_MOE_ROUTING_SIMULATION_STRATEGY: str = ""

# Tool and timeout settings
VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
VLLM_SLEEP_WHEN_IDLE: bool = False
VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300

# Cache and memory settings
VLLM_KV_CACHE_LAYOUT: Optional[str] = None
VLLM_COMPUTE_NANS_IN_LOGITS: bool = False
VLLM_USE_NVFP4_CT_EMULATIONS: bool = False

# CUDA specific settings
VLLM_USE_CUDNN_PREFILL: bool = False
VLLM_USE_TRTLLM_ATTENTION: Optional[str] = None
VLLM_HAS_FLASHINFER_CUBIN: bool = False
VLLM_USE_TRTLLM_FP4_GEMM: bool = False
VLLM_ENABLE_CUDAGRAPH_GC: bool = False

# Network settings
VLLM_LOOPBACK_IP: str = ""
VLLM_PROCESS_NAME_PREFIX: str = "VLLM"

# Attention and cache management
VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = False
VLLM_ENABLE_RESPONSES_API_STORE: bool = False
VLLM_ALLREDUCE_USE_SYMM_MEM: bool = False

# Configuration folder
VLLM_TUNED_CONFIG_FOLDER: Optional[str] = None