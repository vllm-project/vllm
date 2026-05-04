# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable Triton sparse MLA path."""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

_TRITON_MLA_SPARSE_ENV = "VLLM_TRITON_MLA_SPARSE"
_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE"
_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE"
_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV = (
    "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH"
)
_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV = "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE"
_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV = "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE"

_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_FALSE_VALUES = {"0", "false", "no", "off"}

logger = init_logger(__name__)


def _optional_env_flag(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    value = raw_value.lower()
    if value in _ENV_TRUE_VALUES:
        return True
    if value in _ENV_FALSE_VALUES:
        return False
    return None


def _is_sm12x_device(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)[0] == 12


def triton_sparse_mla_configured() -> bool | None:
    return _optional_env_flag(_TRITON_MLA_SPARSE_ENV)


def is_triton_sparse_mla_enabled_for_platform() -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)


def is_triton_sparse_mla_enabled(device: torch.device) -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return _is_sm12x_device(device)


def _uses_speculative_decoding(vllm_config) -> bool:
    return bool(getattr(vllm_config, "speculative_config", None))


def triton_sparse_mla_cudagraphs_allowed(vllm_config=None) -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV)
    if configured is not None:
        return configured
    return not (
        vllm_config is not None and _uses_speculative_decoding(vllm_config)
    )


def disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config) -> None:
    if not is_triton_sparse_mla_enabled_for_platform():
        return
    if triton_sparse_mla_cudagraphs_allowed(vllm_config):
        logger.warning_once(
            "Keeping vLLM compile and CUDA graphs enabled for the DeepSeek V4 "
            "Triton sparse MLA path because "
            f"{_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV}=1 or speculative "
            "decoding is not configured. This is an "
            "experimental performance mode."
        )
        return

    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if (
        compilation_config.mode == CompilationMode.NONE
        and compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    ):
        return

    logger.warning_once(
        "Disabling vLLM compile and CUDA graphs for the DeepSeek V4 Triton "
        "sparse MLA path because the current Triton sparse MLA path is not "
        "compile/graph-safe yet, or because speculative decoding uses "
        "multi-token sparse MLA decode."
    )
    compilation_config.mode = CompilationMode.NONE
    compilation_config.compile_sizes = []
    compilation_config.compile_ranges_endpoints = []
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    compilation_config.cudagraph_capture_sizes = []
    compilation_config.max_cudagraph_capture_size = 0


def triton_sparse_mla_topk_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV)
    if raw_value is None:
        return 512
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 512


def triton_sparse_mla_query_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV)
    if raw_value is None:
        return 256
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 256


def triton_sparse_mla_head_block_size() -> int | None:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    if value in (1, 2, 4):
        return value
    return None


def triton_sparse_mla_matmul_decode_enabled() -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV)
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)
