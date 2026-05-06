# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform controls for the portable Triton sparse MLA path."""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE = 512
_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE = 256

logger = init_logger(__name__)


def _is_sm12x_device(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)[0] == 12


def is_triton_sparse_mla_enabled_for_platform() -> bool:
    return current_platform.is_device_capability_family(120)


def is_triton_sparse_mla_enabled(device: torch.device) -> bool:
    return _is_sm12x_device(device)


def disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config) -> None:
    if not is_triton_sparse_mla_enabled_for_platform():
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
        "compile/graph-safe yet."
    )
    compilation_config.mode = CompilationMode.NONE
    compilation_config.compile_sizes = []
    compilation_config.compile_ranges_endpoints = []
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    compilation_config.cudagraph_capture_sizes = []
    compilation_config.max_cudagraph_capture_size = 0


def triton_sparse_mla_topk_chunk_size() -> int:
    return _TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE


def triton_sparse_mla_query_chunk_size() -> int:
    return _TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE
