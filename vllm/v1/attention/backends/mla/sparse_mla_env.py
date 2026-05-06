# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable Triton sparse MLA path."""

import torch

import vllm.envs as envs
from vllm.platforms import current_platform


def _is_sm12x_device(device: torch.device) -> bool:
    if not current_platform.is_cuda():
        return False
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    capability = current_platform.get_device_capability(device_id=index)
    return capability is not None and capability[0] == 12


def triton_sparse_mla_configured() -> bool | None:
    return envs.VLLM_TRITON_MLA_SPARSE


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


def triton_sparse_mla_topk_chunk_size() -> int:
    return envs.VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE


def triton_sparse_mla_query_chunk_size() -> int:
    return envs.VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE


def triton_sparse_mla_head_block_size() -> int | None:
    value = envs.VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE
    if value in (1, 2, 4):
        return value
    return None


def triton_sparse_mla_matmul_decode_enabled() -> bool:
    configured = envs.VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)


def triton_sparse_mla_splitkv_decode_enabled() -> bool:
    return envs.VLLM_TRITON_MLA_SPARSE_SPLITKV_DECODE
