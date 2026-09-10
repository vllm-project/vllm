# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable Triton sparse MLA path."""

import os

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

# Ampere (SM80/SM86) and Ada Lovelace (SM89) reuse the SM12x portable Triton
# DeepSeek-V4 path on this branch: they all lack the SM90/SM100-only FlashMLA +
# DeepGEMM kernels, so attention / indexer / einsum / MHC must run the same
# Triton fallbacks as SM12x. Ada has FP8 tensor cores; Ampere (8.0/8.6) does
# not, so the FP8 Triton kernels upcast to bf16 before the MMA (gated on
# supports_fp8()). NOTE: the MoE FP4 expert GEMM is NOT covered here (no SM 8.x
# has FP4 tensor cores); it falls back to the Marlin WNA16 backend.
_SM8X_FAMILY = 80


def is_ampere_or_ada() -> bool:
    """True iff the current CUDA device is SM 8.x (Ampere 8.0/8.6 or Ada 8.9)."""
    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        _SM8X_FAMILY
    )


def _device_capability_tuple(device: torch.device):
    if not current_platform.is_cuda():
        return None
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    return current_platform.get_device_capability(device_id=index)


def _is_ampere_or_ada_device(device: torch.device) -> bool:
    capability = _device_capability_tuple(device)
    return capability is not None and capability[0] == 8


def _is_sm12x_device(device: torch.device) -> bool:
    capability = _device_capability_tuple(device)
    return capability is not None and capability[0] == 12


def _is_triton_fallback_device(device: torch.device) -> bool:
    """SM12x (Blackwell client) or SM 8.x (Ampere/Ada): the portable Triton path."""
    return _is_sm12x_device(device) or _is_ampere_or_ada_device(device)


def triton_sparse_mla_configured() -> bool | None:
    return envs.VLLM_TRITON_MLA_SPARSE


def is_triton_sparse_mla_enabled_for_platform() -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120) or is_ampere_or_ada()


def is_triton_sparse_mla_enabled(device: torch.device) -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return _is_triton_fallback_device(device)


def triton_sparse_mla_topk_chunk_size() -> int:
    return envs.VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE


def triton_sparse_mla_prefill_topk_chunk_size(
    *,
    combined_topk_size: int,
    compress_ratio: int,
    request_count: int,
) -> int:
    """Choose the Triton sparse MLA prefill topk chunk size.

    Keep explicit user overrides authoritative. The auto path uses a larger
    chunk for SM12x C128A single-request prefill to reduce per-request loop
    overhead, but keeps a smaller chunk for the multi-request shape that is
    unstable near 128K context.
    """

    configured_topk = triton_sparse_mla_topk_chunk_size()
    if os.getenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE") is not None:
        return min(combined_topk_size, configured_topk)
    if (
        current_platform.is_device_capability_family(120) or is_ampere_or_ada()
    ) and compress_ratio == 128:
        if request_count > 1 and combined_topk_size > 1024:
            configured_topk = min(configured_topk, 256)
        elif request_count == 1 and combined_topk_size > 1024:
            configured_topk = max(configured_topk, 1024)
    return min(combined_topk_size, configured_topk)


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
    return current_platform.is_device_capability_family(120) or is_ampere_or_ada()
