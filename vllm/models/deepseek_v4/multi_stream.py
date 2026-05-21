# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-stream helpers for DeepSeek-V4 CSA attention overlap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata


def create_dsv4_aux_stream_list() -> list[torch.cuda.Stream] | None:
    """Create aux streams for DeepSeek-V4 attention overlap.

    CUDA always gets three aux streams. ROCm is opt-in via
    ``VLLM_DSV4_ROCM_MULTI_STREAM`` because multi-stream was previously
    disabled due to hang issues (#41820).
    """
    if current_platform.is_rocm():
        if not envs.VLLM_DSV4_ROCM_MULTI_STREAM:
            return None
    elif not current_platform.is_cuda():
        return None
    return [torch.cuda.Stream() for _ in range(3)]


def _has_decode_tokens(
    attn_metadata: dict[str, AttentionMetadata] | list | None,
    swa_cache_prefix: str,
) -> bool:
    if not isinstance(attn_metadata, dict):
        return False
    swa_metadata = attn_metadata.get(swa_cache_prefix)
    if swa_metadata is None:
        return False
    return swa_metadata.num_decodes > 0


def should_overlap_dsv4_input_gemms(
    num_tokens: int,
    aux_stream_list: list[torch.cuda.Stream] | None,
    attn_metadata: dict[str, AttentionMetadata] | list | None,
    swa_cache_prefix: str,
) -> bool:
    """Whether to overlap fused_wqa_wkv with auxiliary input GEMMs."""
    if aux_stream_list is None:
        return False
    threshold = envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
    if threshold <= 0 or num_tokens > threshold:
        return False
    if current_platform.is_rocm() and envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY:
        return _has_decode_tokens(attn_metadata, swa_cache_prefix)
    return True


def should_overlap_dsv4_indexer(
    aux_stream_list: list[torch.cuda.Stream] | None,
    attn_metadata: dict[str, AttentionMetadata] | list | None,
    swa_cache_prefix: str,
) -> bool:
    """Whether to overlap the indexer with the main attention prep path."""
    if aux_stream_list is None:
        return False
    if current_platform.is_rocm() and envs.VLLM_DSV4_ROCM_MULTI_STREAM_DECODE_ONLY:
        return _has_decode_tokens(attn_metadata, swa_cache_prefix)
    return True
