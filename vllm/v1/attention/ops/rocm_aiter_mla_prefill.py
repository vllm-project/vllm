# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed DCP context preparation for ROCm MLA."""

from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata


def context_row_indices(
    chunk: MLACommonPrefillMetadata.ContextChunk, device: torch.device
) -> torch.Tensor:
    """Map compact request-major rows to the padded rank-major AllGather."""
    assert chunk.padded_local_seq_lens is not None
    assert chunk.local_context_lens_allranks is not None
    assert chunk.local_starts is not None
    segments = []
    offset = 0
    for padded, lengths, start in zip(
        chunk.padded_local_seq_lens,
        chunk.local_context_lens_allranks,
        chunk.local_starts,
    ):
        for rank, length in enumerate(lengths):
            count = min(max(0, length - start), padded)
            if count:
                first = rank * chunk.num_local_context_tokens + offset
                segments.append(torch.arange(first, first + count, dtype=torch.int32))
        offset += padded
    rows = torch.cat(segments) if segments else torch.empty(0, dtype=torch.int32)
    assert rows.numel() == chunk.num_context_tokens
    return rows.to(device=device, non_blocking=True)


def gather_compressed_context(
    cache: torch.Tensor,
    workspace: torch.Tensor,
    block_table: torch.Tensor,
    chunk: MLACommonPrefillMetadata.ContextChunk,
    gather: Callable[[torch.Tensor, torch.Tensor], object],
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    """Gather FP8 bytes, preserving the existing workspace row capacity."""
    assert chunk.local_context_lens_allranks is not None
    assert workspace.is_contiguous()
    rows, width = workspace.shape
    fp8_workspace = workspace.view(cache_dtype).view(-1, width)[:rows]
    local_rows = chunk.num_local_context_tokens
    world_size = len(chunk.local_context_lens_allranks[0])
    offset = rows // (world_size + 1)
    assert offset * (world_size + 1) == rows
    assert local_rows <= offset
    local = fp8_workspace[:local_rows]
    gathered = fp8_workspace[offset : offset + world_size * local_rows]
    assert chunk.padded_local_cu_seq_lens is not None
    ops.cp_gather_cache(
        cache.view(torch.uint8),
        local.view(torch.uint8),
        block_table,
        chunk.padded_local_cu_seq_lens,
        chunk.num_requests,
        chunk.starts,
    )
    gather(gathered.view(torch.uint8), local.view(torch.uint8))
    return gathered


def expand_context(
    gathered: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse row reorganization, dequantization, projection and K/V packing."""
    k = torch.empty(
        (row_indices.numel(), num_heads, qk_nope_head_dim + qk_rope_head_dim),
        device=gathered.device,
        dtype=weight.dtype,
    )
    v = torch.empty(
        (row_indices.numel(), num_heads, v_head_dim),
        device=gathered.device,
        dtype=weight.dtype,
    )
    if row_indices.numel():
        rocm_aiter_ops.gather_kv_b_proj(
            gathered.unsqueeze(1),
            scale,
            cu_seq_lens,
            row_indices,
            cu_seq_lens,
            weight,
            k,
            v,
        )
    return k, v
