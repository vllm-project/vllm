# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA compress → norm → RoPE → store dispatch for head=512.

The head=128 indexer path is handled by the shared triton launcher in
``common/ops/fused_compress_quant_cache.py``; only head=512 (cutedsl)
flows through here. Uses fused C4 or split C128 kernels based on
``compress_ratio``.
"""

from typing import Any

import torch

from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
    compress_kv_sparse_attn_cutedsl,
    fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
    norm_rope_insert_sparse_attn_cutedsl,
)


def compress_norm_rope_store(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    if compress_ratio == 4:
        fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            kv_cache,
            k_cache_metadata.slot_mapping,
            kv_cache.shape[1],  # paged KV cache block size
            kv_cache.stride(0),
            head_size=head_dim,
            state_width=state_width,
            rope_head_dim=rope_head_dim,
            fp8_max=448.0,
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=scale_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
    else:
        compressed_kv = torch.empty(
            (num_actual, head_dim),
            dtype=torch.float32,
            device=state_cache.device,
        )
        compress_kv_sparse_attn_cutedsl(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            compressed_kv,
            head_size=head_dim,
            state_width=state_width,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        norm_rope_insert_sparse_attn_cutedsl(
            compressed_kv,
            positions,
            slot_mapping,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            kv_cache,
            k_cache_metadata.slot_mapping,
            kv_cache.shape[1],  # paged KV cache block size
            kv_cache.stride(0),
            head_size=head_dim,
            rope_head_dim=rope_head_dim,
            fp8_max=448.0,
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=scale_dim,
            compress_ratio=compress_ratio,
        )
