# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm paged attention with Inkling's query-dependent relative bias.

The NVIDIA implementation uses the score-mod hook in tml-fa4.  ROCm Flash
Attention and AITER do not expose an equivalent hook, so this module implements
the same operation directly in Triton.  Query heads belonging to one KV head
are processed together and KV pages are gathered through vLLM's block table.
"""

from __future__ import annotations

import os
from typing import cast

import torch

from vllm.models.inkling.amd.ops.rel_attention_decode import (
    inkling_rel_attention_split_kv_decode,
    use_split_kv_decode,
)
from vllm.models.inkling.common.ops.triton_rel_attention import (
    inkling_triton_rel_attention,
)
from vllm.platforms.rocm import on_gfx950


def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    """Round the scheduling bound up to a power of two."""
    return 1 << max(0, max_seqlen_q - 1).bit_length()


def use_gfx950_gluon_decode(
    *, max_query_len: int, page_size: int, head_dim: int
) -> bool:
    """Use the vendored TokenSpeed CDNA4 decode kernel where it is supported."""
    return (
        os.getenv("INKLING_GFX950_GLUON", "1") == "1"
        and on_gfx950()
        and max_query_len == 1
        and page_size in (64, 128, 256)
        and head_dim in (64, 128)
    )


def use_gfx950_gluon_extend(
    *,
    max_query_len: int,
    max_kv_len: int,
    page_size: int,
    head_dim: int,
    window_left: int,
) -> bool:
    """Use Gluon only for the long full-attention extend regime it wins."""
    return (
        os.getenv("INKLING_GFX950_GLUON", "1") == "1"
        and on_gfx950()
        and max_query_len > 1
        and max_kv_len >= 8192
        and window_left < 0
        and page_size in (64, 128, 256)
        and head_dim in (64, 128)
    )


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Keep the NVIDIA-facing split heuristic as API-compatible metadata.

    The ROCm Triton implementation performs online softmax in one program and
    does not consume the result.  Keeping this function unchanged avoids
    platform-specific scheduling branches in :mod:`inkling.amd.attention`.
    """
    if is_local:
        return 1

    q_rows = max_query_len * (num_heads // num_kv_heads)
    q_tiles = (q_rows + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    target_ctas = (
        256 if q_tiles == 1 and batch_size == 1 else (128 if q_tiles == 1 else 64)
    )
    max_splits = 128
    if q_tiles == 1 and batch_size == 1:
        if num_kv_heads == 8:
            max_splits = 16
        elif num_kv_heads == 4 or max_kv_len <= 8192:
            max_splits = 32
        elif max_kv_len <= 65536:
            max_splits = 64
    return max(
        1,
        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


@torch.no_grad()
def inkling_fa4_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 32,
    max_kv_len: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged varlen attention with Inkling's relative score modification."""
    del num_splits
    if not causal or window_size[1] not in (0, -1):
        raise NotImplementedError("Inkling ROCm attention requires causal masking")
    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected q [T,H,D] and paged K/V [B,P,Hkv,D]")
    if rel_logits.shape != (q.shape[0], q.shape[1], rel_extent):
        raise ValueError(
            f"relative logits have shape {tuple(rel_logits.shape)}, expected "
            f"{(q.shape[0], q.shape[1], rel_extent)}"
        )

    num_kv_heads = key_cache.shape[2]
    if q.shape[1] % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if out is None:
        out = torch.empty_like(q)
    if max_kv_len is None:
        max_kv_len = key_cache.shape[0] * key_cache.shape[1]
    if use_gfx950_gluon_decode(
        max_query_len=max_seqlen_q,
        page_size=key_cache.shape[1],
        head_dim=q.shape[2],
    ):
        # Import only after the architecture guard. The implementation uses
        # gfx950-only CDNA4 Gluon layouts and async-copy operations.
        from vllm.models.inkling.amd.ops.gluon.rel_mha_decode_gfx950 import (
            gluon_rel_mha_decode_gfx950,
        )

        return gluon_rel_mha_decode_gfx950(
            q,
            key_cache,
            value_cache,
            block_table,
            cache_seqlens,
            max_kv_len,
            rel_logits,
            cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            window_left=window_size[0],
            softmax_scale=softmax_scale,
            out=out,
        )
    if use_gfx950_gluon_extend(
        max_query_len=max_seqlen_q,
        max_kv_len=max_kv_len,
        page_size=key_cache.shape[1],
        head_dim=q.shape[2],
        window_left=window_size[0],
    ):
        from vllm.models.inkling.amd.ops.gluon.rel_mha_extend_gfx950 import (
            gluon_rel_mha_extend_gfx950,
        )

        # The copied kernel retains TokenSpeed's cu_seqlens_kv parameter for
        # API compatibility but does not consume it.
        return cast(
            torch.Tensor,
            gluon_rel_mha_extend_gfx950(
                q,
                cu_seqlens_q,
                cu_seqlens_q,
                key_cache,
                value_cache,
                block_table,
                cache_seqlens,
                window_left=window_size[0],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_kv_len,
                rel_logits=rel_logits,
                softmax_scale=softmax_scale,
                out=out,
            ),
        )
    if use_split_kv_decode(
        max_query_len=max_seqlen_q,
        max_kv_len=max_kv_len,
        page_size=key_cache.shape[1],
        window_left=window_size[0],
    ):
        return inkling_rel_attention_split_kv_decode(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            window_left=window_size[0],
            rel_extent=rel_extent,
            rel_logits=rel_logits,
            max_kv_len=max_kv_len,
            out=out,
        )
    return inkling_triton_rel_attention(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        max_kv_len=max_kv_len,
        out=out,
    )
