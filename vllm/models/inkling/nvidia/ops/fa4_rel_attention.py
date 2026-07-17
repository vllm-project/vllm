# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    """Round the FA4 scheduling bound up to a power of two."""
    return 1 << max(0, max_seqlen_q - 1).bit_length()


@cache
def _use_sheared_bias() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (10, 11)


@cache
def _get_score_mod(rel_extent: int) -> Callable:
    """Return the score modification that adds Inkling relative bias."""
    import cutlass.cute as cute
    from cutlass.cute import Float32

    from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        rel_dist_0 = rel_dist[0]
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        rel_bias = rel_logits[global_q_idx[0], h_idx[0], rel_idx]
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
        return scores + rel_bias

    return score_mod_rel_bias


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Return the split-KV cap for Inkling relative attention."""
    capability = current_platform.get_device_capability()
    if capability is not None and capability.major == 9:
        return 1
    if is_local:
        return 1

    q_rows = max_query_len * (num_heads // num_kv_heads)
    q_tiles = (q_rows + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    # Shearing makes split/combine overhead more visible. Multi-tile causal
    # prefill saturates around 64 CTAs. Batch-1 decode at very long context is
    # memory-bound and uses a TP-specific cap measured through 1M KV tokens.
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
        else:
            max_splits = 128
    return max(
        1,
        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged varlen FA4 over the bound K/V cache with the Inkling relative bias.

    ``q`` is ``(num_tokens, num_heads, head_dim)``; ``key_cache`` / ``value_cache``
    are the paged caches ``(num_blocks, block_size, num_kv_heads, head_dim)``;
    ``block_table`` is the per-request page table and ``cache_seqlens`` the
    per-request KV lengths (``seqused_k``). ``rel_logits`` is
    ``(num_tokens, num_heads, rel_extent)``.

    Hopper uses standard FA4's score-mod gather. Blackwell uses tml-fa4's
    sheared relative-bias layout.
    """
    # cute uses (None, None) to mean "no window".
    cute_window = (None, None) if window_size == (-1, -1) else window_size

    rel_logits = rel_logits.contiguous()
    flash_attn_impl: Callable[..., Any]
    if _use_sheared_bias():
        from vllm.third_party.tml_fa4 import (
            flash_attn_varlen_func as tml_flash_attn_varlen_func,
        )

        flash_attn_impl = tml_flash_attn_varlen_func
        bias_kwargs: dict[str, Any] = {"rel_bias": rel_logits}
    else:
        from vllm.vllm_flash_attn.cute import (
            flash_attn_varlen_func as cute_flash_attn_varlen_func,
        )

        flash_attn_impl = cute_flash_attn_varlen_func
        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }

    ret = flash_attn_impl(
        q=q,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=cute_window,
        num_splits=num_splits,
        return_lse=False,
        out=out,
        **bias_kwargs,
    )
    if isinstance(ret, tuple):
        return ret[0]
    return ret


@triton.jit
def _inkling_decode_rel_attention_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    rel_logits_ptr,
    out_ptr,
    softmax_scale,
    stride_q_token: tl.int64,
    stride_q_head: tl.int64,
    stride_q_dim: tl.int64,
    stride_k_block: tl.int64,
    stride_k_token: tl.int64,
    stride_k_head: tl.int64,
    stride_k_dim: tl.int64,
    stride_v_block: tl.int64,
    stride_v_token: tl.int64,
    stride_v_head: tl.int64,
    stride_v_dim: tl.int64,
    stride_block_table_seq: tl.int64,
    stride_rel_token: tl.int64,
    stride_rel_head: tl.int64,
    stride_rel_distance: tl.int64,
    stride_out_token: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.int64,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    query_idx = tl.load(cu_seqlens_q_ptr + seq_idx)
    query_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    if query_idx == query_end:
        return

    kv_len = tl.load(cache_seqlens_ptr + seq_idx)
    if kv_len <= 0:
        return

    kv_group_size: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    kv_head_idx = head_idx // kv_group_size
    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < HEAD_DIM
    q_offsets = (
        query_idx * stride_q_token + head_idx * stride_q_head + offs_d * stride_q_dim
    )
    q = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0)

    key_start = 0
    if WINDOW_LEFT >= 0:
        key_start = tl.maximum(0, kv_len - WINDOW_LEFT - 1)

    running_max = -float("inf")
    running_sum = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for tile_start in range(key_start, kv_len, BLOCK_N):
        key_idx = tile_start + tl.arange(0, BLOCK_N)
        key_mask = key_idx < kv_len
        physical_block = tl.load(
            block_table_ptr + seq_idx * stride_block_table_seq + key_idx // PAGE_SIZE,
            mask=key_mask,
            other=0,
            cache_modifier=".ca",
        ).to(tl.int64)
        page_offset = key_idx % PAGE_SIZE

        key_offsets = (
            physical_block[:, None] * stride_k_block
            + page_offset[:, None] * stride_k_token
            + kv_head_idx * stride_k_head
            + offs_d[None, :] * stride_k_dim
        )
        key = tl.load(
            key_cache_ptr + key_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        scores = tl.sum(q[None, :] * key, axis=1) * softmax_scale

        distance = kv_len - 1 - key_idx
        rel_mask = key_mask & (distance < REL_EXTENT)
        rel_offsets = (
            query_idx * stride_rel_token
            + head_idx * stride_rel_head
            + distance * stride_rel_distance
        )
        relative_bias = tl.load(
            rel_logits_ptr + rel_offsets,
            mask=rel_mask,
            other=0.0,
            cache_modifier=".ca",
        )
        scores += relative_bias
        scores = tl.where(key_mask, scores, -float("inf"))

        value_offsets = (
            physical_block[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head_idx * stride_v_head
            + offs_d[None, :] * stride_v_dim
        )
        value = tl.load(
            value_cache_ptr + value_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )

        tile_max = tl.maximum(tl.max(scores, axis=0), running_max)
        old_scale = tl.exp(running_max - tile_max)
        probabilities = tl.exp(scores - tile_max)
        acc = acc * old_scale + tl.sum(probabilities[:, None] * value, axis=0)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = tile_max

    output_offsets = (
        query_idx * stride_out_token
        + head_idx * stride_out_head
        + offs_d * stride_out_dim
    )
    tl.store(out_ptr + output_offsets, acc / running_sum, mask=dim_mask)


def inkling_triton_decode_rel_attention(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused paged relative attention for a single-token decode step."""
    del num_splits
    if max_seqlen_q != 1:
        raise ValueError("Inkling Triton decode attention requires max_seqlen_q=1")
    if not causal:
        raise NotImplementedError(
            "Inkling Triton decode attention requires causal=True"
        )
    if window_size == (-1, -1):
        window_left = -1
    elif window_size[0] >= 0 and window_size[1] == 0:
        window_left = window_size[0]
    else:
        raise NotImplementedError(
            f"Unsupported Inkling Triton attention window: {window_size}"
        )

    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("Expected q to be 3D and paged K/V caches to be 4D")
    if key_cache.shape != value_cache.shape:
        raise ValueError("Inkling key and value cache shapes must match")
    if key_cache.shape[-1] != q.shape[-1]:
        raise ValueError("Inkling query and KV head dimensions must match")
    if rel_logits.shape[:2] != q.shape[:2] or rel_logits.shape[2] < rel_extent:
        raise ValueError("Inkling relative logits have an incompatible shape")
    if cu_seqlens_q.shape[0] != cache_seqlens.shape[0] + 1:
        raise ValueError("Inkling varlen query and KV metadata do not match")

    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    if num_heads % num_kv_heads != 0:
        raise ValueError("Inkling query heads must be divisible by KV heads")

    result = torch.empty_like(q) if out is None else out
    if result.shape != q.shape:
        raise ValueError("Inkling attention output must have the query shape")
    if q.shape[0] == 0:
        return result

    head_dim = q.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    kv_group_size = num_heads // num_kv_heads
    num_warps = 1 if kv_group_size > 1 else 4
    grid = (cache_seqlens.shape[0], num_heads)
    _inkling_decode_rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        rel_logits,
        result,
        softmax_scale,
        *q.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        block_table.stride(0),
        *rel_logits.stride(),
        *result.stride(),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=8,
        PAGE_SIZE=key_cache.shape[1],
        REL_EXTENT=rel_extent,
        WINDOW_LEFT=window_left,
        num_warps=num_warps,
        num_stages=2,
    )
    return result


def inkling_torch_rel_attention(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correctness-first paged relative attention for ROCm.

    This mirrors :func:`inkling_fa4_rel_attention` using bounded-size PyTorch
    score tensors. It intentionally synchronizes the small varlen metadata to
    the host and is not intended as a performance path.
    """
    del max_seqlen_q, num_splits
    if not causal:
        raise NotImplementedError("Inkling PyTorch attention requires causal=True")
    if window_size == (-1, -1):
        window_left = None
    elif window_size[0] >= 0 and window_size[1] == 0:
        window_left = window_size[0]
    else:
        raise NotImplementedError(
            f"Unsupported Inkling PyTorch attention window: {window_size}"
        )

    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("Expected q to be 3D and paged K/V caches to be 4D")
    if key_cache.shape != value_cache.shape:
        raise ValueError("Inkling key and value cache shapes must match")
    if key_cache.shape[-1] != q.shape[-1]:
        raise ValueError("Inkling query and KV head dimensions must match")
    if rel_logits.shape[:2] != q.shape[:2] or rel_logits.shape[2] < rel_extent:
        raise ValueError("Inkling relative logits have an incompatible shape")

    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    if num_heads % num_kv_heads != 0:
        raise ValueError("Inkling query heads must be divisible by KV heads")
    num_kv_groups = num_heads // num_kv_heads
    block_size = key_cache.shape[1]

    query_starts = cu_seqlens_q.detach().to("cpu").tolist()
    kv_lens = cache_seqlens.detach().to("cpu").tolist()
    if len(query_starts) != len(kv_lens) + 1:
        raise ValueError("Inkling varlen query and KV metadata do not match")
    if query_starts[-1] > q.shape[0]:
        raise ValueError("Inkling query metadata exceeds the query tensor")

    result = torch.empty_like(q) if out is None else out
    if result.shape != q.shape:
        raise ValueError("Inkling attention output must have the query shape")

    # Keep each fp32 score tensor at or below roughly 64 MiB.
    max_score_elements = 16 * 1024 * 1024
    for seq_idx, kv_len in enumerate(kv_lens):
        query_start = query_starts[seq_idx]
        query_end = query_starts[seq_idx + 1]
        query_len = query_end - query_start
        if query_len == 0:
            continue
        if kv_len < query_len or kv_len <= 0:
            raise ValueError("Inkling KV length must cover every query token")

        if window_left is None:
            query_chunk_size = max(
                1, min(query_len, max_score_elements // (num_heads * kv_len))
            )
        else:
            max_keys_per_query = min(kv_len, window_left + 1)
            query_chunk_size = max(
                1,
                min(
                    query_len,
                    max_score_elements // (num_heads * max_keys_per_query),
                ),
            )
            while (
                num_heads
                * query_chunk_size
                * min(kv_len, window_left + query_chunk_size)
                > max_score_elements
            ):
                query_chunk_size = max(1, query_chunk_size // 2)

        for chunk_start in range(0, query_len, query_chunk_size):
            chunk_end = min(query_len, chunk_start + query_chunk_size)
            absolute_query_start = kv_len - query_len + chunk_start
            absolute_query_end = kv_len - query_len + chunk_end
            key_start = (
                0 if window_left is None else max(0, absolute_query_start - window_left)
            )
            key_end = absolute_query_end

            first_block = key_start // block_size
            last_block = (key_end + block_size - 1) // block_size
            physical_blocks = block_table[seq_idx, first_block:last_block].to(
                dtype=torch.long
            )
            if physical_blocks.numel() == 0 or bool((physical_blocks < 0).any()):
                raise ValueError("Inkling block table is missing required KV pages")
            page_offset = key_start - first_block * block_size
            num_keys = key_end - key_start

            key = key_cache.index_select(0, physical_blocks).reshape(
                -1, num_kv_heads, q.shape[-1]
            )[page_offset : page_offset + num_keys]
            value = value_cache.index_select(0, physical_blocks).reshape(
                -1, num_kv_heads, q.shape[-1]
            )[page_offset : page_offset + num_keys]

            query = q[query_start + chunk_start : query_start + chunk_end].float()
            query = query.view(
                chunk_end - chunk_start,
                num_kv_heads,
                num_kv_groups,
                q.shape[-1],
            )
            scores = torch.einsum("qhgd,khd->hgqk", query, key.float())
            scores = scores.reshape(num_heads, chunk_end - chunk_start, num_keys)
            scores.mul_(softmax_scale)

            query_positions = torch.arange(
                absolute_query_start,
                absolute_query_end,
                device=q.device,
            ).view(-1, 1)
            key_positions = torch.arange(key_start, key_end, device=q.device).view(
                1, -1
            )
            distance = query_positions - key_positions
            relative_index = distance.clamp(0, rel_extent - 1)
            relative = rel_logits[
                query_start + chunk_start : query_start + chunk_end
            ].float()
            bias = relative.permute(1, 0, 2).gather(
                2,
                relative_index.unsqueeze(0).expand(num_heads, -1, -1),
            )
            bias.masked_fill_(
                ((distance < 0) | (distance >= rel_extent)).unsqueeze(0),
                0.0,
            )
            scores.add_(bias)

            mask = distance < 0
            if window_left is not None:
                mask = mask | (distance > window_left)
            scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

            probabilities = torch.softmax(scores, dim=-1)
            probabilities = probabilities.view(
                num_kv_heads,
                num_kv_groups,
                chunk_end - chunk_start,
                num_keys,
            )
            attention = torch.einsum(
                "hgqk,khd->qhgd", probabilities, value.float()
            ).reshape(chunk_end - chunk_start, num_heads, q.shape[-1])
            result[query_start + chunk_start : query_start + chunk_end].copy_(
                attention.to(q.dtype)
            )

    return result
