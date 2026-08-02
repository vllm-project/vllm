# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.platforms import current_platform


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
    if _use_sheared_bias():
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func

        bias_kwargs: dict[str, Any] = {"rel_bias": rel_logits}
    else:
        from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }

    ret = flash_attn_varlen_func(
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
