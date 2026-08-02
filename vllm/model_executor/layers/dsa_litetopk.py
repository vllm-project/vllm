# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSA (DeepSeek Sparse Attention) indexer top-k for Blackwell (SM100).

Streams KV in tiles and fuses fp8 MQA scoring (tcgen05 UMMA) + an online
bucketed gate + compact top-k, so the ``[num_q, seq_len]`` logit matrix is
never materialized.

The kernels are JIT-compiled from ``csrc/libtorch_stable/attention/dsa/latest``
by ``litetopk_indexer``; this module is the vLLM-facing layer over it, adapting
the indexer to streaming-aware models and to the dense-selector crossover.

Opt-in via ``VLLM_LITETOPK=1`` together with ``VLLM_DSA_MODE=litetopk`` (or
``litedsa``); falls back to the dense ``fp8_fp4_mqa_logits`` +
``top_k_per_row_prefill`` path when unsupported.
"""

import functools

import torch

from vllm.model_executor.layers import litetopk_indexer as latest_litetopk

# Streaming-aware indexing leaves only 1008 ranked slots after LongCat's
# 16 sink + 1024 local tokens. With a hot-carry sample, zero forward
# headroom can collapse enough newly-hot scores into bucket 0 for the packed
# selector's boundary certificate to be undefined (the CUDA kernel traps
# intentionally in that case). Half a sample span is the kernel's designed
# drift allowance and keeps that boundary representable. Non-streaming
# callers (GLM) retain the adapter's configured/default headroom.
_STREAMING_MIN_HEADROOM = 0.5


@functools.cache
def dsa_litetopk_latest_available() -> bool:
    """Whether the vendored high24-v1 adapter can run on this device."""
    if not latest_litetopk.ENABLED or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def dsa_litetopk_latest_dense_topk(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out_indices: torch.Tensor,
    topk: int,
    *,
    seq_len_hint: int,
    num_init_tokens: int,
    num_local_tokens: int,
) -> bool:
    """Use LiteTopK's exact dense selector when its measured gate accepts it."""
    return latest_litetopk.try_dense_topk(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out_indices,
        topk,
        seq_len_hint=seq_len_hint,
        num_init_tokens=num_init_tokens,
        num_local_tokens=num_local_tokens,
    )


def dsa_litetopk_latest_streaming_indexer(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    out_indices: torch.Tensor,
    num_init_tokens: int,
    num_local_tokens: int,
    *,
    num_reqs: int,
    ke_min_hint: int,
    hot_key: str | None,
) -> bool:
    """Run the vendored production kernel while preserving streaming tokens."""
    if not dsa_litetopk_latest_available():
        return False

    n_forced = num_init_tokens + num_local_tokens
    k_free = topk - n_forced
    if k_free <= 0:
        raise ValueError(
            f"topk={topk} leaves no room for {n_forced} forced streaming tokens"
        )

    if n_forced == 0:
        # GLM has no forced sink/local tokens. Reuse the caller's bounds
        # directly instead of launching add, subtract, and maximum kernels
        # for every indexer layer and every prefill chunk.
        ks_in = cu_seqlen_ks
        ke_in = cu_seqlen_ke
    else:
        ks_in = cu_seqlen_ks + num_init_tokens
        ke_in = torch.maximum(cu_seqlen_ke - num_local_tokens, ks_in)
    inner = (
        out_indices
        if n_forced == 0
        else torch.empty(
            q.shape[0],
            k_free,
            dtype=out_indices.dtype,
            device=out_indices.device,
        )
    )
    ok = latest_litetopk.try_chunk(
        q,
        kv,
        kv_scales,
        weights,
        ks_in,
        ke_in,
        inner,
        k_free,
        num_reqs=num_reqs,
        ke_min_hint=ke_min_hint - num_local_tokens,
        cap=latest_litetopk.MERGE_CAP,
        hot_key=hot_key,
        ks_common_hint=num_init_tokens,
        carry_extent_hint=kv.shape[0] - num_local_tokens,
        headroom=(
            max(latest_litetopk.HEADROOM, _STREAMING_MIN_HEADROOM) if n_forced else None
        ),
    )
    if not ok:
        return False
    if n_forced == 0:
        return True

    out_indices[:, :k_free] = inner
    num_q = q.shape[0]
    dev = q.device
    ks_l = cu_seqlen_ks.to(dev, torch.int64).reshape(-1)[:num_q]
    last = (cu_seqlen_ke.to(dev, torch.int64).reshape(-1)[:num_q] - 1).clamp_min(ks_l)
    col = k_free
    if num_init_tokens > 0:
        idx = ks_l[:, None] + torch.arange(num_init_tokens, device=dev)
        out_indices[:, col : col + num_init_tokens] = torch.minimum(
            idx, last[:, None]
        ).to(out_indices.dtype)
        col += num_init_tokens
    if num_local_tokens > 0:
        idx = last[:, None] - torch.arange(num_local_tokens, device=dev)
        out_indices[:, col : col + num_local_tokens] = torch.maximum(
            idx, ks_l[:, None]
        ).to(out_indices.dtype)
    return True


def dsa_litetopk_latest_stash_dense(
    topk_indices: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    seq_len: int,
    num_init_tokens: int,
    num_local_tokens: int,
    hot_key: str | None,
) -> None:
    """Seed the hot-only adapter from the final dense chunk before crossover."""
    if (
        not dsa_litetopk_latest_available()
        or hot_key is None
        or not latest_litetopk.HOTONLY
        or not (latest_litetopk.MIN_S - 32768 <= seq_len < latest_litetopk.MIN_S)
    ):
        return

    n_forced = num_init_tokens + num_local_tokens
    if n_forced == 0:
        latest_litetopk.stash_carry(hot_key, topk_indices, seq_len)
        return

    k_free = topk_indices.shape[1] - n_forced
    if k_free <= 0:
        return
    ks_in = (cu_seqlen_ks + num_init_tokens).reshape(-1, 1)
    ke_in = (cu_seqlen_ke - num_local_tokens).reshape(-1, 1)
    valid = (topk_indices >= ks_in) & (topk_indices < ke_in)
    try:
        interior = topk_indices.masked_select(valid).view(topk_indices.shape[0], k_free)
    except RuntimeError:
        return
    latest_litetopk.stash_carry(
        hot_key,
        interior,
        seq_len - num_local_tokens,
        min_index=num_init_tokens,
    )
