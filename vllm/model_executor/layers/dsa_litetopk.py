# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSA (DeepSeek Sparse Attention) indexer top-k for Blackwell (SM100).

Pair-swaps the carried HOT12288 set into the ordinary paged-gather prefix,
scores/emits that prefix once, then streams only the suffix through a fixed
bucket gate.  The score scan does not maintain a histogram; a compact
candidate-only postpass rebuilds the selector certificate, so the
``[num_q, seq_len]`` logit matrix is never materialized.

The kernels are JIT-compiled from the wheel-packaged ``litetopk_kernels``
by ``litetopk_indexer``; this module is the vLLM-facing layer over it, adapting
the indexer to streaming-aware models and to the dense-selector crossover.

Opt-in via ``VLLM_LITETOPK=1`` together with ``VLLM_DSA_MODE=litetopk`` (or
``litedsa``); falls back to the dense ``fp8_fp4_mqa_logits`` +
``top_k_per_row_prefill`` path when unsupported.
"""

import functools

import torch

from vllm.model_executor.layers import litetopk_indexer as latest_litetopk


@functools.cache
def dsa_litetopk_latest_available(*, use_fp4: bool, topk: int) -> bool:
    """Whether the exact-once adapter is loaded and ready on this device."""
    return latest_litetopk.production_extension_available(
        use_fp4=use_fp4,
        topk=topk,
    )


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


def dsa_litetopk_latest_prepare_permuted_gather(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    *,
    sequence_length: int,
    query_length: int,
    num_reqs: int,
    common_end: int,
    window_start: int,
    topk: int,
    hot_key: str | None,
):
    """Consume the carry-prepared HOT12288 map and gather the cache once.

    The asynchronous carry is consumed by one main-extension binding that
    plans the pair swap and writes the permuted paged cache workspace.
    """
    if not dsa_litetopk_latest_available(
        use_fp4=dst_k.dtype == torch.uint8,
        topk=topk,
    ):
        return None
    return latest_litetopk.prepare_permuted_gather(
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        sequence_length=sequence_length,
        query_length=query_length,
        num_reqs=num_reqs,
        common_end=common_end,
        window_start=window_start,
        hot_key=hot_key,
    )


def dsa_litetopk_latest_release_pair_swap_workspace(
    device: torch.device,
) -> None:
    """Release the current stream's pair-swap planner state."""
    latest_litetopk.release_pair_swap_workspace(device)


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
    permuted_plan=None,
    q_sf: torch.Tensor | None = None,
) -> bool:
    """Run the vendored production kernel while preserving streaming tokens."""
    if not dsa_litetopk_latest_available(use_fp4=q_sf is not None, topk=topk):
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
    common_ke = ke_min_hint - num_local_tokens
    # The signed-sortable high24 candidate ABI preserves exact order inside
    # bucket zero, so streaming callers no longer need a forced drift margin.
    # Keeping the configured value (zero in production) also avoids widening
    # every fixed-threshold bucket and inflating the candidate slab.
    headroom = latest_litetopk.HEADROOM if n_forced else None
    if permuted_plan is None:
        return False
    ok = latest_litetopk.try_large_exact_once_chunk(
        q,
        kv,
        kv_scales,
        weights,
        ks_in,
        ke_in,
        inner,
        k_free,
        permuted_plan=permuted_plan,
        num_reqs=num_reqs,
        ke_min_hint=common_ke,
        cap=latest_litetopk.MERGE_CAP,
        hot_key=hot_key,
        ks_common_hint=num_init_tokens,
        carry_extent_hint=kv.shape[0] - num_local_tokens,
        headroom=headroom,
        q_sf=q_sf,
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
    use_fp4: bool = False,
) -> None:
    """Seed the hot-only adapter from the final dense chunk before crossover."""
    # The next scheduler step can have a different query length. Refresh from
    # every dense step that could precede a full-Q exact-once chunk, rather
    # than predicting the next length from this step's row count.
    min_s = latest_litetopk.production_min_s(use_fp4)
    seeds_production_boundary = min_s - 32768 <= seq_len < min_s
    if (
        not dsa_litetopk_latest_available(use_fp4=use_fp4, topk=topk_indices.shape[1])
        or hot_key is None
        or not latest_litetopk.HOTONLY
        or not seeds_production_boundary
    ):
        return

    n_forced = num_init_tokens + num_local_tokens
    if n_forced == 0:
        latest_litetopk.stash_carry(
            hot_key,
            topk_indices,
            seq_len,
            next_sequence_length=seq_len + latest_litetopk.FUSED_QUERY_LEN,
        )
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
        next_sequence_length=seq_len + latest_litetopk.FUSED_QUERY_LEN,
    )
