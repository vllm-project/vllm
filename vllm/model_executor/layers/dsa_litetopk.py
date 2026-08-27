# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSA (DeepSeek Sparse Attention) indexer top-k for Blackwell (SM100).

Pair-swaps the carried HOT12288 set into the ordinary paged-gather prefix,
scores/emits that prefix once, then streams only the suffix through a fixed
bucket gate.  The score scan does not maintain a histogram; a compact
candidate-only postpass rebuilds the selector certificate, so the
``[num_q, seq_len]`` logit matrix is never materialized.

The kernels are JIT-compiled from the wheel-packaged ``litetopk_kernels``
by ``litetopk_indexer``; this module is the vLLM-facing layer over it.

Opt-in via ``VLLM_LITETOPK=1``;
falls back to the dense ``fp8_fp4_mqa_logits`` +
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
    *,
    num_reqs: int,
    ke_min_hint: int,
    hot_key: str | None,
    permuted_plan=None,
    q_sf: torch.Tensor | None = None,
    carry_broadcast_src: int | None = None,
    carry_broadcast_extent: int | None = None,
    publish_carry: bool = True,
) -> bool:
    """Run the vendored production kernel."""
    if not dsa_litetopk_latest_available(use_fp4=q_sf is not None, topk=topk):
        return False
    if permuted_plan is None:
        return False
    return latest_litetopk.try_large_exact_once_chunk(
        q,
        kv,
        kv_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out_indices,
        topk,
        permuted_plan=permuted_plan,
        num_reqs=num_reqs,
        ke_min_hint=ke_min_hint,
        cap=latest_litetopk.MERGE_CAP,
        hot_key=hot_key,
        ks_common_hint=0,
        carry_extent_hint=kv.shape[0],
        q_sf=q_sf,
        carry_broadcast_src=carry_broadcast_src,
        carry_broadcast_extent=carry_broadcast_extent,
        _carry_io=publish_carry,
    )


def dsa_litetopk_latest_stash_carry(
    topk_indices: torch.Tensor,
    *,
    seq_len: int,
    hot_key: str | None,
    carry_broadcast_src: int | None = None,
    carry_broadcast_extent: int | None = None,
) -> None:
    """Publish steady-state carry from a globally assembled top-k output."""
    latest_litetopk.stash_carry(
        hot_key,
        topk_indices,
        seq_len,
        broadcast_src=carry_broadcast_src,
        broadcast_extent=carry_broadcast_extent,
    )


def dsa_litetopk_latest_stash_dense(
    topk_indices: torch.Tensor,
    *,
    seq_len: int,
    hot_key: str | None,
    use_fp4: bool = False,
    pcp_world_size: int = 1,
) -> None:
    """Seed the hot-only adapter from the final dense chunk before crossover."""
    # The next scheduler step can have a different query length. Refresh from
    # every dense step that could precede a full-Q exact-once chunk, rather
    # than predicting the next length from this step's row count. Under PCP,
    # DualChunkSwap exposes two FUSED_QUERY_LEN virtual rows per rank, while
    # the global scheduler advances all PCP ranks at once. Consequently a
    # row's sequence extent can advance by 2 * PCP * FUSED_QUERY_LEN between
    # opportunities to seed the carry. Keep the historical 32K floor for
    # PCP<=2 and widen it for PCP4/PCP8 so the crossover cannot be skipped.
    min_s = latest_litetopk.production_min_s(use_fp4)
    seed_window = max(
        32768,
        2 * max(1, int(pcp_world_size)) * latest_litetopk.FUSED_QUERY_LEN,
    )
    seeds_production_boundary = min_s - seed_window <= seq_len < min_s
    if (
        not dsa_litetopk_latest_available(use_fp4=use_fp4, topk=topk_indices.shape[1])
        or hot_key is None
        or not seeds_production_boundary
    ):
        return
    latest_litetopk.stash_carry(
        hot_key,
        topk_indices,
        seq_len,
    )
