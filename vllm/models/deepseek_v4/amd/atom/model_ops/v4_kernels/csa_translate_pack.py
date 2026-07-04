# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused CSA topk translate + packed-write kernel.

Replaces the chain (per CSA layer, per fwd):

    block_idx = topk_local // csa_block_capacity
    slot      = topk_local %  csa_block_capacity
    safe_bid  = batch_id.clamp(0).long()
    safe_blk  = block_idx.long().clamp(0, mnbps-1)
    phys      = block_tables[safe_bid_expanded, safe_blk]      # fancy index
    paged     = swa_pages + phys * csa_block_capacity + slot
    packed_write(paged, kv_indices_csa, kv_indptr_csa, ...)     # triton

with a single triton kernel that does the indexer-topk → paged offset
translation + bounded packed write entirely in registers — no per-layer
[T, K] intermediates, no fancy index, no separate launch.

CG benefits (V4-Pro: 31 CSA layers per fwd):
  - 0 transient [T, K] tensor allocs (was 5+/layer × 31 → 155+/fwd)
  - 1 captured graph node per layer instead of 7-8

Per-token write offset (`skip_prefix_len_per_token[t]`) accommodates the
two-source paged_prefill layout:
  - decode:           skip = window_size       (full SWA prefix)
  - pure prefill:     skip = 0                 (no SWA history in `unified_kv`)
  - chunked prefill:  skip = prior_swa_count   (variable per-token)

Correctness:
  - paged_decode reads `kv_indices_csa[indptr[t] : indptr[t+1]]` whose length
    is exactly `skip_prefix_len_per_token[t] + valid_k[t]`. This kernel writes
    the CSA topk section at the slice HEAD `[indptr[t], indptr[t]+valid_k[t])`,
    where `valid_k[t]` is recovered as `indptr[t+1] - indptr[t] - skip[t]`
    (CPU builder packs exactly this much per token); the SWA prefix (length
    `skip[t]`) occupies the tail, written by `write_v4_paged_decode_indices`.
    The tail `[valid_k, index_topk)` of `topk_local` is uninitialized garbage
    from aiter's `top_k_per_row_*` (writes only `[0, valid_k)`), so the
    `k_offs < valid_k` mask is the sole correctness barrier. CG-padded slots
    (batch_id=-1) bail in the kernel preamble.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _csa_translate_pack_kernel(
    topk_local_ptr,  # [T, index_topk] int32 — indexer raw output
    block_tables_ptr,  # [bs, mnbps] int32 — page table
    positions_ptr,  # [T] int — global token positions (only read under INLINE_SKIP_FROM_POS)
    kv_indptr_csa_ptr,  # [T+1] int32 — packed cumsum; per-token valid_k = indptr[t+1]-indptr[t]-skip[t]
    batch_id_per_token_ptr,  # [T] int32 — token → seq, sentinel -1
    skip_prefix_len_per_token_ptr,  # [T] int32 — per-token write offset; ignored when INLINE_SKIP_FROM_POS
    kv_indices_csa_ptr,  # [total_indices] int32 — destination
    swa_pages,  # i32 — SWA region size, runtime int
    mnbps,  # i32 — max blocks per seq, runtime int
    index_topk: tl.constexpr,
    csa_block_capacity: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INLINE_SKIP_FROM_POS: tl.constexpr,  # True → skip = min(pos+1, WINDOW_SIZE) (decode); False → load from buffer (prefill)
    WINDOW_SIZE: tl.constexpr,  # SWA window; only used under INLINE_SKIP_FROM_POS
):
    pid_t = tl.program_id(0)
    pid_kb = tl.program_id(1)

    # CG-padded slot sentinel: builder fills [actual_T:padded_T] with -1
    # so the captured kernel grid (= padded_T) bails on padded entries.
    bid = tl.load(batch_id_per_token_ptr + pid_t)
    if bid < 0:
        return

    # Per-token valid_k is derived from the indptr delta we already need to
    # load anyway:
    #   kv_indptr_csa[t+1] - kv_indptr_csa[t] = skip[t] + valid_k[t]
    # (CPU builder packs `(prefix_swa_count_or_actual_swa) + csa_valid_k`
    # per token — see `_attach_v4_paged_decode_meta` / `_build_paged_prefill_meta`).
    # Reading the delta replaces the previous chain
    # `min(min((pos+1)//ratio, n_csa_seq), index_topk)` — eliminates the
    # `n_csa_seq` load and the `(pos+1)//ratio` compute, and stays
    # CORRECT under the aiter `top_k_per_row_*` contract which only writes
    # `[0, min(k, row_length))` (the tail is uninitialized garbage from
    # `torch.empty`, never -1 — aiter tests confirm via `compare_topk_results`
    # checking only the head). The `k_offs < valid_k` mask remains the
    # primary correctness barrier; `topk >= 0` is dropped as it was never
    # a reliable filter for the uninitialized tail.
    if INLINE_SKIP_FROM_POS:
        # Decode: skip = `actual_swa_count[t]` = min(positions[t]+1, win).
        # Derived inline so the caller can omit the per-token CPU write +
        # H2D of `v4_skip_prefix_len_csa`. Matches the value the prefill
        # path stores per token, except chunked prefill where skip depends
        # on `chunk_start` (not derivable from pos alone) — that path keeps
        # INLINE_SKIP_FROM_POS=False and loads from the buffer.
        pos = tl.load(positions_ptr + pid_t)
        skip = tl.minimum(pos + 1, WINDOW_SIZE)
    else:
        skip = tl.load(skip_prefix_len_per_token_ptr + pid_t)
    indptr_t = tl.load(kv_indptr_csa_ptr + pid_t)
    indptr_t1 = tl.load(kv_indptr_csa_ptr + pid_t + 1)
    # CSA topk section occupies the slice HEAD; the SWA prefix (length `skip`)
    # sits at the tail, written by write_v4_paged_decode_indices. valid_k =
    # slice_len - skip is unchanged; only write_base moves to the head.
    write_base = indptr_t
    valid_k = indptr_t1 - indptr_t - skip

    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    in_range = k_offs < valid_k

    topk = tl.load(
        topk_local_ptr + pid_t * index_topk + k_offs,
        mask=in_range,
        other=0,
    )

    # Translate seq-local row → physical paged offset.
    blk_idx = topk // csa_block_capacity
    slot = topk % csa_block_capacity
    # Defensive clamp: keep block_tables gather in-bounds even on masked-off
    # lanes (triton speculatively computes addresses).
    blk_safe = tl.minimum(tl.maximum(blk_idx, 0), mnbps - 1)
    phys = tl.load(
        block_tables_ptr + bid * mnbps + blk_safe,
        mask=in_range,
        other=0,
    )
    paged = swa_pages + phys * csa_block_capacity + slot
    tl.store(
        kv_indices_csa_ptr + write_base + k_offs,
        paged,
        mask=in_range,
    )


def csa_translate_pack(
    topk_local: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    kv_indptr_csa: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    skip_prefix_len_per_token: Optional[torch.Tensor],
    kv_indices_csa: torch.Tensor,
    *,
    swa_pages: int,
    csa_block_capacity: int,
    window_size: int = 0,
) -> None:
    """Fused topk translate + packed write into `kv_indices_csa` (in-place).

    Per-token `valid_k` is recovered from `kv_indptr_csa[t+1] - kv_indptr_csa[t]
    - skip[t]`, which equals the Indexer's per-row visibility
    (`min((pos+1)//ratio, n_committed_csa[bid], index_topk)`) by construction
    of the CPU-side indptr builders (see `_attach_v4_paged_decode_meta`,
    `_build_paged_prefill_meta`). aiter's `top_k_per_row_*` writes only
    `[0, valid_k)` and leaves the tail UNINITIALIZED (`torch.empty`, no
    `-1` fill), so the explicit `k_offs < valid_k` mask is what keeps the
    kernel correct — the previous `topk >= 0` check would NOT have been a
    reliable filter for the garbage tail.

    Args:
      topk_local:                  [T, index_topk] int32 — indexer's seq-local
                                   row indices. Cells `[0, valid_k[t])` are
                                   the kernel-written results; `[valid_k[t],
                                   index_topk)` is uninitialized garbage and
                                   never read.
      block_tables:                [bs, mnbps] int32 — logical block → physical.
      positions:                   [T] int — global token positions; used only
                                   under `window_size > 0` (inline skip
                                   derivation, decode path).
      kv_indptr_csa:               [T+1] int32 — per-token packed cumsum
                                   (CG-padded: tail repeats last value
                                   → kv_len=0). The kernel reads both
                                   `[t]` and `[t+1]` so `valid_k[t]` is
                                   recovered as `indptr[t+1] - indptr[t] - skip[t]`.
      batch_id_per_token:          [T] int32 — token → seq, sentinel -1 for
                                   CG-padded slots.
      skip_prefix_len_per_token:   [T] int32 OR None — per-token SWA prefix
                                   length (the tail segment of each token's
                                   slice); used only to recover
                                   `valid_k = slice_len - skip`. Pass None
                                   with `window_size > 0` to derive it inline
                                   as `min(positions[t]+1, window_size)`
                                   (decode shortcut). Pure prefill passes a
                                   zero buffer; chunked prefill passes
                                   `prior_swa_count_per_token`.
      kv_indices_csa:              [total_indices] int32 — destination buffer;
                                   this kernel writes the CSA topk section at
                                   the slice head
                                   `[indptr[t], indptr[t]+valid_k[t])`.
      swa_pages:                   SWA region size — `num_slots * window_size`,
                                   fixed at CG capture time. Keyword-only.
      csa_block_capacity:          `block_size // ratio = 128 // 4 = 32`
                                   (constexpr; triton can strength-reduce
                                   // and %). Keyword-only.
      window_size:                 SWA window. When > 0 the kernel computes
                                   per-token skip inline (decode shortcut);
                                   when 0 the per-token buffer is loaded.
                                   Keyword-only.
    """
    T, index_topk = topk_local.shape
    if T == 0:
        return

    inline_skip = window_size > 0
    if not inline_skip and skip_prefix_len_per_token is None:
        raise ValueError("skip_prefix_len_per_token is required when window_size == 0")
    if kv_indptr_csa.numel() < T + 1:
        raise ValueError(f"kv_indptr_csa.numel()={kv_indptr_csa.numel()} < T+1={T + 1}")
    if batch_id_per_token.numel() < T:
        raise ValueError(
            f"batch_id_per_token.numel()={batch_id_per_token.numel()} < T={T}"
        )
    if not inline_skip and skip_prefix_len_per_token.numel() < T:
        raise ValueError(
            "skip_prefix_len_per_token.numel()="
            f"{skip_prefix_len_per_token.numel()} < T={T}"
        )
    if positions.numel() < T:
        raise ValueError(f"positions.numel()={positions.numel()} < T={T}")
    mnbps = block_tables.size(1)

    # Triton requires a concrete pointer even when the buffer is unused under
    # an INLINE_SKIP_FROM_POS=True constexpr branch. Reuse positions as a
    # dummy — same dtype family (int), valid GPU pointer, never read.
    skip_ptr = (
        skip_prefix_len_per_token
        if skip_prefix_len_per_token is not None
        else positions
    )

    BLOCK_K = min(64, triton.next_power_of_2(index_topk))
    grid = (T, triton.cdiv(index_topk, BLOCK_K))
    _csa_translate_pack_kernel[grid](
        topk_local,
        block_tables,
        positions,
        kv_indptr_csa,
        batch_id_per_token,
        skip_ptr,
        kv_indices_csa,
        swa_pages,
        mnbps,
        index_topk=index_topk,
        csa_block_capacity=csa_block_capacity,
        BLOCK_K=BLOCK_K,
        INLINE_SKIP_FROM_POS=inline_skip,
        WINDOW_SIZE=window_size,
    )


def csa_translate_pack_reference(
    topk_local: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    kv_indptr_csa: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    skip_prefix_len_per_token: Optional[torch.Tensor],
    kv_indices_csa: torch.Tensor,
    *,
    swa_pages: int,
    csa_block_capacity: int,
    window_size: int = 0,
) -> None:
    """Pure-torch reference. Mirrors the kernel — derives per-token valid_k
    inline from the `kv_indptr_csa` delta minus skip. When `window_size > 0`,
    derives per-token skip inline as `min(positions[t]+1, window_size)` and
    ignores `skip_prefix_len_per_token` (which may be None).
    """
    T, _ = topk_local.shape
    indptr = kv_indptr_csa.to(torch.int64)
    poses = positions.to(torch.int64)
    bids = batch_id_per_token.to(torch.int64)
    inline_skip = window_size > 0
    skips = (
        skip_prefix_len_per_token.to(torch.int64)
        if (not inline_skip and skip_prefix_len_per_token is not None)
        else None
    )
    mnbps = block_tables.size(1)
    for t in range(T):
        bid = int(bids[t].item())
        if bid < 0:
            continue
        pos = int(poses[t].item())
        skip_t = min(pos + 1, window_size) if inline_skip else int(skips[t].item())
        # CSA topk at slice HEAD (matches kernel); SWA prefix (len skip_t) at tail.
        base = int(indptr[t].item())
        valid_k = int(indptr[t + 1].item()) - base - skip_t
        if valid_k <= 0:
            continue
        topk = topk_local[t, :valid_k].to(torch.int64)
        blk_idx = (topk // csa_block_capacity).clamp(0, mnbps - 1)
        slot = topk % csa_block_capacity
        phys = block_tables[bid, blk_idx].to(torch.int64)
        paged = swa_pages + phys * csa_block_capacity + slot
        for k in range(valid_k):
            kv_indices_csa[base + k] = int(paged[k].item())
