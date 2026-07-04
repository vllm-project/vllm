# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plugin-local Triton ops for the vLLM DeepSeek V4 bridge.

Kernels here are specific to the bridge's proxy KV-cache layout (the unified
``swa_pages + block_id`` compress region) rather than to native ATOM, so they
live alongside the bridge instead of in ``atom.model_ops.v4_kernels``.

``write_v4_decode_hca_compress_tail`` is the decode-time companion to
``atom.model_ops.v4_kernels.write_v4_paged_decode_indices``: that core kernel
fills the SWA window prefix (placed at the slice TAIL) shared by the SWA / CSA /
HCA index buffers; this one fills the HCA compress section at the slice HEAD
straight from the GPU block table, so the whole decode HCA index segment is
built on-GPU with no per-step host round trip.

Layout note: since the MI355 decode-kernel retune (#1116) the ragged-packed
decode slices place the compress section at the head and the SWA window prefix
at the tail (the kernels read the whole slice and are permutation-invariant
over keys). This kernel must match: it writes HCA entries at ``[hca_indptr[t],
+n_hca)``. (The ``_tail`` in the name is historical.)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_decode_hca_compress_tail_kernel(
    batch_id_per_token_ptr,  # [>=T] int — sentinel -1 in CG pad tail
    positions_ptr,  # [>=T] int — global token position
    hca_indptr_ptr,  # [>=T+1] int32 — ragged (SWA prefix + HCA committed)
    n_committed_hca_per_seq_ptr,  # [num_reqs] int32 — per-seq HCA entry count
    block_tables_ptr,  # [num_reqs, MAX_BLOCKS] int — per-seq paged block ids
    bt_stride_bs,  # block_tables row stride (elements)
    hca_indices_ptr,  # [>=hca_indptr[T]] int32 OUT — HCA compress section (head)
    swa_pages,  # num_slots * cs — boundary into the compress region
    win: tl.constexpr,  # SWA window — per-token prefix length cap
    BLOCK_J: tl.constexpr,  # next_pow2(win) — HCA loop chunk size
):
    """One program per token. Writes the HCA compress segment at the slice
    HEAD ``[hca_indptr[t], +n_hca)``; the j-th committed HCA entry maps to
    physical page ``swa_pages + block_tables[bid, j]``.

    Ragged-packed layout (since the MI355 decode-kernel retune, #1116): the
    compress section occupies the head of each token's slice and the SWA
    window-prefix (length ``n_swa = min(pos+1, win)``) sits at the TAIL,
    written separately by ``write_v4_paged_decode_indices``. The two together
    cover the full per-token HCA segment ``[hca_indptr[t], hca_indptr[t+1])``,
    so no ``-1`` pre-fill is needed. CG-padded tail tokens (batch_id == -1)
    carry a zero-length segment and are skipped.

    Decode analogue of the HCA compress section in
    ``_v4_paged_prefill_indices_kernel``.
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched
    n_hca = tl.load(n_committed_hca_per_seq_ptr + bid)
    # HCA compress section occupies the slice HEAD (offset 0); the SWA window
    # prefix sits at the tail, written by `write_v4_paged_decode_indices`.
    base = tl.load(hca_indptr_ptr + t)
    bt_row_base = bid * bt_stride_bs
    i = tl.arange(0, BLOCK_J)
    for j in tl.range(0, n_hca, BLOCK_J):
        k = j + i
        mask = k < n_hca
        bt = tl.load(block_tables_ptr + bt_row_base + k, mask=mask, other=0)
        tl.store(hca_indices_ptr + base + k, swa_pages + bt, mask=mask)


@triton.jit
def _v4_decode_indices_fused_kernel(
    state_slot_per_seq_ptr,  # [bs] int32
    batch_id_per_token_ptr,  # [>=T] int — sentinel -1 in CG pad tail
    positions_ptr,  # [>=T] int — global token position
    swa_indptr_ptr,  # [>=T+1] int32 — ragged SWA-prefix cumsum
    csa_indptr_ptr,  # [>=T+1] int32 — ragged (SWA + CSA topk)
    hca_indptr_ptr,  # [>=T+1] int32 — ragged (SWA + HCA committed)
    swa_indices_ptr,  # [swa_total] int32 OUT
    csa_indices_ptr,  # [csa_total] int32 OUT (SWA-prefix segment only)
    hca_indices_ptr,  # [hca_total] int32 OUT (SWA prefix tail + HCA head)
    n_committed_hca_per_seq_ptr,  # [num_reqs] int32 — per-seq HCA entry count
    block_tables_ptr,  # [num_reqs, MAX_BLOCKS] int — per-seq paged block ids
    bt_stride_bs,  # block_tables row stride (elements)
    cs,  # win_with_spec — ring-index modulo / SWA-region stride
    swa_pages,  # num_slots * cs — boundary into compress region
    win: tl.constexpr,  # SWA window — max prefix slots
    BLOCK_N: tl.constexpr,  # next_pow2(win)
):
    """Fused decode index build: one program per token writes BOTH the SWA
    window prefix (slice TAIL of swa/csa/hca) and the HCA compress section
    (slice HEAD of hca). Merges ``_v4_paged_decode_indices_kernel`` and
    ``_v4_decode_hca_compress_tail_kernel`` into one launch — the two write
    disjoint regions of each token's slice, so a single program covers both
    with no cross-program race.
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched

    slot = tl.load(state_slot_per_seq_ptr + bid)
    pos = tl.load(positions_ptr + t)

    # --- SWA window prefix (slice TAIL of swa / csa / hca) ---
    n = tl.minimum(pos + 1, win)
    swa_end = tl.load(swa_indptr_ptr + t + 1)
    csa_end = tl.load(csa_indptr_ptr + t + 1)
    hca_end = tl.load(hca_indptr_ptr + t + 1)
    i = tl.arange(0, BLOCK_N)
    mask = i < n
    abs_pos = pos - n + 1 + i
    ring_idx = abs_pos % cs
    paged = slot * cs + ring_idx
    tl.store(swa_indices_ptr + swa_end - n + i, paged, mask=mask)
    tl.store(csa_indices_ptr + csa_end - n + i, paged, mask=mask)
    tl.store(hca_indices_ptr + hca_end - n + i, paged, mask=mask)

    # --- HCA compress section (slice HEAD of hca) ---
    n_hca = tl.load(n_committed_hca_per_seq_ptr + bid)
    base = tl.load(hca_indptr_ptr + t)
    bt_row_base = bid * bt_stride_bs
    for j in tl.range(0, n_hca, BLOCK_N):
        k = j + i
        kmask = k < n_hca
        bt = tl.load(block_tables_ptr + bt_row_base + k, mask=kmask, other=0)
        tl.store(hca_indices_ptr + base + k, swa_pages + bt, mask=kmask)


def write_v4_decode_indices_fused(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor,
    hca_indices: torch.Tensor,
    n_committed_hca_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    T: int,
    win: int,
    cs: int,
    swa_pages: int,
) -> None:
    """Single-launch fusion of ``write_v4_paged_decode_indices`` (SWA window
    prefix) and ``write_v4_decode_hca_compress_tail`` (HCA compress section).

    Both originals are ``grid=(T,)`` one-program-per-token kernels writing
    disjoint regions of each token's ragged slice, so fusing halves the
    per-step Triton host launch overhead with identical output. See those
    functions for the per-segment layout contract.
    """
    if T == 0:
        return
    assert state_slot_per_seq.dim() == 1
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert swa_indptr.dim() == 1 and swa_indptr.shape[0] >= T + 1
    assert csa_indptr.dim() == 1 and csa_indptr.shape[0] >= T + 1
    assert hca_indptr.dim() == 1 and hca_indptr.shape[0] >= T + 1
    assert swa_indices.dim() == 1
    assert csa_indices.dim() == 1
    assert hca_indices.dim() == 1
    assert n_committed_hca_per_seq.dim() == 1
    assert block_tables.dim() == 2

    BLOCK_N = triton.next_power_of_2(win)
    _v4_decode_indices_fused_kernel[(T,)](
        state_slot_per_seq,
        batch_id_per_token,
        positions,
        swa_indptr,
        csa_indptr,
        hca_indptr,
        swa_indices,
        csa_indices,
        hca_indices,
        n_committed_hca_per_seq,
        block_tables,
        block_tables.stride(0),
        cs,
        swa_pages,
        win=win,
        BLOCK_N=BLOCK_N,
    )


def write_v4_decode_hca_compress_tail(
    *,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    hca_indptr: torch.Tensor,
    n_committed_hca_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    hca_indices: torch.Tensor,
    T: int,
    win: int,
    swa_pages: int,
) -> None:
    """In-place GPU fill of the decode HCA compress-section paged offsets.

    Companion to ``write_v4_paged_decode_indices``: that kernel fills the SWA
    window-prefix (at the slice TAIL) of the SWA / CSA / HCA index buffers; this
    one fills the HCA compress section (``swa_pages + block_tables[bid, j]``) at
    the slice HEAD ``[hca_indptr[t], +n_hca)`` in ``hca_indices``. Together they
    write the full per-token HCA segment, so the caller need not pre-fill ``-1``.

    Replaces the prior CPU scatter in the bridge's decode metadata build
    (block-table D2H + numpy ``repeat``/``cumsum``/fancy-index + H2D), so the
    whole HCA index segment is produced on-GPU with no per-step host round trip
    — matching the existing on-GPU prefill build.

    All tensors are GPU tensors. Per-seq inputs are indexed by
    ``batch_id_per_token`` inline (no caller pre-gather).

    Args:
      batch_id_per_token:      ``[>=T]``   int — token→seq map; -1 skipped.
      positions:               ``[>=T]``   int — global token positions (unused
                                           since the compress section moved to the
                                           slice head; kept for call-site parity).
      hca_indptr:              ``[>=T+1]`` int32 — ragged HCA indptr (same one
                                           passed to ``write_v4_paged_decode_indices``).
      n_committed_hca_per_seq: ``[num_reqs]`` int32 — per-seq HCA entry count.
      block_tables:            ``[num_reqs, mnbs]`` int — per-seq paged blocks.
      hca_indices:             ``[>=hca_indptr[T]]`` int32 OUT — compress section
                                           (slice head) written; SWA prefix
                                           (slice tail) left to sibling.
      T:                       int — real token count (grid size).
      win:                     int — SWA window size.
      swa_pages:               int — ``num_slots * cs`` boundary in unified_kv.
    """
    if T == 0:
        return
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert hca_indptr.dim() == 1 and hca_indptr.shape[0] >= T + 1
    assert n_committed_hca_per_seq.dim() == 1
    assert block_tables.dim() == 2
    assert hca_indices.dim() == 1

    BLOCK_J = triton.next_power_of_2(win)
    _v4_decode_hca_compress_tail_kernel[(T,)](
        batch_id_per_token,
        positions,
        hca_indptr,
        n_committed_hca_per_seq,
        block_tables,
        block_tables.stride(0),
        hca_indices,
        swa_pages,
        win=win,
        BLOCK_J=BLOCK_J,
    )
