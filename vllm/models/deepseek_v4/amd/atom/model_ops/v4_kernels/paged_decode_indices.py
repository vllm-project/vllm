# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 paged-decode index scatter — single Triton kernel writes SWA window-
prefix paged offsets into the three ragged-packed destination buffers
(`kv_indices_swa` / `kv_indices_csa` / `kv_indices_hca`).

The ring-index formula `ring = (pos - win + 1 + w) % cs` is computed inline
inside the kernel from `positions[t]` — no `[T, win]` window_topk staging
buffer, no separate CPU build + H2D copy.

Layout: ragged-packed. Each token's slice holds an SWA prefix of length
`n = min(positions[t]+1, win)` plus a per-buffer compress section; the
`swa_indptr` / `csa_indptr` / `hca_indptr` cumsums reflect this ragged
sizing. Within each token's slice the SWA prefix is written at the TAIL
(`[indptr[t+1] - n, indptr[t+1])`) and the compress section (CSA topk /
HCA committed) occupies the head.

Caller contract:
- Grid = T (one program per token).
- `batch_id_per_token[:T]` may carry `-1` sentinels in the CG-padded tail —
  kernel checks and bails (matches `_attach_v4_per_fwd_meta` convention).
- `swa_indptr` / `csa_indptr` / `hca_indptr` must reflect the ragged-packed
  sizing: per-token slot count = `min(positions[t]+1, win) + n_compress[t]`
  where `n_compress[t]` is 0 for SWA, `min(n_committed_csa, index_topk)`
  for CSA, `n_committed_hca` for HCA.
- `swa_indices` / `csa_indices` / `hca_indices` capacity ≥ corresponding
  indptr[T]; this kernel only writes the SWA-prefix segment at the slice
  tail `[indptr[t+1] - n, indptr[t+1])` per token. The compress section is
  filled elsewhere (HCA: numpy fill in caller, CSA: `csa_translate_pack`
  per layer).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_paged_decode_indices_kernel(
    state_slot_per_seq_ptr,  # [bs] int32
    batch_id_per_token_ptr,  # [T+pad] int — sentinel -1 in pad tail
    positions_ptr,  # [T+pad] int — global token position
    swa_indptr_ptr,  # [T+1] int32 — ragged SWA-prefix cumsum
    csa_indptr_ptr,  # [T+1] int32 — ragged (SWA + CSA topk)
    hca_indptr_ptr,  # [T+1] int32 — ragged (SWA + HCA committed)
    swa_indices_ptr,  # [swa_total] int32, output
    csa_indices_ptr,  # [csa_total] int32, output (writes SWA-prefix segment only)
    hca_indices_ptr,  # [hca_total] int32, output (writes SWA-prefix segment only)
    cs,  # win_with_spec — stride into unified_kv SWA region (paper §3.6.1)
    win: tl.constexpr,  # window_size — max SWA prefix slots
    BLOCK_N: tl.constexpr,  # next_pow2(win)
):
    """One program per token. Writes `n = min(positions[t]+1, win)` paged
    offsets to the SWA prefix segment, placed at the TAIL of each token's
    slice in the SWA/CSA/HCA index buffers (the compress section occupies
    the head).

    For token `t`:
        bid = batch_id_per_token[t]                  # bail if -1 (CG pad)
        slot = state_slot_per_seq[bid]
        pos = positions[t]
        n = min(pos + 1, win)
        # i in [0, n) → abs_pos = pos - n + 1 + i ∈ [0, pos]; written at the
        # slice tail (indptr[t+1] - n) so the compress section fills the head.
        for i in range(n):
            abs_pos = pos - n + 1 + i
            ring = abs_pos % cs
            paged = slot * cs + ring
            swa_indices[swa_indptr[t+1] - n + i] = paged
            csa_indices[csa_indptr[t+1] - n + i] = paged
            hca_indices[hca_indptr[t+1] - n + i] = paged
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched

    slot = tl.load(state_slot_per_seq_ptr + bid)
    pos = tl.load(positions_ptr + t)
    # `n` = actual valid SWA prefix count. Cast to match `win` (compile-time
    # int) — pos is i32/i64 from positions buffer.
    n = tl.minimum(pos + 1, win)
    # SWA prefix segment lives at the TAIL of each token's slice (compress
    # section fills the head). Write base = slice END (indptr[t+1]) - n. For
    # the SWA buffer (compress=0) end-n == indptr[t], same as a head write.
    swa_end = tl.load(swa_indptr_ptr + t + 1)
    csa_end = tl.load(csa_indptr_ptr + t + 1)
    hca_end = tl.load(hca_indptr_ptr + t + 1)

    i = tl.arange(0, BLOCK_N)
    mask = i < n
    abs_pos = pos - n + 1 + i  # ∈ [0, pos] for valid i
    ring_idx = abs_pos % cs
    paged = slot * cs + ring_idx

    tl.store(swa_indices_ptr + swa_end - n + i, paged, mask=mask)
    tl.store(csa_indices_ptr + csa_end - n + i, paged, mask=mask)
    tl.store(hca_indices_ptr + hca_end - n + i, paged, mask=mask)


def write_v4_paged_decode_indices(
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
    T: int,
    win: int,
    cs: int,
) -> None:
    """In-place fill SWA / CSA / HCA window-prefix offsets via a single
    Triton kernel. Replaces the prior `_build_window_topk_np` (CPU O(T·win))
    + `index_copy_` chain. All inputs are persistent forward_vars buffers —
    no allocator churn.

    Args (all GPU tensors except T/win/cs):
      state_slot_per_seq:  [bs]   int32 — per-seq state cache slot.
      batch_id_per_token:  [>=T]  int   — token→seq map; -1 sentinel skipped.
      positions:           [>=T]  int   — global token position
                                   (forward_vars["positions"]); used to derive
                                   `n = min(pos+1, win)` per token + the ring
                                   index `(pos - n + 1 + i) % cs`.
      swa_indptr:          [>=T+1] int32 — ragged SWA-prefix cumsum, where
                                   `swa_indptr[t+1] - swa_indptr[t] =
                                    min(positions[t]+1, win)`.
      csa_indptr:          [>=T+1] int32 — ragged CSA buffer indptr (SWA
                                   prefix + CSA topk per token).
      hca_indptr:          [>=T+1] int32 — ragged HCA buffer indptr (SWA
                                   prefix + HCA committed per token).
      swa_indices:         [>=swa_indptr[T]] int32 OUT — fully written by
                                   this kernel (no other source).
      csa_indices:         [>=csa_indptr[T]] int32 OUT — SWA prefix written
                                   here at the slice tail
                                   `[csa_indptr[t+1] - n, csa_indptr[t+1])`;
                                   CSA topk section (slice head) filled
                                   per-layer by `csa_translate_pack`.
      hca_indices:         [>=hca_indptr[T]] int32 OUT — same semantics; HCA
                                   compress section (slice head) filled in the
                                   caller via numpy fill.
      T:                   int — number of real tokens (grid size).
      win:                 int — SWA window size (typically 128 for V4-Pro).
      cs:                  int — `win_with_spec = window_size + max_spec_steps`,
                                 stride into unified_kv SWA region per slot
                                 AND modulo for ring-index wrap.
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

    BLOCK_N = triton.next_power_of_2(win)
    _v4_paged_decode_indices_kernel[(T,)](
        state_slot_per_seq,
        batch_id_per_token,
        positions,
        swa_indptr,
        csa_indptr,
        hca_indptr,
        swa_indices,
        csa_indices,
        hca_indices,
        cs,
        win=win,
        BLOCK_N=BLOCK_N,
    )


def write_v4_paged_decode_indices_reference(
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
    T: int,
    win: int,
    cs: int,
) -> None:
    """Pure-PyTorch reference equivalent of `write_v4_paged_decode_indices`.
    For unit tests and bisect verification. Mirrors the kernel exactly:
    per-token ragged-packed write, no -1 sentinels in output.
    """
    if T == 0:
        return
    bid = batch_id_per_token[:T].long()
    pos_t = positions[:T].long()
    valid = bid >= 0
    # n = min(pos+1, win) per token; clamp invalid rows to 0 to skip writes.
    n_per_tok = torch.minimum(pos_t + 1, torch.full_like(pos_t, win))
    n_per_tok = torch.where(valid, n_per_tok, torch.zeros_like(n_per_tok))
    slot = torch.where(
        valid, state_slot_per_seq[bid.clamp(min=0)].long(), torch.zeros_like(bid)
    )
    for t in range(T):
        n = int(n_per_tok[t].item())
        if n == 0:
            continue
        p = int(pos_t[t].item())
        s = int(slot[t].item())
        i_arr = torch.arange(n, device=positions.device, dtype=torch.long)
        abs_pos = p - n + 1 + i_arr  # [n]
        ring = abs_pos % cs
        paged = (s * cs + ring).to(torch.int32)
        # SWA prefix segment at the slice TAIL (compress section fills the head).
        swa_end = int(swa_indptr[t + 1].item())
        csa_end = int(csa_indptr[t + 1].item())
        hca_end = int(hca_indptr[t + 1].item())
        swa_indices[swa_end - n : swa_end] = paged
        csa_indices[csa_end - n : csa_end] = paged
        hca_indices[hca_end - n : hca_end] = paged
