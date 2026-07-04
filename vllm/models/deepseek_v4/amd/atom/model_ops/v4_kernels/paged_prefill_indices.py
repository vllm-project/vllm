# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 paged-prefill index scatter — single Triton kernel writes the four
per-fwd index buffers consumed by `sparse_attn_v4_paged_prefill`:

  - ``kv_indices_extend``       : per-fwd `kv` tensor row indices for the
                                  in-chunk SWA tail (one shared buffer).
  - ``kv_indices_prefix_swa``   : Dense path — SWA prior-chunk paged offsets
                                  into `unified_kv`.
  - ``kv_indices_prefix_csa``   : CSA path — SWA prefix segment written at the
                                  slice TAIL; the CSA topk HEAD section is filled
                                  per layer by ``csa_translate_pack`` (head-CSA /
                                  tail-SWA convention, matching decode, #1116).
  - ``kv_indices_prefix_hca``   : HCA path — SWA prefix segment + HCA
                                  all-committed compress section, both fully
                                  written.

Replaces the CPU numpy build in
``DeepseekV4AttentionMetadataBuilder._build_paged_prefill_meta`` (per-fwd
`_segment_indices` + cumsum + scatter chain + pinned H2D). The kernel runs
entirely on GPU and is invoked AFTER the caller has computed the four
indptrs via ``torch.cumsum`` (also on GPU).

Caller responsibilities (no copies done here):
  - The CSA slice is fully covered without any ``-1`` pre-fill: this kernel
    writes the SWA prefix at the slice TAIL (length ``prefix_swa_count``) and
    ``csa_translate_pack`` writes the CSA topk at the HEAD (length
    ``valid_k = slice_len - prefix_swa_count``) per layer — together they cover
    ``[indptr[t], indptr[t+1])`` with no gap. (HCA / Dense buffers are likewise
    fully written by this kernel.)
  - Compute and stage the four indptr buffers and the per-seq scalar inputs.

Per-token quantities (kernel-computed from inputs; mirror the formulas in
``_build_paged_prefill_meta``):
  token_pos_in_chunk[t] = positions[t] - chunk_start[bid]
  swa_low[t]            = max(positions[t] - win + 1, 0)
  extend_count[t]       = min(token_pos_in_chunk[t] + 1, win)
  prefix_swa_count[t]   = max(chunk_start[bid] - swa_low[t], 0)

Per-token paged offset for SWA prefix entries (matches the stride/modulo
used by `swa_write` and `_attach_v4_paged_decode_meta`):
  paged[t,k] = state_slot[bid] * cs + ((swa_low[t] + k) % cs)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_paged_prefill_indices_kernel(
    # Per-token inputs.
    positions_ptr,  # [T] int — global token position
    bid_per_token_ptr,  # [T] int — batch id per token (==`np.repeat(arange(bs), tnps)`)
    # Per-seq inputs (indexed by bid).
    chunk_start_per_seq_ptr,  # [bs] int — current chunk's absolute start position
    cu_seqlens_q_per_seq_ptr,  # [bs] int — per-seq prefix sum start in per-fwd kv tensor
    state_slot_per_seq_ptr,  # [bs] int — per-seq SWA ring slot
    n_committed_hca_per_seq_ptr,  # [bs] int — per-seq HCA compress entry count
    block_tables_ptr,  # [bs, MAX_BLOCKS] int — per-seq paged block ids
    bt_stride_bs,  # bytes between block_tables rows
    # Indptrs (already cumsum'd by caller, all length [T+1]).
    extend_indptr_ptr,
    prefix_swa_indptr_ptr,
    prefix_csa_indptr_ptr,
    prefix_hca_indptr_ptr,
    # Output buffers.
    extend_indices_ptr,
    prefix_swa_indices_ptr,
    prefix_csa_indices_ptr,
    prefix_hca_indices_ptr,
    # Constants.
    win: tl.constexpr,
    cs,  # win_with_spec — SWA ring stride (NOT constexpr because varies w/ mtp_k)
    swa_pages,  # state_slot count * cs — boundary into HCA compress section
    HCA_RATIO: tl.constexpr,  # HCA compress ratio (128) for per-token causal cap
    BLOCK_N: tl.constexpr,  # next_pow2(win) — covers SWA prefix and extend segments
):
    """One program per token. Writes four per-token segments:

    - extend         : ``[extend_indptr[t], extend_indptr[t]+extend_count[t])``
    - prefix SWA     : in swa / hca prefix buffers at the slice HEAD
                        ``[*_indptr[t], *_indptr[t]+prefix_swa_count[t])``; in the
                        csa prefix buffer at the slice TAIL
                        ``[csa_indptr[t+1]-prefix_swa_count[t], csa_indptr[t+1])``
    - HCA compress   : ``[prefix_hca_indptr[t]+prefix_swa_count[t], +n_hca[bid])``
                        in prefix_hca_indices

    Per-token bounded segments (extend, SWA prefix) fit in one ``BLOCK_N``
    vector. HCA compress can be up to ``max_model_len // 128`` per token
    (e.g. 8192 at V4-Pro 1M ctx) — looped in ``BLOCK_N`` chunks.
    """
    t = tl.program_id(0)

    bid = tl.load(bid_per_token_ptr + t)
    pos = tl.load(positions_ptr + t)
    chunk_start = tl.load(chunk_start_per_seq_ptr + bid)
    cu_q = tl.load(cu_seqlens_q_per_seq_ptr + bid)
    state_slot = tl.load(state_slot_per_seq_ptr + bid)
    # Per-token CAUSAL HCA visibility: token at `pos` may see only the
    # `(pos+1)//HCA_RATIO` compressed groups committed up to its own position
    # (matches the reference `get_compress_topk_idxs` prefill mask, and mirrors
    # the CSA `(pos+1)//4` cap). Without this cap every token saw the per-seq
    # `n_committed_hca = ctx_end//128`, which over-reads FUTURE groups and makes
    # a token's output depend on the forward's total length (chunked != single).
    n_hca = tl.minimum(
        (pos + 1) // HCA_RATIO, tl.load(n_committed_hca_per_seq_ptr + bid)
    )

    # Per-token derived quantities (single-pass arithmetic).
    token_pos_in_chunk = pos - chunk_start
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(token_pos_in_chunk + 1, win)
    prefix_swa_count = tl.maximum(chunk_start - swa_low, 0)

    i = tl.arange(0, BLOCK_N)

    # ---- Extend kv_indices: rows in per-fwd kv tensor ----
    # row = cu_q + token_pos_in_chunk - extend_count + 1 + k, k in [0, extend_count)
    ext_base = tl.load(extend_indptr_ptr + t)
    ext_mask = i < extend_count
    ext_start_row = cu_q + token_pos_in_chunk - extend_count + 1
    tl.store(extend_indices_ptr + ext_base + i, ext_start_row + i, mask=ext_mask)

    # ---- SWA prefix paged offsets: written to all three prefix buffers ----
    # paged = state_slot * cs + ((swa_low + k) % cs), k in [0, prefix_swa_count)
    swa_base_swa = tl.load(prefix_swa_indptr_ptr + t)
    swa_base_hca = tl.load(prefix_hca_indptr_ptr + t)
    swa_mask = i < prefix_swa_count
    global_pos = swa_low + i
    ring_idx = global_pos - (global_pos // cs) * cs  # global_pos % cs
    paged = state_slot * cs + ring_idx
    tl.store(prefix_swa_indices_ptr + swa_base_swa + i, paged, mask=swa_mask)
    # CSA buffer: the SWA prefix goes at the slice TAIL. `csa_translate_pack`
    # writes the CSA topk section at the slice HEAD
    # `[indptr[t], indptr[t]+valid_k)` (valid_k = slice_len - prefix_swa_count),
    # so the SWA prefix must occupy `[indptr[t+1]-prefix_swa_count, indptr[t+1])`.
    # Writing it at the head (the pre-#1116 layout) collides with the CSA topk
    # head write and leaves the tail uninitialized — #1116 moved decode and
    # csa_translate_pack to this head-CSA / tail-SWA convention but missed this
    # prefill writer, corrupting chunked-prefill CSA slices (prefix_swa_count>0).
    csa_end = tl.load(prefix_csa_indptr_ptr + t + 1)
    csa_tail_base = csa_end - prefix_swa_count
    tl.store(prefix_csa_indices_ptr + csa_tail_base + i, paged, mask=swa_mask)
    tl.store(prefix_hca_indices_ptr + swa_base_hca + i, paged, mask=swa_mask)

    # ---- HCA compress section: block_tables[bid, k] for k in [0, n_hca) ----
    # Written at offset prefix_swa_count past the SWA prefix segment in HCA buffer.
    hca_dst_base = swa_base_hca + prefix_swa_count
    # block_tables row stride is `bt_stride_bs` int32 elements (== max_num_blocks_per_seq).
    bt_row_base = bid * bt_stride_bs
    for j in tl.range(0, n_hca, BLOCK_N):
        k = j + i
        hca_mask = k < n_hca
        bt = tl.load(block_tables_ptr + bt_row_base + k, mask=hca_mask, other=0)
        tl.store(
            prefix_hca_indices_ptr + hca_dst_base + k,
            swa_pages + bt,
            mask=hca_mask,
        )


def write_v4_paged_prefill_indices(
    *,
    positions: torch.Tensor,
    bid_per_token: torch.Tensor,
    chunk_start_per_seq: torch.Tensor,
    cu_seqlens_q_per_seq: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    n_committed_hca_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    extend_indptr: torch.Tensor,
    prefix_swa_indptr: torch.Tensor,
    prefix_csa_indptr: torch.Tensor,
    prefix_hca_indptr: torch.Tensor,
    extend_indices: torch.Tensor,
    prefix_swa_indices: torch.Tensor,
    prefix_csa_indices: torch.Tensor,
    prefix_hca_indices: torch.Tensor,
    T: int,
    win: int,
    cs: int,
    swa_pages: int,
    hca_ratio: int = 128,
) -> None:
    """One-shot GPU build of the V4 paged-prefill index buffers.

    Replaces the CPU numpy build in
    ``DeepseekV4AttentionMetadataBuilder._build_paged_prefill_meta`` (the
    `_segment_indices` + scatter chain). All inputs/outputs are GPU tensors;
    no D2H, no allocator churn beyond the persistent buffers the caller owns.

    Caller is responsible for:
      1. Sizing ``prefix_csa_indices`` so each token's slice is
         ``prefix_swa_count[t] + csa_valid_k[t]`` long. No ``-1`` pre-fill is
         needed: this kernel writes the SWA prefix at the slice tail and
         ``csa_translate_pack`` writes the CSA topk at the head per layer,
         jointly covering the whole slice.
      2. Computing the four indptr cumsums (e.g. via ``torch.cumsum`` over
         the per-token count vectors).
      3. Computing ``bid_per_token`` (e.g.
         ``torch.repeat_interleave(arange(bs), token_num_per_seq)``).

    Per-seq inputs MUST be indexed by ``bid_per_token`` (the kernel reads
    ``chunk_start_per_seq[bid_per_token[t]]`` etc. inline — no per-token
    pre-gather needed by the caller).

    Args (all GPU tensors):
      positions:                 ``[T]``    int — global token positions.
      bid_per_token:             ``[T]``    int — batch id per token.
      chunk_start_per_seq:       ``[bs]``   int — per-seq chunk start.
      cu_seqlens_q_per_seq:      ``[bs]``   int — per-seq cu_seqlens_q[bid]
                                            (NOT the full ``[bs+1]`` cumsum
                                            — caller passes the leading
                                            ``bs`` entries).
      state_slot_per_seq:        ``[bs]``   int — per-seq SWA ring slot.
      n_committed_hca_per_seq:   ``[bs]``   int — per-seq HCA compress count.
      block_tables:              ``[bs, mnbs]`` int — per-seq paged blocks.
      extend_indptr:             ``[T+1]``  int.
      prefix_swa_indptr:         ``[T+1]``  int.
      prefix_csa_indptr:         ``[T+1]``  int.
      prefix_hca_indptr:         ``[T+1]``  int.
      extend_indices:            ``[ext_total]`` int OUT — fully written.
      prefix_swa_indices:        ``[swa_total]`` int OUT — fully written.
      prefix_csa_indices:        ``[csa_total]`` int OUT — SWA prefix
                                  segment written at the slice TAIL; CSA topk
                                  HEAD section filled per layer by
                                  ``csa_translate_pack``.
      prefix_hca_indices:        ``[hca_total]`` int OUT — fully written.
      T:                         int — token count (grid size).
      win:                       int — SWA window size (per-token SWA cap).
      cs:                        int — ``win + max_spec_steps`` (ring stride
                                  and modulo for paged offset).
      swa_pages:                 int — ``num_slots * cs`` boundary in unified_kv.
    """
    if T == 0:
        return
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert bid_per_token.dim() == 1 and bid_per_token.shape[0] >= T
    assert chunk_start_per_seq.dim() == 1
    assert cu_seqlens_q_per_seq.dim() == 1
    assert state_slot_per_seq.dim() == 1
    assert n_committed_hca_per_seq.dim() == 1
    assert block_tables.dim() == 2
    for idp in (extend_indptr, prefix_swa_indptr, prefix_csa_indptr, prefix_hca_indptr):
        assert idp.dim() == 1 and idp.shape[0] >= T + 1
    for idx in (
        extend_indices,
        prefix_swa_indices,
        prefix_csa_indices,
        prefix_hca_indices,
    ):
        assert idx.dim() == 1

    BLOCK_N = triton.next_power_of_2(win)
    _v4_paged_prefill_indices_kernel[(T,)](
        positions,
        bid_per_token,
        chunk_start_per_seq,
        cu_seqlens_q_per_seq,
        state_slot_per_seq,
        n_committed_hca_per_seq,
        block_tables,
        block_tables.stride(0),
        extend_indptr,
        prefix_swa_indptr,
        prefix_csa_indptr,
        prefix_hca_indptr,
        extend_indices,
        prefix_swa_indices,
        prefix_csa_indices,
        prefix_hca_indices,
        win=win,
        cs=cs,
        swa_pages=swa_pages,
        HCA_RATIO=hca_ratio,
        BLOCK_N=BLOCK_N,
    )


def write_v4_paged_prefill_indices_reference(
    *,
    positions: torch.Tensor,
    bid_per_token: torch.Tensor,
    chunk_start_per_seq: torch.Tensor,
    cu_seqlens_q_per_seq: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    n_committed_hca_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    extend_indptr: torch.Tensor,
    prefix_swa_indptr: torch.Tensor,
    prefix_csa_indptr: torch.Tensor,
    prefix_hca_indptr: torch.Tensor,
    extend_indices: torch.Tensor,
    prefix_swa_indices: torch.Tensor,
    prefix_csa_indices: torch.Tensor,
    prefix_hca_indices: torch.Tensor,
    T: int,
    win: int,
    cs: int,
    swa_pages: int,
    hca_ratio: int = 128,
) -> None:
    """Pure-Python equivalent of ``write_v4_paged_prefill_indices``.
    Per-token Python loop — slow but readable; used for unit-test bit-exact
    verification against the Triton kernel and dump-bisect debugging.

    Same caller contract: the SWA prefix is written to the CSA slice TAIL and
    the CSA topk head is filled per layer by ``csa_translate_pack`` — together
    they cover the whole slice, so no ``-1`` pre-fill is needed.
    """
    if T == 0:
        return
    bid_cpu = bid_per_token[:T].cpu().tolist()
    pos_cpu = positions[:T].cpu().tolist()
    cs_per_seq_cpu = chunk_start_per_seq.cpu().tolist()
    cu_q_cpu = cu_seqlens_q_per_seq.cpu().tolist()
    state_slot_cpu = state_slot_per_seq.cpu().tolist()
    n_hca_cpu = n_committed_hca_per_seq.cpu().tolist()
    block_tables_cpu = block_tables.cpu()
    ext_indptr_cpu = extend_indptr.cpu().tolist()
    swa_indptr_cpu = prefix_swa_indptr.cpu().tolist()
    csa_indptr_cpu = prefix_csa_indptr.cpu().tolist()
    hca_indptr_cpu = prefix_hca_indptr.cpu().tolist()
    device = extend_indices.device

    for t in range(T):
        bid = bid_cpu[t]
        pos = pos_cpu[t]
        chunk_start = cs_per_seq_cpu[bid]
        cu_q = cu_q_cpu[bid]
        state_slot = state_slot_cpu[bid]
        # Per-token causal HCA cap (mirrors kernel + reference get_compress_topk_idxs).
        n_hca = min((pos + 1) // hca_ratio, n_hca_cpu[bid])

        token_pos_in_chunk = pos - chunk_start
        swa_low = max(pos - win + 1, 0)
        extend_count = min(token_pos_in_chunk + 1, win)
        prefix_swa_count = max(chunk_start - swa_low, 0)

        # Extend
        ext_base = ext_indptr_cpu[t]
        ext_start_row = cu_q + token_pos_in_chunk - extend_count + 1
        ext_rows = torch.arange(
            ext_start_row,
            ext_start_row + extend_count,
            device=device,
            dtype=extend_indices.dtype,
        )
        extend_indices[ext_base : ext_base + extend_count] = ext_rows

        # SWA prefix (written to swa / csa / hca prefix buffers)
        sb_swa = swa_indptr_cpu[t]
        sb_hca = hca_indptr_cpu[t]
        if prefix_swa_count > 0:
            global_pos = torch.arange(
                swa_low,
                swa_low + prefix_swa_count,
                device=device,
                dtype=prefix_swa_indices.dtype,
            )
            paged = state_slot * cs + (global_pos % cs)
            prefix_swa_indices[sb_swa : sb_swa + prefix_swa_count] = paged
            # CSA: SWA prefix at the slice TAIL (head holds the CSA topk section
            # filled by csa_translate_pack). See the kernel comment above.
            csa_end = csa_indptr_cpu[t + 1]
            prefix_csa_indices[csa_end - prefix_swa_count : csa_end] = paged
            prefix_hca_indices[sb_hca : sb_hca + prefix_swa_count] = paged

        # HCA compress
        if n_hca > 0:
            bt = block_tables_cpu[bid, :n_hca].to(device).to(prefix_hca_indices.dtype)
            hca_dst = sb_hca + prefix_swa_count
            prefix_hca_indices[hca_dst : hca_dst + n_hca] = swa_pages + bt
