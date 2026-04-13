# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused Triton kernels for MoE routing metadata computation.

Replaces the 14-kernel chain (pack_bitmatrix, sum_bitmatrix_rows,
bitmatrix_stage1/2, ragged_meta_memset/compute, copy, index, div_trunc,
etc.) with a 2-kernel pipeline for decode workloads (M <= 64):

  Kernel 1: Sort topk assignments by expert, compute histogram,
            build gather/scatter indices, reorder gate weights,
            compute prefix sum (slice_offs).

  Kernel 2: For each block_m in {16,32,64,128}, compute the
            padded prefix sums (block_offs) and block schedule
            (block_schedule / block_pid_map) needed by matmul_ogs.

The output data structures (RoutingData, GatherIndx, ScatterIndx)
are identical to those produced by the baseline make_routing_data()
path, ensuring drop-in compatibility with matmul_ogs.

Terminology (matching triton_kernels):
  col_sorted_indx / combine_indx:
      combine[sorted_pos] = original_pair_index
      Used as GatherIndx.src_indx, ScatterIndx.dst_indx
  row_sorted_indx / dispatch_indx:
      dispatch[original_pair] = sorted_pos
      Used as GatherIndx.dst_indx, ScatterIndx.src_indx
"""

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# Kernel 1: Sort + histogram + gather/scatter indices + prefix sum
# ---------------------------------------------------------------------------
@triton.jit
def _fused_routing_kernel1(
    # Inputs
    topk_ids_ptr,        # [P_ACTUAL] int16 - flattened topk expert IDs
    topk_weights_ptr,    # [P_ACTUAL] bf16 - flattened topk weights
    P_actual,            # actual number of valid pairs (non-constexpr)
    # Outputs
    gate_scal_ptr,       # [P_ACTUAL] bf16 - gate weights in sorted order
    combine_ptr,         # [P_ACTUAL] int32 - col_sorted_indx
    dispatch_ptr,        # [P_ACTUAL] int32 - row_sorted_indx
    hist_ptr,            # [E] int32 - tokens per expert
    slice_offs_ptr,      # [E+1] int32 - prefix sum of histogram
    # Constants
    P_val: tl.constexpr,          # padded P (power of 2)
    N_EXPERTS_val: tl.constexpr,  # number of experts (power of 2)
    EP1_PAD: tl.constexpr,        # next power of 2 >= E+1
):
    """
    Single-CTA kernel that computes sorting, histogram, and index metadata.

    Uses tl.sort for O(P log P) vectorized sorting instead of O(P^2) loops.

    For each original pair p (0..P_actual-1):
      dispatch[p] = sorted position of pair p within the expert-sorted order
      combine[sorted_pos] = original pair index at that sorted position

    gate_scal[sorted_pos] = topk_weights[combine[sorted_pos]]
                          = weight of the pair at that sorted position
    """
    offs_p = tl.arange(0, P_val)
    offs_e = tl.arange(0, N_EXPERTS_val)

    # Load expert IDs
    ids = tl.load(topk_ids_ptr + offs_p, mask=offs_p < P_actual,
                  other=-1).to(tl.int32)

    # --- Step 1: Sort by expert ID using tl.sort ---
    # Pack (expert_id, pair_index) into a single uint32 for sorting.
    # Use expert_id in upper bits for primary sort key,
    # pair_index in lower bits for stable sort (tie-breaking).
    # Invalid entries (expert_id == -1) get max key to sort to the end.
    is_valid = ids >= 0
    # Map expert_id to sort-friendly value: valid ids -> expert_id,
    # invalid -> 0xFFFF (sorts to end)
    safe_eid = tl.where(is_valid, ids, 0xFFFF)
    sort_keys = (safe_eid.to(tl.uint32) << 16) | offs_p.to(tl.uint32)
    sorted_keys = tl.sort(sort_keys, 0)

    # Extract sorted expert IDs and original pair indices
    sorted_eid = (sorted_keys >> 16).to(tl.int32)
    sorted_orig_idx = (sorted_keys & 0xFFFF).to(tl.int32)

    # combine_indx[sorted_pos] = original_pair_index
    # Store all P_pad entries (including padding) so the dispatch scatter
    # loop below can safely read combine_ptr[sp] for all sp in 0..P_pad-1.
    tl.store(combine_ptr + offs_p, sorted_orig_idx)

    # dispatch_indx[original_pair] = sorted_pos (inverse permutation)
    # Write dispatch by scattering: for each sorted position sp,
    # dispatch[sorted_orig_idx[sp]] = sp
    for sp in tl.static_range(0, P_val):
        orig_idx = tl.load(combine_ptr + sp)
        tl.store(dispatch_ptr + orig_idx, sp)

    # --- Step 2: Compute histogram from sorted order ---
    # Count how many valid pairs map to each expert.
    # Use vectorized comparison against expert IDs.
    hist = tl.zeros((N_EXPERTS_val,), dtype=tl.int32)
    for p in tl.static_range(0, P_val):
        eid = tl.load(topk_ids_ptr + p).to(tl.int32)
        valid = (eid >= 0)
        match = tl.where((offs_e == eid) & valid, 1, 0)
        hist += match

    tl.store(hist_ptr + offs_e, hist, mask=offs_e < N_EXPERTS_val)

    # --- Step 3: Prefix sum of histogram -> slice_offs ---
    # Use exclusive cumsum: slice_offs[e] = sum(hist[0:e])
    # Vectorized approach: compute cumsum of hist, then shift right by 1.
    cum = tl.cumsum(hist, 0)  # cum[e] = sum(hist[0:e+1]) (inclusive)
    # exclusive_sum[e] = cum[e] - hist[e] = sum(hist[0:e])
    exc_sum = cum - hist

    # Store: slice_offs[e] = exc_sum[e] for e in 0..N_EXPERTS-1
    # slice_offs[N_EXPERTS] = cum[N_EXPERTS-1] = total
    tl.store(slice_offs_ptr + offs_e, exc_sum, mask=offs_e < N_EXPERTS_val)
    # Store the total at position N_EXPERTS_val
    # Total = last element of cum
    total = tl.sum(hist, 0)
    if N_EXPERTS_val > 0:
        tl.store(slice_offs_ptr + N_EXPERTS_val, total)

    # --- Step 4: Build gate_scal in sorted order ---
    # gate_scal[sorted_pos] = topk_weights[combine[sorted_pos]]
    weights = tl.load(topk_weights_ptr + offs_p, mask=offs_p < P_actual,
                      other=-1.0)
    # Gather weights by sorted_orig_idx
    for sp in tl.static_range(0, P_val):
        orig = tl.load(combine_ptr + sp)
        w = tl.load(topk_weights_ptr + orig)
        tl.store(gate_scal_ptr + sp, w)


# ---------------------------------------------------------------------------
# Kernel 2: Build RaggedTensorMetadata (block_offs + block_schedule)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_routing_kernel2(
    # Inputs
    hist_ptr,            # [E] int32 - tokens per expert
    # Outputs
    block_offs_ptr,      # [N_BLOCK_SIZES, EP1_PAD] int32
    block_schedule_ptr,  # [N_BLOCK_SIZES, MAX_TILES] int32
    # Strides
    block_offs_stride_m,     # stride between block_offs rows
    block_schedule_stride_m, # stride between block_schedule rows
    # Constants
    N_EXPERTS_val: tl.constexpr,
    EP1_PAD: tl.constexpr,
    MAX_TILES: tl.constexpr,
    FIRST_BLK_LOG2: tl.constexpr,  # log2(16) = 4
    TILE_LIMIT: tl.constexpr,      # max tiles per expert
):
    """
    One CTA per block_m size. Computes padded prefix sums and block schedule.

    For each block_m in {16, 32, 64, 128}:
      block_offs[bm_idx, e] = sum_{j<e} ceil(hist[j] / block_m)
      block_schedule[bm_idx, tile] = (block_within_expert << 16) | expert_id
    """
    bm_idx = tl.program_id(0)
    block_m_log2 = FIRST_BLK_LOG2 + bm_idx

    offs_ep1 = tl.arange(0, EP1_PAD)

    # Compute block_offs: prefix sum of ceil(hist[e] / block_m)
    block_offs_vals = tl.zeros((EP1_PAD,), dtype=tl.int32)
    running_blocks = 0

    for e in tl.static_range(0, N_EXPERTS_val):
        h = tl.load(hist_ptr + e)
        block_offs_vals = tl.where(offs_ep1 == e, running_blocks,
                                   block_offs_vals)
        n_tiles = (h + ((1 << block_m_log2) - 1)) >> block_m_log2
        running_blocks = running_blocks + n_tiles
    block_offs_vals = tl.where(offs_ep1 == N_EXPERTS_val, running_blocks,
                               block_offs_vals)

    # Store block_offs
    base_offs = bm_idx * block_offs_stride_m
    tl.store(block_offs_ptr + base_offs + offs_ep1, block_offs_vals,
             mask=offs_ep1 <= N_EXPERTS_val)

    # Initialize block_schedule to -1 (sentinel)
    sched_base = bm_idx * block_schedule_stride_m
    offs_tiles = tl.arange(0, MAX_TILES)
    tl.store(block_schedule_ptr + sched_base + offs_tiles,
             tl.full((MAX_TILES,), -1, dtype=tl.int32),
             mask=offs_tiles < MAX_TILES)

    # Fill block_schedule entries for each expert
    for e in tl.static_range(0, N_EXPERTS_val):
        h = tl.load(hist_ptr + e)
        n_tiles = (h + ((1 << block_m_log2) - 1)) >> block_m_log2
        tile_start = tl.load(block_offs_ptr + base_offs + e)
        for b in tl.static_range(0, TILE_LIMIT):
            if b < n_tiles:
                val = (b << 16) + e
                tl.store(
                    block_schedule_ptr + sched_base + tile_start + b, val
                )


# ---------------------------------------------------------------------------
# Python dispatch: build RoutingData, GatherIndx, ScatterIndx
# ---------------------------------------------------------------------------

# Maximum M for which the fused routing path is used
FUSED_ROUTING_M_THRESHOLD = 64

# The triton_kernels RaggedTensorMetadata block sizes
_BLOCK_SIZES = [16, 32, 64, 128]
_N_BLOCK_SIZES = len(_BLOCK_SIZES)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def _max_n_tiles(n_slices: int, n_total_rows: int) -> int:
    """
    Compute the maximum number of tiles across all block_m sizes.
    Matches RaggedTensorMetadata.max_n_tiles().
    """
    if n_total_rows <= n_slices:
        return n_total_rows
    min_block = min(_BLOCK_SIZES)
    return n_slices - 1 - ((n_slices - n_total_rows - 1) // min_block)


def fused_make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
):
    """
    Fused replacement for make_routing_data() that computes all MoE
    routing metadata in 2 kernel launches instead of 14.

    Args:
        topk_ids: [M, top_k] int16 tensor of expert assignments.
                  -1 indicates invalid/padding.
        topk_weights: [M, top_k] bf16 tensor of gate weights.
                      -1 at positions where topk_ids == -1.
        num_local_experts: Total number of experts (E).

    Returns:
        Tuple of (RoutingData, GatherIndx, ScatterIndx) with the same
        format as the baseline make_routing_data().
    """
    # Lazy import to avoid circular deps and missing package errors.
    # triton_kernels is guaranteed available when this code is called
    # (gated by has_triton_kernels() in gpt_oss_triton_kernels_moe.py).
    from triton_kernels.matmul_ogs import (
        GatherIndx,
        RoutingData,
        ScatterIndx,
    )
    from triton_kernels.tensor import RaggedTensorMetadata

    device = topk_ids.device
    n_rows, num_topk = topk_ids.shape
    P = n_rows * num_topk
    E = num_local_experts

    assert P <= 0xFFFF, (
        f"Fused routing supports at most 65535 pairs, got {P}. "
        f"Increase FUSED_ROUTING_M_THRESHOLD check or use baseline path."
    )

    # Flatten inputs
    topk_ids_flat = topk_ids.reshape(-1).to(torch.int16)
    # matmul_ogs expects invalid topk_weights to be -1s
    topk_weights_flat = torch.where(
        topk_ids.reshape(-1) == -1,
        torch.tensor(-1.0, dtype=torch.bfloat16, device=device),
        topk_weights.reshape(-1).to(torch.bfloat16),
    )

    # Power-of-2 sizes for Triton constexpr (tl.arange requirement).
    # P_pad must be >= 2 because tl.sort requires at least 2 elements.
    P_pad = _next_power_of_2(max(P, 2))
    E_pad = _next_power_of_2(max(E, 1))
    EP1_PAD = _next_power_of_2(E + 1)

    # Pad inputs if needed
    if P < P_pad:
        pad_size = P_pad - P
        topk_ids_flat = torch.cat([
            topk_ids_flat,
            torch.full((pad_size,), -1, dtype=torch.int16, device=device),
        ])
        topk_weights_flat = torch.cat([
            topk_weights_flat,
            torch.full((pad_size,), -1.0, dtype=torch.bfloat16,
                        device=device),
        ])

    # Allocate outputs for kernel 1
    gate_scal = torch.empty(P_pad, dtype=torch.bfloat16, device=device)
    combine_indx = torch.empty(P_pad, dtype=torch.int32, device=device)
    dispatch_indx = torch.empty(P_pad, dtype=torch.int32, device=device)
    hist = torch.zeros(E_pad, dtype=torch.int32, device=device)
    slice_offs = torch.zeros(EP1_PAD, dtype=torch.int32, device=device)

    # ---- Kernel 1: sort + hist + indices + prefix sum ----
    _fused_routing_kernel1[(1,)](
        topk_ids_flat, topk_weights_flat,
        P,  # actual P (non-constexpr)
        gate_scal, combine_indx, dispatch_indx,
        hist, slice_offs,
        P_val=P_pad,
        N_EXPERTS_val=E_pad,
        EP1_PAD=EP1_PAD,
    )

    # Trim to actual sizes
    gate_scal = gate_scal[:P].contiguous()
    combine_indx = combine_indx[:P].contiguous()
    dispatch_indx = dispatch_indx[:P].contiguous()
    hist_actual = hist[:E].contiguous()

    # ---- Kernel 2: RaggedTensorMetadata ----
    max_n_tiles = _max_n_tiles(E, P)
    max_n_tiles_pad = _next_power_of_2(max(max_n_tiles, 1))

    # Upper bound on tiles per expert (for the inner static_range).
    # For decode: P_max = 512 (M=64, top_k=8), block_m_min = 16
    # -> max tiles per expert = ceil(512/16) = 32.
    # Use 64 for safety margin.
    tile_limit = 64

    assert P <= tile_limit * min(_BLOCK_SIZES), (
        f"tile_limit={tile_limit} too small for P={P} pairs "
        f"(max supported: {tile_limit * min(_BLOCK_SIZES)})"
    )

    block_offs_data = torch.zeros(
        _N_BLOCK_SIZES, EP1_PAD, dtype=torch.int32, device=device
    )
    block_schedule_data = torch.full(
        (_N_BLOCK_SIZES, max_n_tiles_pad),
        -1, dtype=torch.int32, device=device,
    )

    _fused_routing_kernel2[(_N_BLOCK_SIZES,)](
        hist,  # Use E_pad version; extra entries are 0
        block_offs_data,
        block_schedule_data,
        block_offs_data.stride(0),
        block_schedule_data.stride(0),
        N_EXPERTS_val=E_pad,
        EP1_PAD=EP1_PAD,
        MAX_TILES=max_n_tiles_pad,
        FIRST_BLK_LOG2=4,
        TILE_LIMIT=tile_limit,
    )

    # Trim to required shapes
    block_offs_trimmed = block_offs_data[:, :E + 1].contiguous()
    block_schedule_trimmed = (
        block_schedule_data[:, :max_n_tiles].contiguous()
    )
    slice_offs_trimmed = slice_offs[:E + 1].contiguous()

    # Build RaggedTensorMetadata
    ragged_metadata = RaggedTensorMetadata(
        slice_sizes=hist_actual,
        slice_offs=slice_offs_trimmed,
        block_offs_data=block_offs_trimmed,
        block_schedule_data=block_schedule_trimmed,
    )

    # Build output structures.
    # NOTE: The baseline legacy_routing_from_bitmatrix sets expt_hist to
    # ragged_metadata.block_sizes (a static method/function), not the
    # actual histogram tensor. This is because matmul_ogs only checks
    # `expt_hist is not None` and uses expt_data.slice_sizes for the
    # real histogram. We match this behavior exactly.
    routing_data = RoutingData(
        gate_scal=gate_scal,
        expt_hist=ragged_metadata.block_sizes,
        n_expts_tot=E,
        n_expts_act=num_topk,
        expt_data=ragged_metadata,
    )

    # GatherIndx: src_indx=combine_indx, dst_indx=dispatch_indx
    # Performs Y = X[combine_indx, :] (gather from original order to sorted)
    gather_indx = GatherIndx(
        src_indx=combine_indx,
        dst_indx=dispatch_indx,
    )

    # ScatterIndx: src_indx=dispatch_indx, dst_indx=combine_indx
    # Performs Y[combine_indx, :] = X (scatter from sorted order to original)
    scatter_indx = ScatterIndx(
        src_indx=dispatch_indx,
        dst_indx=combine_indx,
    )

    return routing_data, gather_indx, scatter_indx
