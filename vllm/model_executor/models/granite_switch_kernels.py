# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWITCH: the fused switched-LoRA kernel backend for Granite Switch.

Granite Switch checkpoints carry their LoRA adapters baked in, and select one
per token from a control-token signal. SWITCH is the kernel backend that makes
that selection cheap. The name is the design:

    SWITCH = Switched-delta . W-ext-fused . In-place . Tiled
             . Conditional-skip . Heterogeneous-rank

  * Switched-delta     - each token selects exactly one adapter's lora_B delta.
  * W-ext-fused        - the base projection and every adapter's lora_A (shrink)
                         are fused into one wide GEMM (x @ w_ext.T): the static,
                         shared substrate, NOT the switched part.
  * In-place           - the switched delta is accumulated in place into the base
                         columns of x_ext (base and shrink columns are disjoint,
                         so the read-modify-write is hazard-free).
  * Tiled              - block-tiled expand (BLOCK_M tokens x BLOCK_N cols/CTA).
  * Conditional-skip   - per-tile bitmask early-exit when no adapter is present.
  * Heterogeneous-rank - per-adapter / per-module rank tiers (16 .. 512).

Public surface
--------------
build_w_ext(W, lora_A_by_rank)
    Build the extended weight matrix once at load time: the base weight rows
    followed by every adapter's ``lora_A`` (shrink) rows, grouped by rank tier.
    A single GEMM ``x @ w_ext.T`` then produces the base projection in the
    leading ``N_total`` columns and all per-adapter shrink vectors after it.

torch.ops.vllm.granite_switch_lora_expand
    Read each active adapter's shrink columns from ``x_ext``, multiply by its
    packed ``lora_B``, and accumulate the delta in place into the base columns
    ``x_ext[:, :N]``.

torch.ops.vllm.granite_switch_lora_expand_swiglu
    The same expand for a merged gate/up projection, with the SwiGLU activation
    fused into the epilogue so the corrected ``[M, 2H]`` is never materialized.

torch.ops.vllm.granite_switch_lora_shrink_expand
    Shrink-only ("W-less") expand with no base weight, for the Shadow-Residual
    cross-stream shunt.

torch.ops.vllm.granite_switch_compute_per_module_bitmasks
    Per-module, per-tile int64 occupancy bitmasks driving the conditional skip.

FusedLoRAKernelMeta / SRFusedLoRAKernelMeta
    Compute the bitmasks and the per-module kernel-local index vectors once per
    forward and publish them on a shared ``LoRAContext``.

get_switch_lora_expand_config()
    Triton launch parameters (tiling + warps/stages), in the style of vLLM's
    ``get_lora_op_configs`` - a single source of truth returning a dict.

Supported rank tiers: 16, 32, 64, 128, 256, 512. Tiers absent from a
deployment compile to dead code (``tl.static_range(0)`` emits no instructions).

Adapter index convention
------------------------
The ``adapter_indices`` these kernels read are **per-module, kernel-local**
indices, not global adapter ids. ``FusedLoRAKernelMeta`` remaps global ids to
each module's local numbering centrally (per-module ``remap_table``) before
launch: a global adapter that does not apply to this module maps to 0, and the
applicable ones are renumbered 1.. within the module. So each module
independently numbers only the adapters that touch it, grouped by ascending
rank tier:

    0                            base (no LoRA), or adapter not applicable here
    1 .. n_16                    this module's tier-16 adapters
    n_16+1 .. n_16+n_32          this module's tier-32 adapters
    ...

where (n_16, n_32, ..., n_512) are this module's local per-tier counts. Nothing
requires an adapter to be present in every module or to use the same rank across
modules - the per-module remap is what lets ranks and adapter membership vary.

Block sizes
-----------
BLOCK_M and BLOCK_N are the defaults returned by
``get_switch_lora_expand_config`` (currently 32x32), not immutable - the
launcher takes its tiling from that config rather than hardcoding it. The
kernels do NOT use ``@triton.autotune`` (incompatible with torch.compile).
BLOCK_M must match the bitmask tiling in ``_bitmask_reduce_kernel`` below (both
read the same constant). ``block_n`` is a per-launch argument bound at load
time (it determines the precomputed tile/slice tables); it defaults to BLOCK_N
but may differ, and must divide every output slice so no tile straddles a slice
boundary.
"""

import torch
from torch import nn

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

SUPPORTED_RANKS = (16, 32, 64, 128, 256, 512)


def promote_rank(rank: int) -> int:
    """Round a LoRA rank up to the next rank the kernel supports.

    The tier machinery is fixed at six tiers (``slice_col_r`` is built as
    ``[S, 6]`` and ``_na`` as a 6-tuple), so an off-tier rank cannot simply be
    added to SUPPORTED_RANKS. Callers instead promote the rank and zero-pad the
    adapter's lora_A rows / lora_B columns up to it, which is numerically exact:
    the padded rows contribute nothing to the shrink and their lora_B columns
    contribute nothing to the expand.

    Rank 0 (or negative, meaning "not applicable") is returned unchanged.
    """
    if rank <= 0:
        return rank
    for supported in SUPPORTED_RANKS:
        if supported >= rank:
            return supported
    raise ValueError(
        f"LoRA rank {rank} exceeds the largest supported rank {SUPPORTED_RANKS[-1]}."
    )


# Default tiling/launch constants (returned by get_switch_lora_expand_config).
# Validated across 3B/8B prefill+decode in the block-tuning study; no usable
# gain was found from other values.
BLOCK_M: int = 32
BLOCK_N: int = 32
NUM_WARPS: int = 4
NUM_STAGES: int = 1


def get_switch_lora_expand_config() -> dict:
    """Return the expand kernel's tiling/launch parameters.

    Mirrors vLLM's ``get_lora_op_configs`` convention: one place returns the
    kernel config as a dict, and the launcher passes the values as
    ``tl.constexpr``. Currently fixed defaults; a shape-keyed tuned-config
    lookup can be added here later without touching call sites.
    """
    return {
        "block_m": BLOCK_M,
        "block_n": BLOCK_N,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
    }


# ---------------------------------------------------------------------------
# Load-time: build the extended weight matrix
# ---------------------------------------------------------------------------


def build_w_ext(
    W: torch.Tensor,
    lora_A_by_rank: dict,
) -> torch.Tensor:
    """Build W_ext by stacking W with lora_A rows, ordered by ascending rank.

    Args:
        W: ``[N_total, K]`` base weight.
        lora_A_by_rank: ``{rank: Tensor [n_r, S, rank, K]}``, where ``S`` is the
            number of output slices (1 for single-slice layers). Only ranks in
            ``SUPPORTED_RANKS`` are accepted. Rows are appended in
            tier -> adapter -> slice order: for each tier ``r``, for each
            adapter ``a``, ``S * r`` rows, namely ``lora_A[a, 0]`` (slice 0),
            then ``lora_A[a, 1]`` (slice 1), and so on.

    Returns:
        ``[N_total + sum_r(n_r * S * r), K]``, base rows first.

    """
    parts = [W]
    for r in SUPPORTED_RANKS:
        lA = lora_A_by_rank.get(r)
        if lA is not None and lA.shape[0] > 0:
            n_r, S = lA.shape[0], lA.shape[1]
            # [n_r, S, r, K] -> [n_r * S * r, K] in adapter->slice->row order
            parts.append(lA.reshape(n_r * S * r, lA.shape[3]))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Expand kernel
# ---------------------------------------------------------------------------


@triton.jit
def _switch_lora_expand_kernel(
    # x_ext  [M, N_total + sum_r(n_r * S * r)] - base output is cols [0, N),
    # accumulated into in place; shrink cols (>= N) are read. One buffer over
    # disjoint cols is hazard-free, keeps the inductor graph free of
    # clone + slice_scatter, and saves a pointer arg per launch.
    XExt,
    stride_xe_m,
    stride_xe_n,
    # adapter_indices  [M] - kernel-local (post-remap) ids: 0 = base, 1.. = this
    # module's adapters in ascending-rank order.
    AdapIdx,
    # bitmask  [num_tiles_M] int64 - bit a set iff kernel-local adapter (a+1)
    # is present in the tile (computed from the same post-remap ids).
    Bitmask,
    stride_bm_t,
    # SINGLE packed lora_B: contiguous concat over present tiers of
    # [NA_r, N_total, r] (row-major). Tier r block has strides (N*r, r, 1); its
    # base element offset is cumsum_{r'<r}(NA_r' * N * r'), computed inline below.
    LBPACK,
    M,
    N,
    # per-tile slice lookup
    TileSlice,
    SliceColR,
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Unchecked invariant: x_ext's M equals the token count AdapIdx and Bitmask
    # were built from. Routing is per token and the decoder preserves the count,
    # so the loads below are unmasked against their own length; breaking that
    # contract (M_fwd > M_prepare) would read out of bounds.

    # One bit per adapter for this row-tile; bit a set iff adapter (a+1) is
    # present somewhere in the tile. All-zero tile touches no adapter -> skip.
    bitmask = tl.load(Bitmask + pid_m * stride_bm_t).to(tl.int64)
    if bitmask == 0:
        return

    pid_n = tl.program_id(1)
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_range < M
    mask_n = n_range < N

    # This output tile lives in exactly one slice (block_n divides every slice).
    # SliceColR[s] gives, per tier, the first shrink column of that slice's
    # adapters in x_ext - the per-adapter shrink block starts at col_<r> + a*S*r.
    s_idx = tl.load(TileSlice + pid_n).to(tl.int32)
    col_16 = tl.load(SliceColR + s_idx * 6 + 0).to(tl.int32)
    col_32 = tl.load(SliceColR + s_idx * 6 + 1).to(tl.int32)
    col_64 = tl.load(SliceColR + s_idx * 6 + 2).to(tl.int32)
    col_128 = tl.load(SliceColR + s_idx * 6 + 3).to(tl.int32)
    col_256 = tl.load(SliceColR + s_idx * 6 + 4).to(tl.int32)
    col_512 = tl.load(SliceColR + s_idx * 6 + 5).to(tl.int32)

    adap_ids = tl.load(AdapIdx + m_range, mask=mask_m, other=0).to(tl.int32)
    delta = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row_active = adap_ids != 0

    # Kernel-local ids are numbered 1.. across tiers in ascending-rank order, so
    # each tier's ids start after all lower tiers. These offsets convert a
    # tier-local index a to the kernel-local id (and its bit in this module's
    # bitmask).
    id_off_32 = NA_16
    id_off_64 = NA_16 + NA_32
    id_off_128 = NA_16 + NA_32 + NA_64
    id_off_256 = NA_16 + NA_32 + NA_64 + NA_128
    id_off_512 = NA_16 + NA_32 + NA_64 + NA_128 + NA_256

    # Element offset of each tier's block inside the single packed lora_B buffer
    # (tiers concatenated in rank order): cumsum(NA_r * N * r).
    lb_off_16 = 0
    lb_off_32 = NA_16 * N * 16
    lb_off_64 = lb_off_32 + NA_32 * N * 32
    lb_off_128 = lb_off_64 + NA_64 * N * 64
    lb_off_256 = lb_off_128 + NA_128 * N * 128
    lb_off_512 = lb_off_256 + NA_256 * N * 256

    # Per tier, loop over that tier's adapters, bit-testing to skip those absent
    # from the tile; for each active one accumulate shrink @ lora_B into the
    # delta. static_range unrolls at compile time, so absent tiers emit no code.
    # Only the tier-16 loop is annotated; 32..512 are identical.
    r16 = tl.arange(0, 16)
    for a in tl.static_range(NA_16):
        if (bitmask >> a) & 1:  # adapter (a+1) in this tile?
            mask_a = (adap_ids == (a + 1)) & mask_m  # rows that select it
            # shrink: this adapter's r shrink columns for the tile's rows.
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_16 + a * S * 16 + r16[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            # lb: this adapter's lora_B tile [r, BLOCK_N] (rank x out-col);
            # rank is contiguous, each out-col steps by r (layout above).
            lb = tl.load(
                LBPACK
                + lb_off_16
                + a * (N * 16)
                + r16[:, None]
                + n_range[None, :] * 16,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r32 = tl.arange(0, 32)
    for a in tl.static_range(NA_32):
        if (bitmask >> (id_off_32 + a)) & 1:
            mask_a = (adap_ids == (id_off_32 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_32 + a * S * 32 + r32[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_32
                + a * (N * 32)
                + r32[:, None]
                + n_range[None, :] * 32,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r64 = tl.arange(0, 64)
    for a in tl.static_range(NA_64):
        if (bitmask >> (id_off_64 + a)) & 1:
            mask_a = (adap_ids == (id_off_64 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_64 + a * S * 64 + r64[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_64
                + a * (N * 64)
                + r64[:, None]
                + n_range[None, :] * 64,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r128 = tl.arange(0, 128)
    for a in tl.static_range(NA_128):
        if (bitmask >> (id_off_128 + a)) & 1:
            mask_a = (adap_ids == (id_off_128 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_128 + a * S * 128 + r128[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_128
                + a * (N * 128)
                + r128[:, None]
                + n_range[None, :] * 128,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r256 = tl.arange(0, 256)
    for a in tl.static_range(NA_256):
        if (bitmask >> (id_off_256 + a)) & 1:
            mask_a = (adap_ids == (id_off_256 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_256 + a * S * 256 + r256[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_256
                + a * (N * 256)
                + r256[:, None]
                + n_range[None, :] * 256,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r512 = tl.arange(0, 512)
    for a in tl.static_range(NA_512):
        if (bitmask >> (id_off_512 + a)) & 1:
            mask_a = (adap_ids == (id_off_512 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_512 + a * S * 512 + r512[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_512
                + a * (N * 512)
                + r512[:, None]
                + n_range[None, :] * 512,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    # Fold the delta back into the base columns [0, N) of x_ext in place. Only
    # adapter rows are written (row_active); base-only rows already hold the
    # correct projection. Base and shrink columns are disjoint, so this is safe.
    out_ptrs = XExt + m_range[:, None] * stride_xe_m + n_range[None, :] * stride_xe_n
    store_mask = row_active[:, None] & mask_n[None, :]
    existing = tl.load(out_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    tl.store(out_ptrs, (existing + delta).to(XExt.dtype.element_ty), mask=store_mask)


def _granite_switch_lora_expand(
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    """Accumulate the LoRA expand delta in place into ``x_ext[:, :N]``.

    The per-tier adapter counts are passed as six scalars rather than a tuple
    because ``torch.library`` schemas cannot express a fixed-length int tuple.
    """
    M = x_ext.shape[0]
    cfg = get_switch_lora_expand_config()
    grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(N, block_n))
    _switch_lora_expand_kernel[grid](
        x_ext,
        x_ext.stride(0),
        x_ext.stride(1),
        adapter_indices,
        bitmask,
        bitmask.stride(0),
        lb_packed,
        M,
        N,
        tile_slice,
        slice_col_r,
        NA_16=NA_16,
        NA_32=NA_32,
        NA_64=NA_64,
        NA_128=NA_128,
        NA_256=NA_256,
        NA_512=NA_512,
        S=S,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=block_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def _granite_switch_lora_expand_fake(
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    # Pure mutation of ``x_ext`` in place, no allocation and no shape change, so
    # the meta kernel has nothing to declare. ``x_ext`` is already in
    # ``mutates_args``, which is what tells the functionalization pass to clone.
    return


direct_register_custom_op(
    op_name="granite_switch_lora_expand",
    op_func=_granite_switch_lora_expand,
    mutates_args=["x_ext"],
    fake_impl=_granite_switch_lora_expand_fake,
)
granite_switch_lora_expand = torch.ops.vllm.granite_switch_lora_expand


# ---------------------------------------------------------------------------
# Fused gate/up expand + SwiGLU (shared-MLP first projection ONLY)
# ---------------------------------------------------------------------------
#
# The merged gate/up projection is the one module whose output feeds a
# packed-layout kernel (vLLM SiluAndMul). Instead of materializing the corrected
# [M, 2H] gate|up to HBM, re-reading it in a separate activation kernel and
# paying a .contiguous() for x_ext's strided base slice, this kernel applies the
# delta to the gate and up columns and writes silu(gate)*up directly as a
# contiguous [M, H]. The bitmask gates only the delta work; the store always
# runs, since base-only tiles still need their activation written.


@triton.jit
def _accum_slice_delta(
    XExt,
    stride_xe_m,
    stride_xe_n,
    LBPACK,
    m_range,
    mask_m,
    adap_ids,
    n_range,
    mask_n,
    bitmask,
    col_16,
    col_32,
    col_64,
    col_128,
    col_256,
    col_512,
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """LoRA delta [BLOCK_M, BLOCK_N] for ONE output slice.

    Same tier structure as the in-place expand kernel, parameterized by the
    slice's shrink-column bases (col_*) and the output-column range (n_range),
    so it is called once for the gate slice and once for the up slice. Caller
    guarantees bitmask != 0.
    """
    delta = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tier-local index -> kernel-local id / bit position (see expand kernel).
    id_off_32 = NA_16
    id_off_64 = NA_16 + NA_32
    id_off_128 = NA_16 + NA_32 + NA_64
    id_off_256 = NA_16 + NA_32 + NA_64 + NA_128
    id_off_512 = NA_16 + NA_32 + NA_64 + NA_128 + NA_256

    # Element offset of each tier's block in the packed lora_B buffer.
    lb_off_16 = 0
    lb_off_32 = NA_16 * N * 16
    lb_off_64 = lb_off_32 + NA_32 * N * 32
    lb_off_128 = lb_off_64 + NA_64 * N * 64
    lb_off_256 = lb_off_128 + NA_128 * N * 128
    lb_off_512 = lb_off_256 + NA_256 * N * 256

    # Same per-tier accumulation as the in-place expand kernel; see there for the
    # annotated tier-16 loop. Tiers 32..512 repeat the pattern.
    r16 = tl.arange(0, 16)
    for a in tl.static_range(NA_16):
        if (bitmask >> a) & 1:
            mask_a = (adap_ids == (a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_16 + a * S * 16 + r16[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_16
                + a * (N * 16)
                + r16[:, None]
                + n_range[None, :] * 16,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r32 = tl.arange(0, 32)
    for a in tl.static_range(NA_32):
        if (bitmask >> (id_off_32 + a)) & 1:
            mask_a = (adap_ids == (id_off_32 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_32 + a * S * 32 + r32[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_32
                + a * (N * 32)
                + r32[:, None]
                + n_range[None, :] * 32,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r64 = tl.arange(0, 64)
    for a in tl.static_range(NA_64):
        if (bitmask >> (id_off_64 + a)) & 1:
            mask_a = (adap_ids == (id_off_64 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_64 + a * S * 64 + r64[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_64
                + a * (N * 64)
                + r64[:, None]
                + n_range[None, :] * 64,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r128 = tl.arange(0, 128)
    for a in tl.static_range(NA_128):
        if (bitmask >> (id_off_128 + a)) & 1:
            mask_a = (adap_ids == (id_off_128 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_128 + a * S * 128 + r128[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_128
                + a * (N * 128)
                + r128[:, None]
                + n_range[None, :] * 128,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r256 = tl.arange(0, 256)
    for a in tl.static_range(NA_256):
        if (bitmask >> (id_off_256 + a)) & 1:
            mask_a = (adap_ids == (id_off_256 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_256 + a * S * 256 + r256[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_256
                + a * (N * 256)
                + r256[:, None]
                + n_range[None, :] * 256,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r512 = tl.arange(0, 512)
    for a in tl.static_range(NA_512):
        if (bitmask >> (id_off_512 + a)) & 1:
            mask_a = (adap_ids == (id_off_512 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_512 + a * S * 512 + r512[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_512
                + a * (N * 512)
                + r512[:, None]
                + n_range[None, :] * 512,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    return delta


@triton.jit
def _switch_lora_expand_swiglu_kernel(
    # x_ext  [M, 2H + shrink]: gate base = cols [0,H), up base = cols [H,2H),
    # shrink cols follow. Read-only here.
    XExt,
    stride_xe_m,
    stride_xe_n,
    # out  [M, H] contiguous: silu(gate)*up
    Out,
    stride_o_m,
    stride_o_n,
    AdapIdx,
    Bitmask,
    stride_bm_t,
    LBPACK,
    M,
    H,
    N,  # H = gate width; N = N_total = 2H = lora_B output-dim count
    SliceColR,  # [S, 6] int32 (S == 2: row 0 = gate, row 1 = up)
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Same by-design invariant as _switch_lora_expand_kernel: M (x_ext rows)
    # equals the token count behind AdapIdx/Bitmask, because routing is per token
    # and the decoder preserves token count. The unmasked Bitmask load relies on
    # it.
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h_range = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_range < M
    mask_h = h_range < H
    base_mask = mask_m[:, None] & mask_h[None, :]

    gate_n = h_range  # gate output cols [0, H)
    up_n = H + h_range  # up   output cols [H, 2H)

    gate = tl.load(
        XExt + m_range[:, None] * stride_xe_m + gate_n[None, :] * stride_xe_n,
        mask=base_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        XExt + m_range[:, None] * stride_xe_m + up_n[None, :] * stride_xe_n,
        mask=base_mask,
        other=0.0,
    ).to(tl.float32)

    # LoRA delta only for tiles that touch an adapter; the silu(gate)*up store
    # below always runs (base-only tiles still need their activation written).
    bitmask = tl.load(Bitmask + pid_m * stride_bm_t).to(tl.int64)
    if bitmask != 0:
        adap_ids = tl.load(AdapIdx + m_range, mask=mask_m, other=0).to(tl.int32)
        # Per-tier shrink-column bases for each slice: row 0 = gate, row 1 = up.
        g0 = tl.load(SliceColR + 0 * 6 + 0).to(tl.int32)
        g1 = tl.load(SliceColR + 0 * 6 + 1).to(tl.int32)
        g2 = tl.load(SliceColR + 0 * 6 + 2).to(tl.int32)
        g3 = tl.load(SliceColR + 0 * 6 + 3).to(tl.int32)
        g4 = tl.load(SliceColR + 0 * 6 + 4).to(tl.int32)
        g5 = tl.load(SliceColR + 0 * 6 + 5).to(tl.int32)
        u0 = tl.load(SliceColR + 1 * 6 + 0).to(tl.int32)
        u1 = tl.load(SliceColR + 1 * 6 + 1).to(tl.int32)
        u2 = tl.load(SliceColR + 1 * 6 + 2).to(tl.int32)
        u3 = tl.load(SliceColR + 1 * 6 + 3).to(tl.int32)
        u4 = tl.load(SliceColR + 1 * 6 + 4).to(tl.int32)
        u5 = tl.load(SliceColR + 1 * 6 + 5).to(tl.int32)
        gate += _accum_slice_delta(
            XExt,
            stride_xe_m,
            stride_xe_n,
            LBPACK,
            m_range,
            mask_m,
            adap_ids,
            gate_n,
            mask_h,
            bitmask,
            g0,
            g1,
            g2,
            g3,
            g4,
            g5,
            NA_16,
            NA_32,
            NA_64,
            NA_128,
            NA_256,
            NA_512,
            S,
            N,
            BLOCK_M,
            BLOCK_N,
        )
        up += _accum_slice_delta(
            XExt,
            stride_xe_m,
            stride_xe_n,
            LBPACK,
            m_range,
            mask_m,
            adap_ids,
            up_n,
            mask_h,
            bitmask,
            u0,
            u1,
            u2,
            u3,
            u4,
            u5,
            NA_16,
            NA_32,
            NA_64,
            NA_128,
            NA_256,
            NA_512,
            S,
            N,
            BLOCK_M,
            BLOCK_N,
        )

    # SwiGLU epilogue: silu(gate) * up, written as a contiguous [M, H] tile so
    # nothing downstream sees x_ext's strided base columns.
    out_val = (gate * tl.sigmoid(gate)) * up
    tl.store(
        Out + m_range[:, None] * stride_o_m + h_range[None, :] * stride_o_n,
        out_val.to(Out.dtype.element_ty),
        mask=base_mask,
    )


def _granite_switch_lora_expand_swiglu(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    H: int,
    N: int,
) -> None:
    """Fused gate/up LoRA expand + SwiGLU for the shared-MLP first projection.

    Writes ``out[M, H] = silu(gate_corrected) * up_corrected`` where gate/up are
    ``x_ext`` columns [0,H)/[H,2H) plus their LoRA delta. ``S`` must be 2 (gate,
    up), ``N`` is N_total = 2H. ``out`` is a fresh contiguous buffer (mutated).
    """
    M = x_ext.shape[0]
    cfg = get_switch_lora_expand_config()
    grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(H, block_n))
    _switch_lora_expand_swiglu_kernel[grid](
        x_ext,
        x_ext.stride(0),
        x_ext.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        adapter_indices,
        bitmask,
        bitmask.stride(0),
        lb_packed,
        M,
        H,
        N,
        slice_col_r,
        NA_16=NA_16,
        NA_32=NA_32,
        NA_64=NA_64,
        NA_128=NA_128,
        NA_256=NA_256,
        NA_512=NA_512,
        S=S,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=block_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def _granite_switch_lora_expand_swiglu_fake(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    H: int,
    N: int,
) -> None:
    # ``out`` is allocated by the caller and fully written here; the op returns
    # nothing, so there is no output to fake-allocate.
    return


direct_register_custom_op(
    op_name="granite_switch_lora_expand_swiglu",
    op_func=_granite_switch_lora_expand_swiglu,
    mutates_args=["out"],
    fake_impl=_granite_switch_lora_expand_swiglu_fake,
)
granite_switch_lora_expand_swiglu = torch.ops.vllm.granite_switch_lora_expand_swiglu


# ---------------------------------------------------------------------------
# Shrink-only ("W-less") expand - the Shadow-Residual W_cross shunt
# ---------------------------------------------------------------------------
#
# W_cross has no base weight: its output is purely
# ``(x @ lora_A_cross.T) @ lora_B_cross.T``. With no base region to accumulate
# into, this kernel STORES the delta to a fresh, pre-zeroed ``out[M, N]`` rather
# than doing switch_lora_expand's read-modify-write. ``x_ext`` is therefore all
# shrink columns, so ``slice_col_r`` bases start at 0, not N_total. A base token
# (kernel-local id 0) and any bitmask-skipped tile read back exactly zero, which
# keeps the cross-stream injection off the base stream - the SR invariant.
# Otherwise structurally identical to _switch_lora_expand_kernel.


@triton.jit
def _switch_lora_shrink_expand_kernel(
    # x_ext  [M, sum_r(n_r * S * r)] - shrink columns ONLY (no base region).
    XExt,
    stride_xe_m,
    stride_xe_n,
    # out  [M, N] contiguous, PRE-ZEROED - receives delta = shrink @ lora_B.
    Out,
    stride_o_m,
    stride_o_n,
    AdapIdx,
    Bitmask,
    stride_bm_t,
    LBPACK,
    M,
    N,
    TileSlice,
    SliceColR,
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Same by-design invariant as _switch_lora_expand_kernel: M (x_ext rows)
    # equals the token count behind AdapIdx/Bitmask (per-token routing). Tiles
    # with no adapter are skipped; ``out`` is pre-zeroed so they read back zero.
    bitmask = tl.load(Bitmask + pid_m * stride_bm_t).to(tl.int64)
    if bitmask == 0:
        return

    pid_n = tl.program_id(1)
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_range < M
    mask_n = n_range < N

    # This output tile lives in exactly one slice (block_n divides every slice).
    # For the shunt (S == 1) there is a single slice; SliceColR bases start at 0
    # because w_ext_cross has no base region.
    s_idx = tl.load(TileSlice + pid_n).to(tl.int32)
    col_16 = tl.load(SliceColR + s_idx * 6 + 0).to(tl.int32)
    col_32 = tl.load(SliceColR + s_idx * 6 + 1).to(tl.int32)
    col_64 = tl.load(SliceColR + s_idx * 6 + 2).to(tl.int32)
    col_128 = tl.load(SliceColR + s_idx * 6 + 3).to(tl.int32)
    col_256 = tl.load(SliceColR + s_idx * 6 + 4).to(tl.int32)
    col_512 = tl.load(SliceColR + s_idx * 6 + 5).to(tl.int32)

    adap_ids = tl.load(AdapIdx + m_range, mask=mask_m, other=0).to(tl.int32)
    delta = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    id_off_32 = NA_16
    id_off_64 = NA_16 + NA_32
    id_off_128 = NA_16 + NA_32 + NA_64
    id_off_256 = NA_16 + NA_32 + NA_64 + NA_128
    id_off_512 = NA_16 + NA_32 + NA_64 + NA_128 + NA_256

    lb_off_16 = 0
    lb_off_32 = NA_16 * N * 16
    lb_off_64 = lb_off_32 + NA_32 * N * 32
    lb_off_128 = lb_off_64 + NA_64 * N * 64
    lb_off_256 = lb_off_128 + NA_128 * N * 128
    lb_off_512 = lb_off_256 + NA_256 * N * 256

    r16 = tl.arange(0, 16)
    for a in tl.static_range(NA_16):
        if (bitmask >> a) & 1:  # adapter (a+1) in this tile?
            mask_a = (adap_ids == (a + 1)) & mask_m  # rows that select it
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_16 + a * S * 16 + r16[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_16
                + a * (N * 16)
                + r16[:, None]
                + n_range[None, :] * 16,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r32 = tl.arange(0, 32)
    for a in tl.static_range(NA_32):
        if (bitmask >> (id_off_32 + a)) & 1:
            mask_a = (adap_ids == (id_off_32 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_32 + a * S * 32 + r32[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_32
                + a * (N * 32)
                + r32[:, None]
                + n_range[None, :] * 32,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r64 = tl.arange(0, 64)
    for a in tl.static_range(NA_64):
        if (bitmask >> (id_off_64 + a)) & 1:
            mask_a = (adap_ids == (id_off_64 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_64 + a * S * 64 + r64[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_64
                + a * (N * 64)
                + r64[:, None]
                + n_range[None, :] * 64,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r128 = tl.arange(0, 128)
    for a in tl.static_range(NA_128):
        if (bitmask >> (id_off_128 + a)) & 1:
            mask_a = (adap_ids == (id_off_128 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_128 + a * S * 128 + r128[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_128
                + a * (N * 128)
                + r128[:, None]
                + n_range[None, :] * 128,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r256 = tl.arange(0, 256)
    for a in tl.static_range(NA_256):
        if (bitmask >> (id_off_256 + a)) & 1:
            mask_a = (adap_ids == (id_off_256 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_256 + a * S * 256 + r256[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_256
                + a * (N * 256)
                + r256[:, None]
                + n_range[None, :] * 256,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r512 = tl.arange(0, 512)
    for a in tl.static_range(NA_512):
        if (bitmask >> (id_off_512 + a)) & 1:
            mask_a = (adap_ids == (id_off_512 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_512 + a * S * 512 + r512[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_512
                + a * (N * 512)
                + r512[:, None]
                + n_range[None, :] * 512,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    # Shrink-only: plain STORE of the delta into the fresh ``out`` buffer (no base
    # region to read-modify-write). Base rows have delta == 0 and ``out`` is
    # pre-zeroed, so base tokens stay exactly zero. The store casts to ``Out``'s
    # own dtype so a non-bf16 shunt buffer is honored.
    out_ptrs = Out + m_range[:, None] * stride_o_m + n_range[None, :] * stride_o_n
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, delta.to(Out.dtype.element_ty), mask=store_mask)


def _granite_switch_lora_shrink_expand(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    """Shrink-only ("W-less") expand for the Shadow-Residual W_cross shunt.

    Writes ``out[M, N] = (x @ lora_A_cross.T) @ lora_B_cross.T`` for adapter
    tokens and exactly zero for base tokens (kernel-local id 0). ``x_ext`` holds
    only the shrink columns (``w_ext_cross`` built from an empty base ``W``), so
    ``slice_col_r`` bases start at 0. ``N`` is the shunt output width (= hidden
    size). ``out`` is pre-zeroed here so bitmask-skipped tiles read back zero.
    """
    M = x_ext.shape[0]
    out.zero_()
    cfg = get_switch_lora_expand_config()
    grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(N, block_n))
    _switch_lora_shrink_expand_kernel[grid](
        x_ext,
        x_ext.stride(0),
        x_ext.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        adapter_indices,
        bitmask,
        bitmask.stride(0),
        lb_packed,
        M,
        N,
        tile_slice,
        slice_col_r,
        NA_16=NA_16,
        NA_32=NA_32,
        NA_64=NA_64,
        NA_128=NA_128,
        NA_256=NA_256,
        NA_512=NA_512,
        S=S,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=block_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def _granite_switch_lora_shrink_expand_fake(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    # ``out`` is caller-allocated and zeroed-then-written here; nothing to fake.
    return


direct_register_custom_op(
    op_name="granite_switch_lora_shrink_expand",
    op_func=_granite_switch_lora_shrink_expand,
    mutates_args=["out"],
    fake_impl=_granite_switch_lora_shrink_expand_fake,
)
granite_switch_lora_shrink_expand = torch.ops.vllm.granite_switch_lora_shrink_expand

# ---------------------------------------------------------------------------
# Per-module tile bitmasks
#
# The expand kernels take a per-tile int64 bitmask and return early when a tile
# holds no adapter tokens. Bitmasks are computed per module from kernel-local
# (post-remap) adapter indices, so bit ``a`` of module ``i``'s mask corresponds
# exactly to kernel-local adapter ``a + 1`` in module ``i``'s launch. A single
# shared global-id bitmask would be wrong in both directions: false negatives
# from the per-module rank reordering, and false positives from adapters that
# this module remaps to 0 but whose global bit is set.
# ---------------------------------------------------------------------------


@triton.jit
def _or_combine(a, b):
    return a | b


@triton.jit
def _bitmask_reduce_kernel(
    adapter_indices_ptr,
    remap_tables_t_ptr,
    out_ptr,
    M,
    num_modules,
    NA_plus_1,
    BLOCK_M: tl.constexpr,
):
    pid_mod = tl.program_id(0)
    pid_tile = tl.program_id(1)

    offsets = pid_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M
    indices = tl.load(adapter_indices_ptr + offsets, mask=mask, other=0)
    indices = tl.maximum(indices, tl.zeros_like(indices))
    indices = tl.minimum(indices, NA_plus_1 - 1)

    remap_ptrs = remap_tables_t_ptr + indices * num_modules + pid_mod
    remapped = tl.load(remap_ptrs, mask=mask, other=0)

    is_active = remapped > 0
    bits = is_active.to(tl.int64) << tl.maximum(remapped - 1, tl.zeros_like(remapped))
    bitmask = tl.reduce(bits, axis=0, combine_fn=_or_combine)

    num_tiles = tl.cdiv(M, BLOCK_M)
    out_offset = pid_mod * num_tiles + pid_tile
    tl.store(out_ptr + out_offset, bitmask)


def _compute_bitmasks_triton(
    adapter_indices: torch.Tensor,
    remap_tables_t: torch.Tensor,
) -> torch.Tensor:
    """Per-module bitmask computation on the Triton path.

    Reached through the ``vllm::granite_switch_compute_per_module_bitmasks``
    custom op, so it is the path that runs in deployment - including inside the
    ``@support_torch_compile`` forward, where the custom-op boundary keeps
    inductor from tracing into the kernel. ``_compute_bitmasks_reference`` below
    is the equivalent pure-torch implementation used where Triton is
    unavailable; the two must agree bit-for-bit.
    """
    M = adapter_indices.shape[0]
    NA_plus_1, num_modules = remap_tables_t.shape
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M

    out = torch.empty(
        num_modules, num_tiles, dtype=torch.int64, device=adapter_indices.device
    )
    grid = (num_modules, num_tiles)
    _bitmask_reduce_kernel[grid](
        adapter_indices,
        remap_tables_t,
        out,
        M,
        num_modules,
        NA_plus_1,
        BLOCK_M=BLOCK_M,
    )
    return out


def _compute_bitmasks_reference(
    adapter_indices: torch.Tensor,
    remap_tables_t: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch per-module bitmask computation.

    Serves two purposes: it is the readable specification of the bitmask
    contract that ``_compute_bitmasks_triton`` must reproduce bit-for-bit, and
    it is the fallback taken on platforms without Triton. It is correct but
    launches O(BLOCK_M) elementwise ops instead of one reduction, so the Triton
    path is preferred whenever it is available.

    Args:
        adapter_indices: [M] int64, values in [0, NA].
        remap_tables_t: [NA+1, num_modules] int64.

    Returns:
        per_module_bitmasks: [num_modules, num_tiles] int64.

    """
    M = adapter_indices.shape[0]
    num_modules = remap_tables_t.shape[1]

    # Pad to BLOCK_M boundary so tile-reduction reshape works cleanly.
    pad_m = (-M) % BLOCK_M
    if pad_m:
        adapter_indices = torch.nn.functional.pad(adapter_indices, (0, pad_m), value=0)

    # Clamp for safety (no-op in normal execution, guards against stale data).
    NA = remap_tables_t.shape[0] - 1
    adapter_indices = adapter_indices.clamp(0, NA)

    # Gather kernel-local indices for all modules, [M_padded, num_modules] -> T.
    # The transpose is made contiguous here rather than at the end: without it
    # the column-major layout propagates through every op below and the op
    # returns a non-contiguous tensor, contradicting the fake impl (which, like
    # the Triton path, declares a contiguous result) and tripping inductor's
    # stride assertion under torch.compile.
    all_remapped = remap_tables_t[adapter_indices].T.contiguous()

    # Per-token per-module bitmask: bit a set iff kernel-local adapter a+1
    # is present at that token position for that module.
    is_active = all_remapped > 0
    bits = is_active.long() << (all_remapped - 1).clamp(min=0)

    # Reduce per-tile: OR all BLOCK_M token-bitmasks within each tile.
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    cols = bits.view(num_modules, num_tiles, BLOCK_M)
    per_module_bitmasks = cols[:, :, 0].clone()
    for i in range(1, BLOCK_M):
        per_module_bitmasks.bitwise_or_(cols[:, :, i])

    return per_module_bitmasks


def _granite_switch_compute_per_module_bitmasks(
    adapter_indices: torch.Tensor,
    remap_tables_t: torch.Tensor,
) -> torch.Tensor:
    """Per-module, per-tile occupancy bitmasks.

    Args:
        adapter_indices: ``[M]`` int64 global adapter ids, in ``[0, NA]``.
        remap_tables_t: ``[NA + 1, num_modules]`` int64; entry ``[j, i]`` is the
            kernel-local index of global adapter ``j`` in module ``i``, or 0 if
            that adapter does not apply to the module.

    Returns:
        ``[num_modules, num_tiles]`` int64, where ``num_tiles`` tiles the ``M``
        tokens by ``BLOCK_M``.

    """
    if HAS_TRITON:
        return _compute_bitmasks_triton(adapter_indices, remap_tables_t)
    return _compute_bitmasks_reference(adapter_indices, remap_tables_t)


def _granite_switch_compute_per_module_bitmasks_fake(
    adapter_indices: torch.Tensor,
    remap_tables_t: torch.Tensor,
) -> torch.Tensor:
    M = adapter_indices.shape[0]
    num_modules = remap_tables_t.shape[1]
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    return torch.empty(
        num_modules, num_tiles, dtype=torch.int64, device=adapter_indices.device
    )


direct_register_custom_op(
    op_name="granite_switch_compute_per_module_bitmasks",
    op_func=_granite_switch_compute_per_module_bitmasks,
    mutates_args=[],
    fake_impl=_granite_switch_compute_per_module_bitmasks_fake,
)
granite_switch_compute_per_module_bitmasks = (
    torch.ops.vllm.granite_switch_compute_per_module_bitmasks
)


# ---------------------------------------------------------------------------
# Per-forward kernel metadata
# ---------------------------------------------------------------------------


class LoRAContext:
    """Shared per-forward LoRA kernel-metadata context.

    A single instance is created at model level and wired to every
    SwitchedLoRALinear / GraniteLoRAEmbeddedAttention via ``_lora_ctx``.
    Written once per forward in GraniteSwitchModel; read by every layer.
    """

    __slots__ = (
        "adapter_indices",
        "num_tokens",
        "per_module_bitmasks",
        "remapped_indices",
    )

    def __init__(self):
        self.adapter_indices: torch.Tensor | None = None
        self.remapped_indices: torch.Tensor | None = None
        self.per_module_bitmasks: torch.Tensor | None = None
        self.num_tokens: int = 0

    def reset(self):
        self.adapter_indices = None
        self.remapped_indices = None
        self.per_module_bitmasks = None
        self.num_tokens = 0


class FusedLoRAKernelMeta(nn.Module):
    """Compute per-module bitmask metadata for the fused switch-LoRA expand kernel.

    Called once per forward pass. Computes a bitmask for every
    SwitchedLoRALinear module in a single batched operation and stores
    them on the shared LoRAContext.

    The bitmask for module i uses kernel-local (post-remap) adapter indices
    so that bitmask bit a exactly corresponds to kernel-local adapter a+1
    in that module's expand kernel call.  This makes the bitmask exact:
    no false negatives (skipped LoRA) or false positives from non-applicable
    adapters that were remapped to 0 but still had their global bit set.

    register_remap_tables() must be called after finalize_weights() has run
    on all SwitchedLoRALinear instances (i.e. after load_weights()).
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def register_remap_tables(self, all_remap_tables: torch.Tensor) -> None:
        """Register per-module remap tables after weight loading.

        Args:
            all_remap_tables: [num_modules, NA+1] long tensor where row i is
                the remap_table of the i-th SwitchedLoRALinear.  Entry [i, 0]
                = 0 (base), entry [i, j] = kernel-local position of global
                adapter j in module i (0 if non-applicable).

        """
        if all_remap_tables.max() > 64:
            raise ValueError(
                "The int64 tile bitmask addresses at most 64 kernel-local "
                "adapters per module, but this checkpoint needs "
                f"{all_remap_tables.max().item()}. Reduce the number of "
                "adapters that target any single module."
            )

        # Store transposed: [NA+1, num_modules] for row-gather in the custom op.
        self.register_buffer(
            "_remap_tables_t",
            all_remap_tables.T.contiguous(),  # [NA+1, num_modules]
            persistent=False,
        )

    def prepare_and_store(
        self, adapter_indices: torch.Tensor, ctx: LoRAContext
    ) -> None:
        """Compute per-module bitmasks and store on context.

        Args:
            adapter_indices: [num_tokens] with values 0=base, 1..num_adapters.
                             Global adapter indices as returned by the switch.
            ctx: Shared LoRAContext to populate.

        """
        M = adapter_indices.shape[0]

        # Wrapped as a custom op so the bitmask launcher stays opaque to
        # inductor inside the compiled forward.
        per_module_bitmasks = granite_switch_compute_per_module_bitmasks(
            adapter_indices,
            self._remap_tables_t,
        )

        # Per-module kernel-local indices, computed once for all modules here so
        # forward() need not re-gather remap_table[adapter_indices] per module.
        # _remap_tables_t is [NA+1, num_modules]; gather + transpose gives a
        # row-major [num_modules, M] buffer, so remapped_indices[module_idx] is a
        # stride-1 contiguous view (no copy, no launch) - exactly what the expand
        # kernel's AdapIdx slot requires.
        ctx.remapped_indices = self._remap_tables_t[adapter_indices].T.contiguous()

        ctx.per_module_bitmasks = per_module_bitmasks  # [num_modules, num_tiles_M]
        ctx.adapter_indices = adapter_indices
        ctx.num_tokens = M


class SRLoRAContext(LoRAContext):
    """LoRAContext extended with the M-length real-id metadata for the shunt.

    The Shadow-Residual forward runs two token layouts over one set of
    per-module remap tables:

    * the **2M stacked** projections (qkv / o / gate-up / down) run over
      ``cat([h_base, h_adapt])``, so their adapter-index vector is
      ``cat([0] * M, real)``: the base half is all 0 (pristine base projection,
      no delta, base-only K/V), the adapt half carries the real per-token id.
    * the **W_cross shunt** runs over only the ``M`` base-half rows and must be
      keyed by the ``M``-length REAL adapter ids. Keying it off the 2M vector's
      base-half zeros would make the shrink-only kernel (id 0 never fires, and
      the output is pre-zeroed) emit identically zero, silently deleting the
      cross-stream injection with no error anywhere. The separate ``*_shunt``
      fields exist precisely so the two layouts cannot be confused.

    ``remapped_indices`` / ``per_module_bitmasks`` (inherited) hold the **2M**
    metadata consumed by every SwitchedLoRALinear projection; the ``*_shunt``
    fields hold the **M**-length real-id metadata consumed by the WCrossShunt.
    """

    __slots__ = ("per_module_bitmasks_shunt", "remapped_indices_shunt")

    def __init__(self):
        super().__init__()
        self.remapped_indices_shunt: torch.Tensor | None = None
        self.per_module_bitmasks_shunt: torch.Tensor | None = None

    def reset(self):
        super().reset()
        self.remapped_indices_shunt = None
        self.per_module_bitmasks_shunt = None


class SRFusedLoRAKernelMeta(FusedLoRAKernelMeta):
    """Kernel metadata for the dual-stream Shadow-Residual forward.

    ``register_remap_tables`` is inherited unchanged; the module set it is
    given includes both the SwitchedLoRALinear projections and the WCrossShunt
    modules (each with its own ``_module_idx`` / ``remap_table``).
    """

    def prepare_and_store_sr(
        self, adapter_indices: torch.Tensor, ctx: SRLoRAContext
    ) -> None:
        """Compute + store both metadata layouts from the real per-token ids.

        Args:
            adapter_indices: ``[M]`` real ids (0=base, 1..num_adapters), one per
                real input token - as returned by the switch.
            ctx: shared SRLoRAContext to populate.

        """
        M = adapter_indices.shape[0]

        # --- 2M stacked layout (projections): base half 0, adapt half real ---
        zeros = torch.zeros_like(adapter_indices)
        ai_2m = torch.cat([zeros, adapter_indices], dim=0)  # [2M]
        ctx.per_module_bitmasks = granite_switch_compute_per_module_bitmasks(
            ai_2m, self._remap_tables_t
        )
        ctx.remapped_indices = self._remap_tables_t[
            ai_2m
        ].T.contiguous()  # [num_modules, 2M]

        # --- M-length real-id layout (shunt): keyed by real ids, not zeros ---
        ctx.per_module_bitmasks_shunt = granite_switch_compute_per_module_bitmasks(
            adapter_indices, self._remap_tables_t
        )
        ctx.remapped_indices_shunt = self._remap_tables_t[
            adapter_indices
        ].T.contiguous()  # [num_modules, M]

        # Real ids, for the per-token head select.
        ctx.adapter_indices = adapter_indices
        ctx.num_tokens = M
