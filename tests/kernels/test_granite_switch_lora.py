# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics for the three Granite Switch fused-LoRA Triton kernels.

Each kernel is checked against an unfused reference that computes the same
quantity with explicit per-token matmuls, over every rank tier in
``SUPPORTED_RANKS``:

* ``granite_switch_lora_expand`` - accumulates the LoRA delta in place into the
  base columns of ``x_ext``, single-slice and multi-slice (packed qkv / gate-up).
* ``granite_switch_lora_expand_swiglu`` - the same expand fused with
  ``silu(gate) * up`` for the shared-MLP first projection.
* ``granite_switch_lora_shrink_expand`` - the Shadow Residual ``W_cross`` shunt,
  which has no base region and must emit *exactly* zero for base tokens.

The tests build the packed kernel metadata (``w_ext``, ``lb_packed``,
``slice_col_r``, ``tile_slice``, per-tier counts, bitmask) by hand rather than
through ``SwitchedLoRALinear.finalize_weights``, so a layout regression in
either producer shows up as a mismatch instead of cancelling out.

Two deliberate choices keep that honest:

* The reference reads the LoRA shrink vectors out of the same ``x_ext`` the
  kernel reads, so the base ``x @ w_ext.T`` GEMM - which is ``torch.mm``, not
  one of these kernels - is excluded from the comparison. That is what lets the
  tolerance stay tight at bf16.
* ``_assert_shrink_columns`` independently pins ``build_w_ext``'s row order
  against ``x @ lora_A.T``. Without it the kernel and the reference would index
  the *same* ``x_ext`` columns, so a row-ordering bug would agree with itself.
"""

import pytest
import torch

from vllm.model_executor.models.granite_switch_kernels import (
    BLOCK_M,
    SUPPORTED_RANKS,
    build_w_ext,
    granite_switch_lora_expand,
    granite_switch_lora_expand_swiglu,
    granite_switch_lora_shrink_expand,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Granite Switch LoRA kernels are Triton, CUDA/ROCm only",
)

DTYPE = torch.bfloat16
DEVICE = "cuda"
SEED = 11

# bf16 accumulation against an fp32 reference over K <= 512 columns.
RTOL = 2e-2
ATOL = 2e-2


# ---------------------------------------------------------------------------
# Metadata construction (mirrors finalize_weights, independently written)
# ---------------------------------------------------------------------------


def _make_adapter(rank, K, output_slices, scale=0.05):
    """One adapter: ``(rank, A[S, rank, K], B[S][N_s, rank])``.

    ``A`` carries one shrink matrix per output slice, matching the checkpoint
    layout ``lora_A_slices[s][adapter]``; ``B`` likewise per slice.
    """
    S = len(output_slices)
    A = torch.randn(S, rank, K, device=DEVICE, dtype=DTYPE) * scale
    B = [
        torch.randn(N_s, rank, device=DEVICE, dtype=DTYPE) * scale
        for N_s in output_slices
    ]
    return rank, A, B


def _build_meta(adapters, W, output_slices, block_n):
    """Build the packed kernel metadata for one module.

    ``adapters`` must be ordered by ascending rank: kernel-local ids are
    ``1..len(adapters)`` in that order, which is the order ``finalize_weights``
    produces from ``sorted(applicable, key=rank)``.
    """
    S = len(output_slices)
    N_total = W.shape[0]
    ranks = [r for r, _, _ in adapters]
    assert ranks == sorted(ranks), "adapters must be in ascending rank order"

    # {rank: [n_r, S, rank, K]} -> w_ext rows in tier/adapter/slice order.
    A_lists: dict[int, list[torch.Tensor]] = {}
    for r, A, _B in adapters:
        A_lists.setdefault(r, []).append(A)
    lora_A_by_rank = {r: torch.stack(mats, dim=0) for r, mats in A_lists.items()}
    w_ext = build_w_ext(W, lora_A_by_rank)

    # Per-tier applicable-adapter counts; constexprs in the kernel.
    na = tuple(sum(1 for r in ranks if r == R) for R in SUPPORTED_RANKS)

    # slice_col_r[s, t] = tier base column + s * r_t. Tier bases follow w_ext's
    # row order, so each tier block is n_r * S * r wide.
    tier_base = []
    offset = N_total
    for t, R in enumerate(SUPPORTED_RANKS):
        tier_base.append(offset)
        offset += na[t] * S * R
    slice_col_r = torch.tensor(
        [[tier_base[t] + s * SUPPORTED_RANKS[t] for t in range(6)] for s in range(S)],
        dtype=torch.int32,
        device=DEVICE,
    )

    # lora_B merged along N_total per adapter, then packed as a contiguous
    # concat over PRESENT tiers of [n_r, N_total, r]. The kernel derives each
    # tier's offset into this buffer from cumsum(na[t] * N_total * r_t), so the
    # concat order has to be ascending-tier exactly.
    merged = {}  # rank -> [n_r, N_total, rank]
    for R in SUPPORTED_RANKS:
        tier = [torch.cat(B, dim=0) for r, _A, B in adapters if r == R]
        if tier:
            merged[R] = torch.stack(tier, dim=0)
    lb_packed = torch.cat([merged[R].reshape(-1) for R in sorted(merged)]).contiguous()

    # tile_slice[num_tiles_N]: which output slice each N-tile belongs to. Valid
    # only because block_n divides every slice, so no tile straddles a boundary.
    assert all(N_s % block_n == 0 for N_s in output_slices)
    bounds = [0]
    for N_s in output_slices:
        bounds.append(bounds[-1] + N_s)
    tile_slice = torch.zeros(N_total // block_n, dtype=torch.int32, device=DEVICE)
    for t in range(tile_slice.shape[0]):
        col = t * block_n
        for s in range(S):
            if bounds[s] <= col < bounds[s + 1]:
                tile_slice[t] = s
                break

    # Per-adapter shrink-column start for slice 0, used by the reference and by
    # _assert_shrink_columns. Positions within a tier follow `adapters` order.
    col_of = []
    pos_in_tier = dict.fromkeys(SUPPORTED_RANKS, 0)
    for r, _A, _B in adapters:
        t = SUPPORTED_RANKS.index(r)
        col_of.append(tier_base[t] + pos_in_tier[r] * S * r)
        pos_in_tier[r] += 1

    return w_ext, lb_packed, slice_col_r, tile_slice, na, col_of, merged, bounds


def _assert_shrink_columns(x, x_ext, adapters, col_of, S):
    """Pin build_w_ext's row order: adapter i slice s must occupy its own block.

    The kernel and the reference both read shrink vectors from ``x_ext``, so a
    wrong row order in ``w_ext`` would be invisible to their comparison. This
    recomputes each block from ``lora_A`` directly.
    """
    for i, (r, A, _B) in enumerate(adapters):
        for s in range(S):
            c = col_of[i] + s * r
            torch.testing.assert_close(
                x_ext[:, c : c + r].float(),
                (x @ A[s].t()).float(),
                rtol=RTOL,
                atol=ATOL,
            )


def _bitmask(ids, M):
    """Per-tile int64 occupancy: bit ``id - 1`` set iff id appears in the tile."""
    bm = torch.zeros((M + BLOCK_M - 1) // BLOCK_M, dtype=torch.int64, device=ids.device)
    for m in range(M):
        i = int(ids[m].item())
        if i > 0:
            bm[m // BLOCK_M] |= 1 << (i - 1)
    return bm


def _reference_delta(x_ext, ids, adapters, col_of, merged, bounds, S, N_total):
    """Unfused reference delta: per token, per slice, ``(x @ A.T) @ B.T``."""
    M = x_ext.shape[0]
    delta = torch.zeros(M, N_total, dtype=torch.float32, device=x_ext.device)
    pos_in_tier = dict.fromkeys(SUPPORTED_RANKS, 0)
    positions = []
    for r, _A, _B in adapters:
        positions.append(pos_in_tier[r])
        pos_in_tier[r] += 1
    for m in range(M):
        i = int(ids[m].item())
        if i == 0:
            continue
        r, _A, _B = adapters[i - 1]
        B_merged = merged[r][positions[i - 1]].float()  # [N_total, r]
        for s in range(S):
            c = col_of[i - 1] + s * r
            shrink = x_ext[m, c : c + r].float()
            lo, hi = bounds[s], bounds[s + 1]
            delta[m, lo:hi] += shrink @ B_merged[lo:hi].t()
    return delta


def _run_expand(adapters, ids, K, output_slices, block_n):
    """Launch the expand kernel and return ``(kernel_out, reference_out)``."""
    torch.manual_seed(SEED)
    S = len(output_slices)
    N_total = sum(output_slices)
    M = ids.shape[0]
    ids = ids.to(DEVICE)

    W = torch.randn(N_total, K, device=DEVICE, dtype=DTYPE) * 0.02
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE) * 0.01

    (w_ext, lb_packed, slice_col_r, tile_slice, na, col_of, merged, bounds) = (
        _build_meta(adapters, W, output_slices, block_n)
    )
    x_ext = (x @ w_ext.t()).contiguous()
    _assert_shrink_columns(x, x_ext, adapters, col_of, S)

    ref = x_ext[:, :N_total].float() + _reference_delta(
        x_ext, ids, adapters, col_of, merged, bounds, S, N_total
    )

    granite_switch_lora_expand(
        x_ext,
        ids,
        _bitmask(ids, M),
        lb_packed,
        tile_slice,
        slice_col_r,
        *na,
        S,
        block_n,
        N_total,
    )
    return x_ext[:, :N_total].float(), ref


# ---------------------------------------------------------------------------
# granite_switch_lora_expand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", SUPPORTED_RANKS)
def test_expand_single_slice_per_rank_tier(rank):
    """Every rank tier, one adapter, mixed base/adapter tokens over >1 row tile."""
    K, output_slices, block_n = 256, (256,), 32
    torch.manual_seed(SEED)
    adapters = [_make_adapter(rank, K, output_slices)]
    ids = torch.tensor([0, 1] * 24, dtype=torch.int64)  # 48 tokens, 2 row tiles
    out, ref = _run_expand(adapters, ids, K, output_slices, block_n)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("rank", SUPPORTED_RANKS)
def test_expand_packed_qkv_per_rank_tier(rank):
    """S=3 packed qkv with unequal slice widths, every rank tier.

    Multi-slice is where the per-tile slice lookup and the ``s * r`` stride into
    the shrink block matter: a wrong ``tile_slice`` or ``slice_col_r`` reads
    another slice's shrink columns, which single-slice cannot catch.
    """
    K, output_slices, block_n = 256, (128, 64, 64), 32
    torch.manual_seed(SEED)
    adapters = [_make_adapter(rank, K, output_slices)]
    ids = torch.tensor([1, 0, 1, 1] * 12, dtype=torch.int64)  # 48 tokens
    out, ref = _run_expand(adapters, ids, K, output_slices, block_n)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_expand_mixed_rank_tiers():
    """Four adapters spanning three tiers at once (r=16, 16, 32, 64)."""
    K, output_slices, block_n = 512, (128, 128), 32
    torch.manual_seed(SEED)
    adapters = [
        _make_adapter(16, K, output_slices, scale=0.05),
        _make_adapter(16, K, output_slices, scale=0.07),
        _make_adapter(32, K, output_slices),
        _make_adapter(64, K, output_slices),
    ]
    ids = torch.tensor([0, 1, 2, 3, 4, 2, 0, 4] * 8, dtype=torch.int64)  # 64 tokens
    out, ref = _run_expand(adapters, ids, K, output_slices, block_n)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_expand_all_base_leaves_base_output_bit_exact():
    """No adapter token anywhere: every tile is bitmask-skipped.

    Asserted at zero tolerance - the kernel accumulates in place, so "skipped"
    has to mean untouched, not approximately untouched.
    """
    K, output_slices, block_n = 256, (256,), 32
    torch.manual_seed(SEED)
    adapters = [_make_adapter(32, K, output_slices)]
    ids = torch.zeros(64, dtype=torch.int64)
    out, ref = _run_expand(adapters, ids, K, output_slices, block_n)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_expand_ragged_token_count():
    """M not a multiple of BLOCK_M - the tail tile is partially masked."""
    K, output_slices, block_n = 256, (128, 64, 64), 32
    torch.manual_seed(SEED)
    adapters = [
        _make_adapter(16, K, output_slices),
        _make_adapter(32, K, output_slices),
    ]
    ids = torch.tensor([1, 2, 0] * 13 + [1, 0], dtype=torch.int64)  # 41 tokens
    out, ref = _run_expand(adapters, ids, K, output_slices, block_n)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# granite_switch_lora_expand_swiglu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", SUPPORTED_RANKS)
def test_expand_swiglu_matches_dense_reference(rank):
    """Fused gate/up expand + SwiGLU vs. ``silu(gate) * up`` on the dense result.

    The reference applies the activation to the unfused corrected gate/up, so a
    mismatch means the fusion - not the expand - is wrong; the expand itself is
    covered independently above.
    """
    K, H, block_n = 256, 128, 32
    output_slices = (H, H)
    N_total = 2 * H
    torch.manual_seed(SEED)
    adapters = [_make_adapter(rank, K, output_slices)]
    ids = torch.tensor([0, 1, 1, 0] * 12, dtype=torch.int64)  # 48 tokens
    M = ids.shape[0]
    ids = ids.to(DEVICE)

    W = torch.randn(N_total, K, device=DEVICE, dtype=DTYPE) * 0.02
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE) * 0.01
    (w_ext, lb_packed, slice_col_r, _tile, na, col_of, merged, bounds) = _build_meta(
        adapters, W, output_slices, block_n
    )
    x_ext = (x @ w_ext.t()).contiguous()
    _assert_shrink_columns(x, x_ext, adapters, col_of, 2)

    dense = x_ext[:, :N_total].float() + _reference_delta(
        x_ext, ids, adapters, col_of, merged, bounds, 2, N_total
    )
    gate, up = dense[:, :H], dense[:, H:]
    ref = (gate * torch.sigmoid(gate)) * up

    out = torch.empty(M, H, device=DEVICE, dtype=DTYPE)
    granite_switch_lora_expand_swiglu(
        out,
        x_ext,
        ids,
        _bitmask(ids, M),
        lb_packed,
        slice_col_r,
        *na,
        2,
        block_n,
        H,
        N_total,
    )
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)


def test_expand_swiglu_all_base_still_activates():
    """A bitmask-skipped tile must still write ``silu(gate) * up``.

    The plain expand kernel returns early on an empty tile because its output is
    an in-place accumulate; this kernel writes a fresh buffer, so returning
    early would leave the activation unwritten.
    """
    K, H, block_n = 256, 128, 32
    output_slices = (H, H)
    N_total = 2 * H
    torch.manual_seed(SEED)
    adapters = [_make_adapter(16, K, output_slices)]
    ids = torch.zeros(32, dtype=torch.int64).to(DEVICE)
    M = ids.shape[0]

    W = torch.randn(N_total, K, device=DEVICE, dtype=DTYPE) * 0.02
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE) * 0.01
    (w_ext, lb_packed, slice_col_r, _tile, na, _col, _merged, _b) = _build_meta(
        adapters, W, output_slices, block_n
    )
    x_ext = (x @ w_ext.t()).contiguous()

    base = x_ext[:, :N_total].float()
    gate, up = base[:, :H], base[:, H:]
    ref = (gate * torch.sigmoid(gate)) * up

    out = torch.empty(M, H, device=DEVICE, dtype=DTYPE)
    granite_switch_lora_expand_swiglu(
        out,
        x_ext,
        ids,
        _bitmask(ids, M),
        lb_packed,
        slice_col_r,
        *na,
        2,
        block_n,
        H,
        N_total,
    )
    assert torch.count_nonzero(out) > 0
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# granite_switch_lora_shrink_expand (Shadow Residual W_cross shunt)
# ---------------------------------------------------------------------------


def _run_shrink_expand(adapters, ids, K, N, block_n):
    """Launch the shunt kernel and return ``(kernel_out, reference_out)``.

    ``w_ext`` is built from an *empty* base weight, so ``x_ext`` holds only
    shrink columns and tier bases start at 0. The delta is stored to a fresh
    buffer rather than accumulated.
    """
    torch.manual_seed(SEED)
    M = ids.shape[0]
    ids = ids.to(DEVICE)
    W_empty = torch.empty(0, K, device=DEVICE, dtype=DTYPE)

    (w_ext, lb_packed, slice_col_r, tile_slice, na, col_of, merged, bounds) = (
        _build_meta(adapters, W_empty, (N,), block_n)
    )
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    x_ext = (x @ w_ext.t()).contiguous()
    _assert_shrink_columns(x, x_ext, adapters, col_of, 1)

    # tile_slice is empty when N_total == 0; the shunt tiles its own output
    # width N instead, all of it slice 0.
    tile_slice = torch.zeros(N // block_n, dtype=torch.int32, device=DEVICE)

    ref = _reference_delta(x_ext, ids, adapters, col_of, merged, bounds, 1, N)

    out = torch.empty(M, N, device=DEVICE, dtype=DTYPE)
    granite_switch_lora_shrink_expand(
        out,
        x_ext,
        ids,
        _bitmask(ids, M),
        lb_packed,
        tile_slice,
        slice_col_r,
        *na,
        1,
        block_n,
        N,
    )
    return out, ref


@pytest.mark.parametrize("rank", SUPPORTED_RANKS)
def test_shrink_expand_per_rank_tier(rank):
    """Every rank tier; base rows must be exactly zero, not merely small."""
    K, N, block_n = 256, 64, 32
    torch.manual_seed(SEED)
    adapters = [_make_adapter(rank, K, (N,))]
    ids = torch.tensor([0, 1] * 20, dtype=torch.int64)  # 40 tokens, 2 row tiles
    out, ref = _run_shrink_expand(adapters, ids, K, N, block_n)
    torch.testing.assert_close(out.float(), ref, rtol=RTOL, atol=ATOL)
    assert torch.count_nonzero(out[(ids == 0).to(DEVICE)]) == 0


def test_shrink_expand_all_base_tile_is_zero():
    """A tile with no adapter token is skipped; the pre-zeroed buffer stays 0."""
    K, N, block_n = 256, 64, 32
    torch.manual_seed(SEED)
    adapters = [_make_adapter(16, K, (N,))]
    ids = torch.zeros(32, dtype=torch.int64)
    out, ref = _run_shrink_expand(adapters, ids, K, N, block_n)
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=0)


def test_shrink_expand_mixed_rank_tiers():
    """Two adapters across tiers (r=16, r=32) in one launch."""
    K, N, block_n = 512, 128, 32
    torch.manual_seed(SEED)
    adapters = [_make_adapter(16, K, (N,)), _make_adapter(32, K, (N,))]
    ids = torch.tensor([0, 1, 2, 2, 0, 1] * 8, dtype=torch.int64)  # 48 tokens
    out, ref = _run_shrink_expand(adapters, ids, K, N, block_n)
    torch.testing.assert_close(out.float(), ref, rtol=RTOL, atol=ATOL)


def test_shrink_expand_multiple_adapters_same_tier():
    """Two tier-16 adapters (kernel-local ids 1, 2) with disjoint shrink columns."""
    K, N, block_n = 256, 64, 32
    torch.manual_seed(SEED)
    adapters = [
        _make_adapter(16, K, (N,), scale=0.05),
        _make_adapter(16, K, (N,), scale=0.07),
    ]
    ids = torch.tensor([1, 2, 0, 1, 2, 0] * 8, dtype=torch.int64)  # 48 tokens
    out, ref = _run_shrink_expand(adapters, ids, K, N, block_n)
    torch.testing.assert_close(out.float(), ref, rtol=RTOL, atol=ATOL)
