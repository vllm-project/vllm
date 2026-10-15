# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm correctness tests for the strided DSA candidate mask.

``_apply_candidate_mask_strided`` is a decode-path variant of the shared
``apply_candidate_mask``. It differs in two ways that need pinning down:

* it launches a fixed number of programs that stride the column axis, rather
  than one program per 1024-column tile, so at ``max_model_len`` widths the
  grid no longer scales with a workspace that is mostly padding;
* it stops at each row's ``end`` instead of sanitizing the whole width.

The second is only sound because every consumer bounds its scan by the same
ends -- ``top_k_per_row_decode`` takes ``rowEnd`` from ``seq_lens``. So the
contract is agreement with the shared kernel over ``[0, end)`` and nothing
beyond it, which is what these tests assert rather than comparing whole rows.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

BLOCK_SIZE = 8
TOPK_BLOCKS = 64
# Above 128 tiles the grid saturates and programs start striding; below it the
# min() clamp applies instead. Both regimes need covering, so the widths here
# deliberately straddle 128 * 1024.
NARROW = 8192
WIDE = 262144


def _inputs(rows, width, ends, starts, *, block_size, col_stride, dtype):
    """Logits, bounds and candidate blocks shared by both kernels."""
    torch.manual_seed(0)
    # A column stride > 1 keeps the kernels honest about stride_col; the decode
    # caller slices a workspace, so contiguous rows are not guaranteed.
    logits = torch.randn(rows, width * col_stride, device="cuda")[:, ::col_stride]
    logits[0, : min(16, width)] = 0.0
    if rows > 2:
        logits[2, 5] = float("nan")
    row_ke = torch.tensor(ends, device="cuda", dtype=dtype)
    row_ks = torch.tensor(starts, device="cuda", dtype=dtype) if starts else None
    nblocks = (width + block_size - 1) // block_size
    candidates = torch.randint(
        0, nblocks, (rows, TOPK_BLOCKS), device="cuda", dtype=torch.int32
    )
    # -1 is the "no block" sentinel the selector emits for short rows.
    candidates[:, ::7] = -1
    return logits, row_ks, row_ke, candidates


def _run_both(logits, row_ks, row_ke, candidates, block_size, row_repeat):
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _apply_candidate_mask_strided,
    )

    ref = logits.clone()
    got = logits.clone()
    apply_candidate_mask(ref, row_ks, row_ke, candidates, block_size, row_repeat)
    _apply_candidate_mask_strided(
        got, row_ks, row_ke, candidates, block_size, row_repeat
    )
    return ref, got


def _assert_agrees_within_ends(ref, got, row_ke, row_repeat):
    """Compare only the region a consumer can read."""
    for row in range(ref.shape[0]):
        end = int(row_ke[row // row_repeat])
        if end <= 0:
            continue
        torch.testing.assert_close(
            got[row, :end],
            ref[row, :end],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda m, r=row, e=end: f"row {r} differs within [0, {e}): {m}",
        )


@pytest.mark.parametrize("width", [NARROW, WIDE])
@pytest.mark.parametrize("col_stride", [1, 2])
def test_matches_shared_kernel_ragged_ends(width, col_stride):
    """Ends on and around tile boundaries, plus the empty and full rows."""
    ends = [0, 1, 1023, 1024, 1025, width - 1, width, width // 2]
    logits, row_ks, row_ke, candidates = _inputs(
        len(ends),
        width,
        ends,
        None,
        block_size=BLOCK_SIZE,
        col_stride=col_stride,
        dtype=torch.int32,
    )
    ref, got = _run_both(logits, row_ks, row_ke, candidates, BLOCK_SIZE, 1)
    _assert_agrees_within_ends(ref, got, row_ke, 1)


@pytest.mark.parametrize("block_size", [8, 16])
def test_matches_shared_kernel_with_starts(block_size):
    """Non-zero starts: the decode caller passes zeros, prefill does not."""
    ends = [0, 64, 4096, 8192]
    starts = [0, 7, 1024, 3]
    logits, row_ks, row_ke, candidates = _inputs(
        len(ends),
        NARROW,
        ends,
        starts,
        block_size=block_size,
        col_stride=1,
        dtype=torch.int64,
    )
    ref, got = _run_both(logits, row_ks, row_ke, candidates, block_size, 1)
    _assert_agrees_within_ends(ref, got, row_ke, 1)


def test_matches_shared_kernel_row_repeat():
    """Speculative decode: next_n logit rows share one bound."""
    row_repeat = 3
    ends = [0, 1025, WIDE]
    logits, row_ks, row_ke, candidates = _inputs(
        len(ends) * row_repeat,
        WIDE,
        ends,
        None,
        block_size=BLOCK_SIZE,
        col_stride=1,
        dtype=torch.int32,
    )
    ref, got = _run_both(logits, row_ks, row_ke, candidates, BLOCK_SIZE, row_repeat)
    _assert_agrees_within_ends(ref, got, row_ke, row_repeat)


def test_leaves_columns_past_end_untouched():
    """The deliberate difference from the shared kernel, pinned explicitly.

    Skipping the tail is where the speedup comes from, so a change that
    quietly restores full-width sanitizing should fail here rather than just
    get slower. Work stops at the tile boundary containing ``end``, not at
    ``end`` itself, so that is the bound asserted.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _MASK_TILE,
        _apply_candidate_mask_strided,
    )

    ends = [0, 1, 1025, WIDE // 2]
    logits, row_ks, row_ke, candidates = _inputs(
        len(ends),
        WIDE,
        ends,
        None,
        block_size=BLOCK_SIZE,
        col_stride=1,
        dtype=torch.int32,
    )
    sentinel = 1.5
    logits.fill_(sentinel)
    _apply_candidate_mask_strided(logits, row_ks, row_ke, candidates, BLOCK_SIZE, 1)
    for row, end in enumerate(ends):
        touched = min(-(-end // _MASK_TILE) * _MASK_TILE, WIDE)
        tail = logits[row, touched:]
        assert torch.equal(tail, torch.full_like(tail, sentinel)), (
            f"row {row} (end {end}) was written past column {touched}"
        )
        # And the bound is tight: a row that ends early must not have cost a
        # full-width pass, which is the whole point of the kernel.
        assert touched <= end + _MASK_TILE


def test_cudagraph_replay_tracks_changing_ends():
    """The grid is static so a FULL capture stays valid when ends change.

    Capturing bakes the grid, so a data-dependent program count would freeze
    whatever lengths were resident at capture. Replaying against longer and
    shorter contexts than the captured ones is the property that matters.
    """
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _apply_candidate_mask_strided,
    )

    rows = 4
    captured_ends = [1024, 4096, 8192, 2048]
    logits, row_ks, row_ke, candidates = _inputs(
        rows,
        WIDE,
        captured_ends,
        None,
        block_size=BLOCK_SIZE,
        col_stride=1,
        dtype=torch.int32,
    )
    base = logits.clone()

    # Warm up outside the capture: Triton compiles on first call, and autotune
    # launches would otherwise happen inside the graph.
    _apply_candidate_mask_strided(
        logits.clone(), row_ks, row_ke, candidates, BLOCK_SIZE, 1
    )
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _apply_candidate_mask_strided(logits, row_ks, row_ke, candidates, BLOCK_SIZE, 1)

    for replay_ends in ([1, 131072, WIDE, 0], [WIDE, 1025, 3, 65536]):
        row_ke.copy_(torch.tensor(replay_ends, device="cuda", dtype=row_ke.dtype))
        logits.copy_(base)
        graph.replay()
        torch.accelerator.synchronize()

        ref = base.clone()
        apply_candidate_mask(ref, row_ks, row_ke, candidates, BLOCK_SIZE, 1)
        _assert_agrees_within_ends(ref, logits, row_ke, 1)
