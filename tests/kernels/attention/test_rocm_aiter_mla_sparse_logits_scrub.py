# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the bounded decode paged-MQA-logits scrub.

``out_logits`` is a ``[rows, max_model_len]`` fp32 workspace that is reused
across steps, so whatever the paged-MQA-logits kernel does not write still
holds values from earlier steps -- including NaN, which the top-k histogram
cannot rank. vllm#49714 scrubbed the whole workspace with
``nan_to_num_(-inf)``; ``scrub_decode_logits`` scrubs only the window the
top-k actually reads.

The property that matters is that the two agree everywhere the top-k looks:
``top_k_per_row_decode`` scans row ``r`` over
``[0, seq_len[r // next_n] - next_n + r % next_n + 1)``. These tests pin that
window against ``nan_to_num_(-inf)`` for both the 1-D and 2-D ``seq_lens``
forms, pin the sub-crossover fallback to the old path, and pin the launch grid
to the row count so CUDA-graph capture stays valid across steps.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_ops

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="the bounded decode-logits scrub is a ROCm path",
)

NEG_INF = float("-inf")


def _row_end(seq_lens: torch.Tensor, row: int, next_n: int) -> int:
    """The bound ``top_k_per_row_decode`` reads, in plain Python."""
    if seq_lens.dim() == 2:
        end = int(seq_lens.view(-1)[row])
    else:
        end = int(seq_lens[row // next_n]) - next_n + (row % next_n) + 1
    return max(end, 0)


def _poisoned(rows: int, cols: int) -> torch.Tensor:
    """A workspace holding every value the scrub has to decide on."""
    torch.manual_seed(0)
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
    flat = x.view(-1)
    flat[0::7] = float("nan")
    flat[1::11] = float("inf")
    flat[2::13] = NEG_INF
    return x


@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("seq_lens_2d", [False, True])
def test_matches_nan_to_num_over_the_read_window(next_n, seq_lens_2d):
    batch, cols = 5, 4096
    rows = batch * next_n
    logits = _poisoned(rows, cols)
    reference = logits.clone().nan_to_num_(NEG_INF)

    # Lengths that exercise the edges: one empty, one full-width, and one that
    # would go negative for the first row of a multi-token block.
    lens = torch.tensor([0, 1, 37, 2048, cols], dtype=torch.int32, device="cuda")
    if seq_lens_2d:
        seq_lens = torch.stack(
            [(lens - next_n + i + 1).clamp(min=0) for i in range(next_n)], dim=1
        ).contiguous()
    else:
        seq_lens = lens

    # Force the Triton path: the real crossover needs a 40MB workspace.
    with patch.object(sparse_ops, "_SCRUB_MIN_ELEMS", 0):
        sparse_ops.scrub_decode_logits(logits, seq_lens, next_n)

    for row in range(rows):
        end = _row_end(seq_lens, row, next_n)
        torch.testing.assert_close(
            logits[row, :end], reference[row, :end], rtol=0, atol=0, equal_nan=True
        )


def test_leaves_the_unread_tail_alone():
    """Nothing past the top-k bound is touched -- it is never read."""
    next_n, cols = 1, 4096
    logits = _poisoned(4, cols)
    before = logits.clone()
    seq_lens = torch.tensor([16, 512, 0, cols], dtype=torch.int32, device="cuda")

    with patch.object(sparse_ops, "_SCRUB_MIN_ELEMS", 0):
        sparse_ops.scrub_decode_logits(logits, seq_lens, next_n)

    for row in range(logits.shape[0]):
        end = _row_end(seq_lens, row, next_n)
        torch.testing.assert_close(
            logits[row, end:], before[row, end:], rtol=0, atol=0, equal_nan=True
        )


def test_nan_maps_to_neg_inf_not_neg_flt_max():
    """Every case is decided against the original value, as nan_to_num_ does."""
    next_n = 1
    logits = torch.tensor(
        [[float("nan"), float("inf"), NEG_INF, 1.5]],
        dtype=torch.float32,
        device="cuda",
    )
    seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")

    with patch.object(sparse_ops, "_SCRUB_MIN_ELEMS", 0):
        sparse_ops.scrub_decode_logits(logits, seq_lens, next_n)

    flt_max = torch.finfo(torch.float32).max
    assert logits[0, 0].item() == NEG_INF
    assert logits[0, 1].item() == flt_max
    assert logits[0, 2].item() == -flt_max
    assert logits[0, 3].item() == 1.5


def test_below_the_crossover_runs_the_old_path():
    """Small batches stay bit-identical to today's code, kernel never launched."""
    next_n, cols = 1, 1024
    logits = _poisoned(4, cols)
    reference = logits.clone().nan_to_num_(NEG_INF)
    seq_lens = torch.tensor([8, 8, 8, 8], dtype=torch.int32, device="cuda")

    assert logits.numel() < sparse_ops._SCRUB_MIN_ELEMS
    with patch.object(sparse_ops, "_scrub_decode_logits_kernel") as kernel:
        sparse_ops.scrub_decode_logits(logits, seq_lens, next_n)
    kernel.__getitem__.assert_not_called()

    torch.testing.assert_close(logits, reference, rtol=0, atol=0, equal_nan=True)


def test_empty_workspace_is_a_no_op():
    logits = torch.empty(0, 4096, dtype=torch.float32, device="cuda")
    seq_lens = torch.empty(0, dtype=torch.int32, device="cuda")
    with patch.object(sparse_ops, "_scrub_decode_logits_kernel") as kernel:
        sparse_ops.scrub_decode_logits(logits, seq_lens, 1)
    kernel.__getitem__.assert_not_called()


def test_grid_depends_only_on_the_row_count():
    """The grid is a shape, so CUDA-graph capture stays valid across steps."""
    next_n, rows, cols = 1, 64, 4096
    logits = torch.zeros(rows, cols, dtype=torch.float32, device="cuda")

    grids = []

    class _Spy:
        def __getitem__(self, grid):
            grids.append(grid)
            return lambda *a, **kw: None

    with (
        patch.object(sparse_ops, "_SCRUB_MIN_ELEMS", 0),
        patch.object(sparse_ops, "_scrub_decode_logits_kernel", _Spy()),
    ):
        for fill in (1, 977, cols):
            seq_lens = torch.full((rows,), fill, dtype=torch.int32, device="cuda")
            sparse_ops.scrub_decode_logits(logits, seq_lens, next_n)

    assert len(grids) == 3
    assert len(set(grids)) == 1, grids
    assert grids[0][0] == rows
