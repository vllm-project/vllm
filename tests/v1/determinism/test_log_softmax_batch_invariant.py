# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance and correctness tests for the split-reduction log_softmax.

The batch-invariant ``log_softmax`` reduces each row across a fixed number of
column-splits that depends only on the vocab width (``n_cols``), never on the
batch size. These tests assert both that a row's output does not depend on how
many rows share the launch, and that the split reduction still matches a
reference log_softmax.
"""

import pytest
import torch
from utils import skip_if_not_cuda

from vllm.model_executor.layers.batch_invariant import log_softmax
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_if_not_cuda
@pytest.mark.parametrize("vocab_size", [2048, 32000, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log_softmax_batch_invariance(vocab_size: int, dtype: torch.dtype):
    """A row's log_softmax is identical alone vs. embedded in a larger batch.

    Enabled under ``VLLM_BATCH_INVARIANT=1`` by the autouse fixture in
    conftest.py. The split count is a function of ``vocab_size`` only, so the
    per-row reduction order must not change with batch size.
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)

    row = torch.randn(1, vocab_size, dtype=dtype, device=device) * 5.0
    out_single = log_softmax(row, dim=-1)

    # Embed the same row at several positions in a large batch whose size
    # crosses well past the single-row launch.
    batch = torch.randn(512, vocab_size, dtype=dtype, device=device) * 5.0
    for pos in (0, 137, 511):
        batch[pos] = row[0]
    out_batch = log_softmax(batch, dim=-1)

    for pos in (0, 137, 511):
        assert torch.equal(out_single[0], out_batch[pos]), (
            f"log_softmax row differs when batch context changes at pos={pos}"
        )


@skip_if_not_cuda
@pytest.mark.parametrize("vocab_size", [1000, 32000, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log_softmax_matches_reference(vocab_size: int, dtype: torch.dtype):
    """Split-reduction log_softmax stays close to torch's reference."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)

    x = torch.randn(8, vocab_size, dtype=dtype, device=device) * 5.0
    out = log_softmax(x, dim=-1)
    ref = torch.log_softmax(x.float(), dim=-1).to(dtype)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@skip_if_not_cuda
def test_log_softmax_invariant_across_batch_launch_boundaries():
    """Output for a fixed row is stable across many batch sizes.

    Exercises the small-batch decode regime (few rows, wide vocab) that the
    split reduction targets, sweeping batch sizes on both sides of typical SM
    occupancy so any batch-size-dependent reduction order would surface.
    """
    device = torch.device(DEVICE_TYPE)
    vocab_size = 151936
    dtype = torch.bfloat16
    torch.manual_seed(0)

    row = torch.randn(vocab_size, dtype=dtype, device=device) * 5.0
    ref = log_softmax(row.view(1, -1), dim=-1)[0]

    for batch_size in (1, 2, 8, 33, 64, 128, 257):
        batch = torch.randn(batch_size, vocab_size, dtype=dtype, device=device) * 5.0
        batch[batch_size // 2] = row
        out = log_softmax(batch, dim=-1)
        assert torch.equal(out[batch_size // 2], ref), (
            f"log_softmax not batch-invariant at batch_size={batch_size}"
        )
