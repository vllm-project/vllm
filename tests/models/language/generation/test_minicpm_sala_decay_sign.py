# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the MiniCPM-SALA lightning-attention decay SIGN.

This guards a *critical* class of bug that is invisible at TP=1-vs-TP>1
level and does not require a GPU, a checkpoint, or a distributed group to
catch: the sign of the per-head decay rate handed to the reused lightning
kernels.

Root cause it guards against
----------------------------
``vllm.model_executor.layers.lightning_attn`` computes the decay as
``exp(-rate * distance)`` in every path:

  * prefill diag kernel:      ``s_index = where(diff>=0, -s*diff, -inf); exp(s_index)``
  * prefill reduce kernel:    ``exp(-s * block_size)``
  * decode kernel:            ``ratio = exp(-ratio)``

so ``rate`` (== ``s``) MUST be strictly positive for a bounded decay. An
earlier revision multiplied the ALiBi slope by ``-1.0`` at the call site,
producing ``exp(+|s| * distance)`` -- attention weights that GROW with
distance and overflow to Inf/NaN on any non-trivial sequence. Because the
sign lives in ``__init__`` (not a returned value), no correctness test that
stops short of a full forward pass would have caught it. Extracting the
convention into ``build_lightning_decay_rate`` gives this test a seam.
"""

import math

import pytest
import torch

from vllm.model_executor.models.minicpm_sala import (
    build_alibi_slopes,
    build_lightning_decay_rate,
)

pytestmark = pytest.mark.hybrid_model


def test_lightning_decay_rate_is_strictly_positive() -> None:
    """CRITICAL: the kernels apply exp(-rate*distance); a non-positive
    rate inverts the decay into unbounded growth (Inf/NaN)."""
    for num_heads in (24, 32, 40, 64):
        rate = build_lightning_decay_rate(num_heads)
        assert torch.all(rate > 0), (
            f"decay rate for num_heads={num_heads} must be strictly "
            f"positive; got min={rate.min().item()}. A negative rate makes "
            f"exp(-rate*distance) grow with distance and overflow to NaN."
        )


def test_decay_rate_matches_positive_alibi_magnitude() -> None:
    """The seam must be a straight pass-through of the positive slope
    magnitude -- not negated, not re-scaled."""
    for num_heads in (32, 40):
        assert torch.equal(
            build_lightning_decay_rate(num_heads), build_alibi_slopes(num_heads)
        )


def test_positive_rate_actually_decays_over_distance() -> None:
    """End-to-end sanity on the convention: with the correct (positive)
    rate, exp(-rate*distance) is a bounded decay in (0, 1]; with the old
    negated rate it exceeds 1 and blows up. This is the exact arithmetic
    the Triton kernels perform, reproduced in plain PyTorch.

    Uses float64 so long-distance decay stays representable (float32
    underflows to 0.0 for head-0 slope by distance ~124).
    """
    rate = build_lightning_decay_rate(32)[0].item()  # head 0, largest slope
    # float64: the largest slope (~0.84) drives exp(-rate*d) below the float32
    # subnormal floor by distance ~124, underflowing to exactly 0.0. That underflow
    # is correct decay behavior, not a sign bug — use float64 so the arithmetic
    # stays representable and the *convention* is what's under test.
    distances = torch.arange(1, 512, dtype=torch.float64)

    correct = torch.exp(-rate * distances)
    assert torch.all(correct <= 1.0) and torch.all(correct > 0.0)
    assert torch.isfinite(correct).all()
    assert torch.all(correct[1:] <= correct[:-1])

    # The regression: negating the rate produces monotonically growing,
    # overflowing "decay".
    negated_rate = -rate  # the buggy sign an earlier revision produced
    blown_up = torch.exp(-negated_rate * distances)
    assert blown_up.max().item() > 1e30, (
        "sanity: a negated rate should demonstrably explode -- if this "
        "assertion fails the test itself is mischaracterizing the bug"
    )


def test_head0_slope_closed_form() -> None:
    """Anchor the magnitude to the closed-form ALiBi value so a change to
    build_alibi_slopes that happens to stay positive but wrong is caught."""
    assert math.isclose(
        build_lightning_decay_rate(32)[0].item(),
        2 ** (-8 / 32),
        rel_tol=1e-6,
    )
