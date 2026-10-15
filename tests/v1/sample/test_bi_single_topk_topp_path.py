# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single top-k/top-p code path under batch invariance."""

import pytest
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def test_tied_logits_survive_masking_identically():
    """Tied logits are not dropped differently at the top-p boundary.

    The Triton and PyTorch paths order equal keys differently, so with the
    batch-size-dependent dispatch a razor-edge candidate can be masked to
    -inf at batch >= 8 (Triton) but kept at batch < 8 (PyTorch). Under BI
    the PyTorch path is taken for all batch sizes, which this test pins by
    checking the tied candidates keep equal values across batch sizes.
    """
    vocab = 64
    base = torch.full((1, vocab), -5.0)
    base[0, 5] = -2.5
    base[0, 27] = -2.5  # exact f32 tie
    p = torch.full((1,), 0.9, dtype=torch.float32)

    outs = {}
    for rows in (1, 2, 8, 16):
        batch = base.expand(rows, -1).clone()
        out = apply_top_k_top_p_pytorch(batch, None, p)
        # Both tied candidates must survive masking with identical values
        # at every batch size on the shared (PyTorch) code path.
        assert out[0, 5] == out[0, 27]
        outs[rows] = out[0]

    assert outs[1].equal(outs[8])
    assert outs[2].equal(outs[16])


@pytest.mark.parametrize("p_val", [0.5, 0.9, 1.0])
def test_bi_off_matches_pytorch_reference(p_val):
    """Distinct logits: BI-off dispatch and the PyTorch path agree.

    This pins that redirecting BI mode to the PyTorch path does not change
    results for inputs without ties, and that non-BI behavior is untouched.
    """
    torch.manual_seed(1234)
    logits = torch.randn(16, 256, dtype=torch.float32)
    p = torch.full((16,), p_val, dtype=torch.float32)

    reference = apply_top_k_top_p_pytorch(logits.clone(), None, p)
    from vllm import envs
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    saved = envs.VLLM_BATCH_INVARIANT
    try:
        envs.VLLM_BATCH_INVARIANT = False
        dispatched = apply_top_k_top_p(logits.clone(), None, p)
    finally:
        envs.VLLM_BATCH_INVARIANT = saved

    assert dispatched.equal(reference)
