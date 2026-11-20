# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate/up global-scale fold for NVFP4 MoE exports whose w1 and w3 differ.

CPU only. The fused w13 kernels dequantize both halves of w13 with the gate's
global scale; the fold moves the up/gate ratio into the w2 global scale so an
`act(gate) * up` expert computes what the checkpoint encodes.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.modelopt import (
    fold_w13_global_scale_into_w2,
)


def _expert_out(x, gate_q, up_q, w2_q, gs_gate, gs_up, gs_w2):
    """act(x @ gate) * (x @ up) @ w2, with per-tensor global scales applied to
    the quantized (here: unscaled fp32) weights, as the kernels dequantize them."""
    gate = x @ (gate_q * gs_gate).T
    up = x @ (up_q * gs_up).T
    return (F.silu(gate) * up) @ (w2_q * gs_w2).T


@pytest.mark.parametrize("ratio", [0.125, 0.5, 2.0, 8.0, 1.37])
def test_fold_restores_checkpoint_math(ratio):
    torch.manual_seed(0)
    num_experts, hidden, inter = 3, 32, 16
    x = torch.randn(4, hidden)
    gate_q = torch.randn(num_experts, inter, hidden)
    up_q = torch.randn(num_experts, inter, hidden)
    w2_q = torch.randn(num_experts, hidden, inter)
    gs_gate = torch.tensor([0.01, 0.02, 0.03])
    gs_up = gs_gate * ratio
    gs_w2 = torch.tensor([0.05, 0.04, 0.06])

    w13_gs = torch.stack([gs_gate, gs_up], dim=1)
    w2_gs = gs_w2.clone()
    returned = fold_w13_global_scale_into_w2(w13_gs, w2_gs)
    torch.testing.assert_close(returned, torch.full((num_experts,), ratio))
    torch.testing.assert_close(w2_gs, gs_w2 * ratio)

    for e in range(num_experts):
        want = _expert_out(
            x, gate_q[e], up_q[e], w2_q[e], gs_gate[e], gs_up[e], gs_w2[e]
        )
        # The kernel uses the gate's scale for up, and the folded w2 scale.
        got = _expert_out(
            x, gate_q[e], up_q[e], w2_q[e], gs_gate[e], gs_gate[e], w2_gs[e]
        )
        stale = _expert_out(
            x, gate_q[e], up_q[e], w2_q[e], gs_gate[e], gs_gate[e], gs_w2[e]
        )
        scale = want.abs().max()
        assert (got - want).abs().max() <= 1e-5 * scale
        # Without the fold the up half is off by 1/r and the output is wrong.
        assert (stale - want).abs().max() > 1e-2 * scale


def test_fold_keeps_zero_gate_experts_finite():
    """An expert whose gate scale is 0 (an all-zero export) keeps r = 1 and a
    finite w2 scale, as it had before the fold."""
    w13_gs = torch.tensor([[0.0, 0.0], [0.0, 2.0], [1.0, 4.0]])
    w2_gs = torch.tensor([3.0, 5.0, 7.0])
    ratio = fold_w13_global_scale_into_w2(w13_gs, w2_gs)
    torch.testing.assert_close(ratio, torch.tensor([1.0, 1.0, 4.0]))
    torch.testing.assert_close(w2_gs, torch.tensor([3.0, 5.0, 28.0]))
    assert torch.isfinite(w2_gs).all()
