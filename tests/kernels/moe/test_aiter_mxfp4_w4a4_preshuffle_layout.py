# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-layout contracts of the gfx1250 W4A4 MoE preshuffled path, checked
on CPU without launching a kernel:

- the oracle interleaves SILU gate/up rows as [g0, u0, g1, u1, ...], the order
  the fused swiglu epilogue reads;
- process_weights_after_loading shuffles the [E, K/2, N] views into a decode
  view whose last dim is N // 16 (the kernel launcher multiplies it back by 16).
"""

import pytest
import torch

pytest.importorskip("aiter.ops.shuffle")

from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a4_moe import (  # noqa: E402
    AiterW4A4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    _interleave_gate_up_rows,
)


def test_interleave_gate_up_rows_order():
    num_experts, n, inner = 2, 4, 3
    gate = torch.arange(num_experts * n * inner).view(num_experts, n, inner)
    up = gate + 1000
    out = _interleave_gate_up_rows(torch.cat([gate, up], dim=1))
    assert out.shape == (num_experts, 2 * n, inner)
    torch.testing.assert_close(out[:, 0::2], gate)
    torch.testing.assert_close(out[:, 1::2], up)


@pytest.mark.parametrize(("n", "k_half"), [(512, 256), (1536, 2048)])
def test_preshuffled_weight_view_restores_n_with_times_16(n, k_half):
    num_experts = 2
    layer = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        storage = torch.randint(0, 256, (num_experts, n, k_half), dtype=torch.uint8)
        # The oracle hands over [E, K/2, N] views of [E, N, K/2] storage.
        layer.register_parameter(
            name, torch.nn.Parameter(storage.transpose(1, 2), requires_grad=False)
        )
    originals = {name: getattr(layer, name).data.clone() for name in ("w13_weight",)}

    experts = AiterW4A4ExpertsMonolithic.__new__(AiterW4A4ExpertsMonolithic)
    experts.preshuffle_weights = True
    experts.process_weights_after_loading(layer)

    for name in ("w13_weight", "w2_weight"):
        w = getattr(layer, name)
        assert w.dtype == torch.uint8
        assert w.shape == (num_experts, k_half * 16, n // 16)
        # The launcher's correction: preshuffled N = last dim * 16.
        assert w.shape[-1] * 16 == n
    # The shuffle only reorders bytes within each expert.
    before = originals["w13_weight"].reshape(num_experts, -1).sort(dim=-1).values
    after = layer.w13_weight.data.reshape(num_experts, -1).sort(dim=-1).values
    assert torch.equal(before, after)


def test_unshuffled_weights_are_left_alone():
    layer = torch.nn.Module()
    w = torch.nn.Parameter(
        torch.zeros(1, 32, 16, dtype=torch.uint8), requires_grad=False
    )
    layer.register_parameter("w13_weight", w)
    experts = AiterW4A4ExpertsMonolithic.__new__(AiterW4A4ExpertsMonolithic)
    experts.preshuffle_weights = False
    experts.process_weights_after_loading(layer)
    assert layer.w13_weight is w
