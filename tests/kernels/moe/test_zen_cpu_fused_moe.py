# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ZenTorch CPU fused MoE dispatch and forward."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.test_cpu_fused_moe import ref_fused_moe
from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import CPUFusedMOE
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu() or not current_platform.is_zen_cpu():
    pytest.skip("skipping non-Zen CPU tests", allow_module_level=True)

if not has_zentorch_op(["zentorch_fused_moe"]):
    pytest.skip(
        "skipping tests: zentorch_fused_moe op not available",
        allow_module_level=True,
    )

ZENTORCH_ACT = [
    MoEActivation.SILU,
    MoEActivation.GELU,
    MoEActivation.GELU_TANH,
    MoEActivation.SWIGLUOAI,
]
USE_BIAS = [False, True]


@pytest.mark.parametrize("act", ZENTORCH_ACT)
def test_zen_cpu_fused_moe_dispatches_to_zentorch(act: MoEActivation):
    """When zentorch MoE is supported, ``CPUFusedMOE`` selects the zentorch
    forward and records the lowercased activation string."""
    layer = SimpleNamespace(activation=act)
    moe = CPUFusedMOE(layer)

    assert moe.isa == "none"
    assert moe.act == act.value.lower()
    assert moe.forward_method.__func__ is CPUFusedMOE.forward_zentorch


@pytest.mark.parametrize("use_bias", USE_BIAS)
def test_zen_cpu_fused_moe_forward(
    default_vllm_config,
    use_bias: bool,
):
    """The zentorch forward runs end-to-end and returns a valid output tensor."""
    set_random_seed(0)

    act = MoEActivation.SILU
    expert_num = 8
    hidden_size = 128
    intermediate_size = 128
    batch_size = 64
    dtype = torch.bfloat16
    topk_num = expert_num // 2
    up_dim = 2 * intermediate_size

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w13 = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )
    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)

    layer = SimpleNamespace(activation=act, w13_weight=w13, w2_weight=w2)
    if use_bias:
        layer.w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (
            0.5 * up_dim**0.5
        )
        layer.w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )

    moe = CPUFusedMOE(layer)

    score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk_num)
    topk_ids = topk_ids.to(torch.int32)

    output = moe.forward_method(
        layer,
        input,
        topk_weight,
        topk_ids,
        act,
    )

    ref_output = ref_fused_moe(
        input,
        w13,
        w2,
        getattr(layer, "w13_bias", None),
        getattr(layer, "w2_bias", None),
        topk_weight,
        topk_ids,
        act,
    )

    torch.testing.assert_close(output, ref_output)
