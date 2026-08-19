# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ZenTorch CPU fused MoE dispatch and forward."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.moe.test_cpu_fused_moe import (
    UNALIGNED_INTERMEDIATE_DIM,
    _make_moe_config,
    _StubMoELayer,
    ref_fused_moe,
)
from vllm.model_executor.kernels.linear.zentorch_utils import (
    has_zentorch_op,
    is_zentorch_moe_config_supported,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    CPUUnquantizedExperts,
    select_experts,
)
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


def _make_layer_and_config(
    act: MoEActivation,
    *,
    hidden_size: int = 128,
    intermediate_size: int = 128,
    expert_num: int = 8,
    topk_num: int = 4,
    use_bias: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[_StubMoELayer, FusedMoEConfig]:
    up_dim = 2 * intermediate_size
    w13 = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )
    w13_bias = None
    w2_bias = None
    if use_bias:
        w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )

    layer = _StubMoELayer(
        w13,
        w2,
        act,
        w13_bias,
        w2_bias,
    )
    moe_config = _make_moe_config(
        expert_num,
        hidden_size,
        intermediate_size,
        topk_num,
        dtype,
        act,
        use_bias,
    )
    return layer, moe_config


@pytest.mark.parametrize("act", ZENTORCH_ACT)
def test_zen_cpu_fused_moe_dispatches_to_zentorch(act: MoEActivation):
    """When zentorch MoE is supported, experts select the zentorch path."""
    layer, moe_config = _make_layer_and_config(act)
    w13_before = layer.w13_weight.detach().clone()
    w2_before = layer.w2_weight.detach().clone()

    experts = CPUUnquantizedExperts(moe_config, FusedMoEQuantConfig.make())
    experts.process_weights_after_loading(layer)

    assert experts._use_zentorch is True
    torch.testing.assert_close(layer.w13_weight, w13_before)
    torch.testing.assert_close(layer.w2_weight, w2_before)


def test_zen_cpu_fused_moe_config_supported_unaligned_intermediate():
    """Zentorch accepts shapes the grouped-gemm kernel rejects."""
    act = MoEActivation.SILU
    moe_config = _make_moe_config(
        expert_num=8,
        hidden_size=128,
        intermediate_size=UNALIGNED_INTERMEDIATE_DIM,
        topk_num=4,
        dtype=torch.bfloat16,
        act=act,
    )
    assert is_zentorch_moe_config_supported(moe_config)

    supported, reason = CPUUnquantizedExperts.is_supported_config(
        CPUUnquantizedExperts,
        moe_config,
        None,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported, reason


@pytest.mark.parametrize("act", ZENTORCH_ACT)
@pytest.mark.parametrize("use_bias", USE_BIAS)
def test_zen_cpu_fused_moe_forward(
    default_vllm_config,
    act: MoEActivation,
    use_bias: bool,
):
    """The zentorch forward runs end-to-end and matches the reference."""
    set_random_seed(0)

    expert_num = 8
    hidden_size = 128
    batch_size = 64
    dtype = torch.bfloat16
    topk_num = expert_num // 2

    layer, moe_config = _make_layer_and_config(
        act,
        expert_num=expert_num,
        topk_num=topk_num,
        use_bias=use_bias,
    )

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)

    quant_config = (
        biased_moe_quant_config(layer.w13_bias, layer.w2_bias)
        if use_bias
        else FusedMoEQuantConfig.make()
    )
    experts = CPUUnquantizedExperts(moe_config, quant_config)
    experts.process_weights_after_loading(layer)
    assert experts._use_zentorch is True

    topk_weight, topk_ids = select_experts(
        hidden_states=input,
        router_logits=router_logits,
        top_k=topk_num,
        use_grouped_topk=False,
        renormalize=False,
    )

    output = experts.apply(
        hidden_states=input,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        router_logits=router_logits,
        activation=act,
        global_num_experts=expert_num,
        expert_map=None,
        a1q_scale=None,
        apply_router_weight_on_input=False,
    )

    ref_output = ref_fused_moe(
        input,
        layer.w13_weight,
        layer.w2_weight,
        getattr(layer, "w13_bias", None),
        getattr(layer, "w2_bias", None),
        topk_weight,
        topk_ids,
        act,
    )

    atol, rtol = get_default_atol(output), get_default_rtol(output)
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


def test_zen_cpu_fused_moe_skips_without_activation():
    """Layers without an activation attribute fall back from zentorch."""
    layer = SimpleNamespace()
    moe_config = _make_moe_config(
        expert_num=8,
        hidden_size=128,
        intermediate_size=128,
        topk_num=4,
        dtype=torch.bfloat16,
        act=MoEActivation.SILU,
    )
    experts = CPUUnquantizedExperts(moe_config, FusedMoEQuantConfig.make())
    experts.process_weights_after_loading(layer)

    assert experts._use_zentorch is False
