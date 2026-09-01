# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Zen CPU INT8 MoE experts (ZenCPUExpertsInt8)."""

import pytest
import torch

from tests.kernels.moe.test_cpu_fused_moe import (
    _make_moe_config,
    quantize_per_channel,
    ref_fused_moe_int8,
)
from vllm.model_executor.kernels.linear.zentorch_utils import (
    _ZENTORCH_MOE_ACTIVATIONS,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    ZenCPUExpertsInt8,
    select_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

EXPERT_NUM = 8
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 128
BATCH_SIZE = 32
# zentorch names its activations as strings; the experts API is enum-typed.
ZENTORCH_ACT = [act for act in MoEActivation if act.value in _ZENTORCH_MOE_ACTIVATIONS]


@pytest.fixture
def mock_zentorch_fused_moe():
    """Reference zentorch_fused_moe, registered when zentorch is absent."""
    if hasattr(torch.ops.zentorch, "zentorch_fused_moe"):
        yield None
        return

    calls: dict[str, dict] = {}

    def _fused_moe(
        output,
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_id,
        skip_weighted,
        act,
        w13_scales=None,
        w2_scales=None,
        zentorch_op_name="",
    ):
        calls["fused_moe"] = {
            "w13": w13,
            "w2": w2,
            "w13_bias": w13_bias,
            "w2_bias": w2_bias,
            "topk_weights": topk_weights,
            "topk_id": topk_id,
            "skip_weighted": skip_weighted,
            "act": act,
            "w13_scales": w13_scales,
            "w2_scales": w2_scales,
        }
        output.copy_(
            ref_fused_moe_int8(
                input,
                w13,
                w2,
                w13_scales.float(),
                w2_scales.float(),
                None if w13_bias is None else w13_bias.float(),
                None if w2_bias is None else w2_bias.float(),
                topk_weights,
                topk_id,
                MoEActivation(act),
            )
        )

    lib_def = torch.library.Library("zentorch", "DEF")
    lib_def.define(
        "zentorch_fused_moe(Tensor(a!) output, Tensor input, Tensor w13, "
        "Tensor w2, Tensor? w13_bias, Tensor? w2_bias, Tensor topk_weights, "
        "Tensor topk_id, bool skip_weighted, str act, "
        "Tensor? w13_scales=None, Tensor? w2_scales=None, *, "
        "str zentorch_op_name='zentorch::zentorch_fused_moe') -> ()"
    )
    lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
    lib_impl.impl("zentorch_fused_moe", _fused_moe)

    yield calls

    lib_impl._destroy()
    lib_def._destroy()


def _make_int8_moe_weights(use_bias: bool, dtype: torch.dtype = torch.bfloat16):
    up_dim = 2 * INTERMEDIATE_SIZE
    w13_fp = torch.randn((EXPERT_NUM, up_dim, HIDDEN_SIZE), dtype=dtype) / (
        0.5 * HIDDEN_SIZE**0.5
    )
    w2_fp = torch.randn((EXPERT_NUM, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype) / (
        0.5 * INTERMEDIATE_SIZE**0.5
    )
    w13, w13_scale = quantize_per_channel(w13_fp)
    w2, w2_scale = quantize_per_channel(w2_fp)
    w13_scale = w13_scale.to(torch.bfloat16).float()
    w2_scale = w2_scale.to(torch.bfloat16).float()

    w13_bias = w2_bias = None
    if use_bias:
        w13_bias = torch.randn((EXPERT_NUM, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((EXPERT_NUM, HIDDEN_SIZE), dtype=dtype) / (
            0.5 * HIDDEN_SIZE**0.5
        )
    return w13, w2, w13_scale, w2_scale, w13_bias, w2_bias


def test_zen_int8_first_in_cpu_backend_priority():
    """The Zen experts are preferred over the generic CPU int8 experts."""
    assert backend_to_kernel_cls(Int8MoeBackend.CPU)[0] is ZenCPUExpertsInt8


def test_zen_int8_support_predicates(mock_zentorch_fused_moe, monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)
    assert not ZenCPUExpertsInt8._supports_current_device()
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    assert ZenCPUExpertsInt8._supports_current_device()
    assert ZenCPUExpertsInt8._supports_quant_scheme(
        kInt8StaticChannelSym, kInt8DynamicTokenSym
    )
    assert not ZenCPUExpertsInt8._supports_quant_scheme(kInt8StaticChannelSym, None)
    for act in ZENTORCH_ACT:
        assert ZenCPUExpertsInt8._supports_activation(act)
    assert not ZenCPUExpertsInt8._supports_activation(MoEActivation.RELU2)
    assert not ZenCPUExpertsInt8._supports_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE
    )
    assert not ZenCPUExpertsInt8._supports_no_act_and_mul()
    assert ZenCPUExpertsInt8._supports_parallel_config(
        FusedMoEParallelConfig.make_no_parallel()
    )


@pytest.mark.parametrize("act", ZENTORCH_ACT)
@pytest.mark.parametrize("use_bias", [False, True])
def test_zen_int8_dispatch_contract(
    default_vllm_config,
    mock_zentorch_fused_moe,
    act: MoEActivation,
    use_bias: bool,
):
    """apply() hands zentorch the weights, scales and biases it expects."""
    if mock_zentorch_fused_moe is None:
        pytest.skip("real zentorch is installed; the recorded call is unavailable")
    set_random_seed(0)

    topk_num = EXPERT_NUM // 2
    w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = _make_int8_moe_weights(use_bias)
    moe_config = _make_moe_config(
        EXPERT_NUM,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        topk_num,
        torch.bfloat16,
        act,
        use_bias,
    )
    # A layer carries per-channel scales as [E, N, 1].
    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=w13_scale.unsqueeze(-1),
        w2_scale=w2_scale.unsqueeze(-1),
        a1_scale=None,
        a2_scale=None,
        w1_bias=w13_bias,
        w2_bias=w2_bias,
        per_act_token_quant=True,
    )

    input = torch.randn((BATCH_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16) / (
        0.5 * HIDDEN_SIZE**0.5
    )
    router_logits = torch.randn((BATCH_SIZE, EXPERT_NUM), dtype=torch.bfloat16)

    experts = ZenCPUExpertsInt8(moe_config, quant_config)
    # zentorch_fused_moe takes no expert_map.
    assert not experts.supports_expert_map()
    output = experts.apply(
        hidden_states=input,
        w1=w13,
        w2=w2,
        router_logits=router_logits,
        activation=act,
        global_num_experts=EXPERT_NUM,
        expert_map=None,
        a1q_scale=None,
        apply_router_weight_on_input=False,
    )

    call = mock_zentorch_fused_moe["fused_moe"]
    assert call["w13_scales"].shape == (EXPERT_NUM, 2 * INTERMEDIATE_SIZE)
    assert call["w2_scales"].shape == (EXPERT_NUM, HIDDEN_SIZE)
    assert call["w13_scales"].dtype == torch.bfloat16
    assert call["w2_scales"].dtype == torch.bfloat16
    assert call["act"] == act.value.lower()
    assert call["skip_weighted"] is False
    assert call["topk_weights"].dtype == torch.float32
    assert call["topk_id"].dtype == torch.int32
    if use_bias:
        assert call["w13_bias"].shape == (EXPERT_NUM, 2 * INTERMEDIATE_SIZE)
        assert call["w2_bias"].shape == (EXPERT_NUM, HIDDEN_SIZE)
    else:
        assert call["w13_bias"] is None and call["w2_bias"] is None

    topk_weights, topk_ids = select_experts(
        hidden_states=input,
        router_logits=router_logits,
        use_grouped_topk=False,
        top_k=topk_num,
        renormalize=False,
        scoring_func="softmax",
    )
    ref = ref_fused_moe_int8(
        input,
        w13,
        w2,
        w13_scale,
        w2_scale,
        None if w13_bias is None else w13_bias.float(),
        None if w2_bias is None else w2_bias.float(),
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
        act,
    )
    torch.testing.assert_close(output, ref)
