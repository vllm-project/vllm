# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import torch

import vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe as fi_cutlass_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    make_fp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    prepare_fp8_moe_layer_for_fi,
)


def test_flashinfer_cutlass_ptpc_quant_config_preserves_dynamic_scales():
    w1_scale = torch.ones((2, 4), dtype=torch.float32)
    w2_scale = torch.ones((2, 3), dtype=torch.float32)

    quant_config = make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.FLASHINFER_CUTLASS,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )

    assert quant_config.per_act_token_quant
    assert quant_config.per_out_ch_quant
    assert quant_config.a1_scale is None
    assert quant_config.a2_scale is None
    assert quant_config.w1_scale is w1_scale
    assert quant_config.w2_scale is w2_scale


def test_flashinfer_cutlass_ptpc_apply_passes_activation_and_weight_scales(
    monkeypatch,
):
    core_module = types.ModuleType("flashinfer.fused_moe.core")

    class ActivationType:
        Swiglu = "swiglu"
        Relu2 = "relu2"

    core_module.ActivationType = ActivationType
    fused_moe_module = types.ModuleType("flashinfer.fused_moe")
    fused_moe_module.core = core_module
    flashinfer_module = types.ModuleType("flashinfer")
    flashinfer_module.fused_moe = fused_moe_module
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe_module)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core_module)

    captured_kwargs = {}

    def fake_flashinfer_cutlass_fused_moe(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(
        fi_cutlass_moe,
        "flashinfer_cutlass_fused_moe",
        fake_flashinfer_cutlass_fused_moe,
    )

    w1_scale = torch.ones((2, 4), dtype=torch.float32)
    w2_scale = torch.ones((2, 3), dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )
    experts = object.__new__(FlashInferExperts)
    experts.quant_config = quant_config
    experts.use_deepseek_fp8_block_scale = False
    experts.out_dtype = torch.bfloat16
    experts.tp_size = 1
    experts.tp_rank = 0
    experts.ep_size = 1
    experts.ep_rank = 0
    experts.max_capture_size = 1

    hidden_states = torch.ones((3, 5), dtype=torch.bfloat16)
    w1 = torch.empty((2, 4, 5), dtype=torch.uint8)
    w2 = torch.empty((2, 5, 3), dtype=torch.uint8)
    output = torch.empty((3, 5), dtype=torch.bfloat16)
    topk_weights = torch.ones((3, 1), dtype=torch.float32)
    topk_ids = torch.zeros((3, 1), dtype=torch.int64)
    a1q_scale = torch.ones((3, 1), dtype=torch.float32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    quant_scales = captured_kwargs["quant_scales"]
    assert quant_scales[0] is w1_scale
    assert quant_scales[1] is w2_scale
    assert captured_kwargs["input_sf"] is a1q_scale
    assert captured_kwargs["fc1_expert_weights"] is w1
    assert captured_kwargs["fc2_expert_weights"] is w2


def test_flashinfer_cutlass_prepare_pads_and_swaps_per_channel_w13_scales():
    intermediate = 3
    hidden_size = 5
    padded_intermediate = 16
    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(
            is_act_and_mul=True,
            intermediate_size_per_partition=intermediate,
        ),
        activation=MoEActivation.SILU,
    )

    w13 = torch.arange(
        1,
        1 + 2 * intermediate * hidden_size,
        dtype=torch.uint8,
    ).reshape(1, 2 * intermediate, hidden_size)
    w2 = torch.ones((1, hidden_size, intermediate), dtype=torch.uint8)
    w13_scale = torch.zeros((1, 2 * intermediate, 1), dtype=torch.float32)
    w13_scale[0, :intermediate, 0] = torch.tensor([1.0, 2.0, 3.0])
    w13_scale[0, intermediate:, 0] = torch.tensor([11.0, 12.0, 13.0])
    w2_scale = torch.ones((1, hidden_size, 1), dtype=torch.float32)

    _, padded_w2, padded_w13_scale, padded_w2_scale = prepare_fp8_moe_layer_for_fi(
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w13_input_scale=None,
        w2_scale=w2_scale,
        w2_input_scale=None,
        per_out_ch_quant=True,
    )

    expected_w13_scale = torch.zeros(
        (1, 2 * padded_intermediate, 1),
        dtype=torch.float32,
    )
    expected_w13_scale[0, :intermediate, 0] = torch.tensor([11.0, 12.0, 13.0])
    expected_w13_scale[
        0, padded_intermediate : padded_intermediate + intermediate, 0
    ] = torch.tensor([1.0, 2.0, 3.0])

    assert padded_w2.shape == (1, hidden_size, padded_intermediate)
    assert padded_w13_scale.shape == expected_w13_scale.shape
    assert torch.equal(padded_w13_scale, expected_w13_scale)
    assert padded_w2_scale is w2_scale
    assert layer.moe_config.intermediate_size_per_partition == padded_intermediate


def test_flashinfer_cutlass_prepare_uses_quant_flag_not_scale_shape():
    intermediate = 3
    hidden_size = 5
    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(
            is_act_and_mul=True,
            intermediate_size_per_partition=intermediate,
        ),
        activation=MoEActivation.SILU,
    )

    w13 = torch.arange(
        1,
        1 + 2 * intermediate * hidden_size,
        dtype=torch.uint8,
    ).reshape(1, 2 * intermediate, hidden_size)
    w2 = torch.ones((1, hidden_size, intermediate), dtype=torch.uint8)
    w13_scale = torch.ones((1, 2 * intermediate, 1), dtype=torch.float32)
    w2_scale = torch.ones((1, hidden_size, 1), dtype=torch.float32)

    _, _, out_w13_scale, out_w2_scale = prepare_fp8_moe_layer_for_fi(
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w13_input_scale=None,
        w2_scale=w2_scale,
        w2_input_scale=None,
        per_out_ch_quant=False,
    )

    assert out_w13_scale is w13_scale
    assert out_w2_scale is w2_scale
