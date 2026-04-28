# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from enum import IntEnum

import torch

import vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe as fi_trtllm_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    prepare_fp8_moe_layer_for_fi,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)


def test_flashinfer_trtllm_ptpc_quant_config_preserves_dynamic_scales():
    w1_scale = torch.ones((2, 4), dtype=torch.float32)
    w2_scale = torch.ones((2, 3), dtype=torch.float32)

    quant_config = make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
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
    assert quant_config.g1_alphas is not None
    assert quant_config.g1_alphas is not w1_scale
    assert torch.equal(quant_config.g1_alphas, w1_scale)


def test_flashinfer_cutlass_does_not_claim_ptpc_quant_scheme():
    assert not FlashInferExperts._supports_quant_scheme(
        kFp8StaticChannelSym,
        kFp8DynamicTokenSym,
    )


def test_flashinfer_trtllm_claims_ptpc_quant_scheme(monkeypatch):
    monkeypatch.setattr(
        fi_trtllm_moe,
        "has_flashinfer_trtllm_fp8_per_channel_scale_routed_moe",
        lambda: True,
    )

    assert TrtLlmFp8ExpertsModular._supports_quant_scheme(
        kFp8StaticChannelSym,
        kFp8DynamicTokenSym,
    )


def test_flashinfer_trtllm_ptpc_supports_swigluoai_alias():
    assert TrtLlmFp8ExpertsModular._supports_activation(MoEActivation.SWIGLUOAI)


def test_flashinfer_trtllm_ptpc_apply_uses_trtllm_routed_moe(
    monkeypatch,
):
    core_module = types.ModuleType("flashinfer.fused_moe.core")

    class ActivationType(IntEnum):
        Swiglu = 3
        Relu2 = 4

    core_module.ActivationType = ActivationType  # type: ignore[attr-defined]
    fused_moe_module = types.ModuleType("flashinfer.fused_moe")
    fused_moe_module.core = core_module  # type: ignore[attr-defined]
    flashinfer_module = types.ModuleType("flashinfer")
    flashinfer_module.fused_moe = fused_moe_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe_module)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core_module)

    captured_kwargs = {}
    packed_topk_ids = torch.full((3, 1), 7, dtype=torch.int32)

    def fake_pack_topk_ids_weights(topk_ids, topk_weights):
        captured_kwargs["pack_topk_ids"] = topk_ids
        captured_kwargs["pack_topk_weights"] = topk_weights
        return packed_topk_ids

    def fake_trtllm_fp8_per_channel_scale_routed_moe(**kwargs):
        captured_kwargs.update(kwargs)
        return torch.full((3, 5), 3, dtype=torch.bfloat16)

    monkeypatch.setattr(
        fi_trtllm_moe,
        "trtllm_moe_pack_topk_ids_weights",
        fake_pack_topk_ids_weights,
    )
    monkeypatch.setattr(
        fi_trtllm_moe,
        "flashinfer_trtllm_fp8_per_channel_scale_routed_moe",
        fake_trtllm_fp8_per_channel_scale_routed_moe,
    )

    w1_scale = torch.ones((2, 4), dtype=torch.float32)
    w1_gate_scale = torch.full((2, 4), 2.0, dtype=torch.float32)
    w2_scale = torch.ones((2, 3), dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        w1_scale=w1_scale,
        g1_alphas=w1_gate_scale,
        w2_scale=w2_scale,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )
    experts = object.__new__(TrtLlmFp8ExpertsModular)
    experts.quant_config = quant_config
    experts.ep_rank = 0
    experts.max_capture_size = 1
    experts.local_num_experts = 2
    experts.topk = 1
    experts.intermediate_size_per_partition = 3
    experts.routing_method_type = RoutingMethodType.TopK
    experts.moe_config = types.SimpleNamespace(
        routing_method=RoutingMethodType.TopK,
    )

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
        activation=MoEActivation.SWIGLUOAI,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert captured_kwargs["pack_topk_ids"] is topk_ids
    assert captured_kwargs["pack_topk_weights"] is topk_weights
    assert captured_kwargs["topk_ids"] is packed_topk_ids
    assert captured_kwargs["hidden_states_scale"] is a1q_scale
    assert captured_kwargs["gemm1_weights"].dtype == torch.float8_e4m3fn
    assert captured_kwargs["gemm1_per_channel_weight_scale"] is w1_scale
    assert captured_kwargs["gemm1_per_channel_gate_weight_scale"] is w1_gate_scale
    assert captured_kwargs["gemm2_weights"].dtype == torch.float8_e4m3fn
    assert captured_kwargs["gemm2_per_channel_weight_scale"] is w2_scale
    assert captured_kwargs["intermediate_size"] == 3
    assert captured_kwargs["use_routing_scales_on_input"] is False
    assert captured_kwargs["routing_method_type"] == int(RoutingMethodType.TopK)
    assert captured_kwargs["activation_type"] == ActivationType.Swiglu.value
    assert torch.equal(output, torch.full((3, 5), 3, dtype=torch.bfloat16))


def test_flashinfer_trtllm_ptpc_prepare_uses_trtllm_weight_layout(monkeypatch):
    flashinfer_module = types.ModuleType("flashinfer")
    shuffle_calls: list[tuple[tuple[int, ...], int]] = []

    def fake_reorder_rows_for_gated_act_gemm(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[0] // 2
        out = torch.empty_like(x)
        out[0::2] = x[:half]
        out[1::2] = x[half:]
        return out

    def fake_shuffle_matrix_a(
        x: torch.Tensor,
        epilogue_tile_m: int,
    ) -> torch.Tensor:
        shuffle_calls.append((tuple(x.shape), epilogue_tile_m))
        return x.clone()

    flashinfer_module.reorder_rows_for_gated_act_gemm = (  # type: ignore[attr-defined]
        fake_reorder_rows_for_gated_act_gemm
    )
    flashinfer_module.shuffle_matrix_a = (  # type: ignore[attr-defined]
        fake_shuffle_matrix_a
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)

    intermediate = 16
    hidden_size = 2
    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(
            is_act_and_mul=True,
            intermediate_size_per_partition=intermediate,
        ),
        activation=MoEActivation.SILU,
    )
    w13 = torch.arange(
        2 * intermediate * hidden_size,
        dtype=torch.uint8,
    ).reshape(1, 2 * intermediate, hidden_size)
    w2 = torch.arange(
        hidden_size * intermediate,
        dtype=torch.uint8,
    ).reshape(1, hidden_size, intermediate)
    w13_scale = torch.arange(
        2 * intermediate,
        dtype=torch.float32,
    ).reshape(1, 2 * intermediate, 1)
    w2_scale = torch.arange(hidden_size, dtype=torch.float32).reshape(1, hidden_size, 1)

    out_w13, out_w2, out_w13_scale, out_w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
        per_out_ch_quant=True,
    )

    expected_w31 = torch.cat(
        [w13[:, intermediate:], w13[:, :intermediate]],
        dim=1,
    )
    expected_w13 = fake_reorder_rows_for_gated_act_gemm(expected_w31[0]).unsqueeze(0)
    expected_w31_scale = torch.cat(
        [w13_scale[:, intermediate:], w13_scale[:, :intermediate]],
        dim=1,
    ).squeeze(-1)
    expected_w13_scale = fake_reorder_rows_for_gated_act_gemm(
        expected_w31_scale[0].reshape(2 * intermediate, 1)
    ).reshape(1, 2 * intermediate)

    assert out_w13.dtype == torch.float8_e4m3fn
    assert out_w2.dtype == torch.float8_e4m3fn
    assert torch.equal(out_w13.view(torch.uint8), expected_w13)
    assert torch.equal(out_w2.view(torch.uint8), w2)
    assert torch.equal(out_w13_scale, expected_w13_scale)
    assert torch.equal(out_w2_scale, w2_scale.squeeze(-1))
    assert shuffle_calls == [
        ((2 * intermediate, hidden_size), 128),
        ((hidden_size, intermediate), 128),
    ]


def test_flashinfer_prepare_pads_and_swaps_per_channel_w13_scales():
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


def test_flashinfer_prepare_uses_quant_flag_not_scale_shape():
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
