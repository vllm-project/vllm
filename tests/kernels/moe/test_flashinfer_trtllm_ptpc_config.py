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
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsModular,
    pack_topk_ids_weights,
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


def _install_fake_flashinfer(monkeypatch, **submodules):
    modules = {"flashinfer": types.ModuleType("flashinfer")}
    monkeypatch.setitem(sys.modules, "flashinfer", modules["flashinfer"])
    for name, attrs in submodules.items():
        full_name = "flashinfer"
        for part in name.split("."):
            parent = modules[full_name]
            full_name = f"{full_name}.{part}"
            if full_name not in modules:
                modules[full_name] = types.ModuleType(full_name)
                monkeypatch.setitem(sys.modules, full_name, modules[full_name])
                setattr(parent, part, modules[full_name])
        for attr, value in attrs.items():
            setattr(modules[full_name], attr, value)


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

    monkeypatch.setattr(
        fi_trtllm_moe,
        "has_flashinfer_trtllm_fp8_per_channel_scale_routed_moe",
        lambda: False,
    )
    assert not TrtLlmFp8ExpertsModular._supports_quant_scheme(
        kFp8StaticChannelSym,
        kFp8DynamicTokenSym,
    )


def test_pack_topk_ids_weights_roundtrips_expert_ids_and_bf16_weights():
    topk_ids = torch.tensor([[3, 17], [0, 2047]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25, 0.75], [1.0, -0.5]], dtype=torch.float32)

    packed = pack_topk_ids_weights(topk_ids, topk_weights)

    assert packed.dtype == torch.int32
    assert packed.is_contiguous()
    assert torch.equal(packed >> 16, topk_ids.to(torch.int32))
    unpacked_weights = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    assert torch.equal(unpacked_weights, topk_weights.to(torch.bfloat16))


def test_flashinfer_trtllm_ptpc_apply_uses_trtllm_routed_moe(monkeypatch):
    class ActivationType(IntEnum):
        Gelu = 0
        Relu = 1
        Silu = 2
        Swiglu = 3
        Geglu = 4
        Relu2 = 5

    _install_fake_flashinfer(
        monkeypatch, **{"fused_moe.core": {"ActivationType": ActivationType}}
    )

    captured_kwargs = {}

    def fake_trtllm_fp8_per_channel_scale_routed_moe(**kwargs):
        captured_kwargs.update(kwargs)
        return torch.full((3, 5), 3, dtype=torch.bfloat16)

    monkeypatch.setattr(
        fi_trtllm_moe,
        "flashinfer_trtllm_fp8_per_channel_scale_routed_moe",
        fake_trtllm_fp8_per_channel_scale_routed_moe,
    )

    w1_scale = torch.ones((2, 6), dtype=torch.float32)
    w2_scale = torch.ones((2, 5), dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )
    unit_scales = torch.ones(2, dtype=torch.float32)
    experts = object.__new__(TrtLlmFp8ExpertsModular)
    experts.quant_config = quant_config
    experts.ptpc_unit_scales = unit_scales
    experts.ep_rank = 0
    experts.local_num_experts = 2
    experts.topk = 1
    experts.intermediate_size_per_partition = 3
    experts.routing_method_type = RoutingMethodType.TopK
    experts.moe_config = types.SimpleNamespace(max_num_tokens=16, dp_size=1)

    hidden_states = torch.ones((3, 5), dtype=torch.bfloat16)
    w1 = torch.empty((2, 6, 5), dtype=torch.uint8)
    w2 = torch.empty((2, 5, 3), dtype=torch.uint8)
    output = torch.empty((3, 5), dtype=torch.bfloat16)
    topk_weights = torch.ones((3, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1], [0]], dtype=torch.int64)
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

    assert torch.equal(
        captured_kwargs["topk_ids"], pack_topk_ids_weights(topk_ids, topk_weights)
    )
    assert captured_kwargs["hidden_states"] is hidden_states
    assert captured_kwargs["hidden_states_scale"].shape == (3, 1)
    assert torch.equal(captured_kwargs["hidden_states_scale"], a1q_scale)
    assert captured_kwargs["gemm1_weights"].dtype == torch.float8_e4m3fn
    assert captured_kwargs["gemm1_per_channel_weight_scale"] is w1_scale
    assert captured_kwargs["output1_scale_scalar"] is unit_scales
    assert captured_kwargs["output1_scale_gate_scalar"] is unit_scales
    assert captured_kwargs["gemm2_weights"].dtype == torch.float8_e4m3fn
    assert captured_kwargs["gemm2_per_channel_weight_scale"] is w2_scale
    assert captured_kwargs["output2_scale_scalar"] is unit_scales
    assert captured_kwargs["num_experts"] == 2
    assert captured_kwargs["top_k"] == 1
    assert captured_kwargs["intermediate_size"] == 3
    assert captured_kwargs["local_expert_offset"] == 0
    assert captured_kwargs["local_num_experts"] == 2
    assert captured_kwargs["use_routing_scales_on_input"] is False
    assert captured_kwargs["routing_method_type"] == int(RoutingMethodType.TopK)
    assert captured_kwargs["activation_type"] == ActivationType.Swiglu.value
    assert captured_kwargs["tune_max_num_tokens"] == 8192
    assert torch.equal(output, torch.full((3, 5), 3, dtype=torch.bfloat16))


def test_flashinfer_trtllm_ptpc_prepare_permutes_scales_with_weights(monkeypatch):
    index_calls: list[tuple[str, tuple[int, ...]]] = []

    def interleave_indices(num_rows: int) -> torch.Tensor:
        idx = torch.empty(num_rows, dtype=torch.long)
        idx[0::2] = torch.arange(num_rows // 2)
        idx[1::2] = torch.arange(num_rows // 2, num_rows)
        return idx

    def fake_gated_row_indices(x: torch.Tensor) -> torch.Tensor:
        index_calls.append(("gated", tuple(x.shape)))
        return interleave_indices(x.shape[0])

    def fake_shuffle_row_indices(x: torch.Tensor, epilogue_tile_m: int):
        index_calls.append(("shuffle", tuple(x.shape)))
        assert epilogue_tile_m == 128
        return torch.arange(x.shape[0]).flip(0)

    _install_fake_flashinfer(
        monkeypatch,
        **{
            "fused_moe.core": {
                "get_reorder_rows_for_gated_act_gemm_row_indices": (
                    fake_gated_row_indices
                )
            },
            "utils": {"get_shuffle_matrix_a_row_indices": fake_shuffle_row_indices},
        },
    )

    intermediate = 16
    hidden_size = 2
    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(
            is_act_and_mul=True,
            intermediate_size_per_partition=intermediate,
        ),
        activation=MoEActivation.SILU,
    )
    w13 = torch.arange(2 * intermediate * hidden_size, dtype=torch.uint8).reshape(
        1, 2 * intermediate, hidden_size
    )
    w2 = torch.arange(hidden_size * intermediate, dtype=torch.uint8).reshape(
        1, hidden_size, intermediate
    )
    w13_scale = torch.arange(2 * intermediate, dtype=torch.float32).reshape(
        1, 2 * intermediate, 1
    )
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

    w31 = torch.cat([w13[:, intermediate:], w13[:, :intermediate]], dim=1)
    gated_idx = interleave_indices(2 * intermediate)
    shuffle_idx = torch.arange(2 * intermediate).flip(0)
    expected_w13 = w31[:, gated_idx][:, shuffle_idx]
    w31_scale = torch.cat(
        [w13_scale[:, intermediate:], w13_scale[:, :intermediate]], dim=1
    ).squeeze(-1)
    expected_w13_scale = w31_scale[:, gated_idx][:, shuffle_idx]
    w2_shuffle_idx = torch.arange(hidden_size).flip(0)

    assert out_w13.dtype == torch.float8_e4m3fn
    assert out_w2.dtype == torch.float8_e4m3fn
    assert torch.equal(out_w13.view(torch.uint8), expected_w13)
    assert torch.equal(out_w2.view(torch.uint8), w2[:, w2_shuffle_idx])
    assert out_w13_scale.shape == (1, 2 * intermediate)
    assert torch.equal(out_w13_scale, expected_w13_scale)
    assert out_w2_scale.shape == (1, hidden_size)
    assert torch.equal(out_w2_scale, w2_scale.squeeze(-1)[:, w2_shuffle_idx])
    assert index_calls == [
        ("gated", (2 * intermediate, hidden_size)),
        ("shuffle", (2 * intermediate, hidden_size)),
        ("shuffle", (hidden_size, intermediate)),
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

    w13 = torch.arange(1, 1 + 2 * intermediate * hidden_size, dtype=torch.uint8)
    w13 = w13.reshape(1, 2 * intermediate, hidden_size)
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

    expected_w13_scale = torch.zeros((1, 2 * padded_intermediate, 1))
    expected_w13_scale[0, :intermediate, 0] = torch.tensor([11.0, 12.0, 13.0])
    expected_w13_scale[
        0, padded_intermediate : padded_intermediate + intermediate, 0
    ] = torch.tensor([1.0, 2.0, 3.0])

    assert padded_w2.shape == (1, hidden_size, padded_intermediate)
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

    w13 = torch.arange(1, 1 + 2 * intermediate * hidden_size, dtype=torch.uint8)
    w13 = w13.reshape(1, 2 * intermediate, hidden_size)
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
