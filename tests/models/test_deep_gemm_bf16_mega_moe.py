# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.deepseek_v2 import (
    _is_deep_gemm_mega_moe_requested,
    _scale_mega_moe_output_for_deferred_reduce,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepGemmMegaMoEExperts,
    make_deepseek_v4_expert_params_mapping,
)


class _FakeNvfp4QuantConfig:
    quant_format = "nvfp4-pack-quantized"
    config = {
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "experts": {
                "targets": ["Linear"],
            }
        },
    }

    @staticmethod
    def get_name():
        return "compressed-tensors"


class _FakeMixedNvfp4QuantConfig(_FakeNvfp4QuantConfig):
    ignore = ["re:model.layers.78.*"]

    def get_scheme_dict(self, _layer, layer_name):
        if layer_name.startswith("model.layers.78."):
            return None
        if layer_name.startswith("model.layers.77."):
            return {"format": "nvfp4-pack-quantized"}
        return {"format": "float-quantized"}


def test_mega_moe_request_applies_to_mtp_model_config():
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe")
    )

    assert _is_deep_gemm_mega_moe_requested(vllm_config)


def test_mega_moe_scales_output_when_reduce_is_deferred():
    output = torch.full((2, 4), 8.0)

    scaled = _scale_mega_moe_output_for_deferred_reduce(
        output,
        tp_size=8,
        is_sequence_parallel=False,
        reduce_results=False,
    )

    assert torch.equal(scaled, torch.ones_like(output))


def test_mega_moe_keeps_complete_output_without_deferred_reduce():
    output = torch.full((2, 4), 8.0)

    scaled = _scale_mega_moe_output_for_deferred_reduce(
        output,
        tp_size=8,
        is_sequence_parallel=True,
        reduce_results=False,
    )

    assert torch.equal(scaled, output)


def test_megamoe_mapping_uses_direct_expert_parameter_prefix():
    mapping = make_deepseek_v4_expert_params_mapping(
        1,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    )

    assert mapping == [
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
        ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
    ]


def test_nvfp4_expert_quantization_is_detected():
    quant_config = _FakeNvfp4QuantConfig()

    assert DeepGemmMegaMoEExperts.source_is_nvfp4(quant_config)
    assert (
        DeepGemmMegaMoEExperts.source_weight_block_size_from_quant_config(quant_config)
        is None
    )


def test_ignored_mtp_layer_uses_unquantized_mega_moe_weights():
    quant_config = _FakeMixedNvfp4QuantConfig()
    prefix = "model.layers.78.mtp_block.mlp"

    assert not DeepGemmMegaMoEExperts.source_is_nvfp4(
        quant_config, torch.nn.Identity(), prefix
    )
    assert (
        DeepGemmMegaMoEExperts.source_weight_block_size_from_quant_config(
            quant_config, torch.nn.Identity(), prefix
        )
        is None
    )


def test_mixed_format_uses_current_layer_scheme():
    quant_config = _FakeMixedNvfp4QuantConfig()
    layer = torch.nn.Identity()

    assert DeepGemmMegaMoEExperts.source_is_nvfp4(
        quant_config, layer, "model.layers.77.mlp"
    )
    assert not DeepGemmMegaMoEExperts.source_is_nvfp4(
        quant_config, layer, "model.layers.76.mlp"
    )


def test_kimi_ct_nvfp4_mapping_includes_global_scales():
    from vllm.models.kimi_k3.nvidia.model import (
        make_kimi_k3_mega_moe_expert_params_mapping,
    )

    mapping = make_kimi_k3_mega_moe_expert_params_mapping(1, source_nvfp4=True)

    assert mapping == [
        (f"experts.w13_{suffix}", f"experts.0.w1.{suffix}", 0, "w1")
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        )
    ] + [
        (f"experts.w2_{suffix}", f"experts.0.w2.{suffix}", 0, "w2")
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        )
    ] + [
        (f"experts.w13_{suffix}", f"experts.0.w3.{suffix}", 0, "w3")
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        )
    ]


def test_kimi_mtp_selects_nvfp4_mega_moe_mapping(monkeypatch):
    from vllm.models.kimi_k3.nvidia import model as kimi_model
    from vllm.models.kimi_k3.nvidia import mtp as kimi_mtp

    draft = object.__new__(kimi_mtp.KimiK3MTP)
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(
        linear_attn_config=None,
        q_lora_rank=None,
        is_moe=True,
        num_experts=1,
    )
    draft.model = torch.nn.Module()
    draft.model.mtp_start_layer_idx = 0
    draft.model.num_mtp_layers = 0

    moe = object.__new__(kimi_model.KimiMoE)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = True
    moe.experts = torch.nn.Module()
    moe.experts.source_nvfp4 = True
    moe.experts.finalize_weights = lambda: None
    draft.model.moe = moe

    mapping_args: dict[str, object] = {}

    def make_mapping(num_experts, source_nvfp4=False):
        mapping_args.update(
            num_experts=num_experts,
            source_nvfp4=source_nvfp4,
        )
        return []

    monkeypatch.setattr(
        kimi_mtp,
        "make_kimi_k3_mega_moe_expert_params_mapping",
        make_mapping,
    )
    monkeypatch.setattr(kimi_mtp, "get_pp_missing_layer_names", lambda _: set())

    assert draft.load_weights([]) == set()
    assert mapping_args == {"num_experts": 1, "source_nvfp4": True}


def test_kimi_mega_moe_preserves_activation_transform_kwarg():
    from vllm.models.kimi_k3.nvidia.model import KimiK3MegaMoEExperts

    experts = KimiK3MegaMoEExperts.__new__(KimiK3MegaMoEExperts)
    experts.activation = "situ"

    assert experts._transform_weights_kwargs() == {"activation": "situ"}


def test_bf16_mega_moe_weights_are_loaded_and_transformed(monkeypatch):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=2,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        mma_type="bf16xbf16",
    )

    assert experts.w13_weight.dtype == torch.bfloat16
    assert experts.w13_weight.shape == (1, 256, 128)
    assert experts.w13_weight_scale is None
    assert experts.w13_weight_scale_inv is None
    assert experts.w2_weight.dtype == torch.bfloat16
    assert experts.w2_weight.shape == (1, 128, 128)
    assert experts.w2_weight_scale is None
    assert experts.w2_weight_scale_inv is None

    w1 = torch.full((128, 128), 3, dtype=torch.bfloat16)
    w3 = torch.full((128, 128), 7, dtype=torch.bfloat16)
    w2 = torch.full((128, 128), 11, dtype=torch.bfloat16)
    for param, weight, param_name, shard_id in (
        (experts.w13_weight, w1, "experts.w13_weight", "w1"),
        (experts.w13_weight, w3, "experts.w13_weight", "w3"),
        (experts.w2_weight, w2, "experts.w2_weight", "w2"),
    ):
        assert experts.weight_loader(
            param,
            weight,
            param_name,
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    transformed: list[tuple[torch.Tensor, torch.Tensor]] = []

    def transform(l1, l2):
        transformed.append((l1, l2))
        return l1.clone(), l2

    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(transform_weights_for_mega_moe=transform),
    )

    experts.finalize_weights()

    assert len(transformed) == 1
    assert torch.equal(transformed[0][0][0, :128], w1)
    assert torch.equal(transformed[0][0][0, 128:], w3)
    assert torch.equal(transformed[0][1][0], w2)
    assert experts.w13_weight is None
    assert experts.w2_weight is None


def test_block_fp8_source_weights_are_dequantized_for_bf16_mega_moe(monkeypatch):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=1,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        mma_type="bf16xbf16",
        source_weight_block_size=(128, 128),
    )

    fp8 = torch.float8_e4m3fn
    one = torch.ones(128, 128).to(fp8)
    scales = {
        "w1": torch.full((1, 1), 2.0),
        "w3": torch.full((1, 1), 3.0),
        "w2": torch.full((1, 1), 5.0),
    }
    for param, value, param_name, shard_id in (
        (experts.w13_weight, one, "experts.w13_weight", "w1"),
        (experts.w13_weight, one, "experts.w13_weight", "w3"),
        (experts.w2_weight, one, "experts.w2_weight", "w2"),
        (
            experts.w13_weight_scale_inv,
            scales["w1"],
            "experts.w13_weight_scale_inv",
            "w1",
        ),
        (
            experts.w13_weight_scale_inv,
            scales["w3"],
            "experts.w13_weight_scale_inv",
            "w3",
        ),
        (
            experts.w2_weight_scale_inv,
            scales["w2"],
            "experts.w2_weight_scale_inv",
            "w2",
        ),
    ):
        assert param is not None
        assert experts.weight_loader(
            param,
            value,
            param_name,
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(),
    )

    experts.finalize_weights()

    l1 = experts._transformed_l1_weights
    l2 = experts._transformed_l2_weights
    assert isinstance(l1, torch.Tensor)
    assert isinstance(l2, torch.Tensor)
    assert l1.dtype == torch.bfloat16
    assert l2.dtype == torch.bfloat16
    interleaved = l1[0].view(-1, 16, 128)
    assert torch.all(interleaved[:, :8] == 2)
    assert torch.all(interleaved[:, 8:] == 3)
    assert torch.all(l2[0] == 5)
    assert experts.w13_weight_scale_inv is None
    assert experts.w2_weight_scale_inv is None


def test_nvfp4_source_weights_are_dequantized_for_bf16_mega_moe(monkeypatch):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=1,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        mma_type="bf16xbf16",
        source_nvfp4=True,
    )

    packed = torch.full((128, 64), 0x11, dtype=torch.uint8)
    group_scales = torch.ones(128, 8, dtype=torch.float8_e4m3fn)
    values = (
        (experts.w13_weight_packed, packed, "experts.w13_weight_packed", "w1"),
        (experts.w13_weight_packed, packed, "experts.w13_weight_packed", "w3"),
        (experts.w2_weight_packed, packed, "experts.w2_weight_packed", "w2"),
        (experts.w13_weight_scale, group_scales, "experts.w13_weight_scale", "w1"),
        (experts.w13_weight_scale, group_scales, "experts.w13_weight_scale", "w3"),
        (experts.w2_weight_scale, group_scales, "experts.w2_weight_scale", "w2"),
        (
            experts.w13_weight_global_scale,
            torch.tensor(2.0),
            "experts.w13_weight_global_scale",
            "w1",
        ),
        (
            experts.w13_weight_global_scale,
            torch.tensor(4.0),
            "experts.w13_weight_global_scale",
            "w3",
        ),
        (
            experts.w2_weight_global_scale,
            torch.tensor(6.0),
            "experts.w2_weight_global_scale",
            "w2",
        ),
    )
    for param, value, param_name, shard_id in values:
        assert param is not None
        assert experts.weight_loader(
            param,
            value,
            param_name,
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(),
    )

    experts.finalize_weights()

    l1 = experts._transformed_l1_weights
    l2 = experts._transformed_l2_weights
    assert isinstance(l1, torch.Tensor)
    assert isinstance(l2, torch.Tensor)
    assert l1.dtype == torch.bfloat16
    assert l2.dtype == torch.bfloat16
    interleaved = l1[0].view(-1, 16, 128)
    assert torch.all(interleaved[:, :8] == 0.25)
    assert torch.all(interleaved[:, 8:] == 0.125)
    assert torch.allclose(l2[0], torch.full_like(l2[0], 1.0 / 12.0))
    assert experts.w13_weight_packed is None
    assert experts.w2_weight_packed is None


def test_nvfp4_source_weights_are_requantized_for_fp8_fp4_mega_moe(monkeypatch):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=1,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        mma_type="fp8xfp4",
        source_nvfp4=True,
    )

    packed = torch.full((128, 64), 0x11, dtype=torch.uint8)
    group_scales = torch.ones(128, 8, dtype=torch.float8_e4m3fn)
    values = (
        (experts.w13_weight_packed, packed, "experts.w13_weight_packed", "w1"),
        (experts.w13_weight_packed, packed, "experts.w13_weight_packed", "w3"),
        (experts.w2_weight_packed, packed, "experts.w2_weight_packed", "w2"),
        (experts.w13_weight_scale, group_scales, "experts.w13_weight_scale", "w1"),
        (experts.w13_weight_scale, group_scales, "experts.w13_weight_scale", "w3"),
        (experts.w2_weight_scale, group_scales, "experts.w2_weight_scale", "w2"),
        (
            experts.w13_weight_global_scale,
            torch.tensor(2.0),
            "experts.w13_weight_global_scale",
            "w1",
        ),
        (
            experts.w13_weight_global_scale,
            torch.tensor(4.0),
            "experts.w13_weight_global_scale",
            "w3",
        ),
        (
            experts.w2_weight_global_scale,
            torch.tensor(6.0),
            "experts.w2_weight_global_scale",
            "w2",
        ),
    )
    for param, value, param_name, shard_id in values:
        assert param is not None
        assert experts.weight_loader(
            param,
            value,
            param_name,
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    quantized_shapes: list[tuple[int, ...]] = []
    quantized_inputs: list[torch.Tensor] = []

    class FakeDeepGemm:
        @staticmethod
        def per_token_cast_to_fp4(x, **kwargs):
            assert kwargs == {
                "use_ue8m0": True,
                "gran_k": 32,
                "use_packed_ue8m0": False,
            }
            quantized_shapes.append(tuple(x.shape))
            quantized_inputs.append(x.clone())
            return (
                torch.zeros(x.shape[0], x.shape[1] // 2, dtype=torch.int8),
                torch.ones(x.shape[0], x.shape[1] // 32),
            )

        @staticmethod
        def transform_sf_into_required_layout(sf, *_args):
            return sf

        @staticmethod
        def transform_weights_for_mega_moe(l1, l2):
            return l1, l2

    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: FakeDeepGemm)

    experts.finalize_weights()

    assert quantized_shapes == [(128, 128), (128, 128), (128, 128)]
    assert torch.all(quantized_inputs[0] == 0.25)
    assert torch.all(quantized_inputs[1] == 0.125)
    assert torch.allclose(
        quantized_inputs[2], torch.full_like(quantized_inputs[2], 1.0 / 12.0)
    )
    assert isinstance(experts._transformed_l1_weights, tuple)
    assert isinstance(experts._transformed_l2_weights, tuple)
    assert experts._transformed_l1_weights[0].dtype == torch.int8
    assert experts._transformed_l1_weights[1].shape == (1, 256, 4)
    assert experts._transformed_l2_weights[0].dtype == torch.int8
    assert experts._transformed_l2_weights[1].shape == (1, 128, 4)
    assert experts.w13_weight_packed is None
    assert experts.w2_weight_packed is None


def test_bf16_mega_moe_stages_inputs_and_selects_bf16_kernel(monkeypatch):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=2,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        mma_type="bf16xbf16",
    )
    experts._transformed_l1_weights = torch.empty(1, 256, 128)
    experts._transformed_l2_weights = torch.empty(1, 128, 128)
    buffer = SimpleNamespace(
        x=torch.empty(4, 128, dtype=torch.bfloat16),
        topk_idx=torch.empty(4, 2, dtype=torch.int64),
        topk_weights=torch.empty(4, 2, dtype=torch.float32),
    )
    experts.get_symm_buffer = lambda: buffer

    called: list[str] = []

    def bf16_mega_moe(y, *_args, **_kwargs):
        called.append("bf16")
        y.zero_()

    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(bf16_mega_moe=bf16_mega_moe),
    )

    hidden_states = torch.randn(2, 128, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    topk_ids = torch.tensor([[0, 1], [1, 0]])
    output = experts(
        hidden_states,
        topk_weights,
        topk_ids,
        activation_clamp=None,
    )

    assert called == ["bf16"]
    assert torch.equal(buffer.x[:2], hidden_states)
    assert torch.equal(buffer.topk_idx[:2], topk_ids)
    assert torch.equal(buffer.topk_weights[:2], topk_weights)
    assert torch.count_nonzero(output) == 0
