# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("DeepSeek V4 AMD MoE tests require ROCm", allow_module_level=True)

import vllm.models.deepseek_v4.amd.model as amd_model
from vllm._aiter_ops import (
    _rocm_aiter_fused_moe_fake,
    _rocm_aiter_fused_moe_impl,
    rocm_aiter_ops,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    rocm_aiter_fused_experts,
)
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4HeterogeneousSharedRoutedExperts,
    _make_deepseek_v4_weights_mapper,
    _pad_and_expand_native_fp8_shared_expert,
    _should_fuse_shared_expert,
)


def _make_fusion_config():
    quant_config = SimpleNamespace(
        get_name=lambda: "deepseek_v4_fp8",
        moe_quant_algo="",
        weight_block_size=[128, 128],
        is_checkpoint_fp8_serialized=True,
        is_scale_e8m0=True,
        ignored_layers=[],
    )
    hf_config = SimpleNamespace(
        n_routed_experts=384,
        hidden_size=7168,
        moe_intermediate_size=3072,
        num_experts_per_tok=6,
        n_shared_experts=1,
        expert_dtype="fp4",
        hidden_act="silu",
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16),
        quant_config=quant_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        kernel_config=SimpleNamespace(moe_backend="aiter"),
    )


@pytest.mark.parametrize(
    "guard",
    [
        "aiter_moe",
        "heterogeneous_kernel",
        "gfx950",
        "backend",
        "tensor_parallel",
        "routed_experts",
        "hidden_size",
        "intermediate_size",
        "top_k",
        "shared_experts",
        "expert_dtype",
        "activation",
        "model_dtype",
        "moe_quant_algo",
        "block_size",
        "serialized_fp8",
        "scale_dtype",
        "ignored_layers",
        "expert_parallel",
        "eplb",
        "uva_weight_offload",
        "prefetch_weight_offload",
    ],
)
def test_deepseek_v4_shared_expert_fusion_guards(monkeypatch, guard):
    config = _make_fusion_config()
    monkeypatch.setattr(
        amd_model.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: guard != "aiter_moe",
    )
    monkeypatch.setattr(
        amd_model.rocm_aiter_ops,
        "fused_moe_supports_heterogeneous_shared_expert",
        lambda: guard != "heterogeneous_kernel",
    )
    monkeypatch.setattr(amd_model, "on_gfx950", lambda: guard != "gfx950")

    if guard == "backend":
        config.kernel_config.moe_backend = "triton"
    elif guard == "tensor_parallel":
        config.parallel_config.tensor_parallel_size = 4
    elif guard == "routed_experts":
        config.model_config.hf_config.n_routed_experts = 256
    elif guard == "hidden_size":
        config.model_config.hf_config.hidden_size = 4096
    elif guard == "intermediate_size":
        config.model_config.hf_config.moe_intermediate_size = 1536
    elif guard == "top_k":
        config.model_config.hf_config.num_experts_per_tok = 8
    elif guard == "shared_experts":
        config.model_config.hf_config.n_shared_experts = 2
    elif guard == "expert_dtype":
        config.model_config.hf_config.expert_dtype = "fp8"
    elif guard == "activation":
        config.model_config.hf_config.hidden_act = "gelu"
    elif guard == "model_dtype":
        config.model_config.dtype = torch.float16
    elif guard == "moe_quant_algo":
        config.quant_config.moe_quant_algo = "NVFP4"
    elif guard == "block_size":
        config.quant_config.weight_block_size = [64, 128]
    elif guard == "serialized_fp8":
        config.quant_config.is_checkpoint_fp8_serialized = False
    elif guard == "scale_dtype":
        config.quant_config.is_scale_e8m0 = False
    elif guard == "ignored_layers":
        config.quant_config.ignored_layers = ["shared_experts"]
    elif guard == "expert_parallel":
        config.parallel_config.enable_expert_parallel = True
    elif guard == "eplb":
        config.parallel_config.enable_eplb = True
    elif guard == "uva_weight_offload":
        config.offload_config.uva.cpu_offload_gb = 1
    elif guard == "prefetch_weight_offload":
        config.offload_config.prefetch.offload_group_size = 1

    assert not _should_fuse_shared_expert(config)


def test_deepseek_v4_shared_expert_fusion_policy_accepts_supported_config(
    monkeypatch,
):
    config = _make_fusion_config()
    monkeypatch.setattr(
        amd_model.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        amd_model.rocm_aiter_ops,
        "fused_moe_supports_heterogeneous_shared_expert",
        lambda: True,
    )
    monkeypatch.setattr(amd_model, "on_gfx950", lambda: True)

    assert _should_fuse_shared_expert(config)


@pytest.mark.parametrize(
    ("projection", "mapped_projection"),
    [("w1", "w1"), ("w2", "down_proj"), ("w3", "w3")],
)
def test_shared_expert_checkpoint_names_stay_on_native_mlp(
    projection, mapped_projection
):
    name = f"layers.1.ffn.shared_experts.{projection}.scale"
    weight = torch.empty(0)
    mapped_name, _ = next(
        iter(_make_deepseek_v4_weights_mapper("fp4").apply([(name, weight)]))
    )

    assert mapped_name == (
        f"model.layers.1.ffn.shared_experts.{mapped_projection}.weight_scale_inv"
    )
    assert ".experts.384." not in mapped_name


def test_fused_moe_registers_native_mlp_before_derived_routed_buffers(monkeypatch):
    captured = {}

    class FakeGate(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    class FakeSharedExpert(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.gate_up_proj = nn.Module()
            self.down_proj = nn.Module()

    def fake_fused_moe(**kwargs):
        captured.update(kwargs)
        runner = nn.Module()
        runner.routed_experts = nn.Module()
        return runner

    monkeypatch.setattr(amd_model, "GateLinear", FakeGate)
    monkeypatch.setattr(amd_model, "DeepseekV4MLP", FakeSharedExpert)
    monkeypatch.setattr(amd_model, "FusedMoE", fake_fused_moe)
    monkeypatch.setattr(amd_model, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(amd_model, "get_tensor_model_parallel_rank", lambda: 0)
    config = SimpleNamespace(
        routed_scaling_factor=2.5,
        hidden_size=7168,
        n_routed_experts=384,
        num_experts_per_tok=6,
        moe_intermediate_size=3072,
        swiglu_limit=10.0,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        num_hash_layers=0,
        vocab_size=163840,
        topk_method=None,
        n_shared_experts=1,
        hidden_act="silu",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=object(),
    )

    moe = amd_model.DeepseekV4MoE(
        vllm_config, prefix="model.layers.0.ffn", fuse_shared_expert=True
    )

    module_names = [name for name, _ in moe.named_modules()]
    assert module_names.index("shared_experts.gate_up_proj") < module_names.index(
        "experts.routed_experts"
    )
    assert captured["shared_experts"] is None
    assert captured["n_shared_experts"] == 1
    assert captured["routed_scaling_factor"] == 2.5
    assert captured["routed_experts_cls"] is (
        DeepseekV4HeterogeneousSharedRoutedExperts
    )
    assert captured["routed_experts_args"] == {
        "shared_expert": moe.shared_experts,
        "shared_expert_id": 384,
    }


def _native_shared_tensors():
    native_intermediate = 128
    hidden_size = 256
    w13_values = (
        torch.arange(2 * native_intermediate * hidden_size, dtype=torch.int32) % 15
    ) - 7
    w2_values = (
        torch.arange(hidden_size * native_intermediate, dtype=torch.int32) % 13
    ) - 6
    w13 = w13_values.to(torch.float8_e4m3fn).view(2 * native_intermediate, hidden_size)
    w2 = w2_values.to(torch.float8_e4m3fn).view(hidden_size, native_intermediate)
    w13_scale = torch.tensor([[120, 121], [130, 131]], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    w2_scale = torch.tensor([[122], [132]], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    return w13, w2, w13_scale, w2_scale


def test_native_fp8_padding_preserves_bytes_and_expands_e8m0_losslessly():
    w13, w2, w13_scale, w2_scale = _native_shared_tensors()

    padded_w13, padded_w2, expanded_w13_scale, expanded_w2_scale = (
        _pad_and_expand_native_fp8_shared_expert(
            w13, w2, w13_scale, w2_scale, padded_intermediate_size=256
        )
    )

    assert padded_w13.shape == (1, 512, 256)
    assert padded_w2.shape == (1, 256, 256)
    assert expanded_w13_scale.shape == (512, 8)
    assert expanded_w2_scale.shape == (256, 8)
    assert torch.equal(
        padded_w13[0, :128].view(torch.uint8), w13[:128].view(torch.uint8)
    )
    assert torch.equal(
        padded_w13[0, 256:384].view(torch.uint8), w13[128:].view(torch.uint8)
    )
    assert torch.count_nonzero(padded_w13[0, 128:256].view(torch.uint8)) == 0
    assert torch.count_nonzero(padded_w13[0, 384:].view(torch.uint8)) == 0
    assert torch.equal(padded_w2[0, :, :128].view(torch.uint8), w2.view(torch.uint8))
    assert torch.count_nonzero(padded_w2[0, :, 128:].view(torch.uint8)) == 0

    w13_scale_bytes = expanded_w13_scale.view(torch.uint8)
    w2_scale_bytes = expanded_w2_scale.view(torch.uint8)
    expected_gate = w13_scale[:1].view(torch.uint8).repeat_interleave(128, 0)
    expected_gate = expected_gate.repeat_interleave(4, 1)
    expected_up = w13_scale[1:].view(torch.uint8).repeat_interleave(128, 0)
    expected_up = expected_up.repeat_interleave(4, 1)
    expected_w2 = w2_scale.view(torch.uint8).repeat_interleave(128, 0)
    expected_w2 = expected_w2.repeat_interleave(4, 1)
    assert torch.equal(w13_scale_bytes[:128], expected_gate)
    assert torch.equal(w13_scale_bytes[256:384], expected_up)
    assert torch.all(w13_scale_bytes[128:256] == 0x7F)
    assert torch.all(w13_scale_bytes[384:] == 0x7F)
    assert torch.equal(w2_scale_bytes[:, :4], expected_w2)
    assert torch.all(w2_scale_bytes[:, 4:] == 0x7F)


class _NativeSharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        w13, w2, w13_scale, w2_scale = _native_shared_tensors()
        self.gate_up_proj = nn.Module()
        self.gate_up_proj.weight = w13
        self.gate_up_proj.weight_scale_inv = w13_scale
        self.down_proj = nn.Module()
        self.down_proj.weight = w2
        self.down_proj.weight_scale_inv = w2_scale


def _empty_heterogeneous_experts(shared_expert):
    experts = object.__new__(DeepseekV4HeterogeneousSharedRoutedExperts)
    nn.Module.__init__(experts)
    experts._shared_expert_ref = weakref.ref(shared_expert)
    experts.moe_config = SimpleNamespace(intermediate_size_per_partition=256)
    experts.register_buffer("shared_w1", None, persistent=False)
    experts.register_buffer("shared_w2", None, persistent=False)
    experts.register_buffer("shared_w1_scale", None, persistent=False)
    experts.register_buffer("shared_w2_scale", None, persistent=False)
    return experts


def test_native_fp8_postload_uses_gate_up_interleaved_aiter_layout(monkeypatch):
    shared_expert = _NativeSharedExpert()
    experts = _empty_heterogeneous_experts(shared_expert)
    calls: list[tuple[object, ...]] = []

    def shuffle_weight(tensor, lanes, gate_up):
        calls.append(("weight", tensor.shape, lanes, gate_up))
        return tensor

    def shuffle_scale_a16w4(tensor, num_experts, gate_up):
        calls.append(("scale_a16w4", tensor.shape, num_experts, gate_up))
        return tensor

    def shuffle_scale(tensor):
        calls.append(("scale", tensor.shape))
        return tensor

    monkeypatch.setattr(
        amd_model.rocm_aiter_ops, "shuffle_weight_a16w4", shuffle_weight
    )
    monkeypatch.setattr(
        amd_model.rocm_aiter_ops,
        "shuffle_scale_a16w4",
        shuffle_scale_a16w4,
    )
    monkeypatch.setattr(amd_model.rocm_aiter_ops, "shuffle_scale", shuffle_scale)

    experts.prepare_heterogeneous_shared_expert()

    assert calls == [
        ("weight", torch.Size([1, 512, 256]), 16, True),
        ("scale_a16w4", torch.Size([512, 8]), 1, True),
        ("weight", torch.Size([1, 256, 256]), 16, False),
        ("scale", torch.Size([256, 8])),
    ]


def test_heterogeneous_adapter_forwards_topk7_and_separate_fp8_tensors(monkeypatch):
    experts = _empty_heterogeneous_experts(_NativeSharedExpert())
    experts.shared_expert_id = 384
    experts.w13_weight = torch.empty(385, 512, 128, dtype=torch.uint8)
    experts.w2_weight = torch.empty(385, 256, 128, dtype=torch.uint8)
    experts.shared_w1 = torch.empty(1, 512, 256, dtype=torch.float8_e4m3fn)
    experts.shared_w2 = torch.empty(1, 256, 256, dtype=torch.float8_e4m3fn)
    experts.shared_w1_scale = torch.empty(512, 8, dtype=torch.float8_e8m0fnu)
    experts.shared_w2_scale = torch.empty(256, 8, dtype=torch.float8_e8m0fnu)
    quant_config = object()
    experts.quant_method = SimpleNamespace(moe_quant_config=quant_config)
    experts.activation = object()
    experts.rocm_aiter_fmoe_enabled = False
    experts._expert_map = None
    captured = {}

    def fake_fused_experts(**kwargs):
        captured.update(kwargs)
        return torch.ones_like(kwargs["hidden_states"])

    monkeypatch.setattr(amd_model, "rocm_aiter_fused_experts", fake_fused_experts)
    monkeypatch.setattr(amd_model.rocm_aiter_ops, "get_moe_dispatch_policy", lambda: 7)
    x = torch.empty(2, 256)
    topk_weights = torch.tensor(
        [
            [0.25, 0.50, 0.25, 0.50, 0.50, 0.50, 1.0],
            [0.125, 0.375, 0.25, 0.75, 0.50, 0.50, 1.0],
        ]
    )
    topk_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 384], [6, 7, 8, 9, 10, 11, 384]],
        dtype=torch.int32,
    )

    output = experts.forward_modular(x, topk_weights, topk_ids)

    assert output.shape == x.shape
    assert captured["w1"].shape[0] == 385
    assert captured["topk_weights"] is topk_weights
    torch.testing.assert_close(
        captured["topk_weights"][:, :6].sum(-1),
        torch.full((2,), 2.5),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        captured["topk_weights"][:, -1], torch.ones(2), rtol=0, atol=0
    )
    assert torch.all(captured["topk_ids"][:, -1] == 384)
    assert captured["quant_config"] is quant_config
    assert captured["shared_w1"] is experts.shared_w1
    assert captured["shared_w2"] is experts.shared_w2
    assert captured["shared_w1_scale"] is experts.shared_w1_scale
    assert captured["shared_w2_scale"] is experts.shared_w2_scale
    assert captured["shared_expert_id"] == 384


@pytest.mark.parametrize("swiglu_limit", [None, 10.0])
def test_rocm_aiter_adapter_forwards_heterogeneous_contract_and_swiglu_limit(
    monkeypatch, swiglu_limit
):
    captured = {}

    def fake_fused_moe(*args, **kwargs):
        captured.update(kwargs)
        return torch.empty_like(args[0])

    monkeypatch.setattr(amd_model.rocm_aiter_ops, "fused_moe", fake_fused_moe)
    quant_config = SimpleNamespace(
        per_act_token_quant=False,
        use_fp8_w8a8=False,
        use_mxfp4_w4a4=False,
        use_mxfp4_w4a16=True,
        block_shape=None,
        w1_scale=torch.empty(0),
        w2_scale=torch.empty(0),
        a1_scale=None,
        a2_scale=None,
        w1_bias=None,
        w2_bias=None,
    )
    moe_config = SimpleNamespace(
        hidden_dim_unpadded=256,
        intermediate_size_per_partition_unpadded=256,
        intermediate_size_per_partition=256,
        intermediate_pad=None,
        tp_size=1,
        swiglu_limit=swiglu_limit,
    )
    shared = [torch.empty(0) for _ in range(4)]

    rocm_aiter_fused_experts(
        hidden_states=torch.empty(2, 256),
        w1=torch.empty(385, 512, 128),
        w2=torch.empty(385, 256, 128),
        topk_weights=torch.ones(2, 7),
        topk_ids=torch.zeros(2, 7, dtype=torch.int32),
        moe_config=moe_config,
        quant_config=quant_config,
        shared_w1=shared[0],
        shared_w2=shared[1],
        shared_w1_scale=shared[2],
        shared_w2_scale=shared[3],
        shared_expert_id=384,
    )

    assert captured["swiglu_limit"] == (
        0.0 if swiglu_limit is None else float(swiglu_limit)
    )
    assert captured["shared_w1"] is shared[0]
    assert captured["shared_w2"] is shared[1]
    assert captured["shared_w1_scale"] is shared[2]
    assert captured["shared_w2_scale"] is shared[3]
    assert captured["shared_expert_id"] == 384


def test_heterogeneous_custom_op_impl_fake_and_public_signatures_match():
    from torch._subclasses.fake_tensor import FakeTensorMode

    expected = {
        "shared_w1",
        "shared_w2",
        "shared_w1_scale",
        "shared_w2_scale",
        "shared_expert_id",
    }
    for function in (
        _rocm_aiter_fused_moe_impl,
        _rocm_aiter_fused_moe_fake,
        rocm_aiter_ops.fused_moe,
        rocm_aiter_fused_experts,
    ):
        assert expected <= inspect.signature(function).parameters.keys()

    for function in (
        _rocm_aiter_fused_moe_impl,
        _rocm_aiter_fused_moe_fake,
        rocm_aiter_ops.fused_moe,
    ):
        assert inspect.signature(function).parameters["swiglu_limit"].default == 0.0

    hidden_states = torch.empty(2, 256, dtype=torch.float16)
    output = _rocm_aiter_fused_moe_fake(
        hidden_states,
        torch.empty(385, 512, 128),
        torch.empty(385, 256, 128),
        torch.ones(2, 7),
        torch.zeros(2, 7, dtype=torch.int32),
        output_dtype=torch.bfloat16,
        shared_w1=torch.empty(1, 512, 256),
        shared_w2=torch.empty(1, 256, 256),
        shared_w1_scale=torch.empty(512, 8),
        shared_w2_scale=torch.empty(256, 8),
        shared_expert_id=384,
    )
    assert output.shape == hidden_states.shape
    assert output.dtype == torch.bfloat16

    with FakeTensorMode():
        custom_op_output = rocm_aiter_ops.fused_moe(
            torch.empty(2, 256),
            torch.empty(385, 512, 128),
            torch.empty(385, 256, 128),
            torch.ones(2, 7),
            torch.zeros(2, 7, dtype=torch.int32),
            output_dtype=torch.bfloat16,
            shared_w1=torch.empty(1, 512, 256),
            shared_w2=torch.empty(1, 256, 256),
            shared_w1_scale=torch.empty(512, 8),
            shared_w2_scale=torch.empty(256, 8),
            shared_expert_id=384,
        )
    assert custom_op_output.shape == hidden_states.shape
    assert custom_op_output.dtype == torch.bfloat16
