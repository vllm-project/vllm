# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.models.deepseek_v2 import (
    _is_deep_gemm_mega_moe_requested,
    _scale_mega_moe_output_for_deferred_reduce,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepGemmMegaMoEExperts,
    make_deepseek_v4_expert_params_mapping,
)
from vllm.platforms import current_platform


class _FakeNvfp4QuantConfig:
    quant_format = "nvfp4-pack-quantized"
    config = {"format": quant_format}

    @staticmethod
    def get_name():
        return "compressed-tensors"


class _FakeMixedNvfp4QuantConfig(_FakeNvfp4QuantConfig):
    ignore = ["re:model.layers.78.*"]

    def get_scheme_dict(self, _layer, layer_name):
        if layer_name.startswith("model.layers.78."):
            return None
        return {
            "format": "nvfp4-pack-quantized"
            if layer_name.startswith("model.layers.77.")
            else "float-quantized"
        }


class _IdentityDeepGemm:
    @staticmethod
    def transform_sf_into_required_layout(sf, *_args):
        return sf

    @staticmethod
    def transform_weights_for_mega_moe(l1, l2):
        return l1, l2


def _make_experts(
    *,
    mma_type: str = "fp8xfp4",
    intermediate_size: int = 128,
    source_nvfp4: bool = False,
    source_mxfp4: bool = False,
    source_weight_block_size: tuple[int, int] | None = None,
) -> DeepGemmMegaMoEExperts:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    return DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=1,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=intermediate_size,
        mma_type=mma_type,
        source_nvfp4=source_nvfp4,
        source_mxfp4=source_mxfp4,
        source_weight_block_size=source_weight_block_size,
    )


def _load(
    experts: DeepGemmMegaMoEExperts,
    param_name: str,
    value: torch.Tensor,
    shard_id: str,
) -> None:
    param = getattr(experts, param_name)
    assert param is not None
    assert experts.weight_loader(
        param,
        value,
        f"experts.{param_name}",
        shard_id=shard_id,
        expert_id=0,
        return_success=True,
    )


def _load_nvfp4_weights(experts: DeepGemmMegaMoEExperts) -> None:
    packed = torch.full((128, 64), 0x11, dtype=torch.uint8)
    scale = torch.ones(128, 8, dtype=torch.float8_e4m3fn)
    for param_name, value, shard_id in (
        ("w13_weight_packed", packed, "w1"),
        ("w13_weight_packed", packed, "w3"),
        ("w2_weight_packed", packed, "w2"),
        ("w13_weight_scale", scale, "w1"),
        ("w13_weight_scale", scale, "w3"),
        ("w2_weight_scale", scale, "w2"),
        ("w13_weight_global_scale", torch.tensor(2.0), "w1"),
        ("w13_weight_global_scale", torch.tensor(4.0), "w3"),
        ("w2_weight_global_scale", torch.tensor(6.0), "w2"),
    ):
        _load(experts, param_name, value, shard_id)


def _load_mxfp4_weights(experts: DeepGemmMegaMoEExperts) -> None:
    for param_name, shape, value, shard_id in (
        ("w13_weight_packed", (512, 64), 0x11, "w1"),
        ("w13_weight_packed", (512, 64), 0x22, "w3"),
        ("w2_weight_packed", (128, 256), 0x33, "w2"),
        ("w13_weight_scale", (512, 4), 1, "w1"),
        ("w13_weight_scale", (512, 4), 2, "w3"),
        ("w2_weight_scale", (128, 16), 3, "w2"),
    ):
        _load(
            experts,
            param_name,
            torch.full(shape, value, dtype=torch.uint8),
            shard_id,
        )


def _disable_runtime_check(experts, monkeypatch, deep_gemm=_IdentityDeepGemm):
    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: deep_gemm)


def test_mega_moe_request_applies_to_mtp_model_config():
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe")
    )

    assert _is_deep_gemm_mega_moe_requested(vllm_config)


@pytest.mark.parametrize(
    ("is_sequence_parallel", "expected"), [(False, 1.0), (True, 8.0)]
)
def test_mega_moe_deferred_reduction_scaling(is_sequence_parallel, expected):
    output = torch.full((2, 4), 8.0)

    actual = _scale_mega_moe_output_for_deferred_reduce(
        output,
        tp_size=8,
        is_sequence_parallel=is_sequence_parallel,
        reduce_results=False,
    )

    assert torch.equal(actual, torch.full_like(output, expected))


def test_megamoe_mapping_uses_direct_expert_parameter_prefix():
    assert make_deepseek_v4_expert_params_mapping(
        1,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    ) == [
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


@pytest.mark.parametrize(
    ("quant_format", "group_size", "activation_bits"),
    [
        ("nvfp4-pack-quantized", 16, 4),
        ("mxfp4-pack-quantized", 32, 8),
    ],
)
def test_layer_fp4_quantization_is_detected(
    quant_format: str, group_size: int, activation_bits: int
):
    is_nvfp4 = quant_format.startswith("nvfp4")
    quant_config = compressed_tensors.CompressedTensorsConfig.from_config(
        {
            "format": "mixed-precision",
            "config_groups": {
                "experts": {
                    "format": quant_format,
                    "targets": [r"re:.*mlp\..*"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "tensor_group" if is_nvfp4 else "group",
                        "group_size": group_size,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": activation_bits,
                        "type": "float",
                        "strategy": "tensor_group" if is_nvfp4 else "group",
                        "group_size": group_size,
                        "symmetric": True,
                        "dynamic": "local" if is_nvfp4 else True,
                    },
                }
            },
        }
    )
    layer = torch.nn.Identity()
    prefix = "model.layers.3.mlp"

    assert DeepGemmMegaMoEExperts.source_is_fp4(quant_config, layer, prefix)
    assert (
        DeepGemmMegaMoEExperts.source_weight_block_size_from_quant_config(
            quant_config, layer, prefix
        )
        is None
    )


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("model.layers.76.mlp", False),
        ("model.layers.77.mlp", True),
        ("model.layers.78.mtp_block.mlp", False),
    ],
)
def test_mixed_format_uses_current_layer_scheme(prefix: str, expected: bool):
    quant_config = _FakeMixedNvfp4QuantConfig()

    assert (
        DeepGemmMegaMoEExperts.source_is_nvfp4(
            quant_config, torch.nn.Identity(), prefix
        )
        is expected
    )
    if "mtp_block" in prefix:
        assert (
            DeepGemmMegaMoEExperts.source_weight_block_size_from_quant_config(
                quant_config, torch.nn.Identity(), prefix
            )
            is None
        )


def test_kimi_ct_nvfp4_mapping_includes_global_scales():
    from vllm.models.kimi_k3.nvidia.model import (
        make_kimi_k3_mega_moe_expert_params_mapping,
    )

    mapping = make_kimi_k3_mega_moe_expert_params_mapping(1, source_nvfp4=True)
    suffixes = (
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
        "input_global_scale",
    )
    assert mapping == [
        (f"experts.{target}_{suffix}", f"experts.0.{source}.{suffix}", 0, shard)
        for target, source, shard in (
            ("w13", "w1", "w1"),
            ("w2", "w2", "w2"),
            ("w13", "w3", "w3"),
        )
        for suffix in suffixes
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
        kimi_mtp, "make_kimi_k3_mega_moe_expert_params_mapping", make_mapping
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
    experts = _make_experts(mma_type="bf16xbf16")
    for param_name, value, shard_id in (
        ("w13_weight", 3, "w1"),
        ("w13_weight", 7, "w3"),
        ("w2_weight", 11, "w2"),
    ):
        _load(
            experts,
            param_name,
            torch.full((128, 128), value, dtype=torch.bfloat16),
            shard_id,
        )
    _disable_runtime_check(experts, monkeypatch)

    experts.finalize_weights()

    l1 = experts._transformed_l1_weights
    l2 = experts._transformed_l2_weights
    assert isinstance(l1, torch.Tensor)
    assert isinstance(l2, torch.Tensor)
    assert torch.all(l1[0, :128] == 3)
    assert torch.all(l1[0, 128:] == 7)
    assert torch.all(l2[0] == 11)


def test_block_fp8_source_weights_are_dequantized_for_bf16_mega_moe(monkeypatch):
    experts = _make_experts(mma_type="bf16xbf16", source_weight_block_size=(128, 128))
    one = torch.ones(128, 128).to(torch.float8_e4m3fn)
    for param_name, value, shard_id in (
        ("w13_weight", one, "w1"),
        ("w13_weight", one, "w3"),
        ("w2_weight", one, "w2"),
        ("w13_weight_scale_inv", torch.full((1, 1), 2.0), "w1"),
        ("w13_weight_scale_inv", torch.full((1, 1), 3.0), "w3"),
        ("w2_weight_scale_inv", torch.full((1, 1), 5.0), "w2"),
    ):
        _load(experts, param_name, value, shard_id)
    _disable_runtime_check(experts, monkeypatch, SimpleNamespace())

    experts.finalize_weights()

    l1 = experts._transformed_l1_weights
    l2 = experts._transformed_l2_weights
    assert isinstance(l1, torch.Tensor)
    assert isinstance(l2, torch.Tensor)
    interleaved = l1[0].view(-1, 16, 128)
    assert torch.all(interleaved[:, :8] == 2)
    assert torch.all(interleaved[:, 8:] == 3)
    assert torch.all(l2[0] == 5)


def test_nvfp4_source_weights_are_dequantized_for_bf16_mega_moe(monkeypatch):
    experts = _make_experts(mma_type="bf16xbf16", source_nvfp4=True)
    _load_nvfp4_weights(experts)
    _disable_runtime_check(experts, monkeypatch, SimpleNamespace())

    experts.finalize_weights()

    l1 = experts._transformed_l1_weights
    l2 = experts._transformed_l2_weights
    assert isinstance(l1, torch.Tensor)
    assert isinstance(l2, torch.Tensor)
    interleaved = l1[0].view(-1, 16, 128)
    assert torch.all(interleaved[:, :8] == 0.25)
    assert torch.all(interleaved[:, 8:] == 0.125)
    assert torch.allclose(l2[0], torch.full_like(l2[0], 1.0 / 12.0))


def test_nvfp4_source_weights_are_requantized_for_fp8_fp4_mega_moe(monkeypatch):
    experts = _make_experts(source_nvfp4=True)
    _load_nvfp4_weights(experts)
    quantized_inputs = []

    class FakeDeepGemm(_IdentityDeepGemm):
        @staticmethod
        def per_token_cast_to_fp4(x, **kwargs):
            assert kwargs == {
                "use_ue8m0": True,
                "gran_k": 32,
                "use_packed_ue8m0": False,
            }
            quantized_inputs.append(x.clone())
            return (
                torch.zeros(x.shape[0], x.shape[1] // 2, dtype=torch.int8),
                torch.ones(x.shape[0], x.shape[1] // 32),
            )

    _disable_runtime_check(experts, monkeypatch, FakeDeepGemm)

    experts.finalize_weights()

    assert [tuple(x.shape) for x in quantized_inputs] == [(128, 128)] * 3
    assert torch.all(quantized_inputs[0] == 0.25)
    assert torch.all(quantized_inputs[1] == 0.125)
    assert torch.allclose(
        quantized_inputs[2], torch.full_like(quantized_inputs[2], 1.0 / 12.0)
    )


def test_mxfp4_source_weights_are_loaded_for_fp8_fp4_mega_moe(monkeypatch):
    experts = _make_experts(intermediate_size=512, source_mxfp4=True)
    _load_mxfp4_weights(experts)
    _disable_runtime_check(experts, monkeypatch)

    experts.finalize_weights()

    l1 = experts._transformed_l1_weights
    l2 = experts._transformed_l2_weights
    assert isinstance(l1, tuple)
    assert isinstance(l2, tuple)
    assert torch.all(l1[0][0, :512] == 0x11)
    assert torch.all(l1[0][0, 512:] == 0x22)
    assert torch.all(l2[0][0] == 0x33)


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="DeepGEMM MegaMoE requires CUDA SM100",
)
def test_mxfp4_source_weights_finalize_with_deep_gemm():
    experts = _make_experts(intermediate_size=512, source_mxfp4=True).cuda()
    assert experts.w13_weight_packed is not None
    assert experts.w13_weight_scale is not None
    assert experts.w2_weight_packed is not None
    assert experts.w2_weight_scale is not None
    experts.w13_weight_packed.data.fill_(0x11)
    experts.w13_weight_scale.data.fill_(127)
    experts.w2_weight_packed.data.fill_(0x11)
    experts.w2_weight_scale.data.fill_(127)

    experts.finalize_weights()

    assert experts._transformed_l1_weights is not None
    assert experts._transformed_l2_weights is not None


def test_bf16_mega_moe_stages_inputs_and_selects_bf16_kernel(monkeypatch):
    experts = _make_experts(mma_type="bf16xbf16")
    experts._transformed_l1_weights = torch.empty(1, 256, 128)
    experts._transformed_l2_weights = torch.empty(1, 128, 128)
    buffer = SimpleNamespace(
        x=torch.empty(4, 128, dtype=torch.bfloat16),
        topk_idx=torch.empty(4, 1, dtype=torch.int64),
        topk_weights=torch.empty(4, 1, dtype=torch.float32),
    )
    experts.get_symm_buffer = lambda: buffer
    called = []

    def bf16_mega_moe(y, *_args, **_kwargs):
        called.append(True)
        y.zero_()

    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(bf16_mega_moe=bf16_mega_moe),
    )
    hidden_states = torch.randn(2, 128, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[0.7], [0.6]])
    topk_ids = torch.tensor([[0], [0]])

    output = experts(
        hidden_states,
        topk_weights,
        topk_ids,
        activation_clamp=None,
    )

    assert called == [True]
    assert torch.equal(buffer.x[:2], hidden_states)
    assert torch.equal(buffer.topk_idx[:2], topk_ids)
    assert torch.equal(buffer.topk_weights[:2], topk_weights)
    assert torch.count_nonzero(output) == 0
