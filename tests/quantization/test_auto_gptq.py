# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the auto_gptq quantization method works correctly.

Run `pytest tests/quantization/test_auto_gptq.py -v -s`.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
    AutoGPTQMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    is_layer_gptq_quantized,
    is_moe_layer_gptq_quantized,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_padded_intermediate,
)
from vllm.platforms import current_platform

PROMPT = "On the surface of Mars, we found"

MODELS = [
    "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
]


@pytest.mark.skipif(
    not is_quant_method_supported("auto_gptq"),
    reason="auto_gptq is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
def test_auto_gptq_quantization_method(
    model_id: str, monkeypatch, dist_init, workspace_init
):
    """Test that quantization='auto_gptq' loads and runs correctly."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model, vllm_config = load_model_without_vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_gptq",
        model_config_kwargs={"max_model_len": 2048},
    )

    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    assert isinstance(qkv_proj.quant_method, AutoGPTQLinearMethod)

    target_device = torch.device(current_platform.device_type)
    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=target_device)
    positions = torch.arange(input_ids.numel(), device=target_device)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


def test_auto_gptq_config_get_name():
    """Test that AutoGPTQConfig.get_name() returns 'auto_gptq'."""
    assert AutoGPTQConfig.get_name() == "auto_gptq"


def test_auto_gptq_quantizes_every_layer_when_the_module_list_is_unknown():
    """An empty module list means the safetensors metadata could not be read,
    not that the checkpoint holds no quantized layer."""
    assert is_layer_gptq_quantized(
        prefix="model.layers.0.mlp.down_proj",
        quantized_layers=[],
    )
    assert not is_layer_gptq_quantized(
        prefix="model.layers.0.mlp.down_proj",
        quantized_layers=["self_attn.q_proj"],
    )


@pytest.mark.parametrize(
    "dynamic",
    [
        {},
        {r"+:model\.layers\.0\..*": {"desc_act": True}},
        {
            r"+:model\.layers\.0\..*": {
                "desc_act": True,
                "group_size": 32,
            }
        },
    ],
)
def test_auto_gptq_rejects_group_activation_order(dynamic):
    desc_act = not dynamic
    with pytest.raises(ValueError, match="group activation ordering"):
        AutoGPTQConfig(4, 128, desc_act, True, False, dynamic, {})


def test_auto_gptq_normalizes_channelwise_activation_order():
    config = AutoGPTQConfig(4, -1, True, True, False, {}, {})
    assert not config.desc_act

    config = AutoGPTQConfig(
        4,
        128,
        False,
        True,
        False,
        {r"+:model\.layers\.0\..*": {"desc_act": True, "group_size": -1}},
        {},
    )
    override_config(config, "model.layers.0.self_attn.q_proj")
    assert config.group_size == -1
    assert not config.desc_act

    with pytest.raises(ValueError, match="group activation ordering"):
        AutoGPTQConfig(
            4,
            -1,
            True,
            True,
            False,
            {r"+:model\.layers\.0\..*": {"group_size": 128}},
            {},
        )


def test_moe_layer_quantized_only_when_its_experts_are_in_the_checkpoint():
    """Experts left unquantized in the checkpoint must not get a GPTQ method.

    `modules_in_block_to_quantize` is derived from the checkpoint dtypes, so a
    partially quantized model (e.g. one whose MTP layer is stored unquantized)
    only lists the quantized layers' experts. Giving the remaining layer a
    quantized method makes it allocate `w2_qweight` while the checkpoint
    provides `w2_weight`.
    """
    quantized_layers = [
        f"language_model.model.layers.{layer}.mlp.experts.{expert}.{proj}"
        for layer in range(2)
        for expert in range(2)
        for proj in ("gate_proj", "down_proj", "up_proj")
    ]

    assert is_moe_layer_gptq_quantized(
        "language_model.model.layers.0.mlp.experts", quantized_layers
    )
    assert not is_moe_layer_gptq_quantized("mtp.layers.0.mlp.experts", quantized_layers)
    # Block-relative names (as written by optimum) cannot distinguish layers.
    assert is_moe_layer_gptq_quantized(
        "mtp.layers.0.mlp.experts", ["mlp.experts.0.down_proj"]
    )
    # Configs that never mention experts carry no information about them.
    assert is_moe_layer_gptq_quantized(
        "mtp.layers.0.mlp.experts", ["mtp.layers.0.self_attn.q_proj"]
    )
    # The runtime prefix may be the experts' parent (DBRX's `...ffn`).
    assert is_moe_layer_gptq_quantized(
        "transformer.blocks.0.ffn", ["transformer.blocks.0.ffn.experts.mlp.w1"]
    )
    assert not is_moe_layer_gptq_quantized(
        "transformer.blocks.1.ffn", ["transformer.blocks.0.ffn.experts.mlp.w1"]
    )


def _gptq_moe_layer() -> RoutedExperts:
    layer = object.__new__(RoutedExperts)
    layer.moe_config = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=4,
        num_logical_experts=4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        intermediate_size_per_partition_unpadded=128,
    )
    return layer


def _gptq_config(
    dynamic=None, modules_in_block_to_quantize=None, bits=4, group_size=128
):
    full_config = {
        "quant_method": "gptq",
        "bits": bits,
        "group_size": group_size,
        "desc_act": False,
        "sym": True,
    }
    return AutoGPTQConfig(
        bits,
        group_size,
        False,
        True,
        False,
        dynamic or {},
        full_config,
        modules_in_block_to_quantize,
    )


@pytest.mark.parametrize("marlin_supported", [True, False])
def test_auto_gptq_moe_quant_method_follows_checkpoint(
    marlin_supported, monkeypatch, default_vllm_config
):
    """MTP experts stored in BF16 get the unquantized method, quantized ones
    keep a GPTQ method, on both the Marlin path and the WNA16 fallback."""
    from vllm.model_executor.layers.quantization import auto_gptq, moe_wna16

    # Kernel availability is not what is under test here.
    for module in (auto_gptq, moe_wna16):
        monkeypatch.setattr(
            module, "select_wna16_moe_backend", lambda *args, **kwargs: (None, None)
        )
    monkeypatch.setattr(
        auto_gptq,
        "check_moe_marlin_supports_layer",
        lambda *args, **kwargs: marlin_supported,
    )
    quantized_method = (
        AutoGPTQMoEMethod if marlin_supported else moe_wna16.MoeWNA16Method
    )
    layer = _gptq_moe_layer()

    config = _gptq_config(
        modules_in_block_to_quantize=[
            f"model.layers.0.mlp.experts.{expert}.{proj}"
            for expert in range(4)
            for proj in ("gate_proj", "up_proj", "down_proj")
        ]
    )
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.mlp.experts"),
        quantized_method,
    )
    assert isinstance(
        config.get_quant_method(layer, "mtp.layers.0.mlp.experts"),
        UnquantizedFusedMoEMethod,
    )

    # MTP experts the checkpoint did quantize keep the quantized method.
    config = _gptq_config(
        modules_in_block_to_quantize=[
            f"{layer_prefix}.mlp.experts.{expert}.{proj}"
            for layer_prefix in ("model.layers.0", "mtp.layers.0")
            for expert in range(4)
            for proj in ("gate_proj", "up_proj", "down_proj")
        ]
    )
    for prefix in ("model.layers.0.mlp.experts", "mtp.layers.0.mlp.experts"):
        assert isinstance(config.get_quant_method(layer, prefix), quantized_method)

    # Negative dynamic rules exclude the layer on either path.
    config = _gptq_config(dynamic={r"-:.*mtp.*": {}})
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.mlp.experts"),
        quantized_method,
    )
    assert isinstance(
        config.get_quant_method(layer, "mtp.layers.0.mlp.experts"),
        UnquantizedFusedMoEMethod,
    )


def test_auto_gptq_negative_dynamic_rule_excludes_matching_linears(monkeypatch):
    """`-:` rules keep matching linears unquantized.

    Block-relative `modules_in_block_to_quantize` entries also match inside the
    excluded prefixes, so only the rule can tell those layers apart.
    """
    from vllm.model_executor.layers.quantization import auto_gptq

    monkeypatch.setattr(auto_gptq, "verify_marlin_supported", lambda **kwargs: None)
    layer = object.__new__(LinearBase)
    # Block-relative names match inside the MTP prefixes too, so only the
    # dynamic rule can tell the draft layers apart.
    config = _gptq_config(
        dynamic={r"-:.*mtp.*": {}},
        modules_in_block_to_quantize=[
            "layers.0.self_attn.q_proj",
            "layers.0.mlp.shared_expert.gate_proj",
        ],
    )

    for prefix in (
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj",
    ):
        assert isinstance(
            config.get_quant_method(layer, prefix), UnquantizedLinearMethod
        )
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.q_proj"),
        AutoGPTQLinearMethod,
    )


def test_auto_gptq_moe_creates_zero_initialized_expert_biases():
    method = object.__new__(AutoGPTQMoEMethod)
    method.quant_config = AutoGPTQConfig(4, 128, False, True, False, {}, {})
    method.experts_cls = None
    method.moe = SimpleNamespace(w13_num_shards=2)
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=8,
        intermediate_size_per_partition=4,
        params_dtype=torch.float16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.w13_bias.shape == (2, 8)
    assert layer.w2_bias.shape == (2, 8)
    assert torch.count_nonzero(layer.w13_bias) == 0
    assert torch.count_nonzero(layer.w2_bias) == 0


def test_routed_experts_loads_per_expert_biases():
    class Loader:
        quant_config = None
        quant_method = object()
        shared_expert_online_loader = None
        moe_config = SimpleNamespace(
            is_act_and_mul=True,
            tp_rank=0,
            num_experts=1,
            moe_parallel_config=SimpleNamespace(tp_size=1),
        )
        expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
        _get_hidden_dim = staticmethod(RoutedExperts._get_hidden_dim)
        _narrow_expert_data_for_padding = staticmethod(
            RoutedExperts._narrow_expert_data_for_padding
        )
        _load_w13 = RoutedExperts._load_w13
        _loaded_expert_biases = set()

        @staticmethod
        def _map_global_expert_id_to_local_expert_id(expert_id):
            return expert_id

    loader = Loader()
    w13_bias = torch.nn.Parameter(torch.zeros(1, 8), requires_grad=False)
    w2_bias = torch.nn.Parameter(torch.zeros(1, 4), requires_grad=False)

    for shard_id, loaded in (
        ("w1", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("w3", torch.tensor([5.0, 6.0, 7.0, 8.0])),
    ):
        assert RoutedExperts.weight_loader(  # type: ignore[call-overload]
            loader,
            w13_bias,
            loaded,
            weight_name="model.layers.0.mlp.experts.w13_bias",
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    assert RoutedExperts.weight_loader(  # type: ignore[call-overload]
        loader,
        w2_bias,
        torch.tensor([9.0, 10.0, 11.0, 12.0]),
        weight_name="model.layers.0.mlp.experts.w2_bias",
        shard_id="w2",
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(w13_bias, torch.arange(1, 9, dtype=torch.float32).reshape(1, 8))
    assert torch.equal(w2_bias, torch.arange(9, 13, dtype=torch.float32).reshape(1, 4))
    assert loader._loaded_expert_biases == {"w13_bias", "w2_bias"}


def _gptq_marlin_moe_layer(
    bits: int, group_size: int, intermediate_size: int
) -> RoutedExperts:
    num_experts, hidden_size = 64, 2048
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=2,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Default,
        intermediate_size_per_partition_unpadded=intermediate_size,
    )
    moe_config.moe_backend = "marlin"
    layer = RoutedExperts(
        "model.layers.0.mlp.experts",
        torch.bfloat16,
        moe_config,
        _gptq_config(bits=bits, group_size=group_size),
        expert_map_manager=ExpertMapManager(
            max_num_batched_tokens=256,
            top_k=2,
            global_num_experts=num_experts,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=moe_config.moe_parallel_config,
            placement_strategy="linear",
            enable_eplb=False,
        ),
    ).cuda()
    generator = torch.Generator(device="cuda").manual_seed(0)
    for parameter in layer.parameters():
        if parameter.dtype == torch.int32:
            parameter.data.copy_(
                torch.randint(
                    0,
                    2**31 - 1,
                    parameter.shape,
                    dtype=torch.int32,
                    device="cuda",
                    generator=generator,
                )
            )
        else:
            parameter.data.copy_(
                torch.rand(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device="cuda",
                    generator=generator,
                )
            )
    return layer


def _extra_peak_bytes(fn):
    torch.accelerator.synchronize()
    allocated_before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    result = fn()
    torch.accelerator.synchronize()
    return result, torch.accelerator.max_memory_allocated() - allocated_before


def _convert_for_marlin(layer, backend, n_first):
    names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
    names += ("w13_qzeros", "w2_qzeros")
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros = (
        n_first(getattr(layer, name).data) for name in names
    )
    return convert_to_wna16_moe_kernel_format(
        backend=backend,
        layer=layer,
        quant_config=layer.quant_method.quant_config,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_qzeros=w13_qzeros,
        w2_qzeros=w2_qzeros,
    )


def _old_n_first(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(1, 2).contiguous()


def _view_n_first(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(1, 2)


def _assert_two_layout_copies_saved(layer, old_extra: int, new_extra: int):
    # The old path held N-first copies and K-first copies of the packed weights.
    # One extra layout copy of w2 alone, the smallest regression, exceeds the bound.
    w2_bytes = layer.w2_qweight.untyped_storage().nbytes()
    packed_weight_bytes = layer.w13_qweight.untyped_storage().nbytes() + w2_bytes
    assert new_extra < old_extra - 2 * packed_weight_bytes + w2_bytes * 3 // 4


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Marlin MoE repack needs CUDA"
)
@pytest.mark.parametrize(
    "bits, group_size, intermediate_size",
    [(4, 128, 1024), (8, 128, 1024), (4, -1, 1056)],
    ids=["int4", "int8", "int4-padded"],
)
def test_auto_gptq_moe_marlin_prep_skips_layout_copies(
    bits, group_size, intermediate_size, default_vllm_config
):
    """Marlin prep converts GPTQ MoE weights without full-layer layout copies.

    The old path copied the K-first checkpoint weights to N-first and Marlin
    copied them straight back, so the layer briefly held two extra copies.
    The result must match that path exactly, with and without tile padding.
    """
    layer = _gptq_marlin_moe_layer(bits, group_size, intermediate_size)
    method = layer.quant_method
    assert isinstance(method, AutoGPTQMoEMethod)
    assert method.wna16_moe_backend == WNA16MoEBackend.MARLIN
    padded = marlin_moe_padded_intermediate(intermediate_size, group_size)
    assert (padded != intermediate_size) == (group_size == -1)

    expected, old_extra = _extra_peak_bytes(
        lambda: _convert_for_marlin(layer, WNA16MoEBackend.MARLIN, _old_n_first)
    )
    assert expected is not None
    _, new_extra = _extra_peak_bytes(
        lambda: method.process_weights_after_loading(layer)
    )

    _assert_two_layout_copies_saved(layer, old_extra, new_extra)
    names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
    names += ("w13_qzeros", "w2_qzeros")
    for name, value in zip(names, expected):
        assert torch.equal(getattr(layer, name).data, value), name


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Marlin MoE repack needs CUDA"
)
def test_batched_marlin_converts_gptq_views_without_layout_copies(
    default_vllm_config,
):
    """Batched Marlin shares the Marlin conversion, so views must work there too.

    Batched Marlin is chosen only for batched all-to-all activations, which a
    single-GPU test cannot build, so the conversion is called directly.
    """
    layer = _gptq_marlin_moe_layer(bits=4, group_size=128, intermediate_size=1024)
    expected, old_extra = _extra_peak_bytes(
        lambda: _convert_for_marlin(layer, WNA16MoEBackend.BATCHED_MARLIN, _old_n_first)
    )
    converted, new_extra = _extra_peak_bytes(
        lambda: _convert_for_marlin(
            layer, WNA16MoEBackend.BATCHED_MARLIN, _view_n_first
        )
    )

    _assert_two_layout_copies_saved(layer, old_extra, new_extra)
    for value, reference in zip(converted, expected):
        if reference is None:
            assert value is None
        else:
            assert torch.equal(value, reference)
