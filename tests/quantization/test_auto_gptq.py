# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the auto_gptq quantization method works correctly.

Run `pytest tests/quantization/test_auto_gptq.py -v -s`.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
    AutoGPTQMoEMethod,
    _resolve_moe_quant_config,
)

PROMPT = "On the surface of Mars, we found"

MODELS = [
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
]

_MOE_PREFIX = "model.layers.0.mlp.experts"
_ESCAPED_MOE_PREFIX = _MOE_PREFIX.replace(".", r"\.")


@pytest.mark.skipif(
    not is_quant_method_supported("auto_gptq"),
    reason="auto_gptq is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
def test_auto_gptq_quantization_method(vllm_runner, model_id: str, monkeypatch):
    """Test that quantization='auto_gptq' loads and runs correctly."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_gptq",
        max_model_len=2048,
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            for name, submodule in model.named_modules():
                if name == "model.layers.0.self_attn.qkv_proj":
                    assert isinstance(submodule.quant_method, AutoGPTQLinearMethod)
                    break

        llm.apply_model(check_model)

        outputs = llm.generate_greedy([PROMPT], max_tokens=8)
        assert outputs
        assert len(outputs[0][1]) > 0


def test_auto_gptq_config_get_name():
    """Test that AutoGPTQConfig.get_name() returns 'auto_gptq'."""
    assert AutoGPTQConfig.get_name() == "auto_gptq"


def _make_auto_gptq_config(
    dynamic: dict[str, dict[str, int | bool]],
    *,
    group_size: int = 128,
    desc_act: bool = False,
) -> AutoGPTQConfig:
    full_config = {
        "bits": 4,
        "group_size": group_size,
        "desc_act": desc_act,
        "sym": True,
        "quant_method": "gptq",
        "dynamic": dynamic,
    }
    return AutoGPTQConfig.from_config(full_config)


def _make_routed_experts_stub(num_experts: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        moe_config=SimpleNamespace(num_logical_experts=num_experts),
        expert_map_manager=SimpleNamespace(num_fused_shared_experts=0),
    )


def test_auto_gptq_moe_resolves_mixed_expert_group_sizes() -> None:
    """Choose the smallest compatible group while retaining source metadata."""
    dynamic = {
        rf"+:^{_ESCAPED_MOE_PREFIX}\.1\..*_proj$": {
            "bits": 4,
            "group_size": 32,
        },
        rf"+:^{_ESCAPED_MOE_PREFIX}\.2\.down_proj$": {
            "bits": 4,
            "group_size": 32,
        },
    }

    config = _make_auto_gptq_config(dynamic)
    resolved, source_group_sizes = _resolve_moe_quant_config(
        config,
        _make_routed_experts_stub(),
        _MOE_PREFIX,
    )

    assert resolved is not None
    assert resolved.group_size == 32
    assert resolved.full_config["group_size"] == 32
    assert config.full_config["group_size"] == 128
    assert source_group_sizes[(0, "w1")] == 128
    assert source_group_sizes[(1, "w1")] == 32
    assert source_group_sizes[(2, "w1")] == 128
    assert source_group_sizes[(2, "w2")] == 32


def test_auto_gptq_moe_normalizes_larger_group_metadata() -> None:
    """Expand scales, zero points, and sequential group indices losslessly."""
    method = object.__new__(AutoGPTQMoEMethod)
    method.quant_config = _make_auto_gptq_config({}, group_size=32)
    method.source_group_sizes = {(0, "w1"): 128}
    loaded: dict[str, torch.Tensor] = {}

    def weight_loader(
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        loaded[weight_name] = loaded_weight
        return return_success

    wrapped_loader = method.get_weight_loader(weight_loader)
    param = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    scales = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    assert wrapped_loader(
        param, scales, "experts.w13_scales", "w1", 0, return_success=True
    )
    assert torch.equal(
        loaded["experts.w13_scales"],
        scales.repeat_interleave(4, dim=0),
    )

    qzeros = torch.arange(4, dtype=torch.int32).reshape(2, 2)
    assert wrapped_loader(
        param, qzeros, "experts.w13_qzeros", "w1", 0, return_success=True
    )
    assert torch.equal(
        loaded["experts.w13_qzeros"],
        qzeros.repeat_interleave(4, dim=0),
    )

    g_idx = torch.arange(256, dtype=torch.int32) // 128
    assert wrapped_loader(
        param, g_idx, "experts.w13_g_idx", "w1", 0, return_success=True
    )
    assert torch.equal(
        loaded["experts.w13_g_idx"],
        torch.arange(256, dtype=torch.int32) // 32,
    )


def test_auto_gptq_moe_rejects_non_sequential_group_indices() -> None:
    """Only the standard non-act-order g_idx layout can be normalized."""
    method = object.__new__(AutoGPTQMoEMethod)
    method.quant_config = _make_auto_gptq_config({}, group_size=32)
    method.source_group_sizes = {(0, "w1"): 128}

    with pytest.raises(ValueError, match="non-sequential g_idx"):
        method._normalize_group_metadata(
            torch.zeros(256, dtype=torch.int32),
            "experts.w13_g_idx",
            "w1",
            0,
        )


def test_auto_gptq_moe_rejects_mixed_groups_with_desc_act() -> None:
    """Activation-order metadata cannot be expanded losslessly."""
    dynamic = {
        rf"+:^{_ESCAPED_MOE_PREFIX}\.1\..*_proj$": {
            "group_size": 32,
        }
    }

    with pytest.raises(ValueError, match="desc_act=True"):
        _resolve_moe_quant_config(
            _make_auto_gptq_config(dynamic, desc_act=True),
            _make_routed_experts_stub(num_experts=2),
            _MOE_PREFIX,
        )


def test_auto_gptq_moe_rejects_partially_unquantized_shards() -> None:
    """A fused MoE layer cannot mix quantized and unquantized shards."""
    dynamic = {
        rf"-:^{_ESCAPED_MOE_PREFIX}\.1\.down_proj$": {},
    }

    with pytest.raises(ValueError, match="excludes only some expert shards"):
        _resolve_moe_quant_config(
            _make_auto_gptq_config(dynamic),
            _make_routed_experts_stub(num_experts=2),
            _MOE_PREFIX,
        )


def test_auto_gptq_moe_rejects_incompatible_group_sizes() -> None:
    """Only divisible group sizes have a lossless common representation."""
    dynamic = {
        rf"+:^{_ESCAPED_MOE_PREFIX}\.1\..*_proj$": {
            "group_size": 96,
        }
    }

    with pytest.raises(ValueError, match="incompatible group sizes"):
        _resolve_moe_quant_config(
            _make_auto_gptq_config(dynamic),
            _make_routed_experts_stub(num_experts=2),
            _MOE_PREFIX,
        )


def test_auto_gptq_moe_creates_zero_initialized_expert_biases():
    method = object.__new__(AutoGPTQMoEMethod)
    method.quant_config = AutoGPTQConfig(4, 128, False, True, False, {}, {})
    method.input_dtype = None
    method.experts_cls = None
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=8,
        intermediate_size_per_partition=4,
        params_dtype=torch.float16,
        intermediate_size_full=4,
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
        moe_config = SimpleNamespace(
            is_act_and_mul=True,
            tp_rank=0,
            moe_parallel_config=SimpleNamespace(tp_size=1),
        )
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
        assert RoutedExperts.weight_loader(
            loader,
            w13_bias,
            loaded,
            weight_name="model.layers.0.mlp.experts.w13_bias",
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    assert RoutedExperts.weight_loader(
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
