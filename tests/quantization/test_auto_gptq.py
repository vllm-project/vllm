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
)

PROMPT = "On the surface of Mars, we found"

MODELS = [
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
]


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
