# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether gptq models with dynamic quantized can be loaded.

Run `pytest tests/quantization/test_gptq_dynamic.py --forked`.

Note: Only symmetric GPTQ models are supported after consolidation to Marlin.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
)

PROMPT = "On the surface of Mars, we found"

# The first layer is quantized using bits=4, group_size=128
# The second layer is quantized using bits=8, group_size=32
# All other layers (layer index >= 2) are not quantized
# Note: Only symmetric models are supported with Marlin kernels
MODELS = [
    "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symTrue",
]


def _make_dynamic_config(
    dynamic: dict[str, dict[str, int | bool]],
    packed_modules_mapping: dict[str, list[str]],
) -> AutoGPTQConfig:
    config = AutoGPTQConfig(4, 128, False, True, False, dynamic, {})
    config.packed_modules_mapping = packed_modules_mapping
    return config


@pytest.mark.parametrize(
    "packed_modules_mapping",
    [
        {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
        {
            "self_attn.qkv_proj": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ]
        },
    ],
    ids=["module-name", "dotted-module-name"],
)
def test_dynamic_override_resolves_fused_qkv_shards(
    packed_modules_mapping: dict[str, list[str]],
) -> None:
    """Resolve unfused checkpoint rules for either fused mapping format."""
    dynamic = {
        rf"+:^model\.layers\.0\.self_attn\.{projection}$": {
            "bits": 4,
            "group_size": 32,
        }
        for projection in ("q_proj", "k_proj", "v_proj")
    }
    dynamic[r"+:^model\.layers\.0\.self_attn\.notq_proj$"] = {"group_size": 64}
    config = _make_dynamic_config(dynamic, packed_modules_mapping)

    assert (
        get_dynamic_override(
            config,
            "model.layers.0.self_attn.qkv_proj",
            "group_size",
            config.group_size,
        )
        == 32
    )
    assert (
        get_dynamic_override(
            config,
            "model.layers.1.self_attn.qkv_proj",
            "group_size",
            config.group_size,
        )
        == 128
    )
    assert (
        get_dynamic_override(
            config,
            "model.layers.0.self_attn.notqkv_proj",
            "group_size",
            config.group_size,
        )
        == 128
    )


def test_dynamic_override_skips_fused_layer_when_all_shards_are_excluded() -> None:
    """A fused layer is skipped when every checkpoint shard is excluded."""
    dynamic = {
        rf"-:^model\.layers\.0\.self_attn\.{projection}$": {}
        for projection in ("q_proj", "k_proj", "v_proj")
    }
    config = _make_dynamic_config(
        dynamic,
        {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )

    assert (
        get_dynamic_override(
            config,
            "model.layers.0.self_attn.qkv_proj",
        )
        is False
    )


def test_dynamic_override_rejects_partially_excluded_fused_layer() -> None:
    """A fused layer cannot contain both skipped and quantized shards."""
    config = _make_dynamic_config(
        {r"-:^model\.layers\.0\.self_attn\.q_proj$": {}},
        {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )

    with pytest.raises(ValueError, match="does not match across shards"):
        get_dynamic_override(
            config,
            "model.layers.0.self_attn.qkv_proj",
        )


def test_dynamic_override_rejects_mixed_fused_shards() -> None:
    """Reject fused shards whose effective settings differ."""
    config = _make_dynamic_config(
        {r"+:^model\.layers\.0\.self_attn\.q_proj$": {"group_size": 32}},
        {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )

    with pytest.raises(ValueError, match="does not match across shards"):
        get_dynamic_override(
            config,
            "model.layers.0.self_attn.qkv_proj",
            "group_size",
            config.group_size,
        )


@pytest.mark.parametrize(
    ("dynamic", "expected"),
    [
        (
            {
                r"+:^model\.layers\..*\.q_proj$": {"group_size": 64},
                r"+:^model\.layers\.0\.q_proj$": {"group_size": 32},
            },
            64,
        ),
        (
            {
                r"+:^model\.layers\.0\.q_proj$": {"group_size": 32},
                r"+:^model\.layers\..*\.q_proj$": {"group_size": 64},
            },
            32,
        ),
    ],
)
def test_dynamic_override_preserves_first_match_order(dynamic, expected) -> None:
    """The exact-rule index must retain dict insertion order semantics."""
    config = _make_dynamic_config(dynamic, {})

    assert (
        get_dynamic_override(
            config,
            "model.layers.0.q_proj",
            "group_size",
            config.group_size,
        )
        == expected
    )


def test_linear_dynamic_override_does_not_mutate_base_config() -> None:
    """Per-layer overrides must leave the shared model config unchanged."""
    config = _make_dynamic_config(
        {r"+:^model\.layers\.0\.q_proj$": {"group_size": 32}},
        {},
    )
    config.modules_in_block_to_quantize = ["q_proj"]
    layer = object.__new__(LinearBase)

    method = get_linear_quant_method(
        config,
        layer,
        "model.layers.0.q_proj",
        lambda quant_config: SimpleNamespace(quant_config=quant_config),
    )

    assert method.quant_config.group_size == 32
    assert method.quant_config.full_config is not config.full_config
    assert config.group_size == 128


@pytest.mark.parametrize("model_id", MODELS)
def test_gptq_with_dynamic(vllm_runner, model_id: str, monkeypatch):
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    linear_method_cls = AutoGPTQLinearMethod

    with vllm_runner(
        model_id, dtype=torch.float16, max_model_len=2048, enforce_eager=True
    ) as llm:

        def check_model(model):
            for name, submodule in model.named_modules():
                if name == "lm_head":
                    assert isinstance(submodule.quant_method, linear_method_cls)
                elif name == "model.layers.0.self_attn.qkv_proj":
                    # The first layer is quantized using bits=4, group_size=128
                    # desc_act=True
                    assert isinstance(submodule.quant_method, linear_method_cls)
                    config = submodule.quant_method.quant_config
                    assert config.weight_bits == 4
                    assert config.group_size == 128
                    assert config.desc_act
                elif name == "model.layers.1.self_attn.qkv_proj":
                    # The second layer is quantized using bits=8, group_size=32
                    # desc_act=False
                    assert isinstance(submodule.quant_method, linear_method_cls)
                    config = submodule.quant_method.quant_config
                    assert (
                        get_dynamic_override(config, layer_name=name, key="bits") == 8
                    )
                    assert (
                        get_dynamic_override(config, layer_name=name, key="group_size")
                        == 32
                    )
                    assert not get_dynamic_override(
                        config, layer_name=name, key="desc_act"
                    )
                elif (
                    name == "model.layers.2.self_attn.qkv_proj"
                    or name == "model.layers.2.mlp.gate_up_proj"
                ):
                    # All other layers (layer index >= 2) are not quantized
                    assert isinstance(submodule.quant_method, UnquantizedLinearMethod)

        llm.apply_model(check_model)
