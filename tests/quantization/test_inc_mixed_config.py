# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.inc.inc import INCConfig


def make_config(extra):
    config = INCConfig(weight_bits=4, group_size=128, extra_config=extra)
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    return config


def fp8_config():
    return {
        "bits": 8,
        "group_size": [128, 128],
        "data_type": "fp",
        "packing_format": "auto_round:fp8",
    }


def test_inc_mixed_exact_and_default():
    config = make_config({"model.layers.0.q_proj": fp8_config()})
    layer = torch.nn.Linear(128, 128)
    assert config.config_parser.resolve(layer, "model.layers.0.q_proj").is_fp8_block
    assert config.config_parser.resolve(layer, "model.layers.1.q_proj").is_wna16_int
    assert config.get_layer_config(layer, "model.layers.0.q_proj") == (
        8,
        (128, 128),
        True,
    )


def test_inc_mixed_regex_and_suffix():
    config = make_config({r"model\.layers\.0\..*": fp8_config(), "head": fp8_config()})
    layer = torch.nn.Linear(128, 128)
    assert config.config_parser.resolve(layer, "model.layers.0.q_proj").is_fp8_block
    assert config.config_parser.resolve(layer, "nested.head").is_fp8_block


def test_inc_mixed_fused_equal():
    config = make_config(
        {f"model.{name}": fp8_config() for name in ("q_proj", "k_proj", "v_proj")}
    )
    assert config.config_parser.resolve(
        torch.nn.Linear(128, 128), "model.qkv_proj"
    ).is_fp8_block


@pytest.mark.parametrize(
    "field,value", [("data_type", "int"), ("packing_format", "auto_round:auto_gptq")]
)
def test_inc_mixed_fused_rejects_format_difference(field, value):
    overrides = {
        f"model.{name}": fp8_config() for name in ("q_proj", "k_proj", "v_proj")
    }
    overrides["model.k_proj"][field] = value
    with pytest.raises(ValueError, match="consistent quant config"):
        make_config(overrides).config_parser.resolve(
            torch.nn.Linear(128, 128), "model.qkv_proj"
        )


def test_inc_mixed_bf16_override():
    config = make_config({"model.q_proj": {"bits": 16}})
    assert not config.config_parser.resolve(
        torch.nn.Linear(128, 128), "model.q_proj"
    ).quantized
