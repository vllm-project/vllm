# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from transformers import AutoConfig

from vllm.transformers_utils.config import get_config, uses_mrope
from vllm.transformers_utils.configs.bailing_moe_v3_vl import (
    BailingMoeV3TextConfig,
    BailingMoeV3VisionConfig,
    BailingMoeV3VLConfig,
)


def _bailing_v3_vl_config() -> dict[str, Any]:
    return {
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": ("configuration_bailing_moe_v3_vl.BailingMoeV3VLConfig")
        },
        "model_type": "bailing_moe_v3_vl",
        "mrope_section": [8, 12, 12],
        "text_config": {
            "num_hidden_layers": 42,
            "layer_group_size": 6,
            "rope_theta": 6_000_000,
        },
        "vision_config": {"disable_merger_proj": True},
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_bailing_v3_vl_config_loads_without_remote_code(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_config(model_path, _bailing_v3_vl_config())

    config = get_config(model_path, trust_remote_code=False)

    assert isinstance(config, BailingMoeV3VLConfig)
    assert isinstance(config.text_config, BailingMoeV3TextConfig)
    assert isinstance(config.vision_config, BailingMoeV3VisionConfig)
    assert isinstance(
        AutoConfig.from_pretrained(model_path, trust_remote_code=False),
        BailingMoeV3VLConfig,
    )
    assert config.mrope_section == [8, 12, 12]
    assert uses_mrope(config)

    text_config = config.text_config
    assert text_config.model_type == "bailing_hybrid"
    assert text_config.architectures == ["BailingMoeV3ForCausalLM"]
    assert text_config.rope_parameters["mrope_section"] == [8, 12, 12]
    assert text_config.rope_parameters["rope_theta"] == 6_000_000
    assert text_config.layer_types.count("linear_attention") == 35
    assert [
        layer_idx
        for layer_idx, layer_type in enumerate(text_config.layer_types)
        if layer_type == "full_attention"
    ] == [5, 11, 17, 23, 29, 35, 41]
    assert config.vision_config.disable_merger_proj is True

    normalized = config.to_dict()
    restored = BailingMoeV3VLConfig.from_dict(normalized)
    assert isinstance(restored.text_config, BailingMoeV3TextConfig)
    assert isinstance(restored.vision_config, BailingMoeV3VisionConfig)
    assert restored.to_dict() == normalized


def test_bailing_v3_vl_config_preserves_explicit_nested_values():
    config_dict = _bailing_v3_vl_config()
    config_dict["architectures"] = ["CustomBailingVLForConditionalGeneration"]
    text_config = config_dict["text_config"]
    text_config["architectures"] = ["CustomBailingForCausalLM"]
    text_config["layer_types"] = (["linear_attention"] * 5 + ["full_attention"]) * 7
    text_config["rope_theta"] = 1_000_000
    text_config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 1_000_000,
        "mrope_section": [8, 12, 12],
    }
    original = copy.deepcopy(config_dict)

    config = BailingMoeV3VLConfig(**config_dict)

    assert config_dict == original
    assert config.architectures == ["CustomBailingVLForConditionalGeneration"]
    assert config.text_config.architectures == ["CustomBailingForCausalLM"]
    assert config.text_config.layer_types == text_config["layer_types"]
    assert config.text_config.rope_parameters["rope_theta"] == 1_000_000
    assert config.text_config.rope_parameters["mrope_section"] == [8, 12, 12]


def test_bailing_v3_vl_config_normalizes_config_instance():
    text_config = BailingMoeV3TextConfig()

    config = BailingMoeV3VLConfig(
        text_config=text_config,
        mrope_section=[8, 12, 12],
    )

    assert config.text_config is text_config
    assert text_config.rope_parameters["mrope_section"] == [8, 12, 12]


def test_bailing_v3_text_config_trailing_and_legacy_layers():
    expected = ["linear_attention", "linear_attention", "full_attention"] * 2 + [
        "full_attention"
    ] * 2
    text_config = BailingMoeV3TextConfig(
        num_hidden_layers=8,
        layer_group_size=3,
    )
    assert text_config.layer_types == expected

    text_config = BailingMoeV3TextConfig(
        num_hidden_layers=8,
        layer_group_size=3,
        layers_block_type=["mamba", "mamba", "attention"] * 2 + ["attention"] * 2,
    )
    assert text_config.layer_types == expected


def test_bailing_v3_text_config_accepts_legacy_rope_type():
    text_config = BailingMoeV3TextConfig(rope_scaling={"type": "linear", "factor": 2.0})

    assert text_config.rope_parameters["type"] == "linear"
    assert text_config.rope_parameters["rope_type"] == "linear"


def test_bailing_v3_vl_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="layer_group_size must be positive"):
        BailingMoeV3TextConfig(layer_group_size=0)

    with pytest.raises(ValueError, match="must match the layer_group_size schedule"):
        BailingMoeV3TextConfig(
            num_hidden_layers=8,
            layer_group_size=3,
            layer_types=["full_attention"] * 8,
        )

    with pytest.raises(ValueError, match="conflicting M-RoPE sections"):
        BailingMoeV3VLConfig(
            text_config={
                "rope_parameters": {
                    "rope_type": "default",
                    "mrope_section": [4, 14, 14],
                }
            },
            mrope_section=[8, 12, 12],
        )
