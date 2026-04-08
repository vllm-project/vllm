# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from typing import NamedTuple

import pytest
from torch import nn

from vllm.lora.shared import (
    is_in_target_modules,
    is_supported_lora_module,
    parse_fine_tuned_lora_name,
    replace_submodule,
)
from vllm.model_executor.models.utils import WeightsMapper


class TestIsSupportedLoraModule:
    """Tests for is_supported_lora_module (model-definition check)."""

    def test_suffix_match(self):
        assert is_supported_lora_module(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_no_match(self):
        assert not is_supported_lora_module(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_exact_match(self):
        assert is_supported_lora_module("o_proj", ["o_proj"])

    def test_regex_suffix_matching(self):
        """Regex anchors to end — partial suffix should not match."""
        assert not is_supported_lora_module("model.layers.0.self_attn.o_proj", ["proj"])

    def test_empty_supported_modules(self):
        assert not is_supported_lora_module("model.layers.0.self_attn.o_proj", [])

    def test_multiple_supported_modules(self):
        supported = ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert is_supported_lora_module("model.layers.0.self_attn.v_proj", supported)
        assert not is_supported_lora_module("model.layers.0.mlp.gate_proj", supported)


class TestIsInTargetModules:
    """Tests for is_in_target_modules (deployment-time filter)."""

    def test_none_allows_all(self):
        assert is_in_target_modules("model.layers.0.self_attn.o_proj", None)

    def test_suffix_in_target(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_suffix_not_in_target(self):
        assert not is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_empty_target_modules(self):
        assert not is_in_target_modules("model.layers.0.self_attn.o_proj", [])

    def test_exact_name_match(self):
        assert is_in_target_modules("dense1", ["dense1", "dense2"])

    def test_exact_name_no_match(self):
        assert not is_in_target_modules("dense3", ["dense1", "dense2"])


class LoRANameParserTestConfig(NamedTuple):
    name: str
    module_name: str
    is_lora_a: bool
    weights_mapper: WeightsMapper | None = None


def test_parse_fine_tuned_lora_name_valid():
    fixture = [
        LoRANameParserTestConfig(
            "base_model.model.lm_head.lora_A.weight", "lm_head", True, False
        ),
        LoRANameParserTestConfig(
            "base_model.model.lm_head.lora_B.weight", "lm_head", False, False
        ),
        LoRANameParserTestConfig(
            "base_model.model.model.embed_tokens.lora_embedding_A",
            "model.embed_tokens",
            True,
        ),
        LoRANameParserTestConfig(
            "base_model.model.model.embed_tokens.lora_embedding_B",
            "model.embed_tokens",
            False,
        ),
        LoRANameParserTestConfig(
            "base_model.model.model.layers.9.mlp.down_proj.lora_A.weight",
            "model.layers.9.mlp.down_proj",
            True,
        ),
        LoRANameParserTestConfig(
            "base_model.model.model.layers.9.mlp.down_proj.lora_B.weight",
            "model.layers.9.mlp.down_proj",
            False,
        ),
        LoRANameParserTestConfig(
            "language_model.layers.9.mlp.down_proj.lora_A.weight",
            "language_model.layers.9.mlp.down_proj",
            True,
        ),
        LoRANameParserTestConfig(
            "language_model.layers.9.mlp.down_proj.lora_B.weight",
            "language_model.layers.9.mlp.down_proj",
            False,
        ),
        # Test with WeightsMapper
        LoRANameParserTestConfig(
            "base_model.model.model.layers.9.mlp.down_proj.lora_A.weight",
            "language_model.model.layers.9.mlp.down_proj",
            True,
            weights_mapper=WeightsMapper(
                orig_to_new_prefix={"model.": "language_model.model."}
            ),
        ),
        LoRANameParserTestConfig(
            "base_model.model.model.layers.9.mlp.down_proj.lora_B.weight",
            "language_model.model.layers.9.mlp.down_proj",
            False,
            weights_mapper=WeightsMapper(
                orig_to_new_prefix={"model.": "language_model.model."}
            ),
        ),
        LoRANameParserTestConfig(
            "model.layers.9.mlp.down_proj.lora_A.weight",
            "language_model.model.layers.9.mlp.down_proj",
            True,
            weights_mapper=WeightsMapper(
                orig_to_new_prefix={"model.": "language_model.model."}
            ),
        ),
        LoRANameParserTestConfig(
            "model.layers.9.mlp.down_proj.lora_B.weight",
            "language_model.model.layers.9.mlp.down_proj",
            False,
            weights_mapper=WeightsMapper(
                orig_to_new_prefix={"model.": "language_model.model."}
            ),
        ),
    ]
    for name, module_name, is_lora_a, weights_mapper in fixture:
        assert (module_name, is_lora_a) == parse_fine_tuned_lora_name(
            name, weights_mapper
        )


def test_parse_fine_tuned_lora_name_invalid():
    fixture = {
        "base_model.weight",
        "base_model.model.weight",
    }
    for name in fixture:
        with pytest.raises(ValueError, match="unsupported LoRA weight"):
            parse_fine_tuned_lora_name(name)


def test_replace_submodule():
    model = nn.Sequential(
        OrderedDict(
            [
                ("dense1", nn.Linear(764, 100)),
                ("act1", nn.ReLU()),
                ("dense2", nn.Linear(100, 50)),
                (
                    "seq1",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("dense1", nn.Linear(100, 10)),
                                ("dense2", nn.Linear(10, 50)),
                            ]
                        )
                    ),
                ),
                ("act2", nn.ReLU()),
                ("output", nn.Linear(50, 10)),
                ("outact", nn.Sigmoid()),
            ]
        )
    )

    sigmoid = nn.Sigmoid()

    replace_submodule(model, "act1", sigmoid)
    assert dict(model.named_modules())["act1"] == sigmoid

    dense2 = nn.Linear(1, 5)
    replace_submodule(model, "seq1.dense2", dense2)
    assert dict(model.named_modules())["seq1.dense2"] == dense2
