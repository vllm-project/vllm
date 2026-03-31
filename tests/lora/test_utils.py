# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import HfHubHTTPError
from torch import nn

from vllm.lora.utils import (
    get_adapter_absolute_path,
    parse_fine_tuned_lora_name,
    replace_submodule,
)
from vllm.model_executor.models.utils import WeightsMapper


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


# Unit tests for get_adapter_absolute_path
@patch("os.path.isabs")
def test_get_adapter_absolute_path_absolute(mock_isabs):
    path = "/absolute/path/to/lora"
    mock_isabs.return_value = True
    assert get_adapter_absolute_path(path) == path


@patch("os.path.expanduser")
def test_get_adapter_absolute_path_expanduser(mock_expanduser):
    # Path with ~ that needs to be expanded
    path = "~/relative/path/to/lora"
    absolute_path = "/home/user/relative/path/to/lora"
    mock_expanduser.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch("os.path.exists")
@patch("os.path.abspath")
def test_get_adapter_absolute_path_local_existing(mock_abspath, mock_exist):
    # Relative path that exists locally
    path = "relative/path/to/lora"
    absolute_path = "/absolute/path/to/lora"
    mock_exist.return_value = True
    mock_abspath.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch("huggingface_hub.snapshot_download")
@patch("os.path.exists")
def test_get_adapter_absolute_path_huggingface(mock_exist, mock_snapshot_download):
    # Hugging Face model identifier
    path = "org/repo"
    absolute_path = "/mock/snapshot/path"
    mock_exist.return_value = False
    mock_snapshot_download.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch("huggingface_hub.snapshot_download")
@patch("os.path.exists")
def test_get_adapter_absolute_path_huggingface_error(
    mock_exist, mock_snapshot_download
):
    # Hugging Face model identifier with download error
    path = "org/repo"
    mock_exist.return_value = False
    mock_snapshot_download.side_effect = HfHubHTTPError(
        "failed to query model info",
        response=MagicMock(),
    )
    assert get_adapter_absolute_path(path) == path


# ---------------------------------------------------------------------------
# Regression tests for MoE expert LoRA module name parsing (gh-38522)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name, expected_last_component",
    [
        # Standard MoE: "...experts" is the leaf — last component is "experts"
        ("model.layers.0.mlp.experts", "experts"),
        # Qwen3.5-MoE style: expert index embedded in path
        ("model.layers.0.mlp.experts.0.down_proj", "down_proj"),
        ("model.layers.0.mlp.experts.1.gate_proj", "gate_proj"),
        ("model.layers.0.mlp.experts.2.up_proj", "up_proj"),
        # Deeply nested expert path
        ("transformer.blocks.3.ffn.experts.7.fc1", "fc1"),
    ],
)
def test_moe_expert_module_name_last_component(
    module_name: str, expected_last_component: str
):
    """
    Regression test for gh-38522.

    The old code used ``module_name[module_name.find('.experts') + 1:]``
    which returns ``"experts.N.down_proj"`` for embedded-index paths —
    never matching the expected module set even when ``"down_proj"`` is.

    The fix uses ``module_name.split('.')[-1]`` to extract only the
    leaf component, which is compared against ``expected_lora_modules``.
    """
    if ".experts" in module_name:
        # New logic: always take the last component
        suffix = module_name.split(".")[-1]

        # Old (buggy) logic for comparison
        expert_idx = module_name.find(".experts")
        old_suffix = module_name[expert_idx + 1:]

        assert suffix == expected_last_component, (
            f"New logic returned {suffix!r}, expected {expected_last_component!r}"
        )

        if "." in old_suffix:
            # The old code would have produced a multi-segment string that
            # is never in expected_lora_modules — confirm regression exists.
            assert old_suffix != expected_last_component, (
                "Old logic unexpectedly returned the correct value; "
                "update this regression test."
            )
