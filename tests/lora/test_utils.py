# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from unittest.mock import patch

import pytest
from huggingface_hub.utils import HfHubHTTPError
from torch import nn

from vllm.lora.utils import (get_adapter_absolute_path,
                             parse_fine_tuned_lora_name, replace_submodule)


def test_parse_fine_tuned_lora_name_valid():
    fixture = {
        ("base_model.model.lm_head.lora_A.weight", "lm_head", True, False),
        ("base_model.model.lm_head.lora_B.weight", "lm_head", False, False),
        (
            "base_model.model.model.embed_tokens.lora_embedding_A",
            "model.embed_tokens",
            True,
            False,
        ),
        (
            "base_model.model.model.embed_tokens.lora_embedding_B",
            "model.embed_tokens",
            False,
            False,
        ),
        (
            "base_model.model.model.layers.9.mlp.down_proj.lora_A.weight",
            "model.layers.9.mlp.down_proj",
            True,
            False,
        ),
        (
            "base_model.model.model.layers.9.mlp.down_proj.lora_B.weight",
            "model.layers.9.mlp.down_proj",
            False,
            False,
        ),
        (
            "language_model.layers.9.mlp.down_proj.lora_A.weight",
            "language_model.layers.9.mlp.down_proj",
            True,
            False,
        ),
        (
            "language_model.layers.9.mlp.down_proj.lora_B.weight",
            "language_model.layers.9.mlp.down_proj",
            False,
            False,
        ),
    }
    for name, module_name, is_lora_a, is_bias in fixture:
        assert (module_name, is_lora_a,
                is_bias) == parse_fine_tuned_lora_name(name)


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
        OrderedDict([
            ("dense1", nn.Linear(764, 100)),
            ("act1", nn.ReLU()),
            ("dense2", nn.Linear(100, 50)),
            (
                "seq1",
                nn.Sequential(
                    OrderedDict([
                        ("dense1", nn.Linear(100, 10)),
                        ("dense2", nn.Linear(10, 50)),
                    ])),
            ),
            ("act2", nn.ReLU()),
            ("output", nn.Linear(50, 10)),
            ("outact", nn.Sigmoid()),
        ]))

    sigmoid = nn.Sigmoid()

    replace_submodule(model, "act1", sigmoid)
    assert dict(model.named_modules())["act1"] == sigmoid

    dense2 = nn.Linear(1, 5)
    replace_submodule(model, "seq1.dense2", dense2)
    assert dict(model.named_modules())["seq1.dense2"] == dense2


# Unit tests for get_adapter_absolute_path
@patch('os.path.isabs')
def test_get_adapter_absolute_path_absolute(mock_isabs):
    path = '/absolute/path/to/lora'
    mock_isabs.return_value = True
    assert get_adapter_absolute_path(path) == path


@patch('os.path.expanduser')
def test_get_adapter_absolute_path_expanduser(mock_expanduser):
    # Path with ~ that needs to be expanded
    path = '~/relative/path/to/lora'
    absolute_path = '/home/user/relative/path/to/lora'
    mock_expanduser.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch('os.path.exists')
@patch('os.path.abspath')
def test_get_adapter_absolute_path_local_existing(mock_abspath, mock_exist):
    # Relative path that exists locally
    path = 'relative/path/to/lora'
    absolute_path = '/absolute/path/to/lora'
    mock_exist.return_value = True
    mock_abspath.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch('huggingface_hub.snapshot_download')
@patch('os.path.exists')
def test_get_adapter_absolute_path_huggingface(mock_exist,
                                               mock_snapshot_download):
    # Hugging Face model identifier
    path = 'org/repo'
    absolute_path = '/mock/snapshot/path'
    mock_exist.return_value = False
    mock_snapshot_download.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch('huggingface_hub.snapshot_download')
@patch('os.path.exists')
def test_get_adapter_absolute_path_huggingface_error(mock_exist,
                                                     mock_snapshot_download):
    # Hugging Face model identifier with download error
    path = 'org/repo'
    mock_exist.return_value = False
    mock_snapshot_download.side_effect = HfHubHTTPError(
        "failed to query model info")
    assert get_adapter_absolute_path(path) == path
