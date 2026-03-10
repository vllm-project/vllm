# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.utils import HfHubHTTPError
from torch import nn

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import (
    get_adapter_absolute_path,
    is_trainable_tokens_delta,
    parse_fine_tuned_lora_name,
    parse_trainable_tokens_delta_name,
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


# ---- Tests for trainable_tokens_delta helpers ----


def test_is_trainable_tokens_delta():
    assert is_trainable_tokens_delta(
        "base_model.model.language_model.model.embed_tokens"
        ".token_adapter.trainable_tokens_delta"
    )
    assert is_trainable_tokens_delta(
        "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"
    )
    # Should not match standard LoRA weights
    assert not is_trainable_tokens_delta(
        "base_model.model.model.embed_tokens.lora_embedding_A"
    )
    assert not is_trainable_tokens_delta(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )


def test_parse_trainable_tokens_delta_name():
    # With base_model.model. prefix
    assert (
        parse_trainable_tokens_delta_name(
            "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"
        )
        == "model.embed_tokens"
    )

    # VLM-style with language_model prefix
    assert (
        parse_trainable_tokens_delta_name(
            "base_model.model.language_model.model.embed_tokens"
            ".token_adapter.trainable_tokens_delta"
        )
        == "language_model.model.embed_tokens"
    )

    # Without base_model.model. prefix
    assert (
        parse_trainable_tokens_delta_name(
            "model.embed_tokens.token_adapter.trainable_tokens_delta"
        )
        == "model.embed_tokens"
    )


def test_parse_trainable_tokens_delta_name_with_weights_mapper():
    mapper = WeightsMapper(orig_to_new_prefix={"model.": "language_model.model."})
    assert (
        parse_trainable_tokens_delta_name(
            "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta",
            weights_mapper=mapper,
        )
        == "language_model.model.embed_tokens"
    )


def test_from_lora_tensors_trainable_tokens_delta():
    """Test that trainable_tokens_delta is correctly converted to
    equivalent LoRA embedding weights."""
    vocab_size = 1000
    embedding_dim = 64
    token_indices = [500, 501, 502]
    num_tokens = len(token_indices)

    # Simulate the delta tensor from a PEFT checkpoint
    delta = torch.randn(num_tokens, embedding_dim)

    tensors = {
        "base_model.model.model.embed_tokens"
        ".token_adapter.trainable_tokens_delta": delta,
    }

    peft_helper = PEFTHelper(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        trainable_token_indices={"embed_tokens": token_indices},
    )

    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=1,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=torch.float32,
        model_vocab_size=vocab_size,
    )

    lora = lora_model.get_lora("model.embed_tokens")
    assert lora is not None
    assert lora.rank == num_tokens
    assert lora.scaling == 1.0

    # lora_a should be sparse one-hot: [N, vocab_size]
    assert lora.lora_a.shape == (num_tokens, vocab_size)
    for i, idx in enumerate(token_indices):
        assert lora.lora_a[i, idx].item() == 1.0
    # All other entries should be zero
    mask = torch.ones(num_tokens, vocab_size, dtype=torch.bool)
    for i, idx in enumerate(token_indices):
        mask[i, idx] = False
    assert (lora.lora_a[mask] == 0).all()

    # lora_b should be delta transposed: [embedding_dim, N]
    assert lora.lora_b.shape == (embedding_dim, num_tokens)
    torch.testing.assert_close(lora.lora_b, delta.T)


def test_from_lora_tensors_trainable_tokens_delta_missing_config():
    """Test error when trainable_token_indices is missing from config."""
    tensors = {
        "base_model.model.model.embed_tokens"
        ".token_adapter.trainable_tokens_delta": torch.randn(3, 64),
    }
    peft_helper = PEFTHelper(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj"],
    )

    with pytest.raises(ValueError, match="trainable_token_indices not set"):
        LoRAModel.from_lora_tensors(
            lora_model_id=1,
            tensors=tensors,
            peft_helper=peft_helper,
            device="cpu",
            model_vocab_size=1000,
        )
