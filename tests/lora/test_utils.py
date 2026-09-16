# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
import safetensors.torch
import torch
from huggingface_hub.utils import HfHubHTTPError
from torch import nn

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import (
    get_adapter_absolute_path,
    parse_fine_tuned_lora_name,
    parse_trainable_tokens_name,
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


@patch("huggingface_hub.HfApi.snapshot_download")
@patch("os.path.exists")
def test_get_adapter_absolute_path_huggingface(mock_exist, mock_snapshot_download):
    # Hugging Face model identifier
    path = "org/repo"
    absolute_path = "/mock/snapshot/path"
    mock_exist.return_value = False
    mock_snapshot_download.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch("huggingface_hub.HfApi.snapshot_download")
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


@pytest.mark.parametrize(
    "config_indices,weights_mapper,expected_module",
    [
        ([3, 1], None, "model.embed_tokens"),
        ({"embed_tokens": [3, 1]}, None, "model.embed_tokens"),
        ({"model.embed_tokens": [3, 1]}, None, "model.embed_tokens"),
        (
            {"model.embed_tokens": [3, 1]},
            WeightsMapper(orig_to_new_prefix={"model.": "language_model.model."}),
            "language_model.model.embed_tokens",
        ),
    ],
)
def test_trainable_tokens_load_absolute_rows(
    config_indices, weights_mapper, expected_module
):
    """Selected rows keep token order and precision independently of LoRA scaling."""
    name = "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"
    rows = torch.tensor([[1.125, -2.25], [3.5, 4.75]], dtype=torch.float32)
    helper = PEFTHelper(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj"],
        trainable_token_indices=config_indices,
        ensure_weight_tying=True,
    )
    adapter = LoRAModel.from_lora_tensors(
        1,
        {name: rows, "base_model.model.model.q_proj.lora_A.weight": torch.ones(8, 2)},
        helper,
        device="cpu",
        dtype=torch.float16,
        model_vocab_size=8,
        weights_mapper=weights_mapper,
    )
    selected = adapter.trainable_tokens[expected_module]
    assert selected.token_indices.tolist() == [3, 1]
    torch.testing.assert_close(selected.weights, rows)
    assert selected.weights.dtype == torch.float32
    assert len(adapter.loras) == 1
    assert next(iter(adapter.loras.values())).lora_a.dtype == torch.float16
    clone = adapter.clone(2)
    assert clone.ensure_weight_tying
    assert clone.trainable_tokens is not adapter.trainable_tokens
    assert clone.trainable_tokens[expected_module].weights is selected.weights


@pytest.mark.parametrize("extension", ["safetensors", "bin"])
def test_trainable_tokens_local_checkpoint(tmp_path, extension):
    """The real checkpoint module check accepts selected-token tensor keys."""
    name = "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"
    rows = torch.tensor([[1.0, -2.0]])
    path = tmp_path / f"adapter_model.{extension}"
    if extension == "safetensors":
        safetensors.torch.save_file({name: rows}, path)
    else:
        torch.save({name: rows}, path)
    helper = PEFTHelper(
        r=8, lora_alpha=32, target_modules=["q_proj"], trainable_token_indices=[2]
    )
    adapter = LoRAModel.from_local_checkpoint(
        str(tmp_path), {"embed_tokens"}, helper, device="cpu", model_vocab_size=4
    )
    torch.testing.assert_close(
        adapter.trainable_tokens["model.embed_tokens"].weights, rows
    )
    with pytest.raises(ValueError, match="expected target modules"):
        LoRAModel.from_local_checkpoint(
            str(tmp_path), {"q_proj"}, helper, device="cpu", model_vocab_size=4
        )


@pytest.mark.parametrize(
    "indices,rows,vocab_size,error",
    [
        (None, torch.ones(1, 2), 8, "requires trainable_token_indices"),
        ({"lm_head": [1]}, torch.ones(1, 2), 8, "exactly one"),
        (
            {"embed_tokens": [1], "model.embed_tokens": [1]},
            torch.ones(1, 2),
            8,
            "exactly one",
        ),
        ([1], torch.ones(2, 2), 8, "floating-point matrix"),
        ([1], torch.ones(2), 8, "floating-point matrix"),
        ([1], torch.ones(1, 2, dtype=torch.int64), 8, "floating-point matrix"),
        ([8], torch.ones(1, 2), 8, "vocabulary size"),
    ],
)
def test_trainable_tokens_reject_invalid_checkpoint(indices, rows, vocab_size, error):
    helper = PEFTHelper(
        r=8, lora_alpha=32, target_modules=["q_proj"], trainable_token_indices=indices
    )
    with pytest.raises(ValueError, match=error):
        LoRAModel.from_lora_tensors(
            1,
            {"model.embed_tokens.token_adapter.trainable_tokens_delta": rows},
            helper,
            device="cpu",
            model_vocab_size=vocab_size,
        )


@pytest.mark.parametrize("token_weights_first", [True, False])
def test_trainable_tokens_reject_embedding_lora_overlap(token_weights_first):
    helper = PEFTHelper(
        r=8, lora_alpha=32, target_modules=["embed_tokens"], trainable_token_indices=[1]
    )
    entries = [
        ("model.embed_tokens.token_adapter.trainable_tokens_delta", torch.ones(1, 2)),
        ("model.embed_tokens.lora_embedding_A", torch.ones(8, 4)),
    ]
    if not token_weights_first:
        entries.reverse()
    with pytest.raises(ValueError, match="cannot target the same module"):
        LoRAModel.from_lora_tensors(1, dict(entries), helper, device="cpu")


def test_parse_trainable_tokens_name_dropped_by_mapper():
    mapper = WeightsMapper(orig_to_new_prefix={"model.": None})
    with pytest.raises(ValueError, match="cannot be None"):
        parse_trainable_tokens_name(
            "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta",
            mapper,
        )


def test_trainable_tokens_declared_without_saved_rows():
    helper = PEFTHelper(
        r=8, lora_alpha=16, target_modules=["q_proj"], trainable_token_indices=[1]
    )
    with pytest.raises(ValueError, match="no trainable token weights"):
        LoRAModel.from_lora_tensors(1, {}, helper, device="cpu")
