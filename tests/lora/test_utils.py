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


def test_can_replace_layer_supports_subclasses():
    """Test that can_replace_layer works with subclasses of layer types.

    This test verifies that LoRA layer replacement works correctly when
    using subclassed layers (e.g., AscendVocabParallelEmbedding that inherits
    from VocabParallelEmbedding). The fix uses isinstance() instead of
    type() is to properly handle inheritance.

    Fixes: https://github.com/vllm-project/vllm/issues/31767
    """
    from unittest.mock import MagicMock

    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import (
        ColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithLoRA,
        MergedQKVParallelLinearWithLoRA,
        QKVParallelLinearWithLoRA,
        ReplicatedLinearWithLoRA,
        RowParallelLinearWithLoRA,
        VocabParallelEmbeddingWithLoRA,
    )
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    lora_config = MagicMock(spec=LoRAConfig)
    lora_config.fully_sharded_loras = False

    # Test VocabParallelEmbedding subclass
    class CustomVocabParallelEmbedding(VocabParallelEmbedding):
        pass

    mock_custom_vocab = MagicMock(spec=CustomVocabParallelEmbedding)
    mock_custom_vocab.__class__ = CustomVocabParallelEmbedding
    assert VocabParallelEmbeddingWithLoRA.can_replace_layer(
        mock_custom_vocab, lora_config, []
    )

    # Test ColumnParallelLinear subclass
    class CustomColumnParallelLinear(ColumnParallelLinear):
        pass

    mock_custom_col = MagicMock(spec=CustomColumnParallelLinear)
    mock_custom_col.__class__ = CustomColumnParallelLinear
    assert ColumnParallelLinearWithLoRA.can_replace_layer(
        mock_custom_col, lora_config, []
    )

    # Test RowParallelLinear subclass
    class CustomRowParallelLinear(RowParallelLinear):
        pass

    mock_custom_row = MagicMock(spec=CustomRowParallelLinear)
    mock_custom_row.__class__ = CustomRowParallelLinear
    assert RowParallelLinearWithLoRA.can_replace_layer(
        mock_custom_row, lora_config, []
    )

    # Test ReplicatedLinear subclass
    class CustomReplicatedLinear(ReplicatedLinear):
        pass

    mock_custom_rep = MagicMock(spec=CustomReplicatedLinear)
    mock_custom_rep.__class__ = CustomReplicatedLinear
    assert ReplicatedLinearWithLoRA.can_replace_layer(
        mock_custom_rep, lora_config, []
    )

    # Test MergedColumnParallelLinear subclass
    class CustomMergedColumnParallelLinear(MergedColumnParallelLinear):
        pass

    mock_custom_merged = MagicMock(spec=CustomMergedColumnParallelLinear)
    mock_custom_merged.__class__ = CustomMergedColumnParallelLinear
    assert MergedColumnParallelLinearWithLoRA.can_replace_layer(
        mock_custom_merged, lora_config, ["a", "b"]
    )

    # Test QKVParallelLinear subclass with 1 packed module
    class CustomQKVParallelLinear(QKVParallelLinear):
        pass

    mock_custom_qkv = MagicMock(spec=CustomQKVParallelLinear)
    mock_custom_qkv.__class__ = CustomQKVParallelLinear
    assert QKVParallelLinearWithLoRA.can_replace_layer(
        mock_custom_qkv, lora_config, ["q"]
    )

    # Test QKVParallelLinear subclass with 3 packed modules
    assert MergedQKVParallelLinearWithLoRA.can_replace_layer(
        mock_custom_qkv, lora_config, ["q", "k", "v"]
    )
