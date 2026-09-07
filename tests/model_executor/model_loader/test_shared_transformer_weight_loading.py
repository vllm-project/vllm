# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.sharded_state_loader import (
    ShardedStateLoader,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class MockAttention(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)


class MockSharedTransformer(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)


class MockHybridLayer(nn.Module):
    def __init__(self, shared_transformer: MockSharedTransformer):
        super().__init__()
        self.shared_transformer = shared_transformer


class MockHybridModel(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        # Shared transformer block tied across hybrid layers (e.g. 1 and 11)
        self.block0 = MockSharedTransformer(hidden_size)
        self.layers = nn.ModuleList(
            [
                MockHybridLayer(self.block0),  # layer 0
                MockHybridLayer(self.block0),  # layer 1 (tied)
            ]
        )


def test_weights_mapper_resolve_shared_param_name():
    """WeightsMapper.resolve_shared_param_name should resolve tied layer keys."""
    params_dict = {
        "layers.1.shared_transformer.self_attn.qkv_proj.weight": torch.empty(4),
        "model.layers.1.shared_transformer.self_attn.qkv_proj.weight": torch.empty(4),
    }

    # Direct match
    direct = "layers.1.shared_transformer.self_attn.qkv_proj.weight"
    assert WeightsMapper.resolve_shared_param_name(direct, params_dict) == direct

    # Resolving higher layer (e.g. layer 11) to canonical layer 1
    unresolved = "layers.11.shared_transformer.self_attn.qkv_proj.weight"
    resolved = WeightsMapper.resolve_shared_param_name(unresolved, params_dict)
    assert resolved == "layers.1.shared_transformer.self_attn.qkv_proj.weight"

    # Resolving with prefix (model.layers.11...)
    unresolved_prefixed = "model.layers.11.shared_transformer.self_attn.qkv_proj.weight"
    resolved_prefixed = WeightsMapper.resolve_shared_param_name(
        unresolved_prefixed, params_dict
    )
    assert resolved_prefixed == (
        "model.layers.1.shared_transformer.self_attn.qkv_proj.weight"
    )

    # Unrelated name returns None
    assert (
        WeightsMapper.resolve_shared_param_name("layers.11.mlp.weight", params_dict)
        is None
    )


def test_weights_mapper_resolve_param_name():
    """WeightsMapper.resolve_param_name should combine remapping and shared
    resolution.
    """
    mapper = WeightsMapper(
        orig_to_new_stacked={
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
        }
    )
    params_dict = {
        "layers.1.shared_transformer.self_attn.qkv_proj.weight": torch.empty(4),
    }

    # Remaps .self_attn.q_proj to .self_attn.qkv_proj and resolves layer 11 -> layer 1
    input_name = "layers.11.shared_transformer.self_attn.q_proj.weight"
    resolved = mapper.resolve_param_name(input_name, params_dict)
    assert resolved == "layers.1.shared_transformer.self_attn.qkv_proj.weight"


def test_get_subtensor_aliases_shared_transformer():
    """_get_subtensor_aliases should detect tied shared_transformer parameters."""
    shared_param = torch.nn.Parameter(torch.ones(4, 4))
    state_dict = {
        "layers.1.shared_transformer.self_attn.qkv_proj.weight": shared_param,
        "layers.11.shared_transformer.self_attn.qkv_proj.weight": shared_param,
    }

    filtered = ShardedStateLoader._filter_subtensors(state_dict)
    assert "layers.1.shared_transformer.self_attn.qkv_proj.weight" in filtered
    assert "layers.11.shared_transformer.self_attn.qkv_proj.weight" not in filtered

    aliases = ShardedStateLoader._get_subtensor_aliases(state_dict)
    assert (
        aliases.get("layers.11.shared_transformer.self_attn.qkv_proj.weight")
        == "layers.1.shared_transformer.self_attn.qkv_proj.weight"
    )


@pytest.mark.parametrize("rank", [0, 8])
def test_mock_multi_rank_shared_transformer_weight_loading(rank: int):
    """Test loading on multi-node / TP > 8 ranks (e.g. rank 8 on node 2).

    Rank 8 receives a checkpoint containing
    'layers.1.shared_transformer.self_attn.qkv_proj.weight' or
    'layers.11.shared_transformer.self_attn.qkv_proj.weight' without KeyError.
    """
    model = MockHybridModel(hidden_size=4)
    # Name layers as layer 1 and layer 11 to simulate hybrid layout
    named_dict = {
        "layers.1.shared_transformer.self_attn.qkv_proj.weight": (
            model.block0.self_attn.qkv_proj.weight
        ),
        "layers.11.shared_transformer.self_attn.qkv_proj.weight": (
            model.block0.self_attn.qkv_proj.weight
        ),
    }

    class MockModelWrapper(nn.Module):
        def __init__(self, inner_dict):
            super().__init__()
            self._inner_dict = inner_dict

        def state_dict(self, *args, **kwargs):
            return dict(self._inner_dict)

    wrapped_model = MockModelWrapper(named_dict)
    loader = ShardedStateLoader(
        SimpleNamespace(
            load_format="sharded_state",
            model_loader_extra_config=None,
        )
    )

    # Weight tensor to load
    test_tensor = torch.full((12, 4), float(rank + 1))

    # On rank 8, the file contains layer 11's weight
    # On rank 0, the file contains layer 1's weight
    file_key = (
        "layers.11.shared_transformer.self_attn.qkv_proj.weight"
        if rank == 8
        else "layers.1.shared_transformer.self_attn.qkv_proj.weight"
    )

    def mock_iterate_files(paths) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield file_key, test_tensor

    model_config = SimpleNamespace(model="/mock/path", model_weights=None)

    with (
        patch.object(loader, "iterate_over_files", side_effect=mock_iterate_files),
        patch("glob.glob", return_value=["/mock/path/model-rank-8-part-0.safetensors"]),
        patch(
            "vllm.distributed.get_tensor_model_parallel_rank",
            return_value=rank,
        ),
    ):
        # Must not raise KeyError for
        # 'layers.11.shared_transformer.self_attn.qkv_proj.weight'
        loader.load_weights(wrapped_model, model_config)

    # Verify that the parameter was updated with the loaded tensor
    assert torch.equal(model.block0.self_attn.qkv_proj.weight, test_tensor)


def test_auto_weights_loader_shared_transformer():
    """AutoWeightsLoader should safely resolve shared_transformer weights."""
    model = MockHybridModel(hidden_size=4)

    test_weight = torch.full((12, 4), 3.14)
    # Checkpoint contains only layer 11 (the alias key)
    weights = [("layers.1.shared_transformer.self_attn.qkv_proj.weight", test_weight)]
    loader = AutoWeightsLoader(model)
    loaded = loader.load_weights(weights)
    assert "layers.1.shared_transformer.self_attn.qkv_proj.weight" in loaded
    assert torch.equal(model.block0.self_attn.qkv_proj.weight, test_weight)
