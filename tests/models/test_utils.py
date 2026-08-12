# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
    get_spec_layer_idx_from_weight_name,
    skip_spec_layers,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("weight_name", "expected"),
    [
        ("model.layers.16.input_layernorm.weight", 16),
        ("layers.16.input_layernorm.weight", 16),
        ("model.language_model.layers.16.input_layernorm.weight", 16),
        ("model.language_model.layers.15.input_layernorm.weight", None),
        ("language_model.layers.16.input_layernorm.weight", None),
    ],
)
def test_get_spec_layer_idx_from_weight_name(weight_name, expected):
    config = SimpleNamespace(
        num_hidden_layers=16,
        num_nextn_predict_layers=1,
    )

    assert get_spec_layer_idx_from_weight_name(config, weight_name) == expected


@pytest.mark.cpu_test
def test_get_spec_layer_idx_without_mtp_layers():
    config = SimpleNamespace(num_hidden_layers=16)

    assert (
        get_spec_layer_idx_from_weight_name(
            config, "model.language_model.layers.16.input_layernorm.weight"
        )
        is None
    )


@pytest.mark.cpu_test
def test_skip_spec_layers_for_supported_prefixes():
    config = SimpleNamespace(
        num_hidden_layers=16,
        num_nextn_predict_layers=1,
    )
    weights = [
        ("model.layers.15.input_layernorm.weight", torch.empty(0)),
        ("model.layers.16.input_layernorm.weight", torch.empty(0)),
        ("layers.16.input_layernorm.weight", torch.empty(0)),
        (
            "model.language_model.layers.16.input_layernorm.weight",
            torch.empty(0),
        ),
    ]

    remaining = list(skip_spec_layers(weights, config))

    assert [name for name, _ in remaining] == ["model.layers.15.input_layernorm.weight"]


class ModuleWithBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(x)


class ModuleWithNestedBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nested_mod = ModuleWithBatchNorm()

    def forward(self, x):
        return self.nested_mod(x)


@pytest.mark.cpu_test
def test_module_with_batchnorm_can_load():
    """Ensure the auto weight loader can load batchnorm stats."""
    mod = ModuleWithBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithBatchNorm()

    assert not torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert not torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_with_child_containing_batchnorm_can_autoload():
    """Ensure the auto weight loader can load nested modules batchnorm stats."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_skip_prefix():
    """Ensure the auto weight loader can skip prefix."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        # weights needed to be filtered out
        redundant_weights = {
            "prefix.bn.weight": torch.Tensor([1, 2]),
            "prefix.bn.bias": torch.Tensor([3, 4]),
        }
        yield from (mod.state_dict() | redundant_weights).items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod, skip_prefixes=["prefix."])
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_skip_substr():
    """Ensure the auto weight loader can skip prefix."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        # weights needed to be filtered out
        redundant_weights = {
            "nested_mod.0.substr.weight": torch.Tensor([1, 2]),
            "nested_mod.0.substr.bias": torch.Tensor([3, 4]),
            "nested_mod.substr.weight": torch.Tensor([1, 2]),
            "nested_mod.substr.bias": torch.Tensor([3, 4]),
        }
        yield from (mod.state_dict() | redundant_weights).items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod, skip_substrs=["substr."])
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


class raise_if_cuda_sync:
    def __enter__(self):
        self.previous_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exception_type, exception_value, traceback):
        torch.cuda.set_sync_debug_mode(self.previous_debug_mode)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_merge_multimodal_embeddings_no_sync():
    inputs_embeds = torch.zeros(
        [5, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0"
    )
    multimodal_embeddings = [
        torch.ones([3, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0")
    ]
    is_multimodal = torch.tensor([True, False, True, True, False], device="cpu")
    with raise_if_cuda_sync():
        _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )
