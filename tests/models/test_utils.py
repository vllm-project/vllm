# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


class MockKVCacheMethod(BaseKVCacheMethod):
    """Mock KV cache method for testing remapping logic."""

    def __init__(self, remap_rules: dict[str, str] | None = None):
        self.remap_rules = remap_rules or {}

    def remap_kv_scale_name(self, name: str, params_dict: dict) -> str | None:
        for pattern, replacement in self.remap_rules.items():
            if pattern in name:
                new_name = name.replace(pattern, replacement)
                return new_name if new_name in params_dict else None
        return None


class ModuleWithKVCacheMethod(torch.nn.Module):
    """Module with a mock KV cache method."""

    def __init__(self, remap_rules: dict[str, str] | None = None):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.attn_k_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.attn_v_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.quant_method = MockKVCacheMethod(remap_rules)


@pytest.mark.cpu_test
def test_auto_weights_loader_remaps_kv_scale_names():
    """Ensure AutoWeightsLoader applies KV scale remapping from quant method."""
    remap_rules = {
        "k_proj.k_scale": "attn_k_scale",
        "v_proj.v_scale": "attn_v_scale",
    }
    mod = ModuleWithKVCacheMethod(remap_rules)

    def weight_generator():
        yield "k_proj.k_scale", torch.tensor(2.0)
        yield "v_proj.v_scale", torch.tensor(3.0)
        yield "linear.weight", torch.randn(10, 10)

    loader = AutoWeightsLoader(mod)
    loaded = loader.load_weights(weight_generator())

    assert "k_proj.k_scale" not in loaded
    assert "v_proj.v_scale" not in loaded
    assert "attn_k_scale" in loaded
    assert "attn_v_scale" in loaded
    assert torch.allclose(mod.attn_k_scale.data, torch.tensor(2.0))
    assert torch.allclose(mod.attn_v_scale.data, torch.tensor(3.0))


@pytest.mark.cpu_test
def test_auto_weights_loader_skips_remap_when_target_missing():
    """Ensure remapped weights are skipped when target param is missing."""
    remap_rules = {
        "k_proj.k_scale": "nonexistent_scale",
    }
    mod = ModuleWithKVCacheMethod(remap_rules)

    def weight_generator():
        yield "k_proj.k_scale", torch.tensor(2.0)

    loader = AutoWeightsLoader(mod)
    loaded = loader.load_weights(weight_generator())

    assert "k_proj.k_scale" not in loaded
    assert "nonexistent_scale" not in loaded


@pytest.mark.cpu_test
def test_auto_weights_loader_no_remap_without_kv_method():
    """Ensure weights pass through unchanged when no KV method is present."""

    class ModuleWithoutKVMethod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.some_scale = torch.nn.Parameter(torch.tensor(1.0))

    mod = ModuleWithoutKVMethod()

    def weight_generator():
        yield "some_scale", torch.tensor(2.0)
        yield "linear.weight", torch.randn(10, 10)

    loader = AutoWeightsLoader(mod)
    loaded = loader.load_weights(weight_generator())

    assert "some_scale" in loaded
    assert torch.allclose(mod.some_scale.data, torch.tensor(2.0))


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
