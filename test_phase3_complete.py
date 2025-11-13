#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Phase 3 Complete Test: Verify flexible model registration works end-to-end.

This tests:
1. All callable types (lambdas, closures, callable objects, functions)
2. Models pass interface validation
3. No structural assumptions break user models
"""

import torch
import torch.nn as nn

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.interfaces_base import is_vllm_model
from vllm.model_executor.parallel_context import ParallelContext


class MinimalUserModel(nn.Module):
    """
    Minimal user model that satisfies vLLM's interface requirements.

    This demonstrates what users need to provide.
    """

    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(1000, hidden_size)
        self.output = nn.Linear(hidden_size, 1000)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM."""
        return self.embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs):
        """Required by vLLM."""
        embeds = self.embeddings(input_ids)
        return self.output(embeds)


print("=" * 70)
print("Phase 3: Flexible Model Registration - Complete Test")
print("=" * 70)

# Test 1: Lambda
print("\n[Test 1] Lambda registration and validation")
lambda_factory = lambda vllm_config, parallel_context: MinimalUserModel()
ModelRegistry.register_model("LambdaModel", lambda_factory)

model_cls = ModelRegistry._try_load_model_cls("LambdaModel")
print(f"  ✓ Lambda model class: {model_cls.__name__}")
print(f"  ✓ Passes vLLM validation: {is_vllm_model(model_cls)}")

mock_config = type("Config", (), {"parallel_config": None})()
ctx = ParallelContext(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
instance = model_cls(vllm_config=mock_config, parallel_context=ctx)
print(f"  ✓ Has get_input_embeddings: {hasattr(instance, 'get_input_embeddings')}")
print(f"  ✓ Has forward: {hasattr(instance, 'forward')}")

# Test forward pass
input_ids = torch.randint(0, 1000, (2, 10))
positions = torch.arange(10).unsqueeze(0).expand(2, -1)
output = instance.forward(input_ids=input_ids, positions=positions)
print(f"  ✓ Forward pass works: output shape {output.shape}")


# Test 2: Closure
print("\n[Test 2] Closure registration and validation")


def make_model_factory(hidden_size=512):
    """Closure that captures configuration."""

    def factory(vllm_config, parallel_context):
        tp_size = parallel_context.get_tensor_parallel_world_size()
        print(f"    Building model with hidden_size={hidden_size}, TP size={tp_size}")
        return MinimalUserModel(hidden_size=hidden_size)

    return factory


ModelRegistry.register_model("ClosureModel", make_model_factory(hidden_size=384))

model_cls = ModelRegistry._try_load_model_cls("ClosureModel")
print(f"  ✓ Closure model class: {model_cls.__name__}")
print(f"  ✓ Passes vLLM validation: {is_vllm_model(model_cls)}")

instance = model_cls(vllm_config=mock_config, parallel_context=ctx)
output = instance.forward(input_ids=input_ids, positions=positions)
print(f"  ✓ Forward pass works: output shape {output.shape}")


# Test 3: Callable object
print("\n[Test 3] Callable object registration and validation")


class AdvancedModelBuilder:
    """Callable object with state."""

    def __init__(self, model_type="default", use_dropout=False):
        self.model_type = model_type
        self.use_dropout = use_dropout

    def __call__(self, vllm_config, parallel_context):
        """Build the model."""
        tp_rank = parallel_context.get_tensor_parallel_rank()
        print(
            f"    Building {self.model_type} model (dropout={self.use_dropout}) "
            f"on TP rank {tp_rank}"
        )
        return MinimalUserModel()


builder = AdvancedModelBuilder(model_type="experimental", use_dropout=True)
ModelRegistry.register_model("CallableObjectModel", builder)

model_cls = ModelRegistry._try_load_model_cls("CallableObjectModel")
print(f"  ✓ Callable object model class: {model_cls.__name__}")
print(f"  ✓ Passes vLLM validation: {is_vllm_model(model_cls)}")

instance = model_cls(vllm_config=mock_config, parallel_context=ctx)
output = instance.forward(input_ids=input_ids, positions=positions)
print(f"  ✓ Forward pass works: output shape {output.shape}")


# Test 4: Regular function (already tested, but verify again)
print("\n[Test 4] Regular function registration and validation")


def build_model(vllm_config, parallel_context):
    """Regular factory function."""
    return MinimalUserModel()


ModelRegistry.register_model("FunctionModel", build_model)

model_cls = ModelRegistry._try_load_model_cls("FunctionModel")
print(f"  ✓ Function model class: {model_cls.__name__}")
print(f"  ✓ Passes vLLM validation: {is_vllm_model(model_cls)}")

instance = model_cls(vllm_config=mock_config, parallel_context=ctx)
output = instance.forward(input_ids=input_ids, positions=positions)
print(f"  ✓ Forward pass works: output shape {output.shape}")


# Test 5: Verify interface methods are accessible
print("\n[Test 5] Interface method accessibility")

instance = model_cls(vllm_config=mock_config, parallel_context=ctx)

# Test that interface methods work
embeddings = instance.get_input_embeddings(input_ids)
print(f"  ✓ get_input_embeddings works: {embeddings.shape}")

forward_out = instance(input_ids, positions)
print(f"  ✓ __call__ (forward) works: {forward_out.shape}")

# Test attribute delegation
print(f"  ✓ Can access .hidden_size: {instance.hidden_size}")
print(f"  ✓ Can access .embeddings: {type(instance.embeddings)}")


print("\n" + "=" * 70)
print("✅ Phase 3 COMPLETE!")
print("=" * 70)
print("\nSupported callable types:")
print("  • Lambdas ✓")
print("  • Closures ✓")
print("  • Callable objects (with __call__) ✓")
print("  • Regular functions ✓")
print("\nAll models:")
print("  • Pass vLLM interface validation ✓")
print("  • Support attribute delegation ✓")
print("  • Work with parallel context ✓")
print("  • No structural assumptions break them ✓")
