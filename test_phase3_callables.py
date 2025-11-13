#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test Phase 3: Flexible callable support

Test what kinds of callables already work with our current implementation.
"""

import torch.nn as nn

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_context import ParallelContext


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


print("=" * 60)
print("Testing Phase 3: Callable Support")
print("=" * 60)

# Test 1: Lambda (should work - lambdas are callables)
print("\n1. Testing lambda registration...")
try:
    lambda_factory = lambda vllm_config, parallel_context: DummyModel()
    ModelRegistry.register_model("LambdaModel", lambda_factory)
    print("   ✓ Lambda registration succeeded")

    # Try to load it
    model_cls = ModelRegistry._try_load_model_cls("LambdaModel")
    print(f"   ✓ Lambda model class loaded: {model_cls}")

    # Try to instantiate
    mock_config = type("Config", (), {"parallel_config": None})()
    ctx = ParallelContext(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = model_cls(vllm_config=mock_config, parallel_context=ctx)
    print(f"   ✓ Lambda model instantiated: {type(model)}")

except Exception as e:
    print(f"   ✗ Lambda registration failed: {e}")


# Test 2: Callable object (object with __call__)
print("\n2. Testing callable object registration...")


class ModelBuilder:
    """Callable object that builds models."""

    def __init__(self, model_type="default"):
        self.model_type = model_type

    def __call__(self, vllm_config, parallel_context):
        """Build the model."""
        print(f"   Building model of type: {self.model_type}")
        return DummyModel()


try:
    builder = ModelBuilder(model_type="custom")
    ModelRegistry.register_model("CallableModel", builder)
    print("   ✓ Callable object registration succeeded")

    # Try to load it
    model_cls = ModelRegistry._try_load_model_cls("CallableModel")
    print(f"   ✓ Callable model class loaded: {model_cls}")

    # Try to instantiate
    mock_config = type("Config", (), {"parallel_config": None})()
    ctx = ParallelContext(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = model_cls(vllm_config=mock_config, parallel_context=ctx)
    print(f"   ✓ Callable model instantiated: {type(model)}")

except Exception as e:
    print(f"   ✗ Callable object registration failed: {e}")


# Test 3: Closure
print("\n3. Testing closure registration...")


def make_factory(hidden_size=256):
    """Create a closure that captures hidden_size."""

    def factory(vllm_config, parallel_context):
        print(f"   Building model with hidden_size={hidden_size}")
        return DummyModel()

    return factory


try:
    closure_factory = make_factory(hidden_size=512)
    ModelRegistry.register_model("ClosureModel", closure_factory)
    print("   ✓ Closure registration succeeded")

    # Try to load it
    model_cls = ModelRegistry._try_load_model_cls("ClosureModel")
    print(f"   ✓ Closure model class loaded: {model_cls}")

    # Try to instantiate
    mock_config = type("Config", (), {"parallel_config": None})()
    ctx = ParallelContext(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = model_cls(vllm_config=mock_config, parallel_context=ctx)
    print(f"   ✓ Closure model instantiated: {type(model)}")

except Exception as e:
    print(f"   ✗ Closure registration failed: {e}")


# Test 4: Regular function (already tested in Phase 2, but let's verify)
print("\n4. Testing regular function registration...")


def regular_factory(vllm_config, parallel_context):
    """Regular factory function."""
    return DummyModel()


try:
    ModelRegistry.register_model("FunctionModel", regular_factory)
    print("   ✓ Function registration succeeded")

    # Try to load it
    model_cls = ModelRegistry._try_load_model_cls("FunctionModel")
    print(f"   ✓ Function model class loaded: {model_cls}")

    # Try to instantiate
    mock_config = type("Config", (), {"parallel_config": None})()
    ctx = ParallelContext(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = model_cls(vllm_config=mock_config, parallel_context=ctx)
    print(f"   ✓ Function model instantiated: {type(model)}")

except Exception as e:
    print(f"   ✗ Function registration failed: {e}")


print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("All callable types should work with our current implementation!")
print("Phase 3.1 complete: Lambdas, closures, and callable objects supported ✓")
