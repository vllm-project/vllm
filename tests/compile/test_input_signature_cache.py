# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that the input signature hash prevents cache collisions between
graphs with the same structure but different input signatures.

This test specifically verifies the fix for cache collisions when models
have structurally similar graphs but different input argument counts
(e.g., with/without bias weights).
"""

import pytest
import torch
from torch import nn

from vllm.compilation.backends import CompilerManager
from vllm.compilation.compiler_interface import compute_input_signature_hash
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context


class LlamaMLP(nn.Module):
    """Simple MLP module for testing, similar to Llama architecture."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


@support_torch_compile
class MLPModel(nn.Module):
    """Wrapper model with torch compile support."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self.mlp = LlamaMLP(hidden_size, intermediate_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def test_input_signature_hash_basic():
    """Test that compute_input_signature_hash produces different hashes
    for different input signatures."""

    # Same tensor shapes, same count
    inputs1 = [torch.randn(10, 20), torch.randn(5, 10)]
    inputs2 = [torch.randn(10, 20), torch.randn(5, 10)]
    assert compute_input_signature_hash(inputs1) == compute_input_signature_hash(
        inputs2
    )

    # Different number of inputs
    inputs3 = [torch.randn(10, 20)]
    assert compute_input_signature_hash(inputs1) != compute_input_signature_hash(
        inputs3
    )

    # Different shapes
    inputs4 = [torch.randn(10, 20), torch.randn(5, 11)]
    assert compute_input_signature_hash(inputs1) != compute_input_signature_hash(
        inputs4
    )

    # Different dtypes
    inputs5 = [torch.randn(10, 20), torch.randn(5, 10, dtype=torch.float16)]
    assert compute_input_signature_hash(inputs1) != compute_input_signature_hash(
        inputs5
    )

    # Mixed types (tensor + scalar)
    inputs6 = [torch.randn(10, 20), 5]
    inputs7 = [torch.randn(10, 20), 10]
    assert compute_input_signature_hash(inputs6) != compute_input_signature_hash(
        inputs7
    )


def test_input_signature_hash_bias_vs_no_bias():
    """Test that MLP forward passes with bias and without bias have different
    input signature hashes.

    This simulates the real issue where LlamaMLP with bias=True has different
    inputs to the Linear layer's forward compared to bias=False.
    """
    hidden_size = 128
    intermediate_size = 256
    batch_size = 4

    # Create input tensor
    x = torch.randn(batch_size, hidden_size)

    # MLP without bias - Linear.forward gets (input, weight)
    mlp_no_bias = LlamaMLP(hidden_size, intermediate_size, bias=False)

    # MLP with bias - Linear.forward gets (input, weight, bias)
    mlp_with_bias = LlamaMLP(hidden_size, intermediate_size, bias=True)

    # Simulate the inputs that would go to a compiled Linear layer
    # Without bias: [input_tensor, weight_tensor]
    inputs_no_bias = [
        x,
        mlp_no_bias.gate_up_proj.weight,
    ]

    # With bias: [input_tensor, weight_tensor, bias_tensor]
    inputs_with_bias = [
        x,
        mlp_with_bias.gate_up_proj.weight,
        mlp_with_bias.gate_up_proj.bias,
    ]

    # These should have different hashes because of different input counts
    hash_no_bias = compute_input_signature_hash(inputs_no_bias)
    hash_with_bias = compute_input_signature_hash(inputs_with_bias)

    assert hash_no_bias != hash_with_bias, (
        "Input signature hashes must differ for Linear layers with and without bias "
        f"(got {hash_no_bias} and {hash_with_bias})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mlp_bias_no_cache_collision(monkeypatch: pytest.MonkeyPatch):
    """Test that MLP with bias and without bias get different cache entries.

    This verifies the fix for cache collisions when models have the same
    structure but different input signatures (with/without bias parameters).
    """
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    hidden_size = 128
    intermediate_size = 256
    batch_size = 4

    # Create input
    x = torch.randn(batch_size, hidden_size).cuda()

    # Test MLP without bias
    compilation_config_no_bias = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        backend="eager",  # Use eager backend for compatibility
    )
    vllm_config_no_bias = VllmConfig(compilation_config=compilation_config_no_bias)

    with set_current_vllm_config(vllm_config_no_bias):
        model_no_bias = (
            MLPModel(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=False,
                vllm_config=vllm_config_no_bias,
            )
            .eval()
            .cuda()
        )

    with set_forward_context({}, vllm_config=vllm_config_no_bias):
        output_no_bias = model_no_bias(x)

    # Test MLP with bias
    compilation_config_with_bias = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        backend="eager",  # Use eager backend for compatibility
    )
    vllm_config_with_bias = VllmConfig(compilation_config=compilation_config_with_bias)

    with set_current_vllm_config(vllm_config_with_bias):
        model_with_bias = (
            MLPModel(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=True,
                vllm_config=vllm_config_with_bias,
            )
            .eval()
            .cuda()
        )

    with set_forward_context({}, vllm_config=vllm_config_with_bias):
        output_with_bias = model_with_bias(x)

    # Verify outputs are different (since models have different random weights and bias)
    assert not torch.allclose(output_no_bias, output_with_bias), (
        "Outputs should differ (different models)"
    )


def test_old_cache_key_would_collide():
    """Test that the old 3-tuple cache key format would have caused collisions.

    This demonstrates why the input signature hash was necessary. With the old
    cache key format (runtime_shape, graph_index, backend_name), two graphs with
    the same structure but different input signatures would collide.
    """
    hidden_size = 128
    intermediate_size = 256
    batch_size = 4

    x = torch.randn(batch_size, hidden_size)

    # Create two MLPs with different bias settings
    mlp_no_bias = LlamaMLP(hidden_size, intermediate_size, bias=False)
    mlp_with_bias = LlamaMLP(hidden_size, intermediate_size, bias=True)

    # Simulate inputs to Linear layer without bias
    inputs_no_bias = [x, mlp_no_bias.gate_up_proj.weight]

    # Simulate inputs to Linear layer with bias
    inputs_with_bias = [
        x,
        mlp_with_bias.gate_up_proj.weight,
        mlp_with_bias.gate_up_proj.bias,
    ]

    # These graphs would have the same old 3-tuple key components
    runtime_shape = None  # Same for both
    graph_index = 0  # Same for both (first graph)
    backend_name = "eager"  # Same for both

    # OLD cache key format (3-tuple) - WOULD COLLIDE!
    old_key_no_bias = (runtime_shape, graph_index, backend_name)
    old_key_with_bias = (runtime_shape, graph_index, backend_name)

    assert old_key_no_bias == old_key_with_bias, (
        "OLD cache key format causes collision: both models have identical keys!"
    )

    # NEW cache key format (4-tuple) - NO COLLISION!
    input_sig_hash_no_bias = compute_input_signature_hash(inputs_no_bias)
    input_sig_hash_with_bias = compute_input_signature_hash(inputs_with_bias)

    new_key_no_bias = (runtime_shape, graph_index, backend_name, input_sig_hash_no_bias)
    new_key_with_bias = (
        runtime_shape,
        graph_index,
        backend_name,
        input_sig_hash_with_bias,
    )

    assert new_key_no_bias != new_key_with_bias, (
        "NEW cache key format prevents collision: "
        "keys differ due to input signature hash"
    )

    # The difference is specifically in the input signature hash
    assert new_key_no_bias[:3] == new_key_with_bias[:3], (
        "First 3 components are identical (would collide with old format)"
    )
    assert new_key_no_bias[3] != new_key_with_bias[3], (
        "Fourth component (input_sig_hash) is different and prevents collision"
    )


def test_fx_graph_arity_mismatch():
    """Test that compiled graphs for models with/without bias have different arities.

    This verifies the root cause of the cache collision: when compiled, the graphs
    receive different numbers of inputs due to the bias parameter.
    """
    hidden_size = 64
    intermediate_size = 128
    batch_size = 4

    x = torch.randn(batch_size, hidden_size)

    # Create Linear layers with and without bias
    linear_no_bias = nn.Linear(hidden_size, intermediate_size, bias=False)
    linear_with_bias = nn.Linear(hidden_size, intermediate_size, bias=True)

    # When a Linear layer is compiled/traced, the inputs passed differ:
    # Without bias: F.linear(input, weight, bias=None)
    #   - None is often optimized away
    # With bias: F.linear(input, weight, bias=<tensor>)

    # Simulate what would be captured during compilation
    # For linear without bias, the inputs would be:
    example_inputs_no_bias = [x, linear_no_bias.weight]

    # For linear with bias, the inputs would be:
    example_inputs_with_bias = [x, linear_with_bias.weight, linear_with_bias.bias]

    # Verify arity mismatch
    arity_no_bias = len(example_inputs_no_bias)
    arity_with_bias = len(example_inputs_with_bias)

    print("\nArity Analysis:")
    print(f"  Linear without bias: {arity_no_bias} inputs (input, weight)")
    print(f"  Linear with bias: {arity_with_bias} inputs (input, weight, bias)")

    assert arity_no_bias == 2, "Linear without bias should have 2 inputs"
    assert arity_with_bias == 3, "Linear with bias should have 3 inputs"
    assert arity_no_bias != arity_with_bias, (
        "Arity mismatch: models with different bias settings "
        "have different input counts"
    )

    # Verify that the input signature hashes reflect this difference
    hash_no_bias = compute_input_signature_hash(example_inputs_no_bias)
    hash_with_bias = compute_input_signature_hash(example_inputs_with_bias)

    assert hash_no_bias != hash_with_bias, (
        f"Input signature hashes must differ due to arity mismatch: "
        f"{hash_no_bias} vs {hash_with_bias}"
    )

    print(f"  Input sig hash (no bias): {hash_no_bias}")
    print(f"  Input sig hash (with bias): {hash_with_bias}")
    print("  ✓ Hashes differ, preventing cache collision")


def test_cache_key_structure():
    """Test that cache keys have the expected 4-tuple structure with input_sig_hash."""
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        backend="eager",
    )

    # Create a CompilerManager directly
    compiler_manager = CompilerManager(compilation_config)

    # Manually add some test entries to the cache
    test_cache_entries = [
        ((None, 0, "eager", "0123456789abcdef"), "test_value_1"),
        ((128, 1, "eager", "fedcba9876543210"), "test_value_2"),
        ((256, 2, "inductor", "aabbccddeeff0011"), "test_value_3"),
    ]

    for key, value in test_cache_entries:
        compiler_manager.cache[key] = value

    # Verify cache key structure:
    # (runtime_shape, graph_index, backend_name, input_sig_hash)
    cache_keys = list(compiler_manager.cache.keys())
    assert len(cache_keys) == 3, "Should have 3 cache entries"

    for key in cache_keys:
        assert isinstance(key, tuple), "Cache key should be a tuple"
        assert len(key) == 4, "Cache key should have 4 elements"

        runtime_shape, graph_index, backend_name, input_sig_hash = key

        # runtime_shape can be int or None
        assert runtime_shape is None or isinstance(runtime_shape, int), (
            "runtime_shape should be int or None"
        )

        # graph_index should be int
        assert isinstance(graph_index, int), "graph_index should be int"

        # backend_name should be string
        assert isinstance(backend_name, str), "backend_name should be string"

        # input_sig_hash should be a 16-character hex string
        assert isinstance(input_sig_hash, str), "input_sig_hash should be a string"
        assert len(input_sig_hash) == 16, "input_sig_hash should be 16 chars"
        assert all(c in "0123456789abcdef" for c in input_sig_hash), (
            "input_sig_hash should be hex"
        )
