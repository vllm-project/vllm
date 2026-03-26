# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the apply_steering custom op implementation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.steering import apply_steering

# Shorthand for the module whose ``_forward_context`` global we mock.
FC_MODULE = "vllm.forward_context"


def _make_forward_context(num_decode_tokens: int) -> ForwardContext:
    """Build a minimal mock ForwardContext with the given decode count."""
    attn_meta = SimpleNamespace(num_decode_tokens=num_decode_tokens)
    return SimpleNamespace(
        attn_metadata={"layer.0": attn_meta},
    )


class TestApplySteering:
    """Tests call the real Python function directly (not via torch.ops)."""

    def test_steers_only_decode_tokens(self):
        """Only the first num_decode_tokens rows are modified."""
        batch_size, hidden_size = 6, 4
        hidden = torch.zeros(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size)

        ctx = _make_forward_context(num_decode_tokens=3)
        with patch(f"{FC_MODULE}._forward_context", ctx):
            result = apply_steering(hidden, steering_vec)

        # First 3 rows should have steering added
        expected = torch.zeros(batch_size, hidden_size)
        expected[:3] = 1.0
        assert torch.allclose(result, expected), (
            f"Expected first 3 rows steered.\nGot:\n{result}"
        )

    def test_no_context_steers_nothing(self):
        """When forward context is None, no tokens are steered (default=0)."""
        batch_size, hidden_size = 4, 4
        hidden = torch.ones(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size) * 5.0

        with patch(f"{FC_MODULE}._forward_context", None):
            result = apply_steering(hidden, steering_vec)

        assert torch.allclose(result, hidden), (
            "With no forward context, output should equal input."
        )

    def test_zero_decode_tokens(self):
        """num_decode_tokens=0 means no rows are steered."""
        batch_size, hidden_size = 4, 4
        hidden = torch.ones(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size) * 10.0

        ctx = _make_forward_context(num_decode_tokens=0)
        with patch(f"{FC_MODULE}._forward_context", ctx):
            result = apply_steering(hidden, steering_vec)

        assert torch.allclose(result, hidden), (
            "With zero decode tokens, output should equal input."
        )

    def test_all_decode_tokens(self):
        """When num_decode_tokens == batch_size, all rows are steered."""
        batch_size, hidden_size = 4, 4
        hidden = torch.zeros(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size) * 2.0

        ctx = _make_forward_context(num_decode_tokens=batch_size)
        with patch(f"{FC_MODULE}._forward_context", ctx):
            result = apply_steering(hidden, steering_vec)

        expected = torch.ones(batch_size, hidden_size) * 2.0
        assert torch.allclose(result, expected), (
            "All rows should be steered when num_decode_tokens == batch_size."
        )

    def test_zero_vector_noop(self):
        """A zero steering vector produces no change regardless of mask."""
        batch_size, hidden_size = 4, 4
        hidden = torch.randn(batch_size, hidden_size)
        steering_vec = torch.zeros(1, hidden_size)

        ctx = _make_forward_context(num_decode_tokens=batch_size)
        with patch(f"{FC_MODULE}._forward_context", ctx):
            result = apply_steering(hidden, steering_vec)

        assert torch.allclose(result, hidden), (
            "Zero steering vector should produce no change."
        )

    def test_output_shape_and_dtype(self):
        """Output shape and dtype must match input."""
        batch_size, hidden_size = 6, 8
        hidden = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        steering_vec = torch.ones(1, hidden_size, dtype=torch.float32)

        ctx = _make_forward_context(num_decode_tokens=3)
        with patch(f"{FC_MODULE}._forward_context", ctx):
            result = apply_steering(hidden, steering_vec)

        assert result.shape == hidden.shape, (
            f"Shape mismatch: {result.shape} != {hidden.shape}"
        )
        assert result.dtype == hidden.dtype, (
            f"Dtype mismatch: {result.dtype} != {hidden.dtype}"
        )
