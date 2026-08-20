# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for sparse tensor validation.

Simple, fast unit tests that can run without server fixtures.
Run with: pytest tests/multimodal/test_sparse_tensor_validation_unit.py -v
"""

import io

import pytest
import torch


class TestSparseTensorValidationContextManager:
    """Test that torch.sparse.check_sparse_tensor_invariants() works as expected."""

    def test_valid_sparse_tensor_passes(self):
        """Valid sparse tensors should pass validation."""
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0])
        shape = (2, 2)

        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.sparse_coo_tensor(indices, values, shape)
            dense = tensor.to_dense()

        assert dense.shape == shape

    def test_out_of_bounds_indices_rejected(self):
        """Sparse tensors with out-of-bounds indices should be rejected."""
        indices = torch.tensor([[5], [5]])  # Out of bounds for 2x2
        values = torch.tensor([1.0])
        shape = (2, 2)

        with pytest.raises(RuntimeError) as exc_info:  # noqa: SIM117
            with torch.sparse.check_sparse_tensor_invariants():
                tensor = torch.sparse_coo_tensor(indices, values, shape)
                tensor.to_dense()

        assert (
            "index" in str(exc_info.value).lower()
            or "bound" in str(exc_info.value).lower()
        )

    def test_negative_indices_rejected(self):
        """Sparse tensors with negative indices should be rejected."""
        indices = torch.tensor([[-1], [0]])
        values = torch.tensor([1.0])
        shape = (2, 2)

        with pytest.raises(RuntimeError):  # noqa: SIM117
            with torch.sparse.check_sparse_tensor_invariants():
                tensor = torch.sparse_coo_tensor(indices, values, shape)
                tensor.to_dense()

    def test_without_context_manager_allows_invalid(self):
        """
        WITHOUT validation, invalid tensors may not immediately error.

        This demonstrates the vulnerability: PyTorch 2.8.0+ doesn't validate
        by default, which can lead to memory corruption.
        """
        indices = torch.tensor([[100], [100]])  # Way out of bounds
        values = torch.tensor([1.0])
        shape = (2, 2)

        # Without validation context, this might create an invalid tensor
        # (actual behavior depends on PyTorch version)
        tensor = torch.sparse_coo_tensor(indices, values, shape)

        # The tensor object is created, but it's invalid
        assert tensor.is_sparse


class TestTorchLoadWithValidation:
    """Test torch.load() with sparse tensor validation."""

    def test_load_valid_sparse_tensor_with_validation(self):
        """Valid sparse tensors should load successfully with validation."""
        # Create and save a valid sparse tensor
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0])
        tensor = torch.sparse_coo_tensor(indices, values, (2, 2))

        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)

        # Load with validation
        with torch.sparse.check_sparse_tensor_invariants():
            loaded = torch.load(buffer, weights_only=True)
            dense = loaded.to_dense()

        assert dense.shape == (2, 2)

    def test_load_invalid_sparse_tensor_rejected(self):
        """Invalid sparse tensors should be caught when loaded with validation."""
        # Create an invalid sparse tensor (out of bounds)
        indices = torch.tensor([[10], [10]])
        values = torch.tensor([1.0])
        tensor = torch.sparse_coo_tensor(indices, values, (2, 2))

        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)

        # Load with validation - should fail on to_dense()
        with pytest.raises(RuntimeError):  # noqa: SIM117
            with torch.sparse.check_sparse_tensor_invariants():
                loaded = torch.load(buffer, weights_only=True)
                loaded.to_dense()

    def test_load_dense_tensor_unaffected(self):
        """Dense tensors should work normally with the validation context."""
        # Create and save a dense tensor
        tensor = torch.randn(10, 20)

        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)

        # Load with validation (should have no effect on dense tensors)
        with torch.sparse.check_sparse_tensor_invariants():
            loaded = torch.load(buffer, weights_only=True)

        assert loaded.shape == (10, 20)
        assert not loaded.is_sparse


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "--tb=short"])
