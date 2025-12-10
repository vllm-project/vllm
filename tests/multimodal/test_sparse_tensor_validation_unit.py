# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for CVE-2025-62164 sparse tensor validation.

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

        with pytest.raises(RuntimeError) as exc_info:
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

        with pytest.raises(RuntimeError):
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
        with pytest.raises(RuntimeError):
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


class TestCVE202562164VulnerabilityDemo:
    """
    Demonstrates the CVE-2025-62164 vulnerability and fix.

    These tests show what happens with and without the fix.
    """

    def test_vulnerability_demonstration(self):
        """
        Demonstrate the vulnerability: without validation, OOB writes can occur.

        This test shows that PyTorch 2.8.0+ allows creation of invalid sparse
        tensors without validation.
        """
        # Create malicious tensor with indices beyond bounds
        indices = torch.tensor([[999], [999]])
        values = torch.tensor([1.0])
        shape = (10, 10)

        # Without validation, this succeeds (!)
        malicious_tensor = torch.sparse_coo_tensor(indices, values, shape)
        assert malicious_tensor.is_sparse

        # Converting to dense would trigger OOB write without validation
        # (We don't actually call to_dense() here to avoid crashing the test)

    def test_fix_demonstration(self):
        """
        Demonstrate the fix: WITH validation, invalid tensors are rejected.

        This is what our fix implements in the vLLM code.
        """
        indices = torch.tensor([[999], [999]])
        values = torch.tensor([1.0])
        shape = (10, 10)

        # WITH validation, this should be caught
        with pytest.raises(RuntimeError):
            with torch.sparse.check_sparse_tensor_invariants():
                malicious_tensor = torch.sparse_coo_tensor(indices, values, shape)
                malicious_tensor.to_dense()  # This is where it would crash

    def test_attack_chain_blocked(self):
        """
        Full attack chain: malicious tensor -> serialize -> load -> validate.

        This simulates the complete attack path through the API.
        """
        # Step 1: Attacker creates malicious tensor
        indices = torch.tensor([[100], [100]])
        values = torch.tensor([999.0])
        shape = (5, 5)
        malicious = torch.sparse_coo_tensor(indices, values, shape)

        # Step 2: Attacker serializes it
        buffer = io.BytesIO()
        torch.save(malicious, buffer)
        buffer.seek(0)

        # Step 3: Server loads with validation (our fix)
        with pytest.raises(RuntimeError):
            with torch.sparse.check_sparse_tensor_invariants():
                loaded = torch.load(buffer, weights_only=True)
                loaded.to_dense()  # Attack blocked here!


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "--tb=short"])
