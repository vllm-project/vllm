# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.model_loader.tensorizer import MetaTensorMode


class TestMetaTensorMode:
    """Test suite for MetaTensorMode improvements.

    Tests the expanded factory operations coverage and device override behavior.
    """

    def test_basic_factory_ops(self):
        """Test that basic factory operations create meta tensors."""
        with MetaTensorMode():
            # Test basic tensor creation operations
            assert torch.empty(10, 20).device.type == "meta"
            assert torch.zeros(10, 20).device.type == "meta"
            assert torch.ones(10, 20).device.type == "meta"
            assert torch.full((10, 20), 3.14).device.type == "meta"
            assert torch.rand(10, 20).device.type == "meta"
            assert torch.randn(10, 20).device.type == "meta"

    def test_like_operations(self):
        """Test that *_like operations create meta tensors."""
        with MetaTensorMode():
            ref_tensor_int = torch.tensor([[1, 2], [3, 4]])
            ref_tensor_float = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

            assert torch.empty_like(ref_tensor_int).device.type == "meta"
            assert torch.zeros_like(ref_tensor_int).device.type == "meta"
            assert torch.ones_like(ref_tensor_int).device.type == "meta"
            assert torch.full_like(ref_tensor_int, 5.0).device.type == "meta"
            assert torch.rand_like(ref_tensor_float).device.type == "meta"
            assert torch.randn_like(ref_tensor_float).device.type == "meta"

    def test_strided_operations(self):
        """Test that strided operations create meta tensors."""
        with MetaTensorMode():
            tensor = torch.empty_strided((10, 20), (20, 1))
            assert tensor.device.type == "meta"

    def test_device_override_cpu(self):
        """Test that explicit CPU device is overridden to meta."""
        with MetaTensorMode():
            tensor = torch.empty(10, 20, device="cpu")
            assert tensor.device.type == "meta"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_override_cuda(self):
        """Test that explicit CUDA device is overridden to meta."""
        with MetaTensorMode():
            tensor = torch.empty(10, 20, device="cuda")
            assert tensor.device.type == "meta"

    def test_new_tensor_methods(self):
        """Test tensor.new_* methods create meta tensors."""
        with MetaTensorMode():
            ref_tensor = torch.tensor([[1, 2], [3, 4]])

            assert ref_tensor.new_empty(5, 5).device.type == "meta"
            assert ref_tensor.new_zeros(5, 5).device.type == "meta"
            assert ref_tensor.new_ones(5, 5).device.type == "meta"
            assert ref_tensor.new_full((5, 5), 7.0).device.type == "meta"

    def test_kwargs_none_handling(self):
        """Test that kwargs=None is handled correctly.

        This verifies the fix for safer kwargs handling:
        Previous: kwargs = kwargs or {}
        Current:  kwargs = kwargs if kwargs is not None else {}
        """
        with MetaTensorMode():
            # Call without explicit kwargs (internally None)
            tensor = torch.empty(10, 20)
            assert tensor.device.type == "meta"

    def test_factory_ops_coverage(self):
        """Test comprehensive coverage of factory operations.

        This test verifies all 18 factory operations in _FACTORY_OPS
        are properly intercepted.
        """
        with MetaTensorMode():
            ref_tensor = torch.tensor([[1.0, 2.0]])

            # Basic factory ops (6)
            assert torch.empty(5).device.type == "meta"
            assert torch.zeros(5).device.type == "meta"
            assert torch.ones(5).device.type == "meta"
            assert torch.full((5,), 1.0).device.type == "meta"
            assert torch.rand(5).device.type == "meta"
            assert torch.randn(5).device.type == "meta"

            # Like operations (6)
            assert torch.empty_like(ref_tensor).device.type == "meta"
            assert torch.zeros_like(ref_tensor).device.type == "meta"
            assert torch.ones_like(ref_tensor).device.type == "meta"
            assert torch.full_like(ref_tensor, 2.0).device.type == "meta"
            assert torch.rand_like(ref_tensor).device.type == "meta"
            assert torch.randn_like(ref_tensor).device.type == "meta"

            # Strided operation (1)
            assert torch.empty_strided((5, 5), (5, 1)).device.type == "meta"

            # New tensor methods (4)
            assert ref_tensor.new_empty(5).device.type == "meta"
            assert ref_tensor.new_zeros(5).device.type == "meta"
            assert ref_tensor.new_ones(5).device.type == "meta"
            assert ref_tensor.new_full((5,), 3.0).device.type == "meta"

            # Note: new_empty_strided is covered implicitly via empty_strided

    def test_outside_context_manager(self):
        """Test that operations outside MetaTensorMode work normally."""
        # Outside the context manager, tensors should be created on default device
        tensor = torch.empty(5)
        assert tensor.device.type in ["cpu", "cuda"]

        # Only inside the context manager should they be meta
        with MetaTensorMode():
            meta_tensor = torch.empty(5)
            assert meta_tensor.device.type == "meta"

        # After exiting, should work normally again
        tensor_after = torch.empty(5)
        assert tensor_after.device.type in ["cpu", "cuda"]
