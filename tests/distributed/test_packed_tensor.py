# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for packed tensor broadcasting functionality.

Unit tests for packed_broadcast_producer and packed_broadcast_consumer.
These utilities enable efficient batched tensor transfer over NCCL.
"""

import pytest
import torch

from vllm import envs
from vllm.distributed.weight_transfer.nccl_engine import NCCLUpdateInfo
from vllm.distributed.weight_transfer.packed_tensor import (
    packed_broadcast_consumer,
    packed_broadcast_producer,
)


class MockCommunicationGroup:
    """Mock communication group for testing producer broadcast operations."""

    def __init__(self):
        self.broadcasted_tensors: list[torch.Tensor] = []
        self.broadcast_count = 0
        self.device = torch.device("cuda:0")

    def broadcast(self, tensor, src):
        """Mock broadcast that stores the tensor for later verification."""
        self.broadcasted_tensors.append(tensor.clone())
        self.broadcast_count += 1


class MockConsumerCommunicationGroup:
    """Mock communication group for consumer that returns pre-stored tensors."""

    def __init__(self, tensors_to_return: list[torch.Tensor]):
        self.tensors_to_return = tensors_to_return
        self.current_index = 0
        self.device = torch.device("cuda:0")

    def broadcast(self, tensor, src):
        """Mock broadcast that fills the tensor with pre-stored data."""
        if self.current_index < len(self.tensors_to_return):
            tensor.copy_(self.tensors_to_return[self.current_index])
            self.current_index += 1


def create_mock_model_params(
    num_layers: int = 3,
    dtype: torch.dtype = torch.float32,
) -> list[tuple[str, torch.Tensor]]:
    """Create mock model parameters for testing."""
    params = []
    for i in range(num_layers):
        params.append((f"layer{i}.weight", torch.randn(10, 20, dtype=dtype)))
        params.append((f"layer{i}.bias", torch.randn(10, dtype=dtype)))
    return params


def create_state_dict_info(
    params: list[tuple[str, torch.Tensor]],
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Create state dict info (name -> (shape, dtype)) from params."""
    return {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in params}


# --- Unit Tests: NCCLUpdateInfo packed field ---


class TestNCCLUpdateInfoPacked:
    """Test NCCLUpdateInfo dataclass packed field."""

    def test_packed_default_false(self):
        """Test that packed defaults to False."""
        info = NCCLUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
        )
        assert info.packed is False

    def test_packed_can_be_set_true(self):
        """Test that packed can be set to True."""
        info = NCCLUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
            packed=True,
        )
        assert info.packed is True


# --- Unit Tests: packed_broadcast_producer ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastProducer:
    """Test packed_broadcast_producer function."""

    def test_producer_broadcasts_tensors(self, monkeypatch):
        """Test that producer broadcasts all tensors."""
        params = create_mock_model_params()
        params_cuda = [(name, tensor.cuda()) for name, tensor in params]

        mock_group = MockCommunicationGroup()

        # Use a small target size to force multiple batches
        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", 500)
        packed_broadcast_producer(
            iterator=iter(params_cuda),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        # Should have broadcasted some tensors
        assert mock_group.broadcast_count > 0
        assert len(mock_group.broadcasted_tensors) > 0

    def test_producer_single_large_tensor(self, monkeypatch):
        """Test with a single tensor larger than target size."""
        # Create a large tensor
        large_tensor = torch.randn(1000, 1000, dtype=torch.float32).cuda()
        params = [("large_weight", large_tensor)]

        mock_group = MockCommunicationGroup()

        # Small target size to force the tensor to exceed it
        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", 100)
        packed_broadcast_producer(
            iterator=iter(params),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        # Should still broadcast the tensor (at least 1 broadcast)
        assert mock_group.broadcast_count >= 1
        assert len(mock_group.broadcasted_tensors) >= 1

        # Verify the total broadcasted size matches the tensor
        expected_size = large_tensor.numel() * large_tensor.element_size()
        actual_size = sum(t.numel() for t in mock_group.broadcasted_tensors)
        assert actual_size == expected_size

    def test_producer_multiple_batches(self, monkeypatch):
        """Test that tensors are properly batched when exceeding target size."""
        # Create many small tensors
        params = [
            (f"weight_{i}", torch.randn(10, 10, dtype=torch.float32).cuda())
            for i in range(20)
        ]

        mock_group = MockCommunicationGroup()

        # Small target size to force multiple batches
        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", 2000)
        packed_broadcast_producer(
            iterator=iter(params),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        # Should have multiple broadcasts
        assert mock_group.broadcast_count > 1

        # Total size should match sum of all tensors
        expected_total = sum(t.numel() * t.element_size() for _, t in params)
        actual_total = sum(t.numel() for t in mock_group.broadcasted_tensors)
        assert actual_total == expected_total

    def test_producer_empty_iterator(self, monkeypatch):
        """Test producer handles empty iterator gracefully."""
        mock_group = MockCommunicationGroup()

        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", 1000)
        packed_broadcast_producer(
            iterator=iter([]),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        # No broadcasts for empty iterator
        assert mock_group.broadcast_count == 0


# --- Unit Tests: packed_broadcast_consumer ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastConsumer:
    """Test packed_broadcast_consumer function."""

    def test_consumer_receives_tensors(self, monkeypatch):
        """Test that consumer receives and unpacks tensors."""
        params = create_mock_model_params()
        params_cuda = [(name, tensor.cuda()) for name, tensor in params]

        # First, run producer to get the broadcasted tensors
        producer_group = MockCommunicationGroup()

        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", 2000)
        packed_broadcast_producer(
            iterator=iter(params_cuda),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        # Now run consumer with the broadcasted tensors
        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params_cuda)

        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
        )

        # Verify all parameters were unpacked
        assert len(unpacked_tensors) == len(params)

        # Verify each tensor matches the original
        for name, original_tensor in params_cuda:
            assert name in unpacked_tensors
            unpacked = unpacked_tensors[name]
            assert unpacked.shape == original_tensor.shape
            assert unpacked.dtype == original_tensor.dtype
            assert torch.allclose(unpacked, original_tensor, rtol=1e-5, atol=1e-7)


# --- Integration Tests: Producer-Consumer Roundtrip ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastRoundtrip:
    """Test producer-consumer roundtrip behavior."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_roundtrip_different_dtypes(self, dtype, monkeypatch):
        """Test roundtrip with different data types."""
        params = create_mock_model_params(num_layers=2, dtype=dtype)
        params_cuda = [(name, tensor.cuda()) for name, tensor in params]

        producer_group = MockCommunicationGroup()

        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", 1000)
        packed_broadcast_producer(
            iterator=iter(params_cuda),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params_cuda)
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
        )

        # Verify roundtrip preserves data
        for name, original_tensor in params_cuda:
            assert name in unpacked_tensors
            unpacked = unpacked_tensors[name]
            assert unpacked.dtype == dtype
            assert torch.allclose(unpacked, original_tensor, rtol=1e-4, atol=1e-6)

    def test_roundtrip_mixed_dtypes(self, monkeypatch):
        """Test roundtrip with mixed data types."""
        # Create params with mixed dtypes
        params = [
            ("layer1.weight", torch.randn(10, 20, dtype=torch.float32).cuda()),
            ("layer1.bias", torch.randn(10, dtype=torch.float16).cuda()),
            ("layer2.weight", torch.randn(20, 30, dtype=torch.bfloat16).cuda()),
        ]

        producer_group = MockCommunicationGroup()

        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", 500)
        packed_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params)
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
        )

        # Verify all params roundtrip correctly with correct dtypes
        for name, original_tensor in params:
            assert name in unpacked_tensors
            unpacked = unpacked_tensors[name]
            assert unpacked.shape == original_tensor.shape
            assert unpacked.dtype == original_tensor.dtype
            assert torch.allclose(unpacked, original_tensor, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("target_size", [100, 1000, 10000, 100000])
    def test_roundtrip_different_batch_sizes(self, target_size, monkeypatch):
        """Test roundtrip with different target batch sizes."""
        params = create_mock_model_params(num_layers=5)
        params_cuda = [(name, tensor.cuda()) for name, tensor in params]

        producer_group = MockCommunicationGroup()

        monkeypatch.setattr(envs, "VLLM_PACKED_TENSOR_BUFFER_SIZE_BYTES", target_size)
        packed_broadcast_producer(
            iterator=iter(params_cuda),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params_cuda)
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
        )

        # Verify all params roundtrip correctly
        assert len(unpacked_tensors) == len(params)
        for name, original_tensor in params_cuda:
            assert name in unpacked_tensors
            assert torch.allclose(
                unpacked_tensors[name], original_tensor, rtol=1e-5, atol=1e-7
            )
