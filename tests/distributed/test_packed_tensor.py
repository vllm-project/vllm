# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for packed tensor broadcasting functionality.

Unit tests for packed_nccl_broadcast_producer and packed_nccl_broadcast_consumer.
These utilities enable efficient batched tensor transfer over NCCL.
"""

import pytest
import torch

from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferUpdateInfo
from vllm.distributed.weight_transfer.packed_tensor import (
    pack_tensors,
    packed_ipc_consumer,
    packed_ipc_producer,
    packed_nccl_broadcast_consumer,
    packed_nccl_broadcast_producer,
    unpack_tensor,
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


# --- Unit Tests: NCCLWeightTransferUpdateInfo packed field ---


class TestNCCLWeightTransferUpdateInfoPacked:
    """Test NCCLWeightTransferUpdateInfo dataclass packed field."""

    def test_packed_default_false(self):
        """Test that packed defaults to False."""
        info = NCCLWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
        )
        assert info.packed is False

    def test_packed_can_be_set_true(self):
        """Test that packed can be set to True."""
        info = NCCLWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
            packed=True,
        )
        assert info.packed is True


# --- Unit Tests: packed_nccl_broadcast_producer ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastProducer:
    """Test packed_nccl_broadcast_producer function."""

    def test_producer_empty_iterator(self):
        """Test producer handles empty iterator gracefully."""
        mock_group = MockCommunicationGroup()

        packed_nccl_broadcast_producer(
            iterator=iter([]),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=1000,
        )

        # No broadcasts for empty iterator
        assert mock_group.broadcast_count == 0


# --- Integration Tests: Producer-Consumer Roundtrip ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastRoundtrip:
    """Test producer-consumer roundtrip behavior."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_roundtrip_different_dtypes(self, dtype):
        """Test roundtrip with different data types."""
        params = create_mock_model_params(num_layers=2, dtype=dtype)
        params_cuda = [(name, tensor.cuda()) for name, tensor in params]

        buffer_size = 1000
        producer_group = MockCommunicationGroup()

        packed_nccl_broadcast_producer(
            iterator=iter(params_cuda),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=buffer_size,
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params_cuda)
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_nccl_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size,
        )

        # Verify roundtrip preserves data
        for name, original_tensor in params_cuda:
            assert name in unpacked_tensors
            unpacked = unpacked_tensors[name]
            assert unpacked.dtype == dtype
            assert torch.allclose(unpacked, original_tensor, rtol=1e-4, atol=1e-6)

    def test_roundtrip_mixed_dtypes(self):
        """Test roundtrip with mixed data types."""
        # Create params with mixed dtypes
        params = [
            ("layer1.weight", torch.randn(10, 20, dtype=torch.float32).cuda()),
            ("layer1.bias", torch.randn(10, dtype=torch.float16).cuda()),
            ("layer2.weight", torch.randn(20, 30, dtype=torch.bfloat16).cuda()),
        ]

        buffer_size = 500
        producer_group = MockCommunicationGroup()

        packed_nccl_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=buffer_size,
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params)
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_nccl_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size,
        )

        # Verify all params roundtrip correctly with correct dtypes
        for name, original_tensor in params:
            assert name in unpacked_tensors
            unpacked = unpacked_tensors[name]
            assert unpacked.shape == original_tensor.shape
            assert unpacked.dtype == original_tensor.dtype
            assert torch.allclose(unpacked, original_tensor, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("target_size", [100, 100000])
    def test_roundtrip_different_batch_sizes(self, target_size):
        """Test roundtrip with different target batch sizes."""
        params = create_mock_model_params(num_layers=5)
        params_cuda = [(name, tensor.cuda()) for name, tensor in params]

        producer_group = MockCommunicationGroup()

        packed_nccl_broadcast_producer(
            iterator=iter(params_cuda),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=target_size,
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params_cuda)
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_nccl_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=target_size,
        )

        # Verify all params roundtrip correctly
        assert len(unpacked_tensors) == len(params)
        for name, original_tensor in params_cuda:
            assert name in unpacked_tensors
            assert torch.allclose(
                unpacked_tensors[name], original_tensor, rtol=1e-5, atol=1e-7
            )

    def test_roundtrip_non_contiguous_tensors(self):
        """Test roundtrip with non-contiguous tensors from the trainer."""
        # Create non-contiguous tensors (simulating trainer outputs)
        # Transposed tensors are non-contiguous
        weight1 = torch.randn(20, 10, dtype=torch.float32).cuda().T
        # Sliced tensors with step are non-contiguous
        weight2 = torch.randn(40, 30, dtype=torch.float16).cuda()[::2, ::2]
        # Permuted tensors are non-contiguous
        weight3 = torch.randn(5, 10, 15, dtype=torch.bfloat16).cuda().permute(2, 0, 1)

        params = [
            ("layer1.weight", weight1),
            ("layer2.weight", weight2),
            ("layer3.weight", weight3),
        ]

        # Verify tensors are indeed non-contiguous
        for name, tensor in params:
            assert not tensor.is_contiguous(), f"{name} should be non-contiguous"

        buffer_size = 500
        producer_group = MockCommunicationGroup()

        packed_nccl_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=buffer_size,
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        state_dict_info = create_state_dict_info(params)
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor.clone()

        packed_nccl_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size,
        )

        # Verify all non-contiguous params roundtrip correctly
        for name, original_tensor in params:
            assert name in unpacked_tensors
            unpacked = unpacked_tensors[name]
            assert unpacked.shape == original_tensor.shape
            assert unpacked.dtype == original_tensor.dtype
            assert torch.allclose(unpacked, original_tensor, rtol=1e-4, atol=1e-6)


# --- Unit Tests: unpack_tensor ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUnpackTensor:
    """Test the shared unpack_tensor function."""

    def test_unpack_produces_independent_copies(self):
        """Verify unpacked tensors don't share memory with packed buffer."""
        original = torch.randn(10, dtype=torch.float32).cuda()
        packed = original.contiguous().view(torch.uint8).view(-1)

        result = unpack_tensor(
            packed,
            names=["w"],
            shapes=[[10]],
            dtypes=[torch.float32],
            tensor_sizes=[packed.numel()],
        )

        # Mutate the packed buffer
        packed.zero_()

        # Unpacked tensor should be unaffected
        assert torch.allclose(result[0][1], original)


# --- Unit Tests: pack_tensors ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackTensors:
    """Test the shared pack_tensors function."""

    def test_pack_basic(self):
        """Test packing a few tensors into one buffer."""
        params = [
            ("w1", torch.randn(10, 20, dtype=torch.float32).cuda()),
            ("w2", torch.randn(5, dtype=torch.float16).cuda()),
        ]

        chunk = pack_tensors(
            iterator=iter(params),
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=10_000_000,
        )

        assert chunk is not None
        assert len(chunk.names) == 2
        assert chunk.names == ["w1", "w2"]
        assert chunk.shapes == [[10, 20], [5]]
        assert chunk.dtypes == [torch.float32, torch.float16]
        assert chunk.packed_tensor.dtype == torch.uint8

    def test_pack_respects_buffer_limit(self):
        """Test that packing stops when buffer_size_bytes is exceeded."""
        params = [
            (f"w{i}", torch.randn(100, 100, dtype=torch.float32).cuda())
            for i in range(10)
        ]

        chunk = pack_tensors(
            iterator=iter(params),
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=50_000,
        )

        assert chunk is not None
        assert len(chunk.names) < 10

    def test_pack_empty_iterator(self):
        """Test that an empty iterator returns None."""
        chunk = pack_tensors(
            iterator=iter([]),
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=1000,
        )
        assert chunk is None

    def test_pack_single_tensor_larger_than_buffer_warns(self):
        """Test that a tensor exceeding buffer_size_bytes emits a warning."""
        big = torch.randn(1000, 1000, dtype=torch.float32).cuda()
        params = [("big", big)]

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunk = pack_tensors(
                iterator=iter(params),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=100,
            )
        assert chunk is not None
        assert len(chunk.names) == 1
        assert any("exceeds buffer_size_bytes" in str(wi.message) for wi in w)

    def test_pack_unpack_roundtrip(self):
        """Test pack then unpack produces identical tensors."""
        params = [
            ("a", torch.randn(8, 16, dtype=torch.float32).cuda()),
            ("b", torch.randn(4, dtype=torch.float16).cuda()),
            ("c", torch.randn(3, 5, 7, dtype=torch.bfloat16).cuda()),
        ]

        chunk = pack_tensors(
            iterator=iter(params),
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=10_000_000,
        )

        assert chunk is not None
        result = unpack_tensor(
            chunk.packed_tensor,
            chunk.names,
            chunk.shapes,
            chunk.dtypes,
            chunk.tensor_sizes,
        )

        assert len(result) == len(params)
        for (orig_name, orig_tensor), (res_name, res_tensor) in zip(params, result):
            assert orig_name == res_name
            assert res_tensor.shape == orig_tensor.shape
            assert res_tensor.dtype == orig_tensor.dtype
            assert torch.allclose(res_tensor, orig_tensor, rtol=1e-4, atol=1e-6)

    def test_pack_multiple_chunks(self):
        """Test consuming an iterator across multiple pack_tensors calls."""
        params = [
            (f"w{i}", torch.randn(50, 50, dtype=torch.float32).cuda()) for i in range(6)
        ]
        it = iter(params)

        all_names = []
        chunks = []
        while True:
            chunk = pack_tensors(it, lambda x: x[1], buffer_size_bytes=12_000)
            if chunk is None:
                break
            chunks.append(chunk)
            all_names.extend(chunk.names)

        assert len(chunks) > 1
        assert all_names == [f"w{i}" for i in range(6)]


# --- Unit Tests: packed_ipc_producer ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedIpcProducer:
    """Test the packed_ipc_producer generator."""

    def test_producer_yields_chunks(self):
        """Test that the producer yields PackedIpcChunk objects."""
        params = [
            (f"w{i}", torch.randn(50, 50, dtype=torch.float32).cuda()) for i in range(6)
        ]

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid="test-uuid",
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=12_000,
            )
        )

        assert len(chunks) > 1
        assert chunks[0].is_first is True
        assert chunks[-1].is_last is True
        for c in chunks[1:]:
            assert c.is_first is False
        for c in chunks[:-1]:
            assert c.is_last is False

    def test_producer_ipc_handle_has_uuid(self):
        """Test that each chunk's ipc_handle is keyed by the given UUID."""
        params = [("w", torch.randn(10, dtype=torch.float32).cuda())]

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid="my-gpu-uuid",
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        assert "my-gpu-uuid" in chunks[0].ipc_handle

    def test_producer_dtype_names_are_strings(self):
        """Test that dtype_names are string representations."""
        params = [
            ("a", torch.randn(10, dtype=torch.float32).cuda()),
            ("b", torch.randn(10, dtype=torch.float16).cuda()),
        ]

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid="uuid",
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        assert chunks[0].dtype_names == ["float32", "float16"]

    def test_producer_empty_iterator(self):
        """Test producer with empty iterator yields nothing."""
        chunks = list(
            packed_ipc_producer(
                iterator=iter([]),
                gpu_uuid="uuid",
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=1000,
            )
        )
        assert len(chunks) == 0


# --- Integration Tests: IPC Producer-Consumer Roundtrip ---


def _ipc_consumer_worker(queue, done_event, chunk_data_list, device_index):
    """Worker function that runs packed_ipc_consumer in a child process.

    CUDA IPC requires the consumer to be in a separate process from the
    producer. This function is the target of multiprocessing.Process.
    """
    try:
        torch.accelerator.set_device_index(device_index)
        all_results = []
        for cd in chunk_data_list:
            result = packed_ipc_consumer(
                ipc_handle=cd["ipc_handle"],
                names=cd["names"],
                shapes=cd["shapes"],
                dtype_names=cd["dtype_names"],
                tensor_sizes=cd["tensor_sizes"],
                device_index=device_index,
            )
            all_results.extend([(name, tensor.cpu()) for name, tensor in result])
        queue.put(("ok", all_results))
    except Exception as e:
        queue.put(("error", str(e)))
    # Keep the process alive until the parent has finished reading from
    # the queue — torch serializes CPU tensors via fd sharing, which
    # requires this process's resource-sharer server to be running.
    done_event.wait(timeout=60)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedIpcRoundtrip:
    """Test IPC producer-consumer roundtrip using real CUDA IPC.

    These tests spawn a child process for the consumer because
    rebuild_cuda_tensor requires a separate process from the one that
    called reduce_tensor.
    """

    def _get_gpu_uuid(self) -> str:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        return str(props.uuid)

    def _run_roundtrip(self, chunks, device_index, timeout=30):
        """Produce IPC handles in this process, consume in a child."""
        import multiprocessing as mp

        chunk_data_list = [
            {
                "ipc_handle": c.ipc_handle,
                "names": c.names,
                "shapes": c.shapes,
                "dtype_names": c.dtype_names,
                "tensor_sizes": c.tensor_sizes,
            }
            for c in chunks
        ]

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        done_event = ctx.Event()
        proc = ctx.Process(
            target=_ipc_consumer_worker,
            args=(queue, done_event, chunk_data_list, device_index),
        )
        proc.start()
        status, payload = queue.get(timeout=timeout)
        # Signal the child that we've finished reading so it can exit.
        done_event.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            raise RuntimeError("Consumer process timed out")
        if status == "error":
            raise RuntimeError(f"Consumer process failed: {payload}")
        # Reclaim IPC-shared memory now that the child has released it
        torch.cuda.ipc_collect()
        return payload

    def test_roundtrip_basic(self):
        """Test basic IPC producer -> consumer roundtrip."""
        params = [
            ("w1", torch.randn(10, 20, dtype=torch.float32).cuda()),
            ("w2", torch.randn(5, dtype=torch.float16).cuda()),
        ]
        gpu_uuid = self._get_gpu_uuid()
        device_index = torch.cuda.current_device()

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        assert len(chunks) == 1
        result = self._run_roundtrip(chunks, device_index)

        assert len(result) == 2
        for (orig_name, orig_tensor), (res_name, res_tensor) in zip(params, result):
            assert orig_name == res_name
            assert res_tensor.shape == orig_tensor.shape
            assert res_tensor.dtype == orig_tensor.dtype
            assert torch.allclose(res_tensor, orig_tensor.cpu(), rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_roundtrip_dtypes(self, dtype):
        """Test IPC roundtrip with different dtypes."""
        params = create_mock_model_params(num_layers=2, dtype=dtype)
        params_cuda = [(n, t.cuda()) for n, t in params]
        gpu_uuid = self._get_gpu_uuid()
        device_index = torch.cuda.current_device()

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params_cuda),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        result = self._run_roundtrip(chunks, device_index)

        assert len(result) == len(params_cuda)
        for (orig_name, orig_tensor), (res_name, res_tensor) in zip(
            params_cuda, result
        ):
            assert orig_name == res_name
            assert res_tensor.dtype == dtype
            assert torch.allclose(res_tensor, orig_tensor.cpu(), rtol=1e-4, atol=1e-6)

    def test_roundtrip_multiple_chunks(self):
        """Test IPC roundtrip across multiple chunks."""
        params = [
            (f"layer{i}.weight", torch.randn(100, 100, dtype=torch.float32).cuda())
            for i in range(8)
        ]
        gpu_uuid = self._get_gpu_uuid()
        device_index = torch.cuda.current_device()

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=50_000,
            )
        )

        assert len(chunks) > 1
        result = self._run_roundtrip(chunks, device_index)

        assert len(result) == len(params)
        for (orig_name, orig_tensor), (res_name, res_tensor) in zip(params, result):
            assert orig_name == res_name
            assert torch.allclose(res_tensor, orig_tensor.cpu(), rtol=1e-5, atol=1e-7)

    def test_roundtrip_non_contiguous(self):
        """Test IPC roundtrip with non-contiguous tensors."""
        params = [
            ("transposed", torch.randn(20, 10, dtype=torch.float32).cuda().T),
            ("sliced", torch.randn(40, 30, dtype=torch.float16).cuda()[::2, ::2]),
        ]
        gpu_uuid = self._get_gpu_uuid()
        device_index = torch.cuda.current_device()

        for _, t in params:
            assert not t.is_contiguous()

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        result = self._run_roundtrip(chunks, device_index)

        for (orig_name, orig_tensor), (res_name, res_tensor) in zip(params, result):
            assert orig_name == res_name
            assert res_tensor.shape == orig_tensor.shape
            assert res_tensor.dtype == orig_tensor.dtype
            assert torch.allclose(res_tensor, orig_tensor.cpu(), rtol=1e-4, atol=1e-6)

    def test_consumer_wrong_uuid_raises(self):
        """Test that consumer raises ValueError for unknown GPU UUID."""
        params = [("w", torch.randn(10, dtype=torch.float32).cuda())]
        gpu_uuid = self._get_gpu_uuid()

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        c = chunks[0]
        fake_handle = {"fake-uuid-12345": c.ipc_handle[gpu_uuid]}

        with pytest.raises(ValueError, match="IPC handle not found"):
            packed_ipc_consumer(
                ipc_handle=fake_handle,
                names=c.names,
                shapes=c.shapes,
                dtype_names=c.dtype_names,
                tensor_sizes=c.tensor_sizes,
                device_index=torch.cuda.current_device(),
            )
