# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for packed tensor broadcasting functionality.

Unit tests for packed_nccl_broadcast_producer and packed_nccl_broadcast_consumer.
These utilities enable efficient batched tensor transfer over NCCL.
"""

import pytest
import torch

from vllm.distributed.weight_transfer import packed_tensor
from vllm.distributed.weight_transfer.packed_tensor import (
    pack_tensors,
    packed_ipc_consumer,
    packed_ipc_producer,
    packed_nccl_broadcast_consumer,
    packed_nccl_broadcast_producer,
    unpack_tensor,
)
from vllm.model_executor.model_loader.weight_utils import stash_weight_part


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


# Long enough (~1 s on current GPUs) that the host always gets through the
# checked window first; the tests assert that it did.
SLEEP_CYCLES = 2_000_000_000


class DelayedFirstBroadcastGroup:
    """Fills each pack with its sentinel on the current (slot) stream; the
    first pack is zeroed and its sentinel lands only after a GPU sleep, at
    ``first_received``."""

    def __init__(self, sentinels: list[float]):
        self.sentinels = sentinels
        self.first_received = torch.Event()
        self.calls = 0

    def broadcast(self, tensor, src):
        dst = tensor.view(torch.float32)
        if self.calls == 0:
            dst.zero_()
            torch.cuda._sleep(SLEEP_CYCLES)
        dst.fill_(self.sentinels[self.calls])
        if self.calls == 0:
            self.first_received.record()
        self.calls += 1


class SentinelBroadcastGroup:
    """Immediately fills each pack with its own sentinel, recording each pack's
    address and whether ``pending`` (once set) was still incomplete."""

    def __init__(self, sentinels: list[float]):
        self.sentinels = sentinels
        self.pending: torch.Event | None = None
        self.ptrs: list[int] = []
        self.pending_at_receive: list[bool] = []
        self.calls = 0

    def broadcast(self, tensor, src):
        tensor.view(torch.float32).fill_(self.sentinels[self.calls])
        self.ptrs.append(tensor.data_ptr())
        self.pending_at_receive.append(
            self.pending is not None and not self.pending.query()
        )
        self.calls += 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastDeferredConsumer:
    """Consumers that keep unpacked tensors and read them from a later pack's
    callback, as layerwise reload does, on a different slot stream."""

    def test_deferred_read_is_ordered_after_receive(self):
        numel = 4
        state_dict_info = [
            ("a", ([numel], torch.float32)),
            ("b", ([numel], torch.float32)),
        ]
        # A fresh sentinel per run keeps a stale allocator block that held a
        # previous run's value from masking an unordered read.
        expected = float(torch.randint(1, 2**20, ()).item())
        group = DelayedFirstBroadcastGroup(sentinels=[expected, -expected])
        observed = torch.full((numel,), -1.0, device="cuda")
        torch.accelerator.synchronize()

        retained: list[torch.Tensor] = []
        read_enqueued_before_receive: list[bool] = []

        def post_unpack_func(tensor_list):
            [(_, tensor)] = tensor_list
            if not retained:
                retained.append(tensor)
            else:
                read_enqueued_before_receive.append(not group.first_received.query())
                observed.copy_(retained[0])

        packed_nccl_broadcast_consumer(
            iterator=iter(state_dict_info),
            group=group,
            src=0,
            post_unpack_func=post_unpack_func,
            # Smaller than one tensor, so each tensor gets its own pack.
            buffer_size_bytes=numel * 4 - 1,
            num_buffers=2,
        )
        torch.accelerator.synchronize()

        assert group.calls == 2
        assert read_enqueued_before_receive == [True], (
            "the first receive finished before the deferred read was enqueued"
        )
        torch.testing.assert_close(observed, torch.full_like(observed, expected))

    @pytest.mark.parametrize(
        ("num_buffers", "packs_per_layer"), [(2, 4), (2, 5), (3, 5)]
    )
    def test_freed_pack_not_reused_while_replay_pending(
        self, num_buffers, packs_per_layer
    ):
        """Views dropped right after the replay is enqueued (as layerwise
        reload's ``info.reset()`` does) must not let the next receive reuse a
        buffer the replay still reads."""
        num_packs = packs_per_layer + 4
        # A size unique to this run keeps blocks cached by earlier tests from
        # being picked instead of the ones this layer frees.
        numel = 128 * int(torch.randint(1, 64, ()).item())
        state_dict_info = [
            (f"w{i}", ([numel], torch.float32)) for i in range(num_packs)
        ]
        sentinels = torch.randperm(2**20)[:num_packs].float().tolist()
        group = SentinelBroadcastGroup(sentinels)
        observed = torch.full((packs_per_layer, numel), -1.0, device="cuda")
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

        retained: list[torch.Tensor] = []
        replayed = False

        def post_unpack_func(tensor_list):
            nonlocal replayed
            [(_, tensor)] = tensor_list
            if replayed:
                return
            retained.append(tensor)
            if len(retained) == packs_per_layer:
                torch.cuda._sleep(SLEEP_CYCLES)
                for i, t in enumerate(retained):
                    observed[i].copy_(t)
                group.pending = torch.Event()
                group.pending.record()
                retained.clear()
                replayed = True

        packed_nccl_broadcast_consumer(
            iterator=iter(state_dict_info),
            group=group,
            src=0,
            post_unpack_func=post_unpack_func,
            # Smaller than one tensor, so each tensor gets its own pack.
            buffer_size_bytes=numel * 4 - 1,
            num_buffers=num_buffers,
        )
        torch.accelerator.synchronize()

        assert group.calls == num_packs
        # Receives before the host next waits on the replay's slot stream.
        window = range(packs_per_layer, packs_per_layer + num_buffers - 1)
        assert all(group.pending_at_receive[i] for i in window), (
            "the replay finished before every receive in the window was issued"
        )
        pack_bytes = numel * 4
        for i in window:
            for j in range(packs_per_layer):
                assert abs(group.ptrs[i] - group.ptrs[j]) >= pack_bytes, (
                    f"pack {i} reused pack {j}'s buffer while the replay was pending"
                )
        expected = torch.tensor(sentinels[:packs_per_layer], device="cuda")
        torch.testing.assert_close(
            observed, expected.unsqueeze(1).expand_as(observed), rtol=0, atol=0
        )


# --- Unit Tests: unpack_tensor ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestWeightPartsAcrossStreams:
    """The consumer loads alternate buckets on different streams; a weight whose
    parts arrive in different buckets is paired on the second bucket's stream."""

    @pytest.mark.parametrize("num_buffers", [1, 2], ids=["one-stream", "two-streams"])
    def test_stash_pairs_part_parked_on_other_stream(self, monkeypatch, num_buffers):
        n = 1024
        weight = torch.rand(n, n, device="cuda") + 1
        scale = torch.full((1,), 3.0, device="cuda")
        streams = packed_tensor._get_streams(torch.cuda.current_device(), num_buffers)
        monkeypatch.setattr(packed_tensor, "_get_streams", lambda *_: streams)
        # Leave recognizable free blocks in the first stream's pool, where the
        # parked copy of the weight is allocated.
        with torch.cuda.stream(streams[0]):
            junk = [torch.full((n, n), -1.0, device="cuda") for _ in range(3)]
        torch.cuda.synchronize()
        del junk

        class DelayedFirstBroadcast:
            calls = 0

            def broadcast(self, tensor, src):
                self.calls += 1
                if self.calls == 1:  # the first bucket's stream runs late
                    torch.cuda._sleep(int(3e9))
                src_tensor = weight if self.calls == 1 else scale
                tensor.copy_(src_tensor.reshape(-1).view(torch.uint8))

        holder = torch.nn.Module()
        out = torch.zeros(n, n, device="cuda")

        def load(tensors):
            for name, tensor in tensors:
                parts = stash_weight_part(holder, "w", name, tensor)
                if parts is not None:
                    out.copy_(parts["weight"] * parts["scale"])

        group = DelayedFirstBroadcast()
        packed_nccl_broadcast_consumer(
            iterator=iter(
                [("weight", ([n, n], torch.float32)), ("scale", ([1], torch.float32))]
            ),
            group=group,
            src=0,
            post_unpack_func=load,
            buffer_size_bytes=n * n * 4 - 1,  # one bucket per tensor
            num_buffers=num_buffers,
        )
        torch.cuda.synchronize()
        assert group.calls == 2
        torch.testing.assert_close(out, weight * 3, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUnpackTensor:
    """Test the shared unpack_tensor function."""

    def test_unpack_returns_views(self):
        """Unpacked tensors alias the packed buffer; callers that need
        independent storage must clone."""
        packed = torch.randn(10, dtype=torch.float32).cuda().view(torch.uint8)

        result = unpack_tensor(
            packed,
            names=["w"],
            shapes=[[10]],
            dtypes=[torch.float32],
            tensor_sizes=[packed.numel()],
        )

        assert result[0][1].data_ptr() == packed.data_ptr()
        packed.zero_()
        assert torch.count_nonzero(result[0][1]) == 0


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


def _ipc_consumer_worker(cmd_q, ack_q, result_q, done_event, device_index):
    """Worker that consumes chunks streamed one at a time from the parent.

    CUDA IPC requires the consumer to be in a separate process from the
    producer. The producer reuses a single IPC buffer between chunks, so
    the parent must wait for our ack (sent after we copy the chunk to
    CPU) before advancing the producer.
    """
    try:
        torch.accelerator.set_device_index(device_index)
        all_results = []
        while True:
            cd = cmd_q.get()
            if cd is None:
                break
            result = packed_ipc_consumer(
                ipc_handle=cd["ipc_handle"],
                names=cd["names"],
                shapes=cd["shapes"],
                dtype_names=cd["dtype_names"],
                tensor_sizes=cd["tensor_sizes"],
                device_index=device_index,
            )
            # .cpu() forces a GPU→CPU copy off the shared IPC buffer, so
            # the producer is free to overwrite it once we ack.
            all_results.extend([(name, tensor.cpu()) for name, tensor in result])
            del result
            ack_q.put("ack")
        result_q.put(("ok", all_results))
    except Exception as e:
        result_q.put(("error", str(e)))
    # Keep the process alive until the parent has finished reading from
    # the result queue — torch serializes CPU tensors via fd sharing,
    # which requires this process's resource-sharer server to be running.
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

    def _run_roundtrip(self, chunk_iter, device_index, timeout=30):
        """Stream chunks through a child consumer one at a time.

        ``packed_ipc_producer`` reuses a single IPC buffer for every
        chunk, so the producer must not be advanced until the consumer
        has finished reading the current chunk. We enforce that with an
        ack queue: the consumer puts ``"ack"`` after it has copied the
        chunk to CPU, and only then do we pull the next chunk from the
        generator.

        Returns ``(num_chunks, results)``.
        """
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        cmd_q = ctx.Queue()
        ack_q = ctx.Queue()
        result_q = ctx.Queue()
        done_event = ctx.Event()
        proc = ctx.Process(
            target=_ipc_consumer_worker,
            args=(cmd_q, ack_q, result_q, done_event, device_index),
        )
        proc.start()

        num_chunks = 0
        try:
            for chunk in chunk_iter:
                cmd_q.put(
                    {
                        "ipc_handle": chunk.ipc_handle,
                        "names": chunk.names,
                        "shapes": chunk.shapes,
                        "dtype_names": chunk.dtype_names,
                        "tensor_sizes": chunk.tensor_sizes,
                    }
                )
                if ack_q.get(timeout=timeout) != "ack":
                    raise RuntimeError("Consumer did not ack chunk")
                num_chunks += 1
            cmd_q.put(None)
            status, payload = result_q.get(timeout=timeout)
        finally:
            done_event.set()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()

        if status == "error":
            raise RuntimeError(f"Consumer process failed: {payload}")
        # Reclaim IPC-shared memory now that the child has released it
        torch.cuda.ipc_collect()
        return num_chunks, payload

    def test_roundtrip_basic(self):
        """Test basic IPC producer -> consumer roundtrip."""
        params = [
            ("w1", torch.randn(10, 20, dtype=torch.float32).cuda()),
            ("scale", torch.tensor(1.5, dtype=torch.float32).cuda()),
            ("w2", torch.randn(5, dtype=torch.float16).cuda()),
        ]
        gpu_uuid = self._get_gpu_uuid()
        device_index = torch.cuda.current_device()

        num_chunks, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            ),
            device_index,
        )

        assert num_chunks == 1
        assert len(result) == len(params)
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

        _, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params_cuda),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            ),
            device_index,
        )

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

        num_chunks, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=50_000,
            ),
            device_index,
        )

        assert num_chunks > 1
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

        _, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            ),
            device_index,
        )

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pack_unpack_scalar(dtype):
    """Preserve scalar shape, dtype and value alongside a matrix on CPU."""
    params = [
        ("weight", torch.randn(2, 3, dtype=dtype)),
        ("scale", torch.tensor(1.5, dtype=dtype)),
    ]

    chunk = pack_tensors(iter(params), lambda item: item[1], buffer_size_bytes=1024)
    assert chunk is not None

    result = unpack_tensor(
        chunk.packed_tensor,
        chunk.names,
        chunk.shapes,
        chunk.dtypes,
        chunk.tensor_sizes,
    )

    for (expected_name, expected), (name, actual) in zip(params, result, strict=True):
        assert name == expected_name
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
