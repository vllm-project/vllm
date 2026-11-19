# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for packed tensor broadcasting functionality.

Unit tests for packed_nccl_broadcast_producer and packed_nccl_broadcast_consumer.
These utilities enable efficient batched tensor transfer over NCCL.
"""

import pytest
import torch

from vllm.distributed.weight_transfer.packed_tensor import (
    packed_ipc_consumer,
    packed_ipc_producer,
    packed_nccl_broadcast_consumer,
    packed_nccl_broadcast_producer,
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


# Long enough (~1 s on current GPUs) that the host always gets through the
# checked window first; the tests assert that it did.
SLEEP_CYCLES = 2_000_000_000


def roundtrip(params: list[tuple[str, torch.Tensor]], buffer_size_bytes: int):
    producer_group = MockCommunicationGroup()
    packed_nccl_broadcast_producer(
        iterator=iter(params),
        group=producer_group,
        src=0,
        post_iter_func=lambda x: x[1],
        buffer_size_bytes=buffer_size_bytes,
    )
    dst = {name: torch.empty_like(t) for name, t in params}

    def consume(sleep: bool) -> bool:
        torch.accelerator.synchronize()
        if sleep:
            torch.cuda._sleep(SLEEP_CYCLES)
        for t in dst.values():
            t.zero_()
        filled = torch.Event()
        filled.record()
        fill_pending_at_load: list[bool] = []

        def post_unpack_func(tensor_list):
            fill_pending_at_load.append(not filled.query())
            for name, tensor in tensor_list:
                target = dst[name]
                assert (tensor.shape, tensor.dtype) == (target.shape, target.dtype)
                target.copy_(tensor)

        packed_nccl_broadcast_consumer(
            iterator=iter(create_state_dict_info(params).items()),
            group=MockConsumerCommunicationGroup(producer_group.broadcasted_tensors),
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size_bytes,
        )
        torch.accelerator.synchronize()
        for name, t in params:
            assert torch.equal(dst[name], t), name
        return fill_pending_at_load[0]

    consume(sleep=False)
    assert consume(sleep=True), "the fill finished before the first load"


# --- Integration Tests: Producer-Consumer Roundtrip ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastRoundtrip:
    """Test producer-consumer roundtrip behavior."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_roundtrip_different_dtypes(self, dtype):
        """Test roundtrip with different data types."""
        params = create_mock_model_params(num_layers=2, dtype=dtype)
        roundtrip([(n, t.cuda()) for n, t in params], buffer_size_bytes=1000)

    def test_roundtrip_mixed_dtypes(self):
        """Test roundtrip with mixed data types."""
        # Create params with mixed dtypes
        params = [
            ("layer1.weight", torch.randn(10, 20, dtype=torch.float32).cuda()),
            ("layer1.bias", torch.randn(10, dtype=torch.float16).cuda()),
            ("layer2.weight", torch.randn(20, 30, dtype=torch.bfloat16).cuda()),
        ]
        roundtrip(params, buffer_size_bytes=500)

    @pytest.mark.parametrize("target_size", [100, 100000])
    def test_roundtrip_different_batch_sizes(self, target_size):
        """Test roundtrip with different target batch sizes."""
        params = create_mock_model_params(num_layers=5)
        roundtrip([(n, t.cuda()) for n, t in params], buffer_size_bytes=target_size)

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

        roundtrip(params, buffer_size_bytes=500)


class DelayedFirstBroadcastGroup:
    """Fills each pack with its sentinel on the current (slot) stream; the
    first pack is zeroed and its sentinel lands only after a GPU sleep, at
    `first_received`."""

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
    address and whether `pending` (once set) was still incomplete."""

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
        reload's `info.reset()` does) must not let the next receive reuse a
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

    @pytest.mark.parametrize("limit", [16, 100, 1000, 4096])
    def test_chunks_match_ipc_packing(self, limit):
        """Chunks hold tensors back to back and flush before overflowing the
        fixed buffer, so the metadata (and the bytes at those offsets) are
        what earlier versions sent."""
        tensors = [t.cuda() for t in _mixed_tensors()]
        tensors = [t for t in tensors if t.nbytes <= limit]
        expected, cur, size = [], [], 0
        for i, t in enumerate(tensors):  # the IPC rule, spelled out
            if size and size + t.nbytes > limit:
                expected.append(cur)
                cur, size = [], 0
            cur.append(i)
            size += t.nbytes
        if cur:
            expected.append(cur)
        chunks = list(
            packed_ipc_producer(
                iterator=iter([(str(i), t) for i, t in enumerate(tensors)]),
                gpu_uuid="uuid",
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=limit,
            )
        )
        assert [[int(n) for n in c.names] for c in chunks] == expected
        for c in chunks:
            ts = [tensors[int(n)] for n in c.names]
            assert c.tensor_sizes == [t.nbytes for t in ts]
            assert c.shapes == [list(t.shape) for t in ts]

    def test_tensor_over_the_buffer_raises(self):
        with pytest.raises(ValueError, match="exceeds buffer_size_bytes"):
            list(
                packed_ipc_producer(
                    iterator=iter([("big", torch.randn(100).cuda())]),
                    gpu_uuid="uuid",
                    post_iter_func=lambda x: x[1],
                    buffer_size_bytes=100,
                )
            )


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
            for name, tensor in result:
                # independent storage: the producer reuses its IPC buffer
                own = tensor.untyped_storage().nbytes() == tensor.nbytes
                assert own, f"{name} aliases the IPC buffer"
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

        `packed_ipc_producer` reuses a single IPC buffer for every
        chunk, so the producer must not be advanced until the consumer
        has finished reading the current chunk. We enforce that with an
        ack queue: the consumer puts `"ack"` after it has copied the
        chunk to CPU, and only then do we pull the next chunk from the
        generator.

        Returns `(num_chunks, results)`.
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

    def test_roundtrip_misaligned_entries(self):
        """A 3-byte entry shifts every later entry off its dtype's alignment;
        the consumer copies those instead of failing to view them."""
        params = [
            ("tag", torch.tensor([1, 2, 3], dtype=torch.uint8).cuda()),
            ("w", torch.randn(4, 8, dtype=torch.float32).cuda()),
            ("b", torch.randn(5, dtype=torch.bfloat16).cuda()),
            ("i", torch.arange(3, dtype=torch.int64).cuda()),
        ]
        num_chunks, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=self._get_gpu_uuid(),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            ),
            torch.cuda.current_device(),
        )
        assert num_chunks == 1
        for (orig_name, orig_tensor), (res_name, res_tensor) in zip(
            params, result, strict=True
        ):
            assert orig_name == res_name
            assert res_tensor.dtype == orig_tensor.dtype
            assert torch.equal(res_tensor, orig_tensor.cpu())

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


# --- Chunk layout and wire segments ---

from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

from vllm.distributed.weight_transfer import packed_tensor as pt  # noqa: E402
from vllm.distributed.weight_transfer.nccl_engine import (  # noqa: E402
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
    _coalesce,
    _pairs,
)


def _layout_bytes(layout: pt.ChunkLayout, tensors: list[torch.Tensor]) -> list[bytes]:
    """Each chunk's bytes when `tensors` are packed with `layout` (CPU)."""
    out = []
    for k, pieces in enumerate(layout.chunks):
        buf = torch.zeros(layout.sizes[k], dtype=torch.uint8)
        for piece, view in zip(pieces, layout.views(k, buf)):
            view.copy_(pt._source(piece, tensors[piece.index], view))
        out.append(buf.numpy().tobytes())
    return out


def _mixed_tensors(seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 255, (3,), dtype=torch.uint8, generator=g),  # misaligns
        torch.randn(40, 8, generator=g),
        torch.randn(7, generator=g).to(torch.bfloat16),
        torch.randn(64, 32, generator=g),
        torch.randint(-5, 5, (9,), dtype=torch.int64, generator=g),
        torch.randn(5, 3, generator=g, dtype=torch.float64),
    ]


class TestChunkLayout:
    @pytest.mark.parametrize("limit", [16, 100, 1000, 4096, 1 << 20])
    def test_without_segments_matches_packing_before_segments(self, limit):
        """With segments off the wire bytes are what earlier versions packed
        (tensors back to back, a chunk takes tensors until it holds more than
        the buffer), so a peer that predates segments interoperates."""
        tensors = _mixed_tensors()
        layout = pt.chunk_layout(((t.shape, t.dtype) for t in tensors), limit)
        expected, cur = [], []
        for t in tensors:
            cur.append(t.contiguous().view(-1).view(torch.uint8))
            if sum(c.numel() for c in cur) > limit:
                expected.append(torch.cat(cur).numpy().tobytes())
                cur = []
        if cur:
            expected.append(torch.cat(cur).numpy().tobytes())
        assert _layout_bytes(layout, tensors) == expected
        assert all(p.rows is None for chunk in layout.chunks for p in chunk)

    def test_segments_split_only_over_the_buffer(self):
        limit = 1000
        big = torch.randn(10, 100)  # 4000 B, rows of 400 B
        mid = torch.randn(15, 25)  # 1500 B, rows of 100 B
        small = torch.randn(10, 25)  # 1000 B: fits, whole
        layout = pt.chunk_layout(
            [(t.shape, t.dtype) for t in (big, mid, small)], limit, segments=True
        )

        def rows(i):
            return [p.rows for chunk in layout.chunks for p in chunk if p.index == i]

        assert rows(0) == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        assert rows(1) == [(0, 10), (10, 15)]
        assert rows(2) == [None]
        # no piece is larger than the buffer, so no chunk exceeds twice it
        assert all(p.nbytes <= limit for chunk in layout.chunks for p in chunk)
        assert max(layout.sizes) <= 2 * limit
        off = pt.chunk_layout([(big.shape, big.dtype)], limit)
        assert [p.rows for chunk in off.chunks for p in chunk] == [None]

    def test_views_alias_the_buffer_and_misaligned_is_raw(self):
        layout = pt.chunk_layout(
            [((3,), torch.uint8), ((2, 2), torch.float32), ((4,), torch.uint8)], 1024
        )
        buf = torch.zeros(layout.sizes[0], dtype=torch.uint8)
        tag, w, tail = layout.views(0, buf)
        assert tag.data_ptr() == buf.data_ptr() and tag.dtype == torch.uint8
        assert w.dtype == torch.uint8 and w.numel() == 16  # offset 3: raw bytes
        assert tail.data_ptr() == buf.data_ptr() + 19
        piece = layout.chunks[0][1]
        w.copy_(torch.arange(4.0).view(-1).view(torch.uint8))
        got = pt._unpacked(w, piece)
        assert got.dtype == torch.float32 and got.shape == (2, 2)
        assert torch.equal(got.view(-1), torch.arange(4.0))
        assert got.untyped_storage().data_ptr() != buf.untyped_storage().data_ptr()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_scalar_round_trips(self, dtype):
        tensors = [torch.randn(2, 3, dtype=dtype), torch.tensor(1.5, dtype=dtype)]
        layout = pt.chunk_layout(((t.shape, t.dtype) for t in tensors), 1024)
        [raw] = _layout_bytes(layout, tensors)
        buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        for piece, view in zip(layout.chunks[0], layout.views(0, buf)):
            got = pt._unpacked(view, piece)
            assert got.shape == tensors[piece.index].shape
            torch.testing.assert_close(got, tensors[piece.index], rtol=0, atol=0)

    def test_chunks_keep_order_across_several(self):
        tensors = [torch.randn(50, 50) for _ in range(6)]  # 10 kB each
        layout = pt.chunk_layout(((t.shape, t.dtype) for t in tensors), 12_000)
        assert len(layout.chunks) > 1
        assert [p.index for chunk in layout.chunks for p in chunk] == list(range(6))
        assert all(size > 12_000 for size in layout.sizes[:-1])

    def test_row_over_the_buffer_travels_whole_and_warns(self):
        with patch.object(pt.logger, "warning_once") as warn:
            assert pt.wire_segments([3, 1000], torch.float32, 1000) is None
        warn.assert_called_once()

    def test_layout_keeps_full_specs(self):
        layout = pt.chunk_layout([((4, 5), torch.float16)], 8, segments=True)
        assert layout.specs == [((4, 5), torch.float16)]


def _box(*ranges):
    return tuple(slice(a, b) for a, b in ranges)


class TestPlannedHelpers:
    def test_pairs_whole_tensor(self):
        view = torch.arange(12.0).view(3, 4)
        dst = torch.empty(2, 4)
        [(d, s)] = _pairs([(_box((1, 3), (0, 4)), dst)], None, view)
        assert d is dst
        torch.testing.assert_close(s, view[1:3])

    def test_pairs_segment_rows_onto_boxes(self):
        full = torch.arange(40.0).view(10, 4)
        dsts = [torch.empty(5, 4), torch.empty(5, 4)]  # two ranks' rows
        ops = [(_box((0, 5), (0, 4)), dsts[0]), (_box((5, 10), (0, 4)), dsts[1])]
        for lo, hi in [(0, 3), (3, 7), (7, 10)]:  # segments crossing the boxes
            for d, s in _pairs(ops, (lo, hi), full[lo:hi]):
                d.copy_(s)
        torch.testing.assert_close(torch.cat(dsts), full)

    def test_pairs_reshaped_write_needs_the_whole_tensor(self):
        full = torch.arange(24.0).view(6, 4)
        dst = torch.empty(4, 6)  # box (6, 4) written as (4, 6)
        assert _pairs([(_box((0, 6), (0, 4)), dst)], (0, 3), full[:3]) is None

    def test_coalesce_merges_evenly_spaced_and_keeps_w1_w3_apart(self):
        experts, n, k = 6, 4, 8
        src = torch.randn(experts, 2, n, k)  # (expert, w1|w3, n, k) as received
        w13 = torch.zeros(experts, 2 * n, k)
        dsts, srcs, roles = [], [], []
        for e in range(experts):
            for j, role in enumerate(("w1", "w3")):
                dsts.append(w13[e, j * n : (j + 1) * n])
                srcs.append(src[e, j])
                roles.append(f"experts.#.{role}")
        md, ms = _coalesce(dsts, srcs, roles)
        assert len(md) == 2  # one strided copy per role
        for d, s in zip(md, ms):
            d.copy_(s)
        torch.testing.assert_close(w13, src.reshape(experts, 2 * n, k))


class SLOT_KEY:  # noqa: N801
    """Stands for a communicator: any weakly referenceable object."""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSegmentedRoundtrip:
    def _roundtrip(self, tensors, limit, segments, num_buffers=2):
        producer_group = MockCommunicationGroup()
        pairs = [(f"t{i}", t) for i, t in enumerate(tensors)]
        packed_nccl_broadcast_producer(
            iterator=iter(pairs),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=limit,
            num_buffers=num_buffers,
            segments=segments,
        )
        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )
        got: dict[str, torch.Tensor] = {}

        def post_unpack_func(items):
            for name, t in items:
                got[name] = t.clone()

        packed_nccl_broadcast_consumer(
            iterator=iter([(n, (list(t.shape), t.dtype)) for n, t in pairs]),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=limit,
            num_buffers=num_buffers,
            segments=segments,
        )
        torch.accelerator.synchronize()
        for name, t in pairs:
            assert got[name].dtype == t.dtype and got[name].shape == t.shape
            assert torch.equal(got[name], t), name
        return producer_group

    @pytest.mark.parametrize("segments", [False, True])
    def test_roundtrip_with_segments_and_misaligned_entries(self, segments):
        tensors = [t.cuda() for t in _mixed_tensors(1)]
        tensors.insert(2, torch.randn(50, 16, device="cuda"))  # 3200 B
        tensors.insert(3, torch.randint(0, 9, (5,), dtype=torch.uint8).cuda())
        tensors.append(torch.randn(30, 6, device="cuda", dtype=torch.float64))
        group = self._roundtrip(tensors, limit=512, segments=segments)
        if segments:  # no chunk exceeds twice the buffer
            assert max(t.numel() for t in group.broadcasted_tensors) <= 2 * 512

    def test_split_tensor_is_joined_without_a_second_copy(self):
        """The consumer writes segments into one tensor of the full size:
        peak growth is about one tensor plus the receive buffers, not two."""
        mib = 1 << 20
        big = torch.randn(64, 256 * 1024, device="cuda")  # 64 MiB
        limit = 2 * mib
        producer_group = MockCommunicationGroup()
        packed_nccl_broadcast_producer(
            iterator=iter([("big", big)]),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=limit,
            segments=True,
        )
        group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
        torch.accelerator.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        seen = []
        packed_nccl_broadcast_consumer(
            iterator=iter([("big", ([64, 256 * 1024], torch.float32))]),
            group=group,
            src=0,
            post_unpack_func=lambda items: seen.extend(
                t.sum().item() for _, t in items
            ),
            buffer_size_bytes=limit,
            segments=True,
        )
        torch.accelerator.synchronize()
        assert seen == [pytest.approx(big.sum().item(), rel=1e-4)]
        growth = torch.cuda.max_memory_allocated() - base
        assert growth < big.nbytes + 8 * limit, growth / mib

    def test_planned_producer_checks_the_layout(self):
        t = torch.randn(4, 4, device="cuda")
        layout = pt.chunk_layout([((4, 4), torch.float32)], 1024)
        with pytest.raises(ValueError, match="layout it is sent with expects"):
            packed_nccl_broadcast_producer(
                iterator=iter([("x", t.half())]),
                group=MockCommunicationGroup(),
                src=0,
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=1024,
                layout=layout,
            )

    def test_slot_buffers_release_and_regrow_keep_views(self):
        key = SLOT_KEY()
        bufs = pt.slot_buffers(2, 1024, key)
        view = bufs[0][:16].view(torch.float32)
        pt.release_slot_buffers(key)
        assert bufs[0].untyped_storage().nbytes() == 0
        again = pt.slot_buffers(2, 512, key)
        assert again is bufs and bufs[0].untyped_storage().nbytes() == 1024
        view.fill_(3.0)  # the old view writes the regrown storage
        assert bufs[0][:16].view(torch.float32).eq(3.0).all()
        assert pt.slot_buffers(2, 512, SLOT_KEY()) is not bufs  # per group


class _ChunkGroup:
    """Fills each receive with the next pre-packed chunk, on the current
    stream; optionally the first one only after a GPU sleep."""

    def __init__(self, chunks: list[torch.Tensor], delay_first: bool = False):
        self.chunks, self.delay_first, self.calls = chunks, delay_first, 0

    def broadcast(self, tensor, src):
        if self.calls == 0 and self.delay_first:
            torch.cuda._sleep(SLEEP_CYCLES)
        tensor.copy_(self.chunks[self.calls])
        self.calls += 1


class _Plan:
    def __init__(self, dests):
        self.dests = dests

    def get_dest(self, name):
        return self.dests.get(name)


def _planned_engine(group, limit, model=None, segments=True):
    return SimpleNamespace(
        packed_num_buffers=2,
        packed_buffer_size_bytes=limit,
        packed_wire_segments=segments,
        model_update_group=group,
        model=model,
        device=torch.device("cuda"),
    )


def _receive_planned(engine, names, tensors, plan):
    info = NCCLWeightTransferUpdateInfo(
        names=names,
        dtype_names=[str(t.dtype).removeprefix("torch.") for t in tensors],
        shapes=[list(t.shape) for t in tensors],
    )
    for name in ("_planned_layout", "_receive_planned_chunks"):
        setattr(engine, name, getattr(NCCLWeightTransferEngine, name).__get__(engine))
    NCCLWeightTransferEngine._receive_planned(engine, info, 0, plan)


def _packed_chunks(tensors, limit, segments=True):
    layout = pt.chunk_layout(((t.shape, t.dtype) for t in tensors), limit, segments)
    cpu = [t.cpu() for t in tensors]
    return [
        torch.frombuffer(bytearray(b), dtype=torch.uint8).cuda()
        for b in _layout_bytes(layout, cpu)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestConsumerStreamOrdering:
    def test_consumer_waits_for_the_callers_queued_work(self):
        """Work the caller queued before the receive (begin's padding fill)
        runs before the loads, so it can't erase them."""
        weight = torch.randn(1024, device="cuda")
        param = torch.empty(1024, device="cuda")
        group = _ChunkGroup(_packed_chunks([weight], 1 << 16, segments=False))
        torch.accelerator.synchronize()

        def post_unpack_func(items):
            [(_, t)] = items
            param.copy_(t)

        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(SLEEP_CYCLES)
            param.zero_()
            packed_nccl_broadcast_consumer(
                iterator=iter([("w", ([1024], torch.float32))]),
                group=group,
                src=0,
                post_unpack_func=post_unpack_func,
                buffer_size_bytes=1 << 16,
            )
        torch.accelerator.synchronize()
        assert torch.equal(param, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPlannedReceive:
    def test_copies_split_misaligned_cast_and_fallbacks(self):
        limit = 4096
        g = torch.Generator().manual_seed(3)
        tag = torch.randint(0, 255, (3,), dtype=torch.uint8, generator=g)
        experts = torch.randn(16, 8, 32, generator=g)  # 16 KiB: split
        bias = torch.randn(10, generator=g)  # misaligned after `tag`
        w = torch.randn(512, 16, generator=g)  # 32 KiB, split, reshaped: joined
        fb_small = torch.randn(5, generator=g)
        names = ["tag", "experts", "bias", "w", "fb"]
        tensors = [t.cuda() for t in (tag, experts, bias, w, fb_small)]

        local = torch.zeros(8, 8, 32, device="cuda")  # experts 4..12 on this rank
        bias_dst = torch.zeros(10, device="cuda", dtype=torch.bfloat16)  # a cast
        w_dst = torch.zeros(16, 512, device="cuda")  # written as (16, 512)
        tag_dst = torch.zeros(3, device="cuda", dtype=torch.uint8)
        plan = _Plan(
            {
                "tag": [(((0, 3),), tag_dst)],
                "experts": [(((4, 12), (0, 8), (0, 32)), local)],
                "bias": [(((0, 10),), bias_dst)],
                "w": [(((0, 512), (0, 16)), w_dst)],
            }
        )
        loaded: list[tuple[str, torch.Tensor]] = []
        model = SimpleNamespace(load_weights=lambda ws: loaded.extend(ws))
        group = _ChunkGroup(_packed_chunks(tensors, limit))
        _receive_planned(_planned_engine(group, limit, model), names, tensors, plan)
        torch.accelerator.synchronize()
        assert torch.equal(tag_dst, tensors[0])
        assert torch.equal(local, tensors[1][4:12])
        torch.testing.assert_close(bias_dst, tensors[2].to(torch.bfloat16))
        assert torch.equal(w_dst, tensors[3].reshape(16, 512))
        assert [n for n, _ in loaded] == ["fb"]
        assert torch.equal(loaded[0][1], tensors[4])

    def test_copies_wait_for_the_callers_queued_work(self):
        """Work the caller queued before the receive (begin's padding fill)
        runs before the copies into the model, so it can't erase them."""
        limit = 1 << 16
        weight = torch.randn(1024, device="cuda")
        param = torch.empty(1024, device="cuda")
        plan = _Plan({"w": [(((0, 1024),), param)]})
        group = _ChunkGroup(_packed_chunks([weight], limit))
        engine = _planned_engine(group, limit)
        # A first update sizes the shared buffers: growing them later would
        # synchronize the caller's stream and hide a missing wait.
        _receive_planned(engine, ["w"], [weight], plan)
        group.calls = 0
        torch.accelerator.synchronize()
        # a side stream, as a worker's current stream is: the legacy default
        # stream would order everything implicitly
        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(SLEEP_CYCLES)
            param.zero_()  # the fill, queued behind the sleep
            _receive_planned(engine, ["w"], [weight], plan)
        torch.accelerator.synchronize()
        assert torch.equal(param, weight)
