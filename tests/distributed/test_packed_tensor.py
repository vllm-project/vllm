# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for packed tensor broadcasting functionality.

Unit tests for packed_nccl_broadcast_producer and packed_nccl_broadcast_consumer.
These utilities enable efficient batched tensor transfer over NCCL.
"""

import pytest
import torch

from vllm.distributed.weight_transfer.packed_tensor import (
    ALIGN,
    DEFAULT_PACKED_NUM_BUFFERS,
    Chunk,
    PackedIpcConsumer,
    Piece,
    _Unpacker,
    chunks,
    packed_ipc_producer,
    packed_nccl_broadcast_consumer,
    packed_nccl_broadcast_producer,
)
from vllm.utils.math_utils import round_down, round_up


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
    """Fills each receive with the next of `chunks`, the one at `delay` after
    a GPU sleep (`filled` is recorded after it)."""

    def __init__(self, chunks: list[torch.Tensor], delay: int | None = None):
        self.chunks, self.delay, self.calls = chunks, delay, 0
        self.filled = torch.Event()
        self.device = torch.device("cuda:0")

    def broadcast(self, tensor, src):
        if self.calls == self.delay:
            torch.cuda._sleep(SLEEP_CYCLES)
        tensor.copy_(self.chunks[self.calls])
        if self.calls == self.delay:
            self.filled.record()
        self.calls += 1


# Copied as bytes: CUDA `_foreach_copy_` does not implement them.
QUANT_DTYPES = [torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2]


def _randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    if dtype in QUANT_DTYPES:
        return torch.randint(0, 256, shape, dtype=torch.uint8).view(dtype)
    return torch.randn(*shape, dtype=dtype)


def create_mock_model_params(
    num_layers: int = 3,
    dtype: torch.dtype = torch.float32,
) -> list[tuple[str, torch.Tensor]]:
    """Create mock model parameters for testing."""
    params = []
    for i in range(num_layers):
        params.append((f"layer{i}.weight", _randn(10, 20, dtype=dtype)))
        params.append((f"layer{i}.bias", _randn(10, dtype=dtype)))
    params.append(("strided", _randn(20, 10, dtype=dtype).T))
    return params


def staged(params: list[tuple[str, torch.Tensor]]):
    """Yield `params` through one reused staging tensor, as a source may."""
    device = params[0][1].device
    stage = torch.empty(max(t.nbytes for _, t in params), dtype=torch.uint8)
    stage = stage.to(device)
    for name, t in params:
        stage[: t.nbytes].copy_(_flat(t))
        yield name, stage[: t.nbytes].view(t.dtype).view(t.shape)


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


def roundtrip(
    params: list[tuple[str, torch.Tensor]],
    buffer_size_bytes: int,
    stage: bool = False,
) -> MockCommunicationGroup:
    """Round-trip `params` (again behind a slow fill), sent through one staging
    tensor if `stage`, and check they arrive intact. Returns the producer's
    group."""
    producer_group = MockCommunicationGroup()
    packed_nccl_broadcast_producer(
        iterator=staged(params) if stage else iter(params),
        group=producer_group,
        src=0,
        post_iter_func=lambda x: x[1],
        buffer_size_bytes=buffer_size_bytes,
    )
    dst = {n: torch.empty(t.shape, dtype=t.dtype, device=t.device) for n, t in params}

    def consume(sleep: bool) -> bool:
        torch.accelerator.synchronize()
        if sleep:
            torch.cuda._sleep(SLEEP_CYCLES)
        for t in dst.values():
            _flat(t).zero_()
        filled = torch.Event()
        filled.record()
        fill_pending_at_load: list[bool] = []

        def post_unpack_func(tensor_list):
            fill_pending_at_load.append(not filled.query())
            for name, tensor in tensor_list:
                target = dst[name]
                assert (tensor.shape, tensor.dtype) == (target.shape, target.dtype)
                _flat(target).copy_(_flat(tensor))

        packed_nccl_broadcast_consumer(
            iterator=iter(create_state_dict_info(params).items()),
            group=MockConsumerCommunicationGroup(producer_group.broadcasted_tensors),
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=buffer_size_bytes,
        )
        torch.accelerator.synchronize()
        for name, t in params:
            assert torch.equal(_flat(dst[name]), _flat(t)), name
        return fill_pending_at_load[0]

    consume(sleep=False)
    # The first load can only overlap the fill if the first tensor completes
    # within the first `DEFAULT_PACKED_NUM_BUFFERS` chunks.
    layout = chunks(((t.shape, t.dtype) for _, t in params), buffer_size_bytes)
    first_load = next(
        k
        for k, chunk in enumerate(layout)
        if any(p.end == params[p.position][1].nbytes for p in chunk.pieces)
    )
    fill_pending = consume(sleep=True)
    if first_load < DEFAULT_PACKED_NUM_BUFFERS:
        assert fill_pending, "the fill finished before the first load"
    return producer_group


# --- Integration Tests: Producer-Consumer Roundtrip ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedBroadcastRoundtrip:
    """Test producer-consumer roundtrip behavior."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, *QUANT_DTYPES]
    )
    def test_roundtrip_different_dtypes(self, dtype):
        """Test roundtrip with different data types, also strided."""
        params = create_mock_model_params(num_layers=2, dtype=dtype)
        roundtrip([(n, t.cuda()) for n, t in params], buffer_size_bytes=1000)

    @pytest.mark.parametrize("stage", [False, True])
    def test_roundtrip_mixed_dtypes(self, stage):
        """Test roundtrip with mixed data types."""
        # Create params with mixed dtypes
        params = [
            ("layer1.weight", torch.randn(10, 20, dtype=torch.float32).cuda()),
            ("layer1.bias", torch.randn(10, dtype=torch.float16).cuda()),
            ("layer2.weight", torch.randn(20, 30, dtype=torch.bfloat16).cuda()),
        ]
        roundtrip(params, buffer_size_bytes=500, stage=stage)

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
    callback, as layerwise reload does."""

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
            # Exactly one tensor, so each tensor gets its own chunk.
            buffer_size_bytes=numel * 4,
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
            # Exactly one tensor, so each tensor gets its own chunk.
            buffer_size_bytes=numel * 4,
            num_buffers=num_buffers,
        )
        torch.accelerator.synchronize()

        assert group.calls == num_packs
        # Receives before the host next waits for the replay.
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
                device_index=torch.cuda.current_device(),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=12_000,
            )
        )

        assert len(chunks) > 1
        # each tensor is named once, by the chunk holding its first bytes
        assert [n for c in chunks for n in c.names] == [n for n, _ in params]

    def test_producer_ipc_handle_has_uuid(self):
        """The first chunk carries the handle, keyed by the given UUID; the
        rest carry none. Only the last chunk is marked last."""
        params = [("w", torch.randn(10, dtype=torch.float32).cuda())]

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid="my-gpu-uuid",
                device_index=torch.cuda.current_device(),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=16,
            )
        )

        assert len(chunks) == 3
        assert list(chunks[0].ipc_handle) == ["my-gpu-uuid"]
        assert [c.ipc_handle for c in chunks[1:]] == [None, None]
        assert [c.last for c in chunks] == [False, False, True]

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
                device_index=torch.cuda.current_device(),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        assert chunks[0].dtype_names == ["float32", "float16"]

    @pytest.mark.parametrize("names", [[], ["e"]])
    def test_producer_without_bytes_exports_nothing(self, names):
        """No tensors yield nothing; only empty ones, one chunk without a
        handle."""
        chunks = list(
            packed_ipc_producer(
                iterator=iter([(n, torch.empty(0, 3).cuda()) for n in names]),
                gpu_uuid="uuid",
                device_index=torch.cuda.current_device(),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=1000,
            )
        )
        assert [(c.names, c.ipc_handle, c.last) for c in chunks] == [
            (names, None, True)
        ][: len(names)]


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
        uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        consumer = PackedIpcConsumer(device_index, uuid)
        while True:
            cd = cmd_q.get()
            if cd is None:
                break
            result = consumer.receive(
                ipc_handle=cd["ipc_handle"],
                names=cd["names"],
                shapes=cd["shapes"],
                dtype_names=cd["dtype_names"],
                chunk_capacity=cd["chunk_capacity"],
                last=cd["last"],
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
        consumer.close()
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
        """Stream chunks through a child consumer, advancing the producer only
        after its ack; returns `(num_chunks, results)`."""
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
                        "chunk_capacity": chunk.chunk_capacity,
                        "last": chunk.last,
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

    @pytest.mark.parametrize("stage", [False, True])
    def test_roundtrip_basic(self, stage):
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
                iterator=staged(params) if stage else iter(params),
                gpu_uuid=gpu_uuid,
                device_index=torch.cuda.current_device(),
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

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, *QUANT_DTYPES]
    )
    def test_roundtrip_dtypes(self, dtype):
        """Test IPC roundtrip with different dtypes, also strided."""
        params = create_mock_model_params(num_layers=2, dtype=dtype)
        params_cuda = [(n, t.cuda()) for n, t in params]
        gpu_uuid = self._get_gpu_uuid()
        device_index = torch.cuda.current_device()

        _, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params_cuda),
                gpu_uuid=gpu_uuid,
                device_index=torch.cuda.current_device(),
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
            assert torch.equal(_flat(res_tensor), _flat(orig_tensor.cpu()))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_roundtrip_multiple_chunks(self, device):
        """Test IPC roundtrip across multiple chunks, also from CPU sources."""
        params = [
            (f"layer{i}.weight", torch.randn(100, 100, device=device)) for i in range(8)
        ]
        gpu_uuid = self._get_gpu_uuid()
        device_index = torch.cuda.current_device()

        num_chunks, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                device_index=torch.cuda.current_device(),
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
                device_index=torch.cuda.current_device(),
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

    @pytest.mark.parametrize("buffer_size_bytes", [10_000_000, 64, 50])
    def test_roundtrip_odd_sized_and_split_entries(self, buffer_size_bytes):
        """Odd-sized entries, a 0-dim one and a parameter; with a small buffer
        the parameter is split and the 192-byte stream ends in a chunk of only
        the empty tensor."""
        params = [
            ("tag", torch.tensor([1, 2, 3], dtype=torch.uint8).cuda()),
            ("w", torch.nn.Parameter(torch.randn(4, 8, device="cuda"))),
            ("b", torch.randn(5, dtype=torch.bfloat16).cuda()),
            ("s", torch.tensor(2.5).cuda()),
            ("i", torch.arange(2, dtype=torch.int64).cuda()),
            ("e", torch.empty(0, 3).cuda()),
        ]
        num_chunks, result = self._run_roundtrip(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=self._get_gpu_uuid(),
                device_index=torch.cuda.current_device(),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=buffer_size_bytes,
            ),
            torch.cuda.current_device(),
        )
        assert num_chunks == {10_000_000: 1, 64: 4, 50: 5}[buffer_size_bytes]
        for (orig_name, orig_tensor), (res_name, res_tensor) in zip(
            params, result, strict=True
        ):
            assert orig_name == res_name
            assert res_tensor.dtype == orig_tensor.dtype
            assert torch.equal(res_tensor, orig_tensor.detach().cpu())

    def test_consumer_wrong_uuid_raises(self):
        """Test that consumer raises ValueError for unknown GPU UUID."""
        params = [("w", torch.randn(10, dtype=torch.float32).cuda())]
        gpu_uuid = self._get_gpu_uuid()

        chunks = list(
            packed_ipc_producer(
                iterator=iter(params),
                gpu_uuid=gpu_uuid,
                device_index=torch.cuda.current_device(),
                post_iter_func=lambda x: x[1],
                buffer_size_bytes=10_000_000,
            )
        )

        c = chunks[0]
        fake_handle = {"fake-uuid-12345": c.ipc_handle[gpu_uuid]}

        with pytest.raises(ValueError, match="IPC handle not found"):
            PackedIpcConsumer(torch.cuda.current_device(), gpu_uuid).receive(
                ipc_handle=fake_handle,
                names=c.names,
                shapes=c.shapes,
                dtype_names=c.dtype_names,
                chunk_capacity=c.chunk_capacity,
                last=c.last,
            )


# --- Chunk layout ---


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def _layout_bytes(layout: list[Chunk], tensors: list[torch.Tensor]) -> list[bytes]:
    """Each chunk's bytes when `tensors` are packed with `layout` (CPU)."""
    out = []
    for nbytes, pieces, _ in layout:
        buf = torch.zeros(nbytes, dtype=torch.uint8)
        for p in pieces:
            buf[p.offset : p.offset + p.end - p.start] = _flat(tensors[p.position])[
                p.start : p.end
            ]
        out.append(buf.numpy().tobytes())
    return out


def _mixed_tensors(seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 255, (3,), dtype=torch.uint8, generator=g),  # odd size
        torch.randn(40, 8, generator=g),
        torch.randn(7, generator=g).to(torch.bfloat16),
        torch.tensor(1.5),  # 0-dim
        torch.empty(0, 4),  # no bytes
        torch.randn(64, 32, generator=g),
        torch.randint(-5, 5, (9,), dtype=torch.int64, generator=g),
        torch.randn(5, 3, generator=g, dtype=torch.float64),
    ]


class TestChunkLayout:
    @pytest.mark.parametrize("limit", [16, 100, 1 << 20])
    def test_layout_is_the_aligned_stream_cut_into_buffer_sized_chunks(self, limit):
        """The wire bytes are every tensor back to back, each starting
        `ALIGN`-aligned, cut into chunks of the buffer size (rounded down to
        `ALIGN`), without a chunk's trailing padding."""
        tensors = _mixed_tensors()
        layout = list(chunks(((t.shape, t.dtype) for t in tensors), limit))
        stream = b""
        for t in tensors:
            stream = stream.ljust(round_up(len(stream), ALIGN), b"\0")
            stream += _flat(t).numpy().tobytes()
        size = round_down(limit, ALIGN)
        expected = [stream[i : i + size] for i in range(0, len(stream), size)]
        got = _layout_bytes(layout, tensors)
        assert len(got) == len(expected)
        for b, e in zip(got, expected):
            assert b == e[: len(b)] and len(e) - len(b) < ALIGN
        # a tensor inside one chunk starts aligned, so it views as its dtype
        for _, pieces, _ in layout:
            assert all(p.offset % ALIGN == 0 for p in pieces)

    def test_a_full_chunk_is_yielded_before_the_next_spec_is_pulled(self):
        """Also when only the padding is left: the IPC consumer derives each
        chunk from the specs received with it and before."""
        pulled = []

        def specs():
            for spec in [((7,), torch.float32), ((2,), torch.float32)]:
                pulled.append(spec)
                yield spec

        layout = chunks(specs(), 32)
        assert next(layout) == Chunk(28, [Piece(0, 0, 28, 0)], full=True)
        assert len(pulled) == 1
        assert next(layout) == Chunk(8, [Piece(1, 0, 8, 0)], full=False)
        with pytest.raises(ValueError, match="at least 16 bytes"):
            next(chunks(specs(), 15))

    def test_specs_fed_per_chunk_give_the_same_layout(self):
        """As the IPC consumer feeds them; a chunk past the end raises."""
        specs = [(t.shape, t.dtype) for t in _mixed_tensors()]
        unpacker = _Unpacker(100, views=False)
        for chunk in chunks(iter(specs), 100):
            started = [specs[p.position] for p in chunk.pieces if p.start == 0]
            unpacker.feed([""] * len(started), started)
            assert unpacker.next_chunk() == chunk
        unpacker.feed([""], specs[:1])
        with pytest.raises(ValueError, match="past the end"):
            unpacker.next_chunk()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestChunkedRoundtrip:
    def test_roundtrip_with_split_and_odd_sized_tensors(self):
        """Including parameters requiring grad, split and whole."""
        tensors = [t.cuda() for t in _mixed_tensors(1)]
        tensors.insert(2, torch.nn.Parameter(torch.randn(50, 16, device="cuda")))
        tensors.insert(3, torch.nn.Parameter(torch.randn(5, device="cuda")))
        tensors.append(torch.randn(30, 6, device="cuda", dtype=torch.float64).t())
        params = [(f"t{i}", t) for i, t in enumerate(tensors)]
        group = roundtrip(params, buffer_size_bytes=512)
        assert max(t.numel() for t in group.broadcasted_tensors) <= 512

    @pytest.mark.parametrize(
        "first, buffer_size_bytes, sent",
        [
            (torch.randn(8), 16, [16, 16]),  # two full chunks, then none
            (torch.arange(10, dtype=torch.uint8), 64, [10]),  # no padding
        ],
    )
    def test_roundtrip_ending_with_empty_tensors(self, first, buffer_size_bytes, sent):
        """Trailing empty tensors add no padding, and a chunk of only empty
        tensors is not broadcast; they still arrive."""
        params = [
            ("first", first.cuda()),
            ("empty", torch.empty(0, 4, device="cuda")),
            ("none", torch.empty(0, device="cuda", dtype=torch.int8)),
        ]
        group = roundtrip(params, buffer_size_bytes=buffer_size_bytes)
        assert [t.numel() for t in group.broadcasted_tensors] == sent

    def test_unpack_waits_for_each_receive(self):
        """Each chunk is unpacked only after its (slow) receive ends."""
        big = torch.randn(8, 64, device="cuda")  # 2048 B: chunks of 1024, 1024
        group = MockConsumerCommunicationGroup(_packed_chunks([big], 1024), delay=1)
        got: list[torch.Tensor] = []
        fill_pending: list[bool] = []

        def post_unpack_func(items):
            fill_pending.append(not group.filled.query())
            got.extend(t.clone() for _, t in items)

        torch.accelerator.synchronize()
        packed_nccl_broadcast_consumer(
            iterator=iter([("big", ([8, 64], torch.float32))]),
            group=group,
            src=0,
            post_unpack_func=post_unpack_func,
            buffer_size_bytes=1024,
        )
        torch.accelerator.synchronize()
        assert fill_pending == [True], "the receive ended before the unpack"
        assert len(got) == 1 and torch.equal(got[0], big)

    def test_split_tensor_is_joined_without_a_second_copy(self):
        """The consumer writes a split tensor's bytes into one tensor of its
        full size: peak growth is about one tensor plus the receive buffers,
        also over two consecutive ones."""
        mib = 1 << 20
        shape = [32, 256 * 1024]  # 32 MiB
        params = [(n, torch.randn(shape, device="cuda")) for n in ("a", "b")]
        limit = 2 * mib
        producer_group = MockCommunicationGroup()
        packed_nccl_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            buffer_size_bytes=limit,
        )
        group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
        torch.accelerator.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        seen: list = []
        packed_nccl_broadcast_consumer(
            iterator=iter([(n, (shape, torch.float32)) for n, _ in params]),
            group=group,
            src=0,
            post_unpack_func=lambda items: seen.extend(
                t.sum().item() for _, t in items
            ),
            buffer_size_bytes=limit,
        )
        torch.accelerator.synchronize()
        assert seen == [pytest.approx(t.sum().item(), rel=1e-4) for _, t in params]
        growth = torch.cuda.max_memory_allocated() - base
        assert growth < params[0][1].nbytes + 8 * limit, growth / mib


def _packed_chunks(tensors, limit):
    layout = list(chunks(((t.shape, t.dtype) for t in tensors), limit))
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
        # The first run loads the kernels: loading one waits for the device
        # to go idle, which would end the caller's sleep before the receive.
        for run in range(2):
            group = MockConsumerCommunicationGroup(_packed_chunks([weight], 1 << 16))
            torch.accelerator.synchronize()
            zeroed = torch.Event()
            zero_pending: list[bool] = []

            def post_unpack_func(items, zeroed=zeroed, zero_pending=zero_pending):
                [(_, t)] = items
                zero_pending.append(not zeroed.query())
                param.copy_(t)

            with torch.cuda.stream(torch.cuda.Stream()):
                torch.cuda._sleep(SLEEP_CYCLES if run else 1)
                param.zero_()
                zeroed.record()
                packed_nccl_broadcast_consumer(
                    iterator=iter([("w", ([1024], torch.float32))]),
                    group=group,
                    src=0,
                    post_unpack_func=post_unpack_func,
                    buffer_size_bytes=1 << 16,
                )
            torch.accelerator.synchronize()
            assert torch.equal(param, weight)
        assert zero_pending == [True], "the caller's work ended before the load"
