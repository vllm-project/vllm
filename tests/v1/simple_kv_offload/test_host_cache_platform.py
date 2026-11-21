# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The generic KV-cache offload path must follow the active platform.

On non-CUDA platforms the host cache is pageable (no CUDA host registration)
and transfers cannot go through the cuMemcpyBatchAsync backend, so the worker
installs an asynchronous torch-copy backend instead. Without that split,
EngineCore dies at host-cache allocation with `AssertionError: Torch not
compiled with CUDA enabled`. That backend copies KV blocks between the GPU
cache and the host cache on a background thread: it gathers the source blocks,
moves them to the destination device and writes them there; disk mode has no
such fallback and must refuse a non-CUDA platform.
"""

from __future__ import annotations

import gc
import threading
import time
import types
import weakref
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.v1.simple_kv_offload.copy_backend import TorchCopyBackend

# Pure host-side unit tests: no distributed or device state to tear down.
pytestmark = pytest.mark.skip_global_cleanup

NUM_BLOCKS = 8


class _NoMixedDeviceOps(TorchDispatchMode):
    """Fail on any op whose tensor arguments span more than one device.

    Real tensors reject a non-scalar write whose source lives on another
    device ("Expected all tensors to be on the same device"), but meta kernels
    skip that check and accept the block setitem. Watching the dispatched ops
    is what turns a meta-device run into a device-alignment test.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        devices: set[str] = set()

        def walk(obj) -> None:
            if isinstance(obj, torch.Tensor):
                devices.add(str(obj.device))
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    walk(item)

        walk(args)
        walk(kwargs)
        if len(devices) > 1:
            raise RuntimeError(f"cross-device op {func}: {sorted(devices)}")
        return func(*args, **kwargs)


def _fake_platform(is_cuda_alike: bool):
    platform = MagicMock()
    platform.is_cuda_alike.return_value = is_cuda_alike
    platform.device_type = "xpu"
    return platform


def _gpu_caches() -> dict[str, torch.Tensor]:
    return {
        "layer0": torch.empty((NUM_BLOCKS, 2, 32, 16), dtype=torch.bfloat16),
        "layer1": torch.empty((NUM_BLOCKS, 2, 32, 16), dtype=torch.bfloat16),
    }


def _patch_platform(monkeypatch, is_cuda_alike: bool):
    from vllm.v1.simple_kv_offload import worker as worker_mod

    platform = _fake_platform(is_cuda_alike)
    monkeypatch.setattr("vllm.platforms.current_platform", platform)
    # The worker resolves the platform through its own module binding.
    monkeypatch.setattr(worker_mod, "current_platform", platform, raising=False)
    return worker_mod


def _stub_worker(worker_mod) -> Any:
    """A worker object with only the state the CPU-mode init touches."""
    worker = object.__new__(worker_mod.SimpleCPUOffloadWorker)
    worker.num_cpu_blocks = NUM_BLOCKS
    worker.cpu_hugepage_size = None
    worker.load_stream = None
    worker.store_stream = None
    worker.cpu_kv_caches = None
    worker._backend = None
    worker._store_compute_done = None
    worker._store_events = []
    worker._load_events = []
    worker._connector_metadata = None
    return worker


def _forbid_cuda(monkeypatch) -> MagicMock:
    """Any torch.cuda.cudart() call fails like the XPU torch build."""
    cudart = MagicMock(
        side_effect=AssertionError("Torch not compiled with CUDA enabled")
    )
    monkeypatch.setattr(torch.cuda, "cudart", cudart)
    return cudart


def _wait_for_events(
    events: list[tuple[int, Any]], count: int, timeout: float = 30.0
) -> None:
    """Poll until the background thread has posted `count` completions."""
    deadline = time.monotonic() + timeout
    while len(events) < count:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"only {len(events)}/{count} copy events posted in {timeout}s"
            )
        time.sleep(0.001)
    for _, event in events:
        event.synchronize()


class _GateEvent:
    """CopyEvent stand-in whose synchronize() blocks until released.

    Lets a test hold the background thread inside the wait before any byte
    moves, proving launch_copy() returned first.
    """

    def __init__(self) -> None:
        self._gate = threading.Event()
        self.synchronize_calls = 0

    def query(self) -> bool:
        return self._gate.is_set()

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        assert self._gate.wait(timeout=30.0), "gate never released"

    def release(self) -> None:
        self._gate.set()


class _ArmDeviceCheckOnCopyThread:
    """wait_event that arms _NoMixedDeviceOps on the copy thread.

    Dispatch modes are thread-local, so wrapping launch_copy() in the mode on
    the test thread would watch nothing: the copy runs on the backend's
    background thread. Its synchronize() is guaranteed to run there before
    any copy op, which is where the mode has to be entered; the mode stays
    armed on that daemon thread, which the test shuts down afterwards.
    """

    def __init__(self) -> None:
        self._mode = _NoMixedDeviceOps()

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        self._mode.__enter__()


def test_non_cuda_platform_allocates_pageable_caches_without_torch_cuda(monkeypatch):
    """The regression: allocation must not reach torch.cuda on a CUDA-less torch."""
    worker_mod = _patch_platform(monkeypatch, is_cuda_alike=False)
    cudart = _forbid_cuda(monkeypatch)
    pin_tensor = MagicMock()
    monkeypatch.setattr(worker_mod, "pin_tensor", pin_tensor, raising=False)

    mappings: list[torch.Tensor] = []
    alloc_host_buffer = worker_mod.alloc_host_buffer

    def tracking_alloc(nbytes: int, hugepage_size: int | None = None):
        mapping = alloc_host_buffer(nbytes, hugepage_size)
        mappings.append(mapping)
        return mapping

    monkeypatch.setattr(worker_mod, "alloc_host_buffer", tracking_alloc)

    caches = _stub_worker(worker_mod)._alloc_cpu_caches(_gpu_caches(), pin_memory=True)

    expected = _gpu_caches()
    assert set(caches) == set(expected)
    for name, cache in caches.items():
        assert cache.shape == (NUM_BLOCKS,) + expected[name].shape[1:]
        assert cache.dtype == expected[name].dtype
        assert cache.nbytes == expected[name].nbytes
        assert cache.is_pinned() is False

    assert not cudart.called
    assert not pin_tensor.called

    # Pageable means plain host memory: dropping the per-layer views frees the
    # mapping itself, not just one view of it.
    assert len(mappings) == 1
    mapping = weakref.ref(mappings.pop())
    del caches
    gc.collect()
    assert mapping() is None


def test_non_cuda_platform_installs_torch_copy_backend(monkeypatch):
    """CPU mode on XPU must not build the CUDA DMA backend or CUDA streams."""
    worker_mod = _patch_platform(monkeypatch, is_cuda_alike=False)
    cudart = _forbid_cuda(monkeypatch)
    dma = MagicMock()
    monkeypatch.setattr(worker_mod, "DmaCopyBackend", dma, raising=False)

    stub = _stub_worker(worker_mod)
    stub._init_cpu_mode(
        _gpu_caches(), NUM_BLOCKS * 2 * 32 * 16 * 2, torch.device("cpu")
    )

    assert isinstance(stub._backend, TorchCopyBackend)
    assert not dma.called
    assert not cudart.called
    assert stub.load_stream is None and stub.store_stream is None
    assert set(stub.cpu_kv_caches) == {"layer0", "layer1"}


def test_cuda_alike_platform_keeps_dma_backend(monkeypatch):
    """Counter-case: CUDA/HIP must keep host registration and the DMA backend."""
    worker_mod = _patch_platform(monkeypatch, is_cuda_alike=True)
    pin_tensor = MagicMock()
    monkeypatch.setattr(worker_mod, "pin_tensor", pin_tensor)
    monkeypatch.setattr(worker_mod, "unpin_tensor", MagicMock())
    dma = MagicMock()
    monkeypatch.setattr(worker_mod, "DmaCopyBackend", dma)
    # The pin call is gated on PIN_MEMORY (true only where CUDA pinning
    # exists); this test simulates the CUDA side of that gate.
    monkeypatch.setattr(worker_mod, "PIN_MEMORY", True)

    stub = _stub_worker(worker_mod)
    stub._init_cpu_mode(
        _gpu_caches(), NUM_BLOCKS * 2 * 32 * 16 * 2, torch.device("cpu")
    )

    assert dma.return_value.init.called
    assert pin_tensor.called
    registered_bytes = sum(call.args[0].nbytes for call in pin_tensor.call_args_list)
    assert registered_bytes == sum(c.nbytes for c in stub.cpu_kv_caches.values())
    assert not isinstance(stub._backend, TorchCopyBackend)


def test_non_cuda_store_path_uses_accelerator_stream(monkeypatch):
    """Store submission must not call torch.cuda on a non-CUDA platform."""
    worker_mod = _patch_platform(monkeypatch, is_cuda_alike=False)
    cudart = _forbid_cuda(monkeypatch)

    gpu = {"k": torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(1).repeat(1, 4)}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}
    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    worker = _stub_worker(worker_mod)
    worker._backend = backend
    worker._connector_metadata = types.SimpleNamespace(
        store_gpu_blocks=[3], store_cpu_blocks=[0], store_event=7
    )

    current_stream = MagicMock(name="accelerator.current_stream")
    monkeypatch.setattr(torch.accelerator, "current_stream", current_stream)
    event = MagicMock()
    monkeypatch.setattr(worker_mod.torch, "Event", MagicMock(return_value=event))

    worker.wait_for_save()

    assert not cudart.called
    current_stream.assert_called_once_with()
    event.record.assert_called_once_with(current_stream.return_value)
    # The store now runs on the backend's background thread: wait for its
    # completion event instead of assuming launch_copy() already moved bytes.
    _wait_for_events(worker._store_events, 1)
    assert (cpu["k"][0] == gpu["k"][3]).all()
    backend.shutdown()


def test_torch_copy_backend_round_trips_selected_blocks():
    """The fallback moves exactly the requested blocks, both directions."""
    gpu = {
        "k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32),
        "v": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32),
    }
    cpu = {
        "k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32),
        "v": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32),
    }
    for block in range(NUM_BLOCKS):
        gpu["k"][block] = block
        gpu["v"][block] = 100 + block

    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    events: list[tuple[int, Any]] = []
    backend.launch_copy([1, 3], [6, 7], is_store=True, event_idx=0, events_list=events)
    # launch_copy() only enqueues now; the bytes move on the background thread.
    _wait_for_events(events, 1)
    assert (cpu["k"][6] == 1).all() and (cpu["k"][7] == 3).all()
    assert (cpu["v"][6] == 101).all() and (cpu["v"][7] == 103).all()
    assert (cpu["k"][0] == 0).all()  # untouched blocks stay put
    assert len(events) == 1 and events[0][0] == 0 and events[0][1].query()

    for block in [6, 7]:
        gpu["k"][block] = -1
    backend.launch_copy([6, 7], [2, 4], is_store=False, event_idx=1, events_list=events)
    _wait_for_events(events, 2)
    assert (gpu["k"][2] == 1).all() and (gpu["k"][4] == 3).all()
    assert len(events) == 2 and events[-1][1].query()
    backend.shutdown()


def test_torch_copy_backend_load_moves_blocks_to_the_gpu_device():
    """Load: the gather has to reach the destination device before the write.

    The GPU caches live on the meta device, so this runs on a CPU-only host.
    Every dispatched op is checked for mixed-device arguments: the block
    setitem the backend used to do dispatches index_put_ with a meta
    destination and a CPU source, which only meta's lenient kernels accept.
    """
    gpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32, device="meta")}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}

    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    events: list[tuple[int, Any]] = []
    # The mode is armed on the copy thread via the wait_event: dispatch modes
    # are thread-local, so entering the mode around launch_copy() on this
    # thread would watch nothing now that the copy runs on the background
    # thread. A mixed-device op regresses as a copy-thread failure, which
    # surfaces at shutdown instead of an event.
    backend.launch_copy(
        [6, 7],
        [2, 4],
        is_store=False,
        event_idx=0,
        events_list=events,
        wait_event=_ArmDeviceCheckOnCopyThread(),  # type: ignore[arg-type]
    )

    _wait_for_events(events, 1)
    assert len(events) == 1 and events[0][0] == 0 and events[0][1].query()
    backend.shutdown()


def test_torch_copy_backend_store_moves_blocks_to_the_host_device():
    """Store: the same gather-move-write, in the host direction.

    Meta tensors hold no data, so the copy into the host cache stops at
    "Cannot copy out of meta tensor" -- reaching that message is the proof the
    gathered blocks were moved to the destination device. A mixed-device op
    would be rejected before it, and was before the fix. The copy runs on the
    background thread now, so the failure surfaces at shutdown (and on later
    launches) instead of out of launch_copy(); no completion is posted.
    """
    gpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32, device="meta")}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}

    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    events: list[tuple[int, Any]] = []
    backend.launch_copy(
        [1, 3],
        [6, 7],
        is_store=True,
        event_idx=0,
        events_list=events,
        wait_event=_ArmDeviceCheckOnCopyThread(),  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="copy thread failed") as excinfo:
        backend.shutdown()
    # The chained cause is the proof: the move reached the destination device.
    assert "meta tensor" in str(excinfo.value.__cause__)
    assert events == []
    with pytest.raises(RuntimeError, match="copy thread failed"):
        backend.launch_copy(
            [1, 3], [6, 7], is_store=True, event_idx=1, events_list=events
        )


def test_disk_mode_off_cuda_fails_fast(monkeypatch):
    """Disk mode stages through CUDA pinned buffers; off CUDA it must say so."""
    worker_mod = _patch_platform(monkeypatch, is_cuda_alike=False)
    cudart = _forbid_cuda(monkeypatch)

    stub = _stub_worker(worker_mod)
    stub.disk_path = "/nonexistent/kv_offload"
    stub.disk_buffer_slots = 2
    stub.use_page_cache = False
    with pytest.raises(RuntimeError, match="kv_offload_backend='disk'"):
        stub._init_disk_mode(
            _gpu_caches(), NUM_BLOCKS * 2 * 32 * 16 * 2, torch.device("cpu")
        )

    assert not cudart.called
    assert stub._backend is None


def test_launch_copy_returns_before_bytes_move():
    """launch_copy() hands off to the background thread and returns.

    The gate holds the copy thread inside wait_event.synchronize() before it
    can touch any byte, so everything asserted here runs while the copy is
    still queued: the destination is unchanged and no completion is posted.
    Releasing the gate then completes the same copy, proving the handoff.
    """
    gpu = {"k": torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(1).repeat(1, 4)}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}
    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    gate = _GateEvent()
    events: list[tuple[int, Any]] = []
    backend.launch_copy(
        [3],
        [0],
        is_store=True,
        event_idx=0,
        events_list=events,
        wait_event=gate,  # type: ignore[arg-type]
    )

    deadline = time.monotonic() + 30.0
    while gate.synchronize_calls == 0:
        assert time.monotonic() < deadline, "copy thread never reached the wait"
    assert (cpu["k"][0] == 0).all()  # still queued: no byte moved
    assert events == []  # and no completion posted

    gate.release()
    _wait_for_events(events, 1)
    assert (cpu["k"][0] == gpu["k"][3]).all()
    assert events[0][1].query()
    backend.shutdown()


def test_copy_completion_is_observable_via_poll():
    """A queued copy becomes visible through the posted completion event."""
    gpu = {"k": torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(1).repeat(1, 4)}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}
    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    events: list[tuple[int, Any]] = []
    backend.launch_copy([5], [1], is_store=True, event_idx=3, events_list=events)
    _wait_for_events(events, 1)
    assert len(events) == 1 and events[0][0] == 3 and events[0][1].query()
    assert (cpu["k"][1] == 5).all()
    backend.shutdown()


def test_store_waits_for_the_compute_done_event():
    """Stores stay ordered after compute: no byte moves before the wait fires."""
    gpu = {"k": torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(1).repeat(1, 4)}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}
    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    gate = _GateEvent()
    events: list[tuple[int, Any]] = []
    backend.launch_copy(
        [2],
        [4],
        is_store=True,
        event_idx=0,
        events_list=events,
        wait_event=gate,  # type: ignore[arg-type]
    )

    deadline = time.monotonic() + 30.0
    while gate.synchronize_calls == 0:
        assert time.monotonic() < deadline, "copy thread never reached the wait"
    # The thread is parked in the wait: the store must not have started.
    assert (cpu["k"][4] == 0).all()
    assert events == []

    gate.release()
    _wait_for_events(events, 1)
    assert (cpu["k"][4] == 2).all()
    backend.shutdown()


def test_shutdown_drains_queued_copies():
    """Shutdown joins the thread past the queued work instead of dropping it."""
    gpu = {
        "k": torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(1).repeat(1, 4),
    }
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}
    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    events: list[tuple[int, Any]] = []
    for idx in range(NUM_BLOCKS):
        backend.launch_copy(
            [idx], [idx], is_store=True, event_idx=idx, events_list=events
        )
    backend.shutdown()  # no waiting: draining is the shutdown's job

    assert len(events) == NUM_BLOCKS
    assert sorted(i for i, _ in events) == list(range(NUM_BLOCKS))
    assert all(event.query() for _, event in events)
    assert (cpu["k"] == gpu["k"]).all()


def test_stress_mixed_copies_match_the_synchronous_path():
    """Hundreds of queued mixed load/store copies land exactly as serial ones.

    The reference replays each transfer on plain per-block values in launch
    order -- the queue is FIFO, so the background thread applies the same
    order -- without touching the live tensors, keeping the comparison exact
    while copies are still in flight. Every row stays uniform, so one value
    per block models the whole transfer.
    """
    gen = torch.Generator().manual_seed(20261115)
    gpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}
    for block in range(NUM_BLOCKS):
        gpu["k"][block] = block
    ref_gpu_blocks = list(range(NUM_BLOCKS))
    ref_cpu_blocks = [0] * NUM_BLOCKS

    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    n_copies = 300
    events: list[tuple[int, Any]] = []
    for idx in range(n_copies):
        count = int(torch.randint(1, 5, (1,), generator=gen).item())
        src_blocks = torch.randperm(NUM_BLOCKS, generator=gen)[:count].tolist()
        dst_blocks = torch.randperm(NUM_BLOCKS, generator=gen)[:count].tolist()
        is_store = bool(torch.randint(0, 2, (1,), generator=gen).item())
        backend.launch_copy(src_blocks, dst_blocks, is_store, idx, events_list=events)
        if is_store:
            src_model, dst_model = ref_gpu_blocks, ref_cpu_blocks
        else:
            src_model, dst_model = ref_cpu_blocks, ref_gpu_blocks
        for src_block, dst_block in zip(src_blocks, dst_blocks):
            dst_model[dst_block] = src_model[src_block]

    _wait_for_events(events, n_copies)
    assert sorted(i for i, _ in events) == list(range(n_copies))
    for block in range(NUM_BLOCKS):
        assert (gpu["k"][block] == ref_gpu_blocks[block]).all()
        assert (cpu["k"][block] == ref_cpu_blocks[block]).all()
    backend.shutdown()
