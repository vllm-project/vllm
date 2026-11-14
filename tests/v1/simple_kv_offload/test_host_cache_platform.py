# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The generic KV-cache offload path must follow the active platform.

On non-CUDA platforms the host cache is pageable (no CUDA host registration)
and transfers cannot go through the cuMemcpyBatchAsync backend, so the worker
installs a synchronous torch-copy backend instead. Without that split,
EngineCore dies at host-cache allocation with `AssertionError: Torch not
compiled with CUDA enabled`. That backend copies KV blocks between the GPU
cache and the host cache: it gathers the source blocks, moves them to the
destination device and writes them there; disk mode has no such fallback and
must refuse a non-CUDA platform.
"""

from __future__ import annotations

import gc
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
    assert (cpu["k"][0] == gpu["k"][3]).all()  # store ran synchronously


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
    assert (cpu["k"][6] == 1).all() and (cpu["k"][7] == 3).all()
    assert (cpu["v"][6] == 101).all() and (cpu["v"][7] == 103).all()
    assert (cpu["k"][0] == 0).all()  # untouched blocks stay put
    assert len(events) == 1 and events[0][0] == 0 and events[0][1].query()

    for block in [6, 7]:
        gpu["k"][block] = -1
    backend.launch_copy([6, 7], [2, 4], is_store=False, event_idx=1, events_list=events)
    assert (gpu["k"][2] == 1).all() and (gpu["k"][4] == 3).all()
    assert len(events) == 2 and events[-1][1].query()


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
    with _NoMixedDeviceOps():
        backend.launch_copy(
            [6, 7], [2, 4], is_store=False, event_idx=0, events_list=events
        )

    assert len(events) == 1 and events[0][0] == 0 and events[0][1].query()


def test_torch_copy_backend_store_moves_blocks_to_the_host_device():
    """Store: the same gather-move-write, in the host direction.

    Meta tensors hold no data, so the copy into the host cache stops at
    "Cannot copy out of meta tensor" -- reaching that message is the proof the
    gathered blocks were moved to the destination device. A mixed-device op
    would be rejected before it, and was before the fix.
    """
    gpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32, device="meta")}
    cpu = {"k": torch.zeros((NUM_BLOCKS, 4), dtype=torch.int32)}

    backend = TorchCopyBackend()
    backend.init(gpu, cpu, torch.device("cpu"))

    with (
        _NoMixedDeviceOps(),
        pytest.raises(NotImplementedError, match="meta tensor"),
    ):
        backend.launch_copy([1, 3], [6, 7], is_store=True, event_idx=0, events_list=[])


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
