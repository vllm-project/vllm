# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side unit tests for SimpleCPUOffloadConnector.

Covers cross-stream synchronization for both transfer directions. Each transfer
runs on a dedicated stream and must be ordered after compute:
- GPU->CPU store must wait for the compute stream that *writes* the KV blocks,
  else it reads partially written / stale blocks (RAW, #45704).
- CPU->GPU load must wait for the compute stream that *reads* a reused block,
  else it overwrites the block mid-read (WAR); the loaded content clobbers the
  KV a still-running request is attending to.
"""

from __future__ import annotations

import time

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    build_params,
    pin_tensor,
)
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

NUM_BLOCKS = 64
BLOCK_BYTES = 4096
ITERS = 30
# Keep the compute stream busy so the KV write lands late; this makes the
# store-vs-compute race deterministic instead of timing-dependent.
SLEEP_CYCLES = 50_000_000


def _make_backend() -> tuple[DmaCopyBackend, torch.Tensor, torch.Tensor]:
    gpu = {"k": torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cuda")}
    cpu = {"k": torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cpu")}
    pin_tensor(cpu["k"])
    low_pri, _ = torch.cuda.Stream.priority_range()
    backend = DmaCopyBackend()
    backend.init(
        gpu,
        cpu,
        gpu["k"].device,
        torch.cuda.Stream(priority=low_pri),
        torch.cuda.Stream(priority=low_pri),
    )
    return backend, gpu["k"], cpu["k"]


def _drive_store(
    backend: DmaCopyBackend,
    gpu: torch.Tensor,
    cpu: torch.Tensor,
    *,
    with_barrier: bool,
) -> int:
    """Run ITERS store cycles; return how many landed corrupted in the CPU pool.

    Each cycle writes a unique value on a compute stream (after a deliberate
    delay) and then issues the GPU->CPU store. The store is issued *after* the
    write in host program order, mirroring the connector's deferred-store
    assumption. Only the compute-done event creates a real device-side
    happens-before edge.
    """
    block_ids = list(range(gpu.shape[0]))
    compute_stream = torch.cuda.Stream()
    corrupt = 0
    for it in range(ITERS):
        val = (it % 126) + 1  # 1..126; distinct from the zero-initialized pool
        with torch.cuda.stream(compute_stream):
            torch.cuda._sleep(SLEEP_CYCLES)
            gpu.fill_(val)

        wait_event = None
        if with_barrier:
            wait_event = torch.Event()
            wait_event.record(compute_stream)

        store_events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            block_ids,
            block_ids,
            is_store=True,
            event_idx=it,
            events_list=store_events,
            wait_event=wait_event,
        )

        deadline = time.time() + 10.0
        while not store_events and time.time() < deadline:
            time.sleep(0.0005)
        assert store_events, "background copy was never enqueued"
        store_events[0][1].synchronize()

        if int((cpu[:, 0].to(torch.int32) != val).sum().item()):
            corrupt += 1
    return corrupt


def test_store_orders_after_compute_write():
    """The store must wait for the compute event; without it, it races.

    Asserts both directions so the test is self-validating: the no-barrier
    control must actually corrupt (proving the race window is exercised), and
    the fixed path with the compute-done event must be clean.
    """
    backend, gpu, cpu = _make_backend()
    try:
        control = _drive_store(backend, gpu, cpu, with_barrier=False)
        fixed = _drive_store(backend, gpu, cpu, with_barrier=True)
    finally:
        backend.shutdown()

    assert control > 0, (
        "no-barrier store did not race the compute write; the test no longer "
        "exercises the hazard it is meant to guard"
    )
    assert fixed == 0, f"store raced compute even with the barrier: {fixed} corrupt"


def _drive_load(
    backend: DmaCopyBackend,
    gpu: torch.Tensor,
    cpu: torch.Tensor,
    *,
    with_barrier: bool,
) -> int:
    """Run ITERS load cycles; return how many corrupted a live compute read.

    Each cycle mirrors block reuse: the compute stream writes ``old`` into the
    block, keeps reading it after a delay, while a CPU->GPU load overwrites the
    same block with ``new``. Without the compute-done barrier the load can land
    before the read completes, so compute observes ``new`` (a torn read). With
    the barrier the load waits, so compute always observes ``old``.
    """
    block_ids = list(range(gpu.shape[0]))
    compute_stream = torch.cuda.Stream()
    corrupt = 0
    for it in range(ITERS):
        old = (it % 126) + 1  # 1..126
        new = ((it + 63) % 126) + 1  # distinct from old
        cpu.fill_(new)
        seen = torch.empty_like(gpu)
        with torch.cuda.stream(compute_stream):
            gpu.fill_(old)
            torch.cuda._sleep(SLEEP_CYCLES)
            seen.copy_(gpu)  # compute READS the block that load may overwrite

        wait_event = None
        if with_barrier:
            wait_event = torch.Event()
            wait_event.record(compute_stream)

        load_events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            block_ids,
            block_ids,
            is_store=False,
            event_idx=it,
            events_list=load_events,
            wait_event=wait_event,
        )

        deadline = time.time() + 10.0
        while not load_events and time.time() < deadline:
            time.sleep(0.0005)
        assert load_events, "background copy was never enqueued"
        load_events[0][1].synchronize()
        compute_stream.synchronize()

        if int((seen[:, 0].to(torch.int32) != old).sum().item()):
            corrupt += 1
    return corrupt


def test_load_orders_after_compute_read():
    """The load must wait for the compute event; without it, it races.

    Symmetric to the store test: the no-barrier control must corrupt a live
    read (proving the WAR window is exercised), and the barrier path must be
    clean.
    """
    backend, gpu, cpu = _make_backend()
    try:
        control = _drive_load(backend, gpu, cpu, with_barrier=False)
        fixed = _drive_load(backend, gpu, cpu, with_barrier=True)
    finally:
        backend.shutdown()

    assert control > 0, (
        "no-barrier load did not race the compute read; the test no longer "
        "exercises the hazard it is meant to guard"
    )
    assert fixed == 0, f"load raced compute even with the barrier: {fixed} corrupt"


class _RecordingBackend:
    """Captures launch_copy calls without touching the GPU."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def launch_copy(
        self,
        src_blocks,
        dst_blocks,
        is_store,
        event_idx,
        events_list,
        wait_event=None,
    ) -> None:
        self.calls.append({"is_store": is_store, "wait_event": wait_event})


def test_get_finished_gates_both_directions_on_compute_done():
    """get_finished gates both loads and stores on one compute-done event."""
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    recording = _RecordingBackend()
    worker._backend = recording
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        load_event=0,
        load_gpu_blocks=[0],
        load_cpu_blocks=[0],
        store_event=1,
        store_gpu_blocks=[1],
        store_cpu_blocks=[1],
    )

    worker.get_finished(set())

    store_calls = [c for c in recording.calls if c["is_store"]]
    load_calls = [c for c in recording.calls if not c["is_store"]]
    assert len(store_calls) == 1
    assert len(load_calls) == 1
    assert isinstance(store_calls[0]["wait_event"], torch.Event)
    assert isinstance(load_calls[0]["wait_event"], torch.Event)
    # Both directions share the single per-step compute-done event.
    assert load_calls[0]["wait_event"] is store_calls[0]["wait_event"]


def test_build_params_src_access_order():
    """build_params defaults to ANY and honors an explicit STREAM override."""
    gpu = {"k": torch.zeros((4, 64), dtype=torch.int8, device="cuda")}
    cpu = {"k": torch.zeros((4, 64), dtype=torch.int8, device="cpu")}
    stream = torch.cuda.Stream()

    default = build_params(gpu, cpu, stream)
    assert default.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_ANY

    ordered = build_params(
        gpu, cpu, stream, src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
    )
    assert ordered.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
