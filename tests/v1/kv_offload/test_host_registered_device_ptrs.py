# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU KV offload must pass the device address of cudaHostRegister'ed host
memory to batched copies. On NVIDIA vGPU and WSL2 it differs from the host
address; cuMemcpyBatchAsync and the Triton swap kernel fault on the host one.

cudaPointerGetAttributes is faked, so these tests need no GPU.
"""

import contextlib
import mmap
import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch

cudart = pytest.importorskip("cuda.bindings.runtime")

from vllm.v1.kv_offload.cpu import gpu_worker  # noqa: E402
from vllm.v1.kv_offload.cpu import shared_offload_region as region_module  # noqa: E402
from vllm.v1.simple_kv_offload import cuda_mem_ops  # noqa: E402

# Device mappings sit far below host mmap addresses on vGPU and WSL2, so the
# translation must not wrap around.
DEVICE_BASE = 0x1_0026_000_000
PAGE = 64


class _FakeRegistrations:
    """cudaPointerGetAttributes over (host start, size, device base) ranges,
    each mapped on its own like a separate cudaHostRegister call on WSL2."""

    def __init__(self, ranges: list[tuple[int, int, int]]):
        self.ranges = ranges

    def __call__(self, ptr):
        ptr = int(ptr)
        for start, size, device in self.ranges:
            if start <= ptr < start + size:
                attr = SimpleNamespace(
                    type=cudart.cudaMemoryType.cudaMemoryTypeHost,
                    devicePointer=device + ptr - start,
                )
                return cudart.cudaError_t.cudaSuccess, attr
        return _unregistered(ptr)


def _unregistered(ptr):
    attr = SimpleNamespace(
        type=cudart.cudaMemoryType.cudaMemoryTypeUnregistered, devicePointer=0
    )
    return cudart.cudaError_t.cudaSuccess, attr


def _expected(ptrs, ranges):
    out = []
    for ptr in ptrs:
        (device,) = [d + ptr - s for s, n, d in ranges if s <= ptr < s + n]
        out.append(device)
    return out


@pytest.fixture(autouse=True)
def _cuda_platform(monkeypatch):
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: True)


def test_offload_ptrs_use_device_address(monkeypatch):
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    ranges = [(cpu.data_ptr(), cpu.nbytes, DEVICE_BASE)]
    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _FakeRegistrations(ranges))
    device_map = gpu_worker._host_to_device_map([cpu.data_ptr()])
    ids = np.array([0, 3, 7], dtype=np.int64)
    out = np.empty(3, dtype=np.int64)
    gpu_worker.compute_sub_block_ptrs(ids, 1, out, cpu, device_map=device_map)
    assert out.view(np.uint64).tolist() == [DEVICE_BASE + i * PAGE for i in ids]


def test_offload_ptrs_per_registered_chunk(monkeypatch):
    # Two registrations of four rows each, mapped to unrelated device ranges.
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    host = cpu.data_ptr()
    bases = [DEVICE_BASE, DEVICE_BASE + (1 << 30)]
    ranges = [(host, 4 * PAGE, bases[0]), (host + 4 * PAGE, 4 * PAGE, bases[1])]
    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _FakeRegistrations(ranges))
    device_map = gpu_worker._host_to_device_map([host, host + 4 * PAGE])
    ids = np.array([1, 5], dtype=np.int64)
    out = np.empty(4, dtype=np.uint64)
    gpu_worker.compute_sub_block_ptrs(ids, 2, out, cpu, device_map=device_map)
    half = PAGE // 2
    assert out.tolist() == [
        bases[0] + PAGE,
        bases[0] + PAGE + half,
        bases[1] + PAGE,
        bases[1] + PAGE + half,
    ]


@pytest.mark.parametrize("blocks_per_chunk", [1, 2])
def test_offload_ptrs_per_worker_slot(monkeypatch, blocks_per_chunk):
    """TP>1 direct layout: pin_mmap_region registers this rank's slot of every
    row on its own, and each slot gets an unrelated device mapping. Every row
    must translate through its own slot's base, not row 0's."""
    num_rows, num_workers, rank = 455, 2, 1
    slot = 4 * PAGE
    row_stride = num_workers * slot
    backing = torch.zeros((num_rows, row_stride), dtype=torch.int8)
    host = backing.data_ptr()
    cpu = torch.as_strided(
        backing.view(-1), (num_rows, slot), (row_stride, 1), rank * slot
    )
    rng = np.random.default_rng(0)
    device_offsets = rng.permutation(num_rows) * (1 << 21)
    ranges = [
        (host + row * row_stride + rank * slot, slot, DEVICE_BASE + int(off))
        for row, off in enumerate(device_offsets)
    ]
    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _FakeRegistrations(ranges))
    # Rotated like iter_worker_slot_ranges with several backing files.
    starts = [s for s, _, _ in ranges]
    device_map = gpu_worker._host_to_device_map(starts[100:] + starts[:100])

    ids = np.arange(num_rows, dtype=np.int64)
    out = np.empty(num_rows * blocks_per_chunk, dtype=np.int64)
    gpu_worker.compute_sub_block_ptrs(
        ids, blocks_per_chunk, out, cpu, device_map=device_map
    )
    sub = slot // blocks_per_chunk
    host_ptrs = [
        cpu.data_ptr() + row * row_stride + j * sub
        for row in range(num_rows)
        for j in range(blocks_per_chunk)
    ]
    assert out.view(np.uint64).tolist() == _expected(host_ptrs, ranges)


def test_pin_mmap_region_worker_slots_translate(monkeypatch):
    """End to end over a real region: the ranges pin_mmap_region registers for
    this rank's slots are the ones the device map translates through."""
    page = mmap.PAGESIZE
    monkeypatch.setattr(gpu_worker, "MIN_HOST_REGISTER_SAVED_BYTES", page)
    monkeypatch.setattr(
        region_module, "_get_populate_write_fn", lambda _: lambda *a: None
    )
    registered: list[tuple[int, int, int]] = []

    def register(ptr, size):
        registered.append((ptr, size, DEVICE_BASE + len(registered) * (1 << 21)))
        return True

    monkeypatch.setattr(gpu_worker, "host_register", register)
    monkeypatch.setattr(region_module, "host_unregister", lambda ptr: None)
    monkeypatch.setattr(
        cudart, "cudaPointerGetAttributes", _FakeRegistrations(registered)
    )

    num_rows, slot = 6, 2 * page
    region = region_module.SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_chunks=num_rows,
        rank=0,
        kv_bytes_per_chunk=2 * slot,
        cpu_page_size=slot,
    )
    try:
        gpu_worker.pin_mmap_region(region, worker_slot_only=True)
        assert region.is_pinned and len(registered) == num_rows
        cpu = region.create_next_worker_view(slot)
        device_map = gpu_worker._host_to_device_map(region.pinned_addresses)
        ids = np.arange(num_rows, dtype=np.int64)
        out = np.empty(num_rows, dtype=np.int64)
        gpu_worker.compute_sub_block_ptrs(ids, 1, out, cpu, device_map=device_map)
        host_ptrs = [cpu.data_ptr() + row * 2 * slot for row in range(num_rows)]
        assert out.view(np.uint64).tolist() == _expected(host_ptrs, registered)
    finally:
        region.cleanup()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(region.mmap_path)


def test_offload_ptrs_unchanged_for_pageable_memory(monkeypatch):
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _unregistered)
    assert gpu_worker._host_to_device_map([cpu.data_ptr()]) is None
    ids = np.array([2, 6], dtype=np.int64)
    out = np.empty(2, dtype=np.int64)
    gpu_worker.compute_sub_block_ptrs(ids, 1, out, cpu)
    assert out.tolist() == [cpu.data_ptr() + i * PAGE for i in ids]


def test_offload_ptrs_unchanged_when_addresses_match(monkeypatch):
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    host = cpu.data_ptr()
    ranges = [(host, 4 * PAGE, host), (host + 4 * PAGE, 4 * PAGE, host + 4 * PAGE)]
    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _FakeRegistrations(ranges))
    assert gpu_worker._host_to_device_map([host, host + 4 * PAGE]) is None


def test_simple_offload_copy_ptr(monkeypatch):
    cpu = torch.zeros(16, dtype=torch.int8)
    ranges = [(cpu.data_ptr(), cpu.nbytes, DEVICE_BASE)]
    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _FakeRegistrations(ranges))
    assert cuda_mem_ops._copy_ptr(cpu) == DEVICE_BASE
    assert cuda_mem_ops._copy_ptr(cpu[4:]) == DEVICE_BASE + 4

    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _unregistered)
    assert cuda_mem_ops._copy_ptr(cpu) == cpu.data_ptr()
