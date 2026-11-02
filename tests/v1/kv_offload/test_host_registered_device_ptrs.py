# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU KV offload must pass the device address of cudaHostRegister'ed host
memory to batched copies. On NVIDIA vGPU and WSL2 it differs from the host
address; cuMemcpyBatchAsync and the Triton swap kernel fault on the host one.

cudaPointerGetAttributes is faked, so these tests need no GPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

cudart = pytest.importorskip("cuda.bindings.runtime")

from vllm.v1.kv_offload.cpu import gpu_worker  # noqa: E402
from vllm.v1.simple_kv_offload import cuda_mem_ops  # noqa: E402

# Device mappings sit far below host mmap addresses on vGPU and WSL2, so the
# translation must not wrap around.
DEVICE_BASE = 0x1_0026_000_000
PAGE = 64


def _registered(host_base: int, chunk_bytes: int, device_bases: list[int]):
    def fake(ptr):
        off = int(ptr) - host_base
        k = off // chunk_bytes
        attr = SimpleNamespace(
            type=cudart.cudaMemoryType.cudaMemoryTypeHost,
            devicePointer=device_bases[k] + off - k * chunk_bytes,
        )
        return cudart.cudaError_t.cudaSuccess, attr

    return fake


def _unregistered(ptr):
    attr = SimpleNamespace(
        type=cudart.cudaMemoryType.cudaMemoryTypeUnregistered, devicePointer=0
    )
    return cudart.cudaError_t.cudaSuccess, attr


@pytest.fixture(autouse=True)
def _cuda_platform(monkeypatch):
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: True)
    gpu_worker._host_to_device_maps.clear()
    yield
    gpu_worker._host_to_device_maps.clear()


def test_offload_ptrs_use_device_address(monkeypatch):
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    monkeypatch.setattr(
        cudart,
        "cudaPointerGetAttributes",
        _registered(cpu.data_ptr(), 1 << 40, [DEVICE_BASE]),
    )
    ids = np.array([0, 3, 7], dtype=np.int64)
    out = np.empty(3, dtype=np.int64)
    gpu_worker.compute_sub_block_ptrs(ids, 1, out, cpu)
    assert out.view(np.uint64).tolist() == [DEVICE_BASE + i * PAGE for i in ids]


def test_offload_ptrs_per_registered_chunk(monkeypatch):
    # Two registrations of four rows each, mapped to unrelated device ranges.
    monkeypatch.setattr(gpu_worker, "MAX_HOST_REGISTER_CHUNK_BYTES", 4 * PAGE)
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    bases = [DEVICE_BASE, DEVICE_BASE + (1 << 30)]
    monkeypatch.setattr(
        cudart,
        "cudaPointerGetAttributes",
        _registered(cpu.data_ptr(), 4 * PAGE, bases),
    )
    ids = np.array([1, 5], dtype=np.int64)
    out = np.empty(4, dtype=np.uint64)
    gpu_worker.compute_sub_block_ptrs(ids, 2, out, cpu)
    half = PAGE // 2
    assert out.tolist() == [
        bases[0] + PAGE,
        bases[0] + PAGE + half,
        bases[1] + PAGE,
        bases[1] + PAGE + half,
    ]


def test_offload_ptrs_unchanged_for_pageable_memory(monkeypatch):
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _unregistered)
    ids = np.array([2, 6], dtype=np.int64)
    out = np.empty(2, dtype=np.int64)
    gpu_worker.compute_sub_block_ptrs(ids, 1, out, cpu)
    assert out.tolist() == [cpu.data_ptr() + i * PAGE for i in ids]


def test_offload_ptrs_unchanged_when_addresses_match(monkeypatch):
    cpu = torch.zeros((8, PAGE), dtype=torch.int8)
    monkeypatch.setattr(
        cudart,
        "cudaPointerGetAttributes",
        _registered(cpu.data_ptr(), 1 << 40, [cpu.data_ptr()]),
    )
    ids = np.array([1, 4], dtype=np.int64)
    out = np.empty(2, dtype=np.int64)
    gpu_worker.compute_sub_block_ptrs(ids, 1, out, cpu)
    assert out.tolist() == [cpu.data_ptr() + i * PAGE for i in ids]
    assert list(gpu_worker._host_to_device_maps.values()) == [None]


def test_simple_offload_copy_ptr(monkeypatch):
    cpu = torch.zeros(16, dtype=torch.int8)
    monkeypatch.setattr(
        cudart,
        "cudaPointerGetAttributes",
        _registered(cpu.data_ptr(), 1 << 40, [DEVICE_BASE]),
    )
    assert cuda_mem_ops._copy_ptr(cpu) == DEVICE_BASE
    assert cuda_mem_ops._copy_ptr(cpu[4:]) == DEVICE_BASE + 4

    monkeypatch.setattr(cudart, "cudaPointerGetAttributes", _unregistered)
    assert cuda_mem_ops._copy_ptr(cpu) == cpu.data_ptr()
