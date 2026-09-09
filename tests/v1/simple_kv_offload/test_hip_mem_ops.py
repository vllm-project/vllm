# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HIP/ROCm paths in vllm.v1.simple_kv_offload.cuda_mem_ops.

Deliberately not skipped off ROCm: the version gate takes a plain version int
and the library loader is monkeypatched, so nothing here needs a HIP runtime
and the logic stays covered by the default (NVIDIA) CI that runs this
directory.
"""

import ctypes

import numpy as np
import pytest

from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    _CUmemcpyAttributes,
    _load_hip_runtime,
    _num_attrs_for_hip_version,
    copy_blocks,
)


# hipMemcpyBatchAsync rejects numAttrs > 0 on ROCm 7.2.1-7.2.3 and accepts it
# on 7.13+. HIP encodes the version as major*10_000_000 + minor*100_000 + patch.
@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (0, 0),  # unknown (hipRuntimeGetVersion failed) -> conservative 0
        (70_100_000, 0),  # ROCm 7.1
        (70_200_000, 0),  # ROCm 7.2.0
        (70_253_211, 0),  # ROCm 7.2.2 (patch 53211)
        (70_300_000, 0),  # ROCm 7.3
        (71_200_000, 0),  # ROCm 7.12 -- just below the 7.13 cutoff
        (71_300_000, 1),  # ROCm 7.13 -- first release accepting numAttrs > 0
        (71_326_193, 1),  # ROCm 7.13 nightly (patch 26193)
        (71_400_000, 1),  # ROCm 7.14
        (71_500_000, 1),  # ROCm 7.15
        (80_000_000, 1),  # ROCm 8.0
    ],
)
def test_num_attrs_for_hip_version(version: int, expected: int):
    assert _num_attrs_for_hip_version(version) == expected


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        # Devel install: the unversioned symlink is present and preferred.
        ({"libamdhip64.so", "libamdhip64.so.7"}, "libamdhip64.so"),
        # Runtime-only / wheel-packaged ROCm: versioned soname only.
        ({"libamdhip64.so.7"}, "libamdhip64.so.7"),
        ({"libamdhip64.so.6"}, "libamdhip64.so.6"),
    ],
)
def test_load_hip_runtime_falls_back_to_versioned_soname(
    monkeypatch, available: set[str], expected: str
):
    loaded = []

    def fake_cdll(name, mode=0):
        if name not in available:
            raise OSError(f"{name}: cannot open shared object file")
        loaded.append(name)
        return name

    monkeypatch.setattr(cuda_mem_ops.ctypes, "CDLL", fake_cdll)
    assert _load_hip_runtime() == expected
    assert loaded == [expected]


def test_load_hip_runtime_reports_every_candidate_when_all_fail(monkeypatch):
    def fake_cdll(name, mode=0):
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(cuda_mem_ops.ctypes, "CDLL", fake_cdll)
    with pytest.raises(OSError, match="could not load the HIP runtime") as exc:
        _load_hip_runtime()
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        assert name in str(exc.value)


def _fake_params(num_layers: int, bytes_per_block: int = 512) -> BatchMemcpyParams:
    """Minimal params for driving copy_blocks() without a real device.

    Bases are 0 and the batch-memcpy function is monkeypatched, so no memory is
    ever dereferenced; only the descriptor bookkeeping (count/chunking) matters.
    """
    return BatchMemcpyParams(
        src_bases=np.zeros(num_layers, dtype=np.uint64),
        dst_bases=np.zeros(num_layers, dtype=np.uint64),
        bpb=np.full(num_layers, bytes_per_block, dtype=np.uint64),
        num_layers=num_layers,
        attrs=_CUmemcpyAttributes(),
        attrs_idx=ctypes.c_size_t(0),
        num_attrs=0,
        fail_idx=ctypes.c_size_t(0),
        stream_handle=0,
    )


def _record_counts(monkeypatch) -> list[int]:
    """Patch the batch-memcpy call to record the descriptor count per call."""
    counts: list[int] = []

    def fake_fn(dst, src, sizes, count, *rest):
        counts.append(count)
        return 0

    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy", (fake_fn, 0))
    return counts


# ROCm chunks the descriptor batch at the cap (hipMemcpyBatchAsync faults above
# 8192/call); CUDA leaves the cap at 0 and issues a single call.
@pytest.mark.parametrize(
    ("max_desc", "num_layers", "num_blocks", "expected_counts"),
    [
        (8192, 3, 5000, [8192, 6808]),  # 15000 desc -> 8192 + 6808
        (8192, 1, 8192, [8192]),  # exactly at the cap -> single call
        (8192, 1, 8193, [8192, 1]),  # one over -> spills into a second call
        (4096, 61, 135, [4096, 4096, 43]),  # bs=1 Kimi shape (8235 desc)
        (0, 61, 135, [8235]),  # CUDA uncapped -> single call
    ],
)
def test_copy_blocks_chunks_at_descriptor_cap(
    monkeypatch, max_desc, num_layers, num_blocks, expected_counts
):
    counts = _record_counts(monkeypatch)
    monkeypatch.setattr(cuda_mem_ops, "_max_batch_descriptors", max_desc)

    ids = list(range(0, 2 * num_blocks, 2))
    copy_blocks(ids, ids, _fake_params(num_layers))

    assert counts == expected_counts
    assert sum(counts) == num_layers * num_blocks


@pytest.mark.parametrize(
    ("src_ids", "dst_ids", "rocm_counts", "cuda_counts"),
    [
        ([1], [2], [2], [2]),
        ([1, 2, 3], [5, 6, 7], [2], [3, 3]),
        ([1, 2, 3], [2, 4, 6], [3, 3], [3, 3]),
        ([1, 3, 5], [2, 3, 4], [3, 3], [3, 3]),
        ([3, 2, 1], [6, 5, 4], [3, 3], [3, 3]),
        ([1, 1, 2], [2, 4, 5], [3, 1], [3, 3]),
        ([1, 2, 5, 6, 7, 4], [2, 3, 7, 8, 6, 5], [3, 3, 2], [3, 3, 3, 3]),
    ],
)
@pytest.mark.parametrize("is_rocm", [True, False])
def test_copy_blocks_coalesces_contiguous_pairs_only_on_rocm(
    monkeypatch, src_ids, dst_ids, rocm_counts, cuda_counts, is_rocm
):
    """Merged copies preserve mapped bytes and untouched blocks in every region."""
    sources = [np.arange(30, dtype=np.uint8), np.arange(50, dtype=np.uint8)]
    destinations = [np.full_like(src, 255) for src in sources]
    params = _fake_params(2)._replace(
        src_bases=np.array([src.ctypes.data for src in sources], dtype=np.uint64),
        dst_bases=np.array([dst.ctypes.data for dst in destinations], dtype=np.uint64),
        bpb=np.array([3, 5], dtype=np.uint64),
    )
    counts = []

    def copy_descriptors(dst, src, sizes, count, *rest):
        counts.append(count)
        arrays = [
            np.ctypeslib.as_array((ctypes.c_uint64 * count).from_address(ptr))
            for ptr in (src, dst, sizes)
        ]
        for source, destination, size in zip(*arrays):
            ctypes.memmove(int(destination), int(source), int(size))
        return 0

    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy", (copy_descriptors, 0))
    monkeypatch.setattr(cuda_mem_ops, "_max_batch_descriptors", 3)
    monkeypatch.setattr(cuda_mem_ops.current_platform, "is_rocm", lambda: is_rocm)
    copy_blocks(src_ids, dst_ids, params)

    assert counts == (rocm_counts if is_rocm else cuda_counts)
    for source, destination, block_bytes in zip(sources, destinations, params.bpb):
        expected = np.full_like(source, 255)
        block_bytes = int(block_bytes)
        for src_id, dst_id in zip(src_ids, dst_ids):
            expected[dst_id * block_bytes : (dst_id + 1) * block_bytes] = source[
                src_id * block_bytes : (src_id + 1) * block_bytes
            ]
        np.testing.assert_array_equal(destination, expected)


def test_copy_blocks_noop_on_empty(monkeypatch):
    counts = _record_counts(monkeypatch)
    monkeypatch.setattr(cuda_mem_ops, "_max_batch_descriptors", 8192)
    copy_blocks([], [], _fake_params(num_layers=4))
    assert counts == []
