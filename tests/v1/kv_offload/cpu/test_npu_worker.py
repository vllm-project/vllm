# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.kv_offload.cpu.npu_worker import (
    _coalesce_host_pages,
    _iter_transfer_chunks,
    _load_staging_limit_bytes,
)


def test_coalesce_host_pages_coalesces_adjacent_sources():
    output_src = np.empty(3, dtype=np.int64)
    output_dst = np.empty(3, dtype=np.int64)
    output_sizes = np.empty(3, dtype=np.int64)

    num_runs = _coalesce_host_pages(
        np.array([100, 104, 108], dtype=np.int64),
        np.array([4, 4, 4], dtype=np.int64),
        1000,
        output_src,
        output_dst,
        output_sizes,
    )

    assert num_runs == 1
    np.testing.assert_array_equal(output_src[:num_runs], [100])
    np.testing.assert_array_equal(output_dst[:num_runs], [1000])
    np.testing.assert_array_equal(output_sizes[:num_runs], [12])


def test_coalesce_host_pages_handles_disjoint_sources():
    output_src = np.empty(4, dtype=np.int64)
    output_dst = np.empty(4, dtype=np.int64)
    output_sizes = np.empty(4, dtype=np.int64)

    num_runs = _coalesce_host_pages(
        np.array([100, 104, 200, 204], dtype=np.int64),
        np.array([4, 4, 4, 4], dtype=np.int64),
        1000,
        output_src,
        output_dst,
        output_sizes,
    )

    assert num_runs == 2
    np.testing.assert_array_equal(output_src[:num_runs], [100, 200])
    np.testing.assert_array_equal(output_dst[:num_runs], [1000, 1008])
    np.testing.assert_array_equal(output_sizes[:num_runs], [8, 8])


def test_iter_transfer_chunks_bounds_each_chunk():
    sizes = np.array([4, 4, 8, 3], dtype=np.int64)

    chunks = _iter_transfer_chunks(sizes, max_bytes=8)

    assert chunks == [(0, 2), (2, 3), (3, 4)]
    assert all(int(sizes[start:end].sum()) <= 8 for start, end in chunks)


def test_iter_transfer_chunks_allows_single_page_larger_than_limit():
    sizes = np.array([16, 4], dtype=np.int64)

    assert _iter_transfer_chunks(sizes, max_bytes=8) == [(0, 1), (1, 2)]


def test_load_staging_limit_from_environment(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_KV_LOAD_STAGING_BYTES", "1048576")

    assert _load_staging_limit_bytes() == 1048576


def test_load_staging_limit_rejects_non_positive_value(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_KV_LOAD_STAGING_BYTES", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        _load_staging_limit_bytes()
