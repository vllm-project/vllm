# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.kv_offload.cpu.npu_worker import _coalesce_host_pages


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
