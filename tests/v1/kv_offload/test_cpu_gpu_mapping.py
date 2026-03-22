# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.cpu_gpu import build_transfer_indices


def test_build_transfer_indices_whole_blocks_preserves_legacy_skip_behavior():
    src_spec = CPULoadStoreSpec([7])
    dst_spec = GPULoadStoreSpec([3, 4, 5], group_sizes=(3,))

    mapping = build_transfer_indices(
        src_spec,
        dst_spec,
        src_block_size_factor=4,
        dst_block_size_factor=1,
    )

    np.testing.assert_array_equal(
        mapping,
        np.array([[29, 3], [30, 4], [31, 5]], dtype=np.int64),
    )


def test_build_transfer_indices_supports_partial_gpu_ranges():
    src_spec = GPULoadStoreSpec(
        [0, 1],
        group_sizes=(2,),
        block_offsets=[2, 0],
        block_counts=[3, 3],
    )
    dst_spec = GPULoadStoreSpec(
        [5, 6],
        group_sizes=(2,),
        block_offsets=[1, 4],
        block_counts=[3, 3],
    )

    mapping = build_transfer_indices(
        src_spec,
        dst_spec,
        src_block_size_factor=8,
        dst_block_size_factor=8,
    )

    np.testing.assert_array_equal(
        mapping,
        np.array(
            [[2, 41], [3, 42], [4, 43], [8, 52], [9, 53], [10, 54]],
            dtype=np.int64,
        ),
    )
