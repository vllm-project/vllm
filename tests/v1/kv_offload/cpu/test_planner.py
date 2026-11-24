# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the CPU offload copy planner."""

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalPageMapping,
    CopyRun,
    DevicePointers,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.worker import CopyPlanner, TensorRows

PAGE = 0x100


def _device_ptrs(*ptrs: int, size: int = PAGE) -> DevicePointers:
    return DevicePointers(
        ptrs=np.asarray(ptrs, dtype=np.uint64),
        sizes=np.full(len(ptrs), size, dtype=np.uint64),
        group_block_counts=(len(ptrs),),
        group_data_ref_counts=(1,),
        block_indices=(0,),
    )


def _plan(
    planner: CopyPlanner, device_ptrs: DevicePointers, cpu_spec: CPULoadStoreSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_copy_ops = planner._estimate_max_copy_ops(device_ptrs.group_block_counts)
    src = np.empty(num_copy_ops, dtype=np.uint64)
    dst = np.empty(num_copy_ops, dtype=np.uint64)
    sizes = np.empty(num_copy_ops, dtype=np.int64)
    op_idx, _ = planner._fill_ops(device_ptrs, cpu_spec, src, dst, sizes)
    return src[:op_idx], dst[:op_idx], sizes[:op_idx]


@pytest.mark.parametrize("device_to_cpu", [True, False])
def test_direct_descriptors(device_to_cpu: bool) -> None:
    rows = TensorRows(base=0x1000, stride=PAGE, size=PAGE, num_rows=8)
    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)]]
    planner = CopyPlanner([rows], 1, refs, device_to_cpu)

    src, dst, sizes = _plan(
        planner, _device_ptrs(0x2000, 0x3000), CPULoadStoreSpec([0, 3])
    )

    device_addrs = np.array([0x2000, 0x3000], dtype=np.uint64)
    cpu_addrs = np.array([0x1000, 0x1000 + 3 * PAGE], dtype=np.uint64)
    expected_src, expected_dst = (
        (device_addrs, cpu_addrs) if device_to_cpu else (cpu_addrs, device_addrs)
    )
    np.testing.assert_array_equal(src, expected_src)
    np.testing.assert_array_equal(dst, expected_dst)
    np.testing.assert_array_equal(sizes, [PAGE, PAGE])


@pytest.mark.parametrize("device_to_cpu", [True, False])
def test_canonical_descriptors(device_to_cpu: bool) -> None:
    """A canonical page maps two local fragments onto two canonical fragments."""
    mapping = CanonicalPageMapping(
        canonical_page_size_bytes=0x20,
        local_page_size_bytes=0x40,
        runs=(
            CopyRun(
                local_offset=0,
                canonical_offset=0,
                fragment_size=0x10,
                num_fragments=2,
                local_stride=0x20,
                canonical_stride=0x10,
            ),
        ),
        num_writers=1,
        writer_index=0,
        parallelism_agnostic=True,
    )
    ref = CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=0x40, mapping=mapping)
    rows = TensorRows(base=0x1000, stride=0x40, size=0x40, num_rows=4)
    planner = CopyPlanner([rows], 1, [[ref]], device_to_cpu, canonical_layout=True)

    src, dst, sizes = _plan(
        planner, _device_ptrs(0x2000, size=0x40), CPULoadStoreSpec([0])
    )

    local = np.array([0x2000, 0x2020], dtype=np.uint64)
    canonical = np.array([0x1000, 0x1010], dtype=np.uint64)
    expected_src, expected_dst = (
        (local, canonical) if device_to_cpu else (canonical, local)
    )
    np.testing.assert_array_equal(src, expected_src)
    np.testing.assert_array_equal(dst, expected_dst)
    np.testing.assert_array_equal(sizes, [0x10, 0x10])
