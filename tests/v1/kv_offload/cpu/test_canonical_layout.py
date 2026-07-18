# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalPageMapping,
    MappedRun,
)
from vllm.v1.kv_offload.cpu.gpu_worker import (
    _canonical_tensor_areas,
    _ref_copy_expansion,
)


def _ref(mapping: CanonicalPageMapping, tensor_idx: int = 0) -> CanonicalKVCacheRef:
    return CanonicalKVCacheRef(
        tensor_idx=tensor_idx,
        page_size_bytes=mapping.local_page_size_bytes,
        mapping=mapping,
    )


def _nhd_mapping() -> CanonicalPageMapping:
    # 4-token NHD page at tp=4, rank 2: K and V runs of 4x128B fragments
    runs = (
        MappedRun(0, 256, 128, 4, 128, 512),
        MappedRun(512, 2304, 128, 4, 128, 512),
    )
    return CanonicalPageMapping(4096, 1024, runs, runs, True)


def test_expansion_unrolls_runs():
    src, dst, sizes = _ref_copy_expansion(_ref(_nhd_mapping()), gpu_to_cpu=True)
    k_dst = [256, 768, 1280, 1792]
    assert src.tolist() == [0, 128, 256, 384, 512, 640, 768, 896]
    assert dst.tolist() == k_dst + [2048 + o for o in k_dst]
    assert sizes.tolist() == [128] * 8


def test_load_direction_swaps_offsets():
    store = _ref_copy_expansion(_ref(_nhd_mapping()), gpu_to_cpu=True)
    load = _ref_copy_expansion(_ref(_nhd_mapping()), gpu_to_cpu=False)
    assert np.array_equal(store[0], load[1])
    assert np.array_equal(store[1], load[0])
    assert np.array_equal(store[2], load[2])


def test_empty_store_runs_yield_no_ops():
    # Non-writer of a replicated page: stores nothing, loads the whole page
    identity = MappedRun(0, 0, 256, 1, 256, 256)
    mapping = CanonicalPageMapping(256, 256, (), (identity,), True)
    src, dst, sizes = _ref_copy_expansion(_ref(mapping), gpu_to_cpu=True)
    assert len(src) == len(dst) == len(sizes) == 0
    _, _, sizes = _ref_copy_expansion(_ref(mapping), gpu_to_cpu=False)
    assert sizes.tolist() == [256]


def test_canonical_tensor_areas_take_max_per_tensor():
    identity = MappedRun(0, 0, 512, 1, 512, 512)
    small = CanonicalPageMapping(2048, 512, (identity,), (identity,), False)
    refs = [[_ref(_nhd_mapping(), 0), _ref(small, 0)], [_ref(small, 1)]]
    assert _canonical_tensor_areas(refs, 2) == [4096, 2048]
