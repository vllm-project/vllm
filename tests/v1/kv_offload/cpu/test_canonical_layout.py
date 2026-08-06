# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalPageMapping,
    CopyRun,
)
from vllm.v1.kv_offload.cpu.gpu_worker import (
    _canonical_tensor_areas,
    _ref_copy_expansion,
    _sub_block_ids,
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
        CopyRun(0, 256, 128, 4, 128, 512),
        CopyRun(512, 2304, 128, 4, 128, 512),
    )
    return CanonicalPageMapping(4096, 1024, runs, 1, 0, True)


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


def test_writer_rotation_matches_is_writer():
    # Replicas take turns writing shared canonical pages, keyed by the
    # CPU-side sub-block id; the enumeration must agree with is_writer
    identity = CopyRun(0, 0, 256, 1, 256, 256)
    mapping = CanonicalPageMapping(256, 256, (identity,), 2, 1, True)
    ids = _sub_block_ids(
        np.array([3, 7, 9]), blocks_per_chunk=4, count=10, skip_count=2
    )
    assert ids.tolist() == [14, 15, 28, 29, 30, 31, 36, 37, 38, 39]
    mask = ids % mapping.num_writers == mapping.writer_index
    assert mask.tolist() == [mapping.is_writer(int(b)) for b in ids]


def test_canonical_tensor_areas_take_max_per_tensor():
    identity = CopyRun(0, 0, 512, 1, 512, 512)
    small = CanonicalPageMapping(2048, 512, (identity,), 1, 0, False)
    refs = [[_ref(_nhd_mapping(), 0), _ref(small, 0)], [_ref(small, 1)]]
    assert _canonical_tensor_areas(refs, 2) == [4096, 2048]
