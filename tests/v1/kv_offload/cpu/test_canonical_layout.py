# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalPageMapping,
    CopyRun,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
    _build_copy_plan,
    _canonical_block_sizes,
    _canonical_page_ids,
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


def test_copy_plan_unrolls_runs():
    plan = _build_copy_plan(_ref(_nhd_mapping()), gpu_to_cpu=True)
    k_dst = [256, 768, 1280, 1792]
    assert plan.frag_offsets_src.tolist() == [0, 128, 256, 384, 512, 640, 768, 896]
    assert plan.frag_offsets_dst.tolist() == k_dst + [2048 + o for o in k_dst]
    assert plan.frag_sizes.tolist() == [128] * 8
    assert plan.num_frags == 8


def test_load_direction_swaps_offsets():
    store = _build_copy_plan(_ref(_nhd_mapping()), gpu_to_cpu=True)
    load = _build_copy_plan(_ref(_nhd_mapping()), gpu_to_cpu=False)
    assert np.array_equal(store.frag_offsets_src, load.frag_offsets_dst)
    assert np.array_equal(store.frag_offsets_dst, load.frag_offsets_src)
    assert np.array_equal(store.frag_sizes, load.frag_sizes)


def test_writer_rotation_matches_is_writer():
    # Replicas take turns writing shared canonical pages, keyed by the
    # CPU-side canonical page id; the enumeration must agree with is_writer
    identity = CopyRun(0, 0, 256, 1, 256, 256)
    mapping = CanonicalPageMapping(256, 256, (identity,), 2, 1, True)
    ids = _canonical_page_ids(
        np.array([3, 7, 9]), blocks_per_chunk=4, count=10, skip_count=2
    )
    assert ids.tolist() == [14, 15, 28, 29, 30, 31, 36, 37, 38, 39]
    mask = ids % mapping.num_writers == mapping.writer_index
    assert mask.tolist() == [mapping.is_writer(int(b)) for b in ids]


def test_canonical_block_sizes_take_max_per_tensor():
    identity = CopyRun(0, 0, 512, 1, 512, 512)
    small = CanonicalPageMapping(2048, 512, (identity,), 1, 0, False)
    refs = [[_ref(_nhd_mapping(), 0), _ref(small, 0)], [_ref(small, 1)]]
    assert _canonical_block_sizes(refs, 2) == [4096, 2048]


def _tp2_rank_mapping(rank: int) -> CanonicalPageMapping:
    # 4-token NHD page, 4 total heads of 64B, tp=2: rank holds 2 heads,
    # so K and V each scatter as 4 per-token 128B fragments
    runs = (
        CopyRun(0, rank * 128, 128, 4, 128, 256),
        CopyRun(512, 1024 + rank * 128, 128, 4, 128, 256),
    )
    return CanonicalPageMapping(2048, 1024, runs, 1, 0, True)


def _whole_page_mapping() -> CanonicalPageMapping:
    identity = CopyRun(0, 0, 2048, 1, 2048, 2048)
    return CanonicalPageMapping(2048, 2048, (identity,), 1, 0, True)


def _transfer(handler, num_blocks: int, gpu_to_cpu: bool) -> None:
    block_ids = list(range(num_blocks))
    gpu_spec = GPULoadStoreSpec(
        block_ids, group_sizes=(num_blocks,), block_indices=(0,)
    )
    cpu_spec = CPULoadStoreSpec(block_ids)
    src, dst = (gpu_spec, cpu_spec) if gpu_to_cpu else (cpu_spec, gpu_spec)
    assert handler.transfer_async(0, src, dst)
    deadline = time.time() + 30
    while time.time() < deadline:
        if handler.get_finished():
            return
        time.sleep(0.001)
    raise TimeoutError("transfer did not complete")


def _canonical_handler(gpu_tensor, cpu_tensor, mapping, gpu_to_cpu):
    page = mapping.local_page_size_bytes
    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page, mapping=mapping)]]
    return SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu_tensor],
        cpu_tensors=[cpu_tensor],
        blocks_per_chunk=1,
        layer_refs_per_group=refs,
        gpu_to_cpu=gpu_to_cpu,
        canonical_layout=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_roundtrip_assembles_canonical_page_across_ranks():
    """Two TP2-style rank handlers must scatter into one shared canonical
    CPU page such that each rank reloads bit-exact and a whole-page (TP1)
    reader sees both shards — the cross-topology contract."""
    torch.manual_seed(0)
    num_blocks = 4
    gpu_rank = [
        torch.randint(-128, 128, (num_blocks, 1024), dtype=torch.int8, device="cuda")
        for _ in range(2)
    ]
    cpu_canonical = torch.zeros(num_blocks, 2048, dtype=torch.int8, pin_memory=True)

    for rank in (0, 1):
        store = _canonical_handler(
            gpu_rank[rank], cpu_canonical, _tp2_rank_mapping(rank), gpu_to_cpu=True
        )
        _transfer(store, num_blocks, gpu_to_cpu=True)
    torch.accelerator.synchronize()

    # independent oracle: replay each rank's runs in numpy
    expected = np.zeros((num_blocks, 2048), dtype=np.int8)
    for rank in (0, 1):
        local = gpu_rank[rank].cpu().numpy()
        for run in _tp2_rank_mapping(rank).runs:
            for i in range(run.num_fragments):
                lo = run.local_offset + i * run.local_stride
                co = run.canonical_offset + i * run.canonical_stride
                expected[:, co : co + run.fragment_size] = local[
                    :, lo : lo + run.fragment_size
                ]
    assert np.array_equal(cpu_canonical.numpy(), expected)

    # each rank reloads its shard bit-exact
    gpu_back = torch.zeros(num_blocks, 1024, dtype=torch.int8, device="cuda")
    load = _canonical_handler(
        gpu_back, cpu_canonical, _tp2_rank_mapping(0), gpu_to_cpu=False
    )
    _transfer(load, num_blocks, gpu_to_cpu=False)
    torch.accelerator.synchronize()
    assert torch.equal(gpu_back, gpu_rank[0])

    # a whole-page reader (TP1 topology) sees the assembled page
    gpu_full = torch.zeros(num_blocks, 2048, dtype=torch.int8, device="cuda")
    load_full = _canonical_handler(
        gpu_full, cpu_canonical, _whole_page_mapping(), gpu_to_cpu=False
    )
    _transfer(load_full, num_blocks, gpu_to_cpu=False)
    torch.accelerator.synchronize()
    assert torch.equal(gpu_full.cpu(), cpu_canonical)
