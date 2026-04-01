# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random
import time
import uuid
from unittest.mock import patch

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.shared_mmap_region import SharedMmapRegion

NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
GPU_PAGE_SIZES = [512, 1024]
BLOCK_SIZE_FACTORS = [1, 3]
NUM_TENSORS = [4]
SEEDS = [0]
CUDA_DEVICES = ["cuda:0"]
NUM_MAPPINGS = [3]


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("block_size_factor", BLOCK_SIZE_FACTORS)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_tensors", NUM_TENSORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_transfer(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings: int,
    gpu_page_size_bytes: int,
    block_size_factor: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    num_tensors: int,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)

    # build CanonicalKVCacheTensor list: one per tensor
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for i in range(num_tensors):
        gpu_tensor = torch.randint(
            -128,
            127,
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    # one group containing all tensors, one data ref per tensor
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(num_tensors)
        ]
    ]

    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )
    handlers = CpuGpuOffloadingHandlers(
        kv_caches=kv_caches,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
    )

    # select block mappings
    gpu_blocks = random.sample(range(num_gpu_blocks), num_mappings * block_size_factor)
    cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

    # expand cpu blocks to gpu-page granularity for uniform comparison:
    # each cpu block maps to block_size_factor consecutive sub-blocks
    cpu_blocks_expanded = [
        cpu_block * block_size_factor + j
        for cpu_block in cpu_blocks
        for j in range(block_size_factor)
    ]

    # maybe skip some GPU blocks to test reading from the middle of a CPU block
    if not gpu_to_cpu:
        blocks_to_skip = block_size_factor - 1
        gpu_blocks = gpu_blocks[blocks_to_skip:]
        cpu_blocks_expanded = cpu_blocks_expanded[blocks_to_skip:]

    # set transfer direction
    if gpu_to_cpu:
        handler = handlers.gpu_to_cpu_handler
        src_spec = GPULoadStoreSpec(gpu_blocks, group_sizes=(len(gpu_blocks),))
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        dst_to_src = dict(zip(cpu_blocks_expanded, gpu_blocks))
        num_dst_sub_blocks = num_cpu_blocks * block_size_factor
    else:
        handler = handlers.cpu_to_gpu_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(gpu_blocks, group_sizes=(len(gpu_blocks),))
        dst_to_src = dict(zip(gpu_blocks, cpu_blocks_expanded))
        num_dst_sub_blocks = num_gpu_blocks

    # clone src and dst tensors before transfer
    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    # call transfer function
    start_time = time.time()
    assert handler.transfer_async(1, (src_spec, dst_spec))
    assert set({x.job_id for x in handler._transfers}) == {1}

    # wait for transfer to complete
    end_time = time.time() + 10
    while time.time() < end_time:
        finished = handler.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            assert (
                finished[0].transfer_type == ("GPU", "CPU")
                if gpu_to_cpu
                else ("CPU", "GPU")
            )
            assert finished[0].transfer_size == (
                len(gpu_blocks) * handler.group_block_size_in_bytes[0]
            )
            assert finished[0].transfer_time > 0
            assert finished[0].transfer_time < (time.time() - start_time)
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity.
    for src_tensor, dst_tensor, orig_dst_tensor in zip(
        handler.src_tensors,
        handler.dst_tensors,
        orig_dst_tensors,
    ):
        # view both GPU and CPU tensors as (n, gpu_page_size_bytes) for comparison.
        src_view = src_tensor.view(-1, gpu_page_size_bytes)
        dst_view = dst_tensor.view(-1, gpu_page_size_bytes)
        orig_dst_view = orig_dst_tensor.view(-1, gpu_page_size_bytes)
        for dst_sub_block in range(num_dst_sub_blocks):
            src_sub_block = dst_to_src.get(dst_sub_block)
            if src_sub_block is not None:
                expected = src_view[src_sub_block]
            else:
                expected = orig_dst_view[dst_sub_block]
            torch.testing.assert_close(dst_view[dst_sub_block].cpu(), expected.cpu())


# ---------------------------------------------------------------------------
# Helpers for mmap tests
# ---------------------------------------------------------------------------


def _build_kv_caches(
    num_tensors: int,
    num_gpu_blocks: int,
    gpu_page_size_bytes: int,
    device: str,
    fill_value: int | None = None,
) -> CanonicalKVCaches:
    """Build CanonicalKVCaches with int8 GPU tensors.

    If fill_value is given every element is set to that value;
    otherwise random int8 data is used.
    """
    tensors: list[CanonicalKVCacheTensor] = []
    refs: list[CanonicalKVCacheRef] = []
    for i in range(num_tensors):
        if fill_value is not None:
            gpu_tensor = torch.full(
                (num_gpu_blocks, gpu_page_size_bytes),
                fill_value,
                dtype=torch.int8,
                device=device,
            )
        else:
            gpu_tensor = torch.randint(
                -128,
                127,
                (num_gpu_blocks, gpu_page_size_bytes),
                dtype=torch.int8,
                device=device,
            )
        tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor, page_size_bytes=gpu_page_size_bytes
            )
        )
        refs.append(
            CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=gpu_page_size_bytes)
        )
    return CanonicalKVCaches(tensors=tensors, group_data_refs=[refs])


def _wait_for_handler(handler, job_id, timeout=10):
    """Poll handler.get_finished() until the given job completes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for result in handler.get_finished():
            if result.job_id == job_id:
                assert result.success
                return result
        time.sleep(0.01)
    raise TimeoutError(f"Transfer job {job_id} did not complete in time")


# ---------------------------------------------------------------------------
# test_mmap_round_trip — data integrity through the SharedMmapRegion path
# ---------------------------------------------------------------------------

MMAP_GPU_PAGE_SIZES = [512, 1024]
MMAP_NUM_TENSORS = [1, 4]
MMAP_BLOCK_SIZE_FACTORS = [1, 3]


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("block_size_factor", MMAP_BLOCK_SIZE_FACTORS)
@pytest.mark.parametrize("gpu_page_size_bytes", MMAP_GPU_PAGE_SIZES)
@pytest.mark.parametrize("num_tensors", MMAP_NUM_TENSORS)
@torch.inference_mode()
def test_mmap_round_trip(
    default_vllm_config,
    gpu_to_cpu: bool,
    block_size_factor: int,
    gpu_page_size_bytes: int,
    num_tensors: int,
) -> None:
    """Verify data correctness for transfers through mmap-backed CPU tensors."""
    set_random_seed(0)
    num_gpu_blocks = 64
    num_cpu_blocks = 32
    num_workers = 1
    num_mappings = 3
    device = "cuda:0"

    kv_caches = _build_kv_caches(
        num_tensors, num_gpu_blocks, gpu_page_size_bytes, device
    )

    # cpu_page_size: per-worker per-sub-block bytes (no block_size_factor —
    # the factor expands the row count, not the column width)
    cpu_page_size = gpu_page_size_bytes * num_tensors
    instance_id = str(uuid.uuid4())
    # num_cpu_blocks * block_size_factor sub-block rows; each row spans all workers
    total_mmap_bytes = num_cpu_blocks * block_size_factor * num_workers * cpu_page_size

    mmap_region = SharedMmapRegion(
        instance_id=instance_id,
        total_size_bytes=total_mmap_bytes,
        num_blocks=num_cpu_blocks,
        rank=0,
        num_workers=num_workers,
        cpu_page_size=cpu_page_size,
    )
    try:
        with patch("vllm.v1.kv_offload.worker.cpu_gpu.pin_mmap_region"):
            handlers = CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=block_size_factor,
                num_cpu_blocks=num_cpu_blocks,
                mmap_region=mmap_region,
                num_workers=num_workers,
            )

        # each logical cpu_block maps to block_size_factor GPU sub-blocks,
        # so we need num_mappings * block_size_factor GPU blocks total
        gpu_blocks = random.sample(
            range(num_gpu_blocks), num_mappings * block_size_factor
        )
        cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

        if gpu_to_cpu:
            # GPU → CPU: offload and verify CPU got the right data.
            # The handler expands cpu_blocks by block_size_factor, mapping:
            #   gpu_blocks[i*factor + j]  →  cpu_blk*factor + j
            handler = handlers.gpu_to_cpu_handler
            src_spec = GPULoadStoreSpec(gpu_blocks, group_sizes=(len(gpu_blocks),))
            dst_spec = CPULoadStoreSpec(cpu_blocks)
            assert handler.transfer_async(1, (src_spec, dst_spec))
            _wait_for_handler(handler, 1)

            for src_t, dst_t in zip(handler.src_tensors, handler.dst_tensors):
                for i, cpu_blk in enumerate(cpu_blocks):
                    for j in range(block_size_factor):
                        gpu_blk = gpu_blocks[i * block_size_factor + j]
                        cpu_sub_blk = cpu_blk * block_size_factor + j
                        torch.testing.assert_close(
                            dst_t[cpu_sub_blk].cpu(), src_t[gpu_blk].cpu()
                        )
        else:
            # round-trip: GPU → CPU → GPU (different destination GPU blocks)
            store_handler = handlers.gpu_to_cpu_handler
            # snapshot original GPU data before it is offloaded
            orig_gpu_tensors = [t.clone() for t in store_handler.src_tensors]

            store_src = GPULoadStoreSpec(gpu_blocks, group_sizes=(len(gpu_blocks),))
            store_dst = CPULoadStoreSpec(cpu_blocks)
            assert store_handler.transfer_async(99, (store_src, store_dst))
            _wait_for_handler(store_handler, 99)

            new_gpu_blocks = random.sample(
                [b for b in range(num_gpu_blocks) if b not in gpu_blocks],
                num_mappings * block_size_factor,
            )
            load_handler = handlers.cpu_to_gpu_handler
            load_src = CPULoadStoreSpec(cpu_blocks)
            load_dst = GPULoadStoreSpec(
                new_gpu_blocks, group_sizes=(len(new_gpu_blocks),)
            )
            assert load_handler.transfer_async(1, (load_src, load_dst))
            _wait_for_handler(load_handler, 1)

            # each new_gpu_blocks[i] must equal the original gpu_blocks[i]
            for orig_t, dst_t in zip(orig_gpu_tensors, load_handler.dst_tensors):
                for orig_blk, new_blk in zip(gpu_blocks, new_gpu_blocks):
                    torch.testing.assert_close(
                        dst_t[new_blk].cpu(), orig_t[orig_blk].cpu()
                    )
    finally:
        mmap_region.cleanup()
        if os.path.exists(mmap_region.mmap_path):
            os.unlink(mmap_region.mmap_path)


# ---------------------------------------------------------------------------
# test_interleaved_layout — verify raw mmap bytes show correct interleaving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [2, 4])
@pytest.mark.parametrize("num_tensors", [1, 2])
@torch.inference_mode()
def test_interleaved_layout(
    default_vllm_config,
    num_workers: int,
    num_tensors: int,
) -> None:
    """For each block row the mmap should contain
    [worker0 | worker1 | ... | workerN] contiguously."""
    set_random_seed(0)
    gpu_page_size_bytes = 512
    num_gpu_blocks = 32
    num_cpu_blocks = 8
    block_size_factor = 1
    device = "cuda:0"

    cpu_page_size = gpu_page_size_bytes * block_size_factor * num_tensors
    instance_id = str(uuid.uuid4())
    total_mmap_bytes = num_cpu_blocks * num_workers * cpu_page_size

    regions: list[SharedMmapRegion] = []
    try:
        # create one region per rank (same file, MAP_SHARED)
        for rank in range(num_workers):
            regions.append(
                SharedMmapRegion(
                    instance_id=instance_id,
                    total_size_bytes=total_mmap_bytes,
                    num_blocks=num_cpu_blocks,
                    rank=rank,
                    num_workers=num_workers,
                    cpu_page_size=cpu_page_size,
                )
            )

        cpu_blocks_to_use = list(range(min(4, num_cpu_blocks)))

        for rank in range(num_workers):
            # fill every byte with a rank-specific value so we can
            # identify which rank wrote each region
            fill_value = rank + 1
            kv_caches = _build_kv_caches(
                num_tensors,
                num_gpu_blocks,
                gpu_page_size_bytes,
                device,
                fill_value=fill_value,
            )

            with patch("vllm.v1.kv_offload.worker.cpu_gpu.pin_mmap_region"):
                handlers = CpuGpuOffloadingHandlers(
                    kv_caches=kv_caches,
                    block_size_factor=block_size_factor,
                    num_cpu_blocks=num_cpu_blocks,
                    mmap_region=regions[rank],
                    num_workers=num_workers,
                )

            gpu_blocks = list(range(len(cpu_blocks_to_use)))
            src_spec = GPULoadStoreSpec(gpu_blocks, group_sizes=(len(gpu_blocks),))
            dst_spec = CPULoadStoreSpec(cpu_blocks_to_use)

            handler = handlers.gpu_to_cpu_handler
            assert handler.transfer_async(rank, (src_spec, dst_spec))
            _wait_for_handler(handler, rank)

        # ---- verify raw mmap bytes ----
        raw = memoryview(regions[0].mmap_obj)
        for cpu_blk in cpu_blocks_to_use:
            for rank in range(num_workers):
                offset = (cpu_blk * num_workers + rank) * cpu_page_size
                chunk = bytes(raw[offset : offset + cpu_page_size])
                # every byte in this worker's slot should be (rank + 1)
                expected = bytes([rank + 1] * cpu_page_size)
                assert chunk == expected, (
                    f"block {cpu_blk}, rank {rank}: first 16 bytes: {chunk[:16]!r}"
                )

    finally:
        for region in regions:
            region.cleanup()
        mmap_path = f"/dev/shm/vllm_offload_{instance_id}.mmap"
        if os.path.exists(mmap_path):
            os.unlink(mmap_path)


# ---------------------------------------------------------------------------
# test_tp_agnostic_contiguity — all workers' data for a block is contiguous
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [2, 4])
@torch.inference_mode()
def test_tp_agnostic_contiguity(
    default_vllm_config,
    num_workers: int,
) -> None:
    """Each logical CPU block's data from all TP workers must form a single
    contiguous region of num_workers * cpu_page_size bytes in the mmap."""
    set_random_seed(0)
    gpu_page_size_bytes = 512
    num_gpu_blocks = 32
    num_cpu_blocks = 8
    block_size_factor = 2
    num_tensors = 2
    device = "cuda:0"

    cpu_page_size = gpu_page_size_bytes * num_tensors
    instance_id = str(uuid.uuid4())
    total_mmap_bytes = num_cpu_blocks * num_workers * cpu_page_size

    regions: list[SharedMmapRegion] = []
    try:
        for rank in range(num_workers):
            regions.append(
                SharedMmapRegion(
                    instance_id=instance_id,
                    total_size_bytes=total_mmap_bytes,
                    num_blocks=num_cpu_blocks,
                    rank=rank,
                    num_workers=num_workers,
                    cpu_page_size=cpu_page_size,
                )
            )

        cpu_blocks_to_use = list(range(min(4, num_cpu_blocks)))

        for rank in range(num_workers):
            kv_caches = _build_kv_caches(
                num_tensors,
                num_gpu_blocks,
                gpu_page_size_bytes,
                device,
                fill_value=rank + 1,
            )
            with patch("vllm.v1.kv_offload.worker.cpu_gpu.pin_mmap_region"):
                handlers = CpuGpuOffloadingHandlers(
                    kv_caches=kv_caches,
                    block_size_factor=block_size_factor,
                    num_cpu_blocks=num_cpu_blocks,
                    mmap_region=regions[rank],
                    num_workers=num_workers,
                )

            gpu_blocks = list(range(len(cpu_blocks_to_use)))
            src_spec = GPULoadStoreSpec(gpu_blocks, group_sizes=(len(gpu_blocks),))
            dst_spec = CPULoadStoreSpec(cpu_blocks_to_use)

            handler = handlers.gpu_to_cpu_handler
            assert handler.transfer_async(rank, (src_spec, dst_spec))
            _wait_for_handler(handler, rank)

        # ---- verify contiguity ----
        raw = memoryview(regions[0].mmap_obj)
        row_size = num_workers * cpu_page_size

        for cpu_blk in cpu_blocks_to_use:
            row_offset = cpu_blk * row_size
            row_data = bytes(raw[row_offset : row_offset + row_size])

            # the row must be [rank0_data | rank1_data | ... | rankN_data]
            for rank in range(num_workers):
                worker_start = rank * cpu_page_size
                worker_data = row_data[worker_start : worker_start + cpu_page_size]
                expected = bytes([rank + 1] * cpu_page_size)
                assert worker_data == expected, (
                    f"block {cpu_blk}, rank {rank}: "
                    f"first 16 bytes: {worker_data[:16]!r}"
                )

    finally:
        for region in regions:
            region.cleanup()
        mmap_path = f"/dev/shm/vllm_offload_{instance_id}.mmap"
        if os.path.exists(mmap_path):
            os.unlink(mmap_path)
