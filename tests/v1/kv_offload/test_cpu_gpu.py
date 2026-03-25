# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time

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

NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
GPU_PAGE_SIZES = [512, 1024]
BLOCK_SIZE_FACTORS = [1, 3]
NUM_TENSORS = [4]
SEEDS = [0]
DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:0"]
NUM_MAPPINGS = [3]


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("block_size_factor", BLOCK_SIZE_FACTORS)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_tensors", NUM_TENSORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
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
