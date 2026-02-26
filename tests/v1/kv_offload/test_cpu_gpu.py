# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers

BACKENDS_TO_TEST = [FlashAttentionBackend]

if not current_platform.is_rocm():
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    BACKENDS_TO_TEST.append(FlashInferBackend)

    from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnMLABackend

    BACKENDS_TO_TEST.append(FlashAttnMLABackend)

NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
KERNEL_BLOCK_SIZES = [16]
LOGICAL_BLOCK_SIZES = [16, 32]
LOGICAL_BLOCKS_PER_CPU_BLOCK = [1, 3]
HEAD_SIZES = [64]
NUM_HEADS = [8]
NUM_LAYERS = [4]
DTYPES = [torch.bfloat16]
SEEDS = [0]
CUDA_DEVICES = ["cuda:0"]
NUM_MAPPINGS = [3]


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("kernel_block_size", KERNEL_BLOCK_SIZES)
@pytest.mark.parametrize("logical_block_size", LOGICAL_BLOCK_SIZES)
@pytest.mark.parametrize("logical_blocks_per_cpu_block", LOGICAL_BLOCKS_PER_CPU_BLOCK)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_transfer(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings: int,
    head_size: int,
    num_heads: int,
    kernel_block_size: int,
    logical_block_size: int,
    logical_blocks_per_cpu_block: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    num_layers: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)

    # create per-layer GPU KV caches based on available attn_backends
    attn_backends_list = BACKENDS_TO_TEST

    assert logical_block_size % kernel_block_size == 0
    kernel_blocks_per_gpu_block = logical_block_size // kernel_block_size
    num_gpu_kernel_blocks = num_gpu_blocks * kernel_blocks_per_gpu_block

    gpu_caches = {}
    attn_backends = {}
    for i in range(num_layers):
        layer_name = f"layer {i}"

        attn_backend = attn_backends_list[i % len(attn_backends_list)]
        attn_backends[layer_name] = attn_backend

        gpu_cache_shape = attn_backend.get_kv_cache_shape(
            num_gpu_kernel_blocks, kernel_block_size, num_heads, head_size
        )
        gpu_caches[layer_name] = torch.rand(gpu_cache_shape, dtype=dtype, device=device)

    # create handler
    cpu_block_size = logical_blocks_per_cpu_block * logical_block_size
    kernel_blocks_per_cpu_block = cpu_block_size // kernel_block_size
    handlers = CpuGpuOffloadingHandlers(
        attn_backends=attn_backends,
        gpu_block_size=logical_block_size,
        cpu_block_size=cpu_block_size,
        num_cpu_blocks=num_cpu_blocks,
        gpu_caches=gpu_caches,
    )

    # select block mappings
    gpu_blocks = random.sample(
        range(num_gpu_blocks), num_mappings * logical_blocks_per_cpu_block
    )
    cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

    # convert gpu blocks to kernel block size
    gpu_blocks_in_kernel_block_size = []
    for gpu_block in gpu_blocks:
        base_block_id = gpu_block * kernel_blocks_per_gpu_block
        for i in range(kernel_blocks_per_gpu_block):
            gpu_blocks_in_kernel_block_size.append(i + base_block_id)

    # convert cpu blocks to gpu block size
    cpu_blocks_in_kernel_block_size = []
    for cpu_block in cpu_blocks:
        base_block_id = cpu_block * kernel_blocks_per_cpu_block
        for i in range(kernel_blocks_per_cpu_block):
            cpu_blocks_in_kernel_block_size.append(i + base_block_id)

    # maybe skip some GPU block to test reading from the middle of a CPU block
    if not gpu_to_cpu:
        gpu_blocks_to_skip = logical_blocks_per_cpu_block - 1
        gpu_blocks = gpu_blocks[gpu_blocks_to_skip:]
        kernel_blocks_to_skip = gpu_blocks_to_skip * kernel_blocks_per_gpu_block
        gpu_blocks_in_kernel_block_size = gpu_blocks_in_kernel_block_size[
            kernel_blocks_to_skip:
        ]
        cpu_blocks_in_kernel_block_size = cpu_blocks_in_kernel_block_size[
            kernel_blocks_to_skip:
        ]

    # set transfer direction
    if gpu_to_cpu:
        handler = handlers.gpu_to_cpu_handler
        src_spec_class = GPULoadStoreSpec
        dst_spec_class = CPULoadStoreSpec
        src_blocks = gpu_blocks
        dst_blocks = cpu_blocks
        src_blocks_in_kernel_block_size = gpu_blocks_in_kernel_block_size
        dst_blocks_in_kernel_block_size = cpu_blocks_in_kernel_block_size
        dst_size_in_kernel_blocks = num_cpu_blocks * kernel_blocks_per_cpu_block
    else:
        handler = handlers.cpu_to_gpu_handler
        src_spec_class = CPULoadStoreSpec
        dst_spec_class = GPULoadStoreSpec
        src_blocks = cpu_blocks
        dst_blocks = gpu_blocks
        src_blocks_in_kernel_block_size = cpu_blocks_in_kernel_block_size
        dst_blocks_in_kernel_block_size = gpu_blocks_in_kernel_block_size
        dst_size_in_kernel_blocks = num_gpu_blocks * kernel_blocks_per_gpu_block

    # build dst -> src mapping
    dst_to_src = {}
    for src_block, dst_block in zip(
        src_blocks_in_kernel_block_size, dst_blocks_in_kernel_block_size
    ):
        dst_to_src[dst_block] = src_block

    # build transfer specs
    src_spec = src_spec_class(src_blocks)
    dst_spec = dst_spec_class(dst_blocks)

    # clone src and dst tensors before transfer
    orig_src_caches = [x.clone() for x in handler.src_tensors]
    orig_dst_caches = [x.clone() for x in handler.dst_tensors]

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
            assert (
                finished[0].transfer_size
                == handler.total_block_size_in_bytes
                * handler.dst_block_size_factor
                * len(dst_blocks)
            )
            assert finished[0].transfer_time > 0
            assert finished[0].transfer_time < (time.time() - start_time)
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_caches, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors
    for dst_block in range(dst_size_in_kernel_blocks):
        src_block_candidate = dst_to_src.get(dst_block)
        for src_cache, dst_cache, orig_dst_cache in zip(
            handler.src_tensors,
            handler.dst_tensors,
            orig_dst_caches,
        ):
            if src_block_candidate is not None:
                expected_value = src_cache[src_block_candidate]
            else:
                expected_value = orig_dst_cache[dst_block]
            torch.testing.assert_close(dst_cache[dst_block].cpu(), expected_value.cpu())
