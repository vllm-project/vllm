# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import typing

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.pynccl import register_nccl_symmetric_ops
from vllm.distributed.device_communicators.pynccl_allocator import (
    get_nccl_mem_pool,
    is_symmetric_memory_enabled,
)
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

test_size_elements = 4 * 1024 * 1024


def nccl_symm_mem_allreduce_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        pynccl_comm = cuda_communicator.pynccl_comm
        if get_nccl_mem_pool() is None:
            pytest.skip(
                "NCCL allocator compilation failed (probably missing NCCL headers)."
            )
        if not is_symmetric_memory_enabled():
            pytest.skip("NCCL symmetric memory allreduce is disabled.")

        register_nccl_symmetric_ops(pynccl_comm)
        input = torch.randint(1, 23, (test_size_elements,), dtype=dtype, device=device)
        input_clone = input.clone()
        output = torch.ops.vllm.all_reduce_symmetric_with_copy(input)
        assert output is not None

        group = get_tp_group().device_group
        dist.all_reduce(input_clone, group=group)
        torch.testing.assert_close(output, input_clone, atol=2.5, rtol=0.1)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NCCLSymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_nccl_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch, world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    # Enable SymmMemCommunicator
    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    mp.spawn(nccl_symm_mem_allreduce_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()
