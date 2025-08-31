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
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tp_group,
                                             init_distributed_environment,
                                             initialize_model_parallel)
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

test_size_elements = 4 * 1024 * 1024


def symm_mem_allreduce_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables({
            'RANK': str(local_rank),
            'LOCAL_RANK': str(local_rank),
            'WORLD_SIZE': str(world_size),
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': '12345',
        })

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(CudaCommunicator,
                                        get_tp_group().device_communicator)
        symm_mem_comm = cuda_communicator.symm_mem_comm
        if symm_mem_comm is None or symm_mem_comm.disabled:
            pytest.skip("SymmMemCommunicator is not available or disabled.")

        inp_direct_symm_mem = torch.randint(1,
                                            23, (test_size_elements, ),
                                            dtype=dtype,
                                            device=device)
        if not symm_mem_comm.should_use_symm_mem(inp_direct_symm_mem):
            pytest.skip(
                "SymmMemCommunicator isn't used for this world and input size."
            )

        original_inp_direct_symm_mem = inp_direct_symm_mem.clone()
        out_direct_symm_mem = symm_mem_comm.all_reduce(inp_direct_symm_mem)
        assert out_direct_symm_mem is not None

        group = get_tensor_model_parallel_group().device_group
        dist.all_reduce(original_inp_direct_symm_mem, group=group)
        torch.testing.assert_close(out_direct_symm_mem,
                                   original_inp_direct_symm_mem,
                                   atol=2.5,
                                   rtol=0.1)

        # Test tensor_model_parallel_all_reduce which should use symm_mem
        inp_tensor_parallel = torch.randint(-23,
                                            1, (test_size_elements, ),
                                            dtype=dtype,
                                            device=device)
        original_inp_tensor_parallel = inp_tensor_parallel.clone()
        out_tensor_parallel = tensor_model_parallel_all_reduce(
            inp_tensor_parallel)
        dist.all_reduce(original_inp_tensor_parallel, group=group)
        torch.testing.assert_close(out_tensor_parallel,
                                   original_inp_tensor_parallel,
                                   atol=2.5,
                                   rtol=0.1)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.")
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"],
                    reason="Only test on CUDA")
def test_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch, tp_size,
                            pipeline_parallel_size):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    # Enable SymmMemCommunicator
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1")

    mp.spawn(symm_mem_allreduce_worker, args=(world_size, ), nprocs=world_size)
    cleanup_dist_env_and_memory()
