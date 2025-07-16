# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import typing

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tp_group)
from vllm.platforms import current_platform

from ..utils import (ensure_model_parallel_initialized,
                     init_test_distributed_environment, multi_process_parallel)

torch.manual_seed(42)
random.seed(44)

test_size_elements = 4 * 1024 * 1024


@ray.remote(num_gpus=1, max_calls=1)
def symm_mem_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        init_test_distributed_environment(tp_size, pp_size, rank,
                                          distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        dtype = torch.bfloat16

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
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.parametrize("test_target", [symm_mem_allreduce])
def test_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch, tp_size,
                            pipeline_parallel_size, test_target):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    # Enable SymmMemCommunicator
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1")

    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size,
                           test_target)
