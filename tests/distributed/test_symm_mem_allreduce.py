# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import random
import types
import typing

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

test_size_elements = 1024 * 1024


def symm_mem_allreduce_worker(local_rank: int, world_size: int, q: mp.Queue):
    monkeypatch = pytest.MonkeyPatch()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))

    with monkeypatch.context() as m, set_current_vllm_config(config):
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
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
        symm_mem_comm = cuda_communicator.symm_mem_comm
        if symm_mem_comm is None or symm_mem_comm.disabled:
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator is not available or disabled.")
            return

        inp_direct_symm_mem = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )
        if not symm_mem_comm.should_use_symm_mem(inp_direct_symm_mem):
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator isn't used for this world and input size.")
            return

        original_inp_direct_symm_mem = inp_direct_symm_mem.clone()
        out_direct_symm_mem = symm_mem_comm.all_reduce(inp_direct_symm_mem)
        assert out_direct_symm_mem is not None

        group = get_tp_group().device_group
        dist.all_reduce(original_inp_direct_symm_mem, group=group)
        torch.testing.assert_close(
            out_direct_symm_mem, original_inp_direct_symm_mem, atol=2.5, rtol=0.1
        )

        # Test tensor_model_parallel_all_reduce which should use symm_mem
        inp_tensor_parallel = torch.randint(
            -23, 1, (test_size_elements,), dtype=dtype, device=device
        )
        original_inp_tensor_parallel = inp_tensor_parallel.clone()
        out_tensor_parallel = tensor_model_parallel_all_reduce(inp_tensor_parallel)
        dist.all_reduce(original_inp_tensor_parallel, group=group)
        torch.testing.assert_close(
            out_tensor_parallel, original_inp_tensor_parallel, atol=2.5, rtol=0.1
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_symm_mem_allreduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pipeline_parallel_size
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    q = mp.get_context("spawn").Queue()
    mp.spawn(symm_mem_allreduce_worker, args=(world_size, q), nprocs=world_size)
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
        if val is not None:
            pytest.skip(val)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_dp_with_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch):
    world_size = 4
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    # Verify that the DataParallel runs without error
    engine_args = EngineArgs(
        model="distilbert/distilgpt2",
        enforce_eager=True,
        enable_prefix_caching=True,
        data_parallel_size=2,
        tensor_parallel_size=2,
        data_parallel_backend="mp",
    )
    LLMEngine.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "capability,world_size,force_multimem,expected",
    [
        # Capabilities that list the world size as multimem-capable.
        ("9.0", 4, None, True),
        ("9.0", 8, None, True),
        # Same capability, world size that falls back to two-shot.
        ("9.0", 2, None, False),
        ("10.0", 4, None, False),
        # Unknown capability must not raise; two-shot is the safe default.
        ("8.0", 2, None, False),
        ("8.0", 8, None, False),
        # Explicit override wins either way (used by benchmarks/tests).
        ("9.0", 2, True, True),
        ("9.0", 8, False, False),
    ],
)
def test_uses_multimem(capability, world_size, force_multimem, expected):
    """`_uses_multimem` decides whether a multicast pointer is required.

    Guards against re-introducing a multicast capability check that disables
    the two-shot path, which never uses multicast.
    """
    comm = object.__new__(SymmMemCommunicator)
    comm.device_capability = capability
    comm.world_size = world_size
    assert comm._uses_multimem(force_multimem) is expected


@pytest.mark.parametrize(
    "capability,world_size,expected_op",
    [
        # Multimem world size for this capability -> multimem kernel.
        ("9.0", 4, "multimem_all_reduce_"),
        # Same capability, world size 2 falls back to two-shot.
        ("9.0", 2, "two_shot_all_reduce_"),
        ("10.0", 4, "two_shot_all_reduce_"),
    ],
)
def test_all_reduce_dispatch(monkeypatch, capability, world_size, expected_op):
    """`all_reduce` must pick the kernel `_uses_multimem` advertises.

    The guard in ``__init__`` and the dispatch here have to agree; they were
    previously separate expressions and disagreed about whether multicast was
    required.
    """
    called = []

    class _FakeOps:
        @staticmethod
        def multimem_all_reduce_(*args, **kwargs):
            called.append("multimem_all_reduce_")

        @staticmethod
        def two_shot_all_reduce_(*args, **kwargs):
            called.append("two_shot_all_reduce_")

    monkeypatch.setattr(torch.ops, "symm_mem", _FakeOps, raising=False)

    comm = object.__new__(SymmMemCommunicator)
    comm.disabled = False
    comm.dtype = torch.bfloat16
    comm.max_size = 1 << 20
    comm.device_capability = capability
    comm.world_size = world_size
    comm.force_multimem = None
    comm.group = types.SimpleNamespace(group_name="test_group")
    comm.buffer = torch.zeros(1024, dtype=torch.bfloat16)

    inp = torch.ones(1024, dtype=torch.bfloat16)
    out = comm.all_reduce(inp)

    assert called == [expected_op]
    assert out is not None and out.shape == inp.shape
