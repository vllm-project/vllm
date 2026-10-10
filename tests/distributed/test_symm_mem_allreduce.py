# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import random
import typing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators import symm_mem
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
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


@pytest.mark.parametrize(
    ("same_node", "local_fabric", "peer_fabric", "enabled"),
    [
        (True, False, False, True),
        (False, False, False, False),
        (False, False, True, False),
        (False, True, False, False),
        (False, True, True, True),
    ],
)
def test_symm_mem_skips_unsupported_cross_node_rendezvous(
    monkeypatch, same_node, local_fabric, peer_fabric, enabled
):
    """Never enter potentially blocking rendezvous without support on all ranks."""
    group = SimpleNamespace(group_name="test")
    backend = SimpleNamespace(
        empty=Mock(return_value=torch.empty(1)),
        rendezvous=Mock(return_value=SimpleNamespace(multicast_ptr=1)),
    )
    fabric_probe = Mock(return_value=local_fabric)
    monkeypatch.setattr(symm_mem, "symm_mem_available", True)
    monkeypatch.setattr(symm_mem, "torch_symm_mem", backend, raising=False)
    monkeypatch.setattr(symm_mem.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        symm_mem.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(as_version_str=lambda: "10.0"),
    )
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda _device: None)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(symm_mem.dist, "get_world_size", lambda _group: 4)
    monkeypatch.setattr(
        symm_mem, "in_the_same_node_as", lambda *_args, **_kwargs: [True, same_node]
    )
    monkeypatch.setattr(symm_mem, "has_mnnvl_fabric_support", fabric_probe)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    def agree_on_fabric(support, *, op, group):
        assert op == dist.ReduceOp.MIN
        assert support.device.type == "cpu"
        if not peer_fabric:
            support.zero_()

    agreement = Mock(side_effect=agree_on_fabric)
    monkeypatch.setattr(symm_mem.dist, "all_reduce", agreement)
    communicator = symm_mem.SymmMemCommunicator(group, "cuda:0", max_size_override=2)

    assert communicator.disabled is not enabled
    assert backend.empty.call_count == int(enabled)
    assert backend.rendezvous.call_count == int(enabled)
    assert fabric_probe.call_count == int(not same_node)
    assert agreement.call_count == int(not same_node)


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
