# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.ucc_communicator import UCCCommunicator
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

test_size_elements = 4 * 1024 * 1024


def _select_device_and_dtype(local_rank: int):
    if current_platform.is_cuda():
        device = torch.device(f"cuda:{local_rank}")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


def ucc_allreduce_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device, dtype = _select_device_and_dtype(local_rank)

        # Set device only for CUDA
        if current_platform.is_cuda():
            torch.cuda.set_device(device)
        # set_default_device may not exist in all torch versions
        if hasattr(torch, "set_default_device"):
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

        # Check if UCC is available
        if not UCCCommunicator.is_ucc_available():
            pytest.skip("UCC backend is not available in PyTorch.")

        # Create reference device group from TP group
        group = get_tensor_model_parallel_group().device_group  # pyright: ignore[reportDeprecated]

        # Try to create a UCC process group
        try:
            ucc_group = dist.new_group(backend="ucc")
        except Exception:
            pytest.skip("Failed to create UCC process group.")

        # Initialize UCC communicator
        ucc_communicator = UCCCommunicator(group=ucc_group, device=device)

        if ucc_communicator.disabled:
            pytest.skip("UCCCommunicator is disabled.")

        # Test direct UCC allreduce
        inp_direct_ucc = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )

        if not ucc_communicator.should_use_ucc_allreduce(inp_direct_ucc):
            pytest.skip(
                "UCCCommunicator isn't used for this world size and input size."
            )

        original_inp_direct_ucc = inp_direct_ucc.clone()
        out_direct_ucc = ucc_communicator.all_reduce(inp_direct_ucc)
        assert out_direct_ucc is not None

        # Compare with regular allreduce
        dist.all_reduce(original_inp_direct_ucc, group=group)

        # Tolerance based on dtype
        if dtype == torch.float32:
            atol, rtol = 1e-3, 1e-4
        else:
            atol, rtol = 2.5, 0.1
        torch.testing.assert_close(
            out_direct_ucc, original_inp_direct_ucc, atol=atol, rtol=rtol
        )

        # Test different reduction operations
        for op in [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN]:
            inp_op_test = torch.randint(1, 10, (1024,), dtype=dtype, device=device)
            original_inp_op_test = inp_op_test.clone()

            out_ucc_op = ucc_communicator.all_reduce(inp_op_test, op=op)
            if out_ucc_op is not None:
                dist.all_reduce(original_inp_op_test, op=op, group=group)
                torch.testing.assert_close(
                    out_ucc_op, original_inp_op_test, atol=atol, rtol=rtol
                )

        # Test tensor size threshold (avoid huge allocation by using meta)
        small_tensor = torch.ones(100, dtype=dtype, device=device)
        large_tensor = torch.empty(
            513 * 1024 * 1024, dtype=torch.uint8, device="meta"
        )  # > 512MB, meta device

        assert ucc_communicator.should_use_ucc_allreduce(small_tensor) is True
        assert ucc_communicator.should_use_ucc_allreduce(large_tensor) is False

        # Test device mismatch handling
        cpu_tensor = torch.ones(100, dtype=dtype, device="cpu")
        out_cpu = ucc_communicator.all_reduce(cpu_tensor)
        if out_cpu is not None:
            assert out_cpu.device == device


def ucc_availability_worker(local_rank: int, world_size: int):
    """Test UCC availability detection"""
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device, _ = _select_device_and_dtype(local_rank)
        if current_platform.is_cuda():
            torch.cuda.set_device(device)

        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12347",
            }
        )

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        # Test static method
        is_available = UCCCommunicator.is_ucc_available()
        assert isinstance(is_available, bool)

        if not is_available:
            pytest.skip("UCC backend is not available in PyTorch.")

        # Test with non-UCC group (should disable communicator)
        gloo_group = dist.new_group(backend="gloo")
        ucc_comm_with_gloo = UCCCommunicator(group=gloo_group, device=device)
        assert ucc_comm_with_gloo.disabled is True


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
def test_ucc_allreduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pipeline_parallel_size
):
    world_size = tp_size * pipeline_parallel_size

    # For CUDA, ensure enough GPUs; for CPU, proceed.
    if current_platform.is_cuda() and world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    mp.spawn(ucc_allreduce_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
def test_ucc_availability(
    monkeypatch: pytest.MonkeyPatch, tp_size, pipeline_parallel_size
):
    world_size = tp_size * pipeline_parallel_size

    if current_platform.is_cuda() and world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    mp.spawn(ucc_availability_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()


def test_ucc_communicator_initialization():
    """Basic check that static availability method works."""
    is_available = UCCCommunicator.is_ucc_available()
    assert isinstance(is_available, bool)


def test_ucc_static_methods():
    """Test static methods of UCCCommunicator"""
    # Test is_ucc_available static method
    is_available = UCCCommunicator.is_ucc_available()
    assert isinstance(is_available, bool)
    # The method should not crash regardless of environment
    # and should return a boolean value
