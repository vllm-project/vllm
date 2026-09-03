# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import logging
import random
import typing

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from tests.utils import ensure_current_vllm_config
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
        with ensure_current_vllm_config():
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
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    # Enable SymmMemCommunicator
    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    mp.spawn(nccl_symm_mem_allreduce_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()


def nccl_symm_mem_allgather_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
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
                "MASTER_PORT": "12346",
            }
        )

        init_distributed_environment()
        with ensure_current_vllm_config():
            initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        if get_nccl_mem_pool() is None:
            pytest.skip(
                "NCCL allocator compilation failed (probably missing NCCL headers)."
            )
        if not is_symmetric_memory_enabled():
            pytest.skip("NCCL symmetric memory is disabled.")

        per_rank_size = test_size_elements // world_size
        input_tensor = torch.randint(
            1, 23, (per_rank_size,), dtype=dtype, device=device
        )
        output = cuda_communicator.all_gatherv(input_tensor, dim=0)

        group = get_tp_group().device_group
        expected = torch.empty(test_size_elements, dtype=dtype, device=device)
        dist.all_gather_into_tensor(expected, input_tensor, group=group)
        torch.testing.assert_close(output, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NCCL symmetric memory is only available for CUDA platforms.",
)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_nccl_symm_mem_allgather(monkeypatch: pytest.MonkeyPatch, world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    mp.spawn(nccl_symm_mem_allgather_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()


def test_bounded_symm_scratch_reuses_capacity(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import vllm.distributed.device_communicators.pynccl_allocator as allocator

    monkeypatch.setenv("VLLM_NCCL_SYMM_MEM_BOUNDED_SCRATCH", "1")
    monkeypatch.setenv("VLLM_NCCL_SYMM_MEM_ALIAS_DIAGNOSTICS", "1")
    monkeypatch.setattr(
        allocator,
        "nccl_symm_mem_context",
        lambda _communicator: contextlib.nullcontext(),
    )

    communicator = object.__new__(CudaCommunicator)
    communicator.pynccl_comm = object()
    communicator.unique_name = "test"
    device = torch.device("cpu")
    caplog.set_level(logging.WARNING)

    first = communicator._get_symm_scratch("ag_out", (3, 4), torch.float32, device)
    second = communicator._get_symm_scratch("ag_out", (2, 6), torch.float32, device)
    assert first.shape == (3, 4)
    assert second.shape == (2, 6)
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert second.untyped_storage().nbytes() == 16 * second.element_size()
    assert "NCCL bounded symmetric scratch live alias before reuse" in caplog.text

    different_role = communicator._get_symm_scratch(
        "rs_in", (3, 4), torch.float32, device
    )
    assert different_role.untyped_storage().data_ptr() != (
        second.untyped_storage().data_ptr()
    )

    first_storage_ptr = first.untyped_storage().data_ptr()
    communicator.seal_bounded_symm_scratch("unit_test")
    grown = communicator._get_symm_scratch("ag_out", (5, 4), torch.float32, device)
    assert grown.shape == (5, 4)
    assert grown.untyped_storage().nbytes() == 32 * grown.element_size()
    assert len(communicator._symm_scratch_bufs) == 2
    assert len(communicator._symm_scratch_retired) == 1
    assert (
        communicator._symm_scratch_retired[0].untyped_storage().data_ptr()
        == first_storage_ptr
    )
    assert "NCCL bounded symmetric scratch late grow after seal" in caplog.text


def test_bounded_symm_scratch_covers_symmetric_allreduce(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.distributed.device_communicators.cuda_communicator as cuda_module
    import vllm.distributed.device_communicators.pynccl_allocator as allocator

    class FakePyNcclCommunicator:
        world_size = 8

        @staticmethod
        def all_reduce(input_: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
            output.copy_(input_)
            return output

    monkeypatch.setenv("VLLM_NCCL_SYMM_MEM_BOUNDED_SCRATCH", "1")
    monkeypatch.setattr(
        allocator,
        "nccl_symm_mem_context",
        lambda _communicator: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        cuda_module, "should_nccl_symm_mem_allreduce", lambda *_args: True
    )

    communicator = object.__new__(CudaCommunicator)
    communicator.pynccl_comm = FakePyNcclCommunicator()
    communicator.fi_ar_comm = None
    communicator.unique_name = "test"
    communicator._bounded_symm_scratch_enabled = True
    communicator._symm_scratch_sealed = False
    input_ = torch.arange(12, dtype=torch.float32).view(3, 4)

    output = communicator.all_reduce(input_)
    assert torch.equal(output, input_)
    cache = communicator._symm_scratch_bufs
    assert ("flat_capacity", "ar_in", input_.dtype, input_.device) in cache
    assert ("flat_capacity", "ar_out", input_.dtype, input_.device) in cache
    ar_out = cache[("flat_capacity", "ar_out", input_.dtype, input_.device)]
    assert output.untyped_storage().data_ptr() != (ar_out.untyped_storage().data_ptr())

    first_result = output.clone()
    second_input = input_ + 100
    second_output = communicator.all_reduce(second_input)
    assert torch.equal(output, first_result)
    assert torch.equal(second_output, second_input)
    assert second_output.untyped_storage().data_ptr() != (
        ar_out.untyped_storage().data_ptr()
    )


def nccl_symm_mem_reduce_scatter_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
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
                "MASTER_PORT": "12347",
            }
        )

        init_distributed_environment()
        with ensure_current_vllm_config():
            initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        if get_nccl_mem_pool() is None:
            pytest.skip(
                "NCCL allocator compilation failed (probably missing NCCL headers)."
            )
        if not is_symmetric_memory_enabled():
            pytest.skip("NCCL symmetric memory is disabled.")

        per_rank_size = test_size_elements // world_size
        input_tensor = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )
        input_clone = input_tensor.clone()
        output = cuda_communicator.reduce_scatter(input_tensor, dim=0)

        group = get_tp_group().device_group
        expected = torch.empty(per_rank_size, dtype=dtype, device=device)
        dist.reduce_scatter_tensor(expected, input_clone, group=group)
        torch.testing.assert_close(output, expected, atol=2.5, rtol=0.1)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NCCL symmetric memory is only available for CUDA platforms.",
)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_nccl_symm_mem_reduce_scatter(monkeypatch: pytest.MonkeyPatch, world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    mp.spawn(nccl_symm_mem_reduce_scatter_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()
