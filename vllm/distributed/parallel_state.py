# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Tensor and pipeline parallel groups."""
import contextlib
from multiprocessing import resource_tracker, shared_memory
from typing import List, Optional

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True

# Tensor model parallel group that the current rank belongs to.
_TP_DEVICE_GROUP: Optional[ProcessGroup] = None
_TP_CPU_GROUP: Optional[ProcessGroup] = None
_TP_PYNCCL_COMMUNICATOR = None
_TP_CA_COMMUNICATOR = None
# Pipeline model parallel group that the current rank belongs to.
_PP_DEVICE_GROUP: Optional[ProcessGroup] = None
_PP_CPU_GROUP: Optional[ProcessGroup] = None
_PP_PYNCCL_COMMUNICATOR = None

# when people blindly call `torch.distributed.all_reduce` etc,
# it will use this group. It is initialized with the `backend`
# parameter of `init_distributed_environment` below.
# Essentially, this is `torch.distributed.group.WORLD`.
# We leave a line here to note that this is device-specific.
# Note that this variable is not safe to use, because when users
# call `init_distributed_environment` first, and then destroy
# the process group themselves, this variable will keep a reference to the
# destroyed process group, which is not useful.
_DEVICE_WORLD_GROUP = None

# duing `init_distributed_environment`, we will also initialize a
# group with `gloo` backend, to allow direct coordination between
# processes through the CPU.
_CPU_WORLD_GROUP = None

# In summary, after calling `init_distributed_environment`, we will
# always have two groups: one for device-specific (and is the default)
# and one for CPU. All processes will be part of both groups.

# A list of global ranks for each pipeline group to ease calculation of the
# source rank when broadcasting from the first or last pipeline stage.
_PP_GLOBAL_RANKS: Optional[List[int]] = None

_LOCAL_RANK = -1


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def get_pp_pynccl_communicator():
    global _PP_PYNCCL_COMMUNICATOR
    return _PP_PYNCCL_COMMUNICATOR


def get_tp_pynccl_communicator():
    global _TP_PYNCCL_COMMUNICATOR
    return _TP_PYNCCL_COMMUNICATOR


def get_tp_ca_communicator():
    global _TP_CA_COMMUNICATOR
    return _TP_CA_COMMUNICATOR


def get_local_rank():
    global _LOCAL_RANK
    return _LOCAL_RANK


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d "
        "distributed_init_method=%s backend=%s", world_size, rank, local_rank,
        distributed_init_method, backend)
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment")
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank)
        global _DEVICE_WORLD_GROUP, _CPU_WORLD_GROUP
        _DEVICE_WORLD_GROUP = torch.distributed.group.WORLD
        ranks = list(range(torch.distributed.get_world_size()))
        _CPU_WORLD_GROUP = torch.distributed.new_group(ranks=ranks,
                                                       backend="gloo")
        # set the local rank
        # local_rank is not available in torch ProcessGroup,
        # see https://github.com/pytorch/pytorch/issues/122816
        if local_rank == -1:
            # local rank not set, this usually happens in single-node
            # setting, where we can use rank as local rank
            if distributed_init_method == "env://":
                local_rank = envs.LOCAL_RANK
            else:
                local_rank = rank
        global _LOCAL_RANK
        _LOCAL_RANK = local_rank
        # A small all_reduce for warmup.
        data = torch.zeros(1)
        if torch.cuda.is_available():
            data = data.to(device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del data


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend()

    if (world_size !=
            tensor_model_parallel_size * pipeline_model_parallel_size):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    num_tensor_model_parallel_groups: int = (world_size //
                                             tensor_model_parallel_size)
    num_pipeline_model_parallel_groups: int = (world_size //
                                               pipeline_model_parallel_size)
    rank = torch.distributed.get_rank()

    # Build the tensor model-parallel groups.
    global _TP_DEVICE_GROUP, _TP_CPU_GROUP
    global _TP_PYNCCL_COMMUNICATOR, _TP_CA_COMMUNICATOR
    assert _TP_DEVICE_GROUP is None, (
        "tensor model parallel group is already initialized")
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size))
        group = torch.distributed.new_group(ranks, backend=backend)
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            _TP_DEVICE_GROUP = group
            _TP_CPU_GROUP = cpu_group

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    if tensor_model_parallel_size > 1:
        _TP_PYNCCL_COMMUNICATOR = PyNcclCommunicator(
            group=_TP_CPU_GROUP,
            device=_LOCAL_RANK,
        )

    # Initialize a custom fast all-reduce implementation.
    if _ENABLE_CUSTOM_ALL_REDUCE:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
        _TP_CA_COMMUNICATOR = CustomAllreduce(
            group=_TP_CPU_GROUP,
            device=_LOCAL_RANK,
        )

    # Build the pipeline model-parallel groups.
    global _PP_DEVICE_GROUP, _PP_CPU_GROUP
    global _PP_PYNCCL_COMMUNICATOR
    global _PP_GLOBAL_RANKS
    assert _PP_DEVICE_GROUP is None, (
        "pipeline model parallel group is already initialized")
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group = torch.distributed.new_group(ranks, backend=backend)
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            _PP_DEVICE_GROUP = group
            _PP_CPU_GROUP = cpu_group
            _PP_GLOBAL_RANKS = ranks

    if pipeline_model_parallel_size > 1:
        _PP_PYNCCL_COMMUNICATOR = PyNcclCommunicator(
            group=_PP_CPU_GROUP,
            device=_LOCAL_RANK,
        )


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend()
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size,
                                  pipeline_model_parallel_size, backend)
        return

    assert (
        get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    ), ("tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    assert (get_pipeline_model_parallel_world_size(
    ) == pipeline_model_parallel_size), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{get_pipeline_model_parallel_world_size()=} vs. "
        f"{pipeline_model_parallel_size=}")


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (_TP_DEVICE_GROUP is not None and _PP_DEVICE_GROUP is not None)


def get_cpu_world_group():
    """Get the CPU world group."""
    assert _CPU_WORLD_GROUP is not None, ("CPU world group is not initialized")
    return _CPU_WORLD_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TP_DEVICE_GROUP is not None, (
        "tensor model parallel group is not initialized")
    return _TP_DEVICE_GROUP


def get_tensor_model_parallel_cpu_group():
    """Get the tensor model parallel cpu group the caller rank belongs to."""
    assert _TP_CPU_GROUP is not None, (
        "tensor model parallel cpu group is not initialized")
    return _TP_CPU_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PP_DEVICE_GROUP is not None, (
        "pipeline model parallel group is not initialized")
    return _PP_DEVICE_GROUP


def get_pipeline_model_parallel_cpu_group():
    """Get the pipeline model parallel cpu group the caller rank belongs to."""
    assert _PP_CPU_GROUP is not None, (
        "pipeline model parallel cpu group is not initialized")
    return _PP_CPU_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(
        group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return torch.distributed.get_world_size(
        group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return torch.distributed.get_rank(
        group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PP_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    return _PP_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PP_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PP_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PP_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PP_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that precedes the caller in the pipeline"""
    assert _PP_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PP_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP_DEVICE_GROUP
    if _TP_DEVICE_GROUP:
        torch.distributed.destroy_process_group(_TP_DEVICE_GROUP)
    _TP_DEVICE_GROUP = None
    global _TP_CPU_GROUP
    if _TP_CPU_GROUP:
        torch.distributed.destroy_process_group(_TP_CPU_GROUP)
    _TP_CPU_GROUP = None
    global _TP_PYNCCL_COMMUNICATOR
    _TP_PYNCCL_COMMUNICATOR = None

    global _PP_DEVICE_GROUP
    if _PP_DEVICE_GROUP:
        torch.distributed.destroy_process_group(_PP_DEVICE_GROUP)
    _PP_DEVICE_GROUP = None
    global _PP_GLOBAL_RANKS
    _PP_GLOBAL_RANKS = None


def is_in_the_same_node(pg: ProcessGroup):
    """
    This is a collective operation that checks if all processes in the group
    are in the same node. It tests if all processes are attached to the same
    memory system (shared access to shared memory).
    """
    assert torch.distributed.get_backend(
        pg) != torch.distributed.Backend.NCCL, (
            "is_in_the_same_node should be tested with a non-NCCL group.")
    # local rank inside the group
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    # global ranks of the processes in the group
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == 0:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[:len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list([shm.name],
                                                        src=ranks[0],
                                                        group=pg)
                is_in_the_same_node[0] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(recv,
                                                        src=ranks[0],
                                                        group=pg)
                name = recv[0]
                shm = shared_memory.SharedMemory(name=name)
                if shm.buf[:len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == 0:
            if shm:
                shm.unlink()
        else:
            if shm:
                # fix to https://stackoverflow.com/q/62748654/9191338
                resource_tracker.unregister(
                    shm._name, "shared_memory")  # type: ignore[attr-defined]
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return is_in_the_same_node.sum().item() == world_size
