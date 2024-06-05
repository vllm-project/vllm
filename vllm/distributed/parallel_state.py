# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Tensor and pipeline parallel groups."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from torch.distributed import Backend, ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


class GroupCoordinator:
    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_pynccl: bool  # a hint of whether to use PyNccl
    use_custom_allreduce: bool  # a hint of whether to use CustomAllreduce
    # communicators are only created for world size > 1
    pynccl_comm: Optional[PyNcclCommunicator]  # PyNccl communicator
    ca_comm: Optional[CustomAllreduce]  # Custom allreduce communicator

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool,
        use_custom_allreduce: bool,
    ):

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_pynccl = use_pynccl
        self.use_custom_allreduce = use_custom_allreduce

        self.pynccl_comm: Optional[PyNcclCommunicator]
        if use_pynccl and self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )
        else:
            self.pynccl_comm = None

        self.ca_comm: Optional[CustomAllreduce]
        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
            )
        else:
            self.ca_comm = None

    @contextmanager
    def graph_capture(
            self, graph_capture_context: Optional[GraphCaptureContext] = None):
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        ca_comm = self.ca_comm
        maybe_ca_context = nullcontext(
        ) if ca_comm is None else ca_comm.capture()
        with torch.cuda.stream(stream), maybe_ca_context:
            # In graph mode, we have to be very careful about the collective
            # operations. The current status is:
            #     allreduce \ Mode   |  Eager  |  Graph  |
            # --------------------------------------------
            # custom allreduce       | enabled | enabled |
            # PyNccl                 | disabled| enabled |
            # torch.distributed      | enabled | disabled|
            #
            # Note that custom allreduce will have a runtime check, if the
            #  tensor size is too large, it will fallback to the next
            #  available option.
            # In summary: When using CUDA graph, we use
            #  either custom all-reduce kernel or pynccl. When not using
            #  CUDA graph, we use either custom all-reduce kernel or
            #  PyTorch NCCL. We always prioritize using custom all-reduce
            #  kernel but fall back to PyTorch or pynccl if it is
            #  disabled or not supported.
            pynccl_comm = self.pynccl_comm
            maybe_pynccl_context: Any
            if not pynccl_comm:
                maybe_pynccl_context = nullcontext()
            else:
                maybe_pynccl_context = pynccl_comm.change_state(
                    enable=True, stream=torch.cuda.current_stream())
            with maybe_pynccl_context:
                yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        NOTE: This operation will be applied in-place or out-of-place. 
        Always assume this function modifies its input, but use the return
        value as the output.
        """
        ca_comm = self.ca_comm

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        if ca_comm is not None:
            out = ca_comm.custom_all_reduce(input_)
            if out is not None:
                return out
        pynccl_comm = self.pynccl_comm
        if (pynccl_comm is not None and not pynccl_comm.disabled):
            pynccl_comm.all_reduce(input_)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty((world_size, ) + input_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        torch.distributed.all_gather_into_tensor(output_tensor,
                                                 input_,
                                                 group=self.device_group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def destroy(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None


_WORLD: Optional[GroupCoordinator] = None


def get_world() -> GroupCoordinator:
    assert _WORLD is not None, ("world group is not initialized")
    return _WORLD


_TP: Optional[GroupCoordinator] = None


def get_tp() -> GroupCoordinator:
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP


_PP: Optional[GroupCoordinator] = None


def get_pp() -> GroupCoordinator:
    assert _PP is not None, (
        "pipeline model parallel group is not initialized")
    return _PP


logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True

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
    global _TP
    assert _TP is None, ("tensor model parallel group is already initialized")
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)
    global _TP
    _TP = GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=_LOCAL_RANK,
        torch_distributed_backend=backend,
        use_pynccl=True,
        use_custom_allreduce=_ENABLE_CUSTOM_ALL_REDUCE,
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
    return (_TP is not None and _PP_DEVICE_GROUP is not None)


def get_cpu_world_group():
    """Get the CPU world group."""
    assert _CPU_WORLD_GROUP is not None, ("CPU world group is not initialized")
    return _CPU_WORLD_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP


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
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP_DEVICE_GROUP
    if _PP_DEVICE_GROUP:
        torch.distributed.destroy_process_group(_PP_DEVICE_GROUP)
    _PP_DEVICE_GROUP = None
    global _PP_GLOBAL_RANKS
    _PP_GLOBAL_RANKS = None
