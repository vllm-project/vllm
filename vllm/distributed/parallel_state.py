# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to 
 initialize the model parallel groups and disaggregated prefill parallel 
 groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""
import time
import contextlib
import pickle
import logging
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch
import queue

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.distributed.group_coordinator import GroupCoordinator
import vllm.distributed.distributed_kv as dist_kv




_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, ("world group is not initialized")
    return _WORLD


def init_world_group(ranks: List[List[int]], local_rank: int,
                     backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=False,
        use_tpu_communicator=False,
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
) -> GroupCoordinator:
    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=True,
        use_custom_allreduce=use_custom_allreduce,
        use_tpu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
    )


_TP: Optional[GroupCoordinator] = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP


# kept for backward compatibility
get_tensor_model_parallel_group = get_tp_group

_PP: Optional[GroupCoordinator] = None


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, (
        "pipeline model parallel group is not initialized")
    return _PP


# kept for backward compatibility
get_pipeline_model_parallel_group = get_pp_group

_DISAGG: Optional[dist_kv.DistributedKVCoordinator] = None


def get_disagg_group() -> dist_kv.DistributedKVCoordinator:
    assert _DISAGG is not None, (
        "disaggregated prefill parallel group is not initialized")
    return _DISAGG


@contextmanager
def graph_capture():
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    with get_tp_group().graph_capture() as context, get_pp_group(
    ).graph_capture(context):
        yield context


logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def include_decoding_groups_if_disagg_enabled(
    groups: List[List[int]],
    world_size: int,
) -> List[List[int]]:
    """
        Include the distributed group for decode
        Only for disaggregated prefill
        
        Example:
            Original group: [ [0,1], [2,3] ], world_size = 4
            Extended: [ [0,1], [2,3], [4,5], [6,7] ]
        Arguments:
            groups: original distributed group
            world_size: the vLLM world size, which is half of torch.distributed.get_world_size()
    """

    if dist_kv.IS_DISTRIBUTED_KV_INSTANCE:
        new_groups = []
        for group in groups:
            new_groups.append([rank for rank in group])
        for group in groups:
            new_groups.append([rank + world_size for rank in group])
        return new_groups
    else:
        return groups


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
        maybe_disagg_world_size = world_size
        maybe_disagg_rank = rank
        if dist_kv.IS_DISTRIBUTED_KV_INSTANCE:
            maybe_disagg_world_size = world_size * 2
            logger.debug("Disaggregated prefill enabled.")
            if dist_kv.IS_KV_PREFILL_INSTANCE:
                # for prefill, the ranks are [0, world_size)
                maybe_disagg_rank = rank
            else:
                # this is decode instance.
                # offset global rank by tp * pp (which is world_size)
                maybe_disagg_rank = rank + world_size

        logger.debug(
            f"Before: world size {maybe_disagg_world_size}, rank {maybe_disagg_rank}"
        )

        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=maybe_disagg_world_size,
            rank=maybe_disagg_rank)
        logger.debug("torch.distributed initialized")
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

    global _WORLD
    if _WORLD is None:
        ranks = [[i for i in range(world_size)]]
        # offset the distributed group
        if dist_kv.IS_DISTRIBUTED_KV_INSTANCE:
            ranks = include_decoding_groups_if_disagg_enabled(
                ranks, world_size)

        _WORLD = init_world_group(ranks, local_rank, backend)
        logger.debug("_WORLD initialized for rank %d",
                     torch.distributed.get_rank())
        time.sleep(5)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size")


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


    Disaggregated prefill will also initialize its process group using this function.
    Changes:
        - vLLM world size: unchanged (tp * pp)
        - torch.distributed.get_world_size():
            - 2 * tp * pp
            - Why: torch.distributed package sees 2 vLLM instances (prefill and decode)
        - Global rank:
            - [0, tp * pp) for prefill
            - [tp * pp, 2 * tp * pp) for decode
        - Parallel groups
            - Extend _WORLD, _TP and _PP using `include_decoding_groups_if_disagg_enabled`
            - Add a new parallel group `_DISAGG` for disaggregated prefill
                - [ [0, tp * pp], [1, tp * pp + 1], .. ]
        - Local rank: unchanged
    """

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    if dist_kv.IS_DISTRIBUTED_KV_INSTANCE:
        # Disaggregated prefill enabled
        # The world_size for this vLLM instance is tp * pp, but torch.distributed contains 2 vLLM instances, its world size is 2 * tp * pp
        # Adjust the world_size to match.
        world_size = world_size // 2

    if (world_size
            != tensor_model_parallel_size * pipeline_model_parallel_size):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = (world_size //
                                             tensor_model_parallel_size)
    global _TP
    assert _TP is None, ("tensor model parallel group is already initialized")
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)
    group_ranks = include_decoding_groups_if_disagg_enabled(
        group_ranks, world_size)
    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    use_message_queue_broadcaster=True)
    logger.debug("_TP initialized for rank %d", torch.distributed.get_rank())

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = (world_size //
                                               pipeline_model_parallel_size)
    global _PP
    assert _PP is None, (
        "pipeline model parallel group is already initialized")
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    group_ranks = include_decoding_groups_if_disagg_enabled(
        group_ranks, world_size)
    # pipeline parallel does not need custom allreduce
    _PP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    use_custom_allreduce=False)
    logger.debug("_PP initialized for rank %d", torch.distributed.get_rank())

    if dist_kv.IS_DISTRIBUTED_KV_INSTANCE:
        global _DISAGG
        logger.debug("Disaggregated prefill enabled, create _DISAGG group")
        group_ranks = []
        for i in range(world_size):
            # prefill local rank: i
            # decode global rank: i + world_size
            group_ranks.append([i, i + world_size])
        logger.debug("Distributed group is %s", str(group_ranks))
        _DISAGG = dist_kv.DistributedKVCoordinator(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            torch_distributed_backend=backend,
        )
        # follow by a warmup, to warmup nccl
        # necessary, as NCCL may not be warmed up when tp and pp are both 1.
        temp_tensor = torch.tensor([1.]).to(_DISAGG.device)
        if dist_kv.IS_KV_PREFILL_INSTANCE:
            _DISAGG.send(temp_tensor)
        else:
            recv_tensor = _DISAGG.recv(temp_tensor.shape, temp_tensor.dtype)
            assert torch.allclose(temp_tensor, recv_tensor)
        logger.debug("_DISAGG initialized for rank %d",
                     torch.distributed.get_rank())


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size,
                                  pipeline_model_parallel_size, backend)
        return

    assert (
        get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    ), ("tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    pp_world_size = get_pp_group().world_size
    assert (pp_world_size == pipeline_model_parallel_size), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}")


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (_TP is not None and _PP is not None)


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decode to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _DISAGG
    if _DISAGG:
        _DISAGG.destroy()
    _DISAGG = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> List[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    assert torch.distributed.get_backend(
        pg) != torch.distributed.Backend.NCCL, (
            "in_the_same_node_as should be tested with a non-NCCL group.")
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
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[:len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list([shm.name],
                                                        src=ranks[source_rank],
                                                        group=pg)
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(recv,
                                                        src=ranks[source_rank],
                                                        group=pg)
                name = recv[0]
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch("multiprocessing.resource_tracker.register",
                           lambda *args, **kwargs: None):
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
        if rank == source_rank and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return [x == 1 for x in is_in_the_same_node.tolist()]
