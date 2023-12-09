# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Tensor and pipeline parallel groups."""
import contextlib

import torch
import enum

from typing import Optional
from contextlib import contextmanager
from vllm.model_executor.parallel_utils import cupy_utils


class ActiveModel(enum.Enum):
    TARGET = enum.auto()
    DRAFT = enum.auto()


# Tensor model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Draft model tensor parallel group for speculative decoding
_DRAFT_MODEL_TP_GROUP = None
# Pipeline model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None

# A list of global ranks for each pipeline group to ease calculation of the
# source rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

_ACTIVE_MODEL: ActiveModel = ActiveModel.TARGET


# This context manager is used to mark which model
# is currently active so that the corresponding distributed group can be selected.
@contextmanager
def MarkActiveModel(new_value):
    global _ACTIVE_MODEL

    # Save the original value
    original_value = _ACTIVE_MODEL

    try:
        # Temporarily change the global variable
        _ACTIVE_MODEL = new_value
        yield
    finally:
        # Restore the original value after the block
        _ACTIVE_MODEL = original_value


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    draft_model_tp_size: Optional[int] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
        draft_model_tp_size: number of GPUs used for draft model's tensor
            model parallelism.

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
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, (
        "tensor model parallel group is already initialized")
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups.
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, (
        "pipeline model parallel group is already initialized")
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

    # Build the tensor parallel groups for draft model
    global _DRAFT_MODEL_TP_GROUP
    if draft_model_tp_size:
        assert _DRAFT_MODEL_TP_GROUP is None, (
            "draft model tensor parallel group is already initialized")
        # place draft model on the first n devices
        ranks = range(draft_model_tp_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _DRAFT_MODEL_TP_GROUP = group


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    draft_model_tp_size: Optional[int] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size,
                                  pipeline_model_parallel_size,
                                  draft_model_tp_size)
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
    return (_TENSOR_MODEL_PARALLEL_GROUP is not None
            and _PIPELINE_MODEL_PARALLEL_GROUP is not None)


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, (
        "tenosr model parallel group is not initialized")
    if _ACTIVE_MODEL == ActiveModel.DRAFT:
        assert _DRAFT_MODEL_TP_GROUP is not None
        return _DRAFT_MODEL_TP_GROUP
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, (
        "pipeline model parallel group is not initialized")
    return _PIPELINE_MODEL_PARALLEL_GROUP


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
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    if _TENSOR_MODEL_PARALLEL_GROUP:
        torch.distributed.destroy_process_group(_TENSOR_MODEL_PARALLEL_GROUP)
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    if _PIPELINE_MODEL_PARALLEL_GROUP:
        torch.distributed.destroy_process_group(_PIPELINE_MODEL_PARALLEL_GROUP)
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None
    global _DRAFT_MODEL_TP_GROUP
    if _DRAFT_MODEL_TP_GROUP:
        torch.distributed.destroy_process_group(_DRAFT_MODEL_TP_GROUP)
    _DRAFT_MODEL_TP_GROUP = None

    # Destroy the cupy states if any.
    cupy_utils.destroy_process_group()


# Whether to use cupy for nccl all reduce.
# We use cupy for all reduce when using CUDA graph, because torch.distributed
# is not well supported by CUDA graph.
_ENABLE_CUPY_FOR_ALL_REDUCE = False


@contextlib.contextmanager
def with_cupy_nccl_for_all_reduce():
    """use CuPy nccl instead of torch.distributed for all reduce"""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        # No-op.
        # NOTE(woosuk): We don't initialize CuPy when tp_size is 1.
        yield
    else:
        global _ENABLE_CUPY_FOR_ALL_REDUCE
        old = _ENABLE_CUPY_FOR_ALL_REDUCE
        _ENABLE_CUPY_FOR_ALL_REDUCE = True

        stream = torch.cuda.current_stream()
        with cupy_utils.set_cupy_stream(stream):
            yield
        _ENABLE_CUPY_FOR_ALL_REDUCE = old


def is_cupy_nccl_enabled_for_all_reduce():
    """check if CuPy nccl is enabled for all reduce"""
    global _ENABLE_CUPY_FOR_ALL_REDUCE
    return _ENABLE_CUPY_FOR_ALL_REDUCE
