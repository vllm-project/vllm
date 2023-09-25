import torch

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)

def tenosr_model_parallel_all_reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_

def tensor_model_parallel_all_gather(input_):
    """All-gather the input tensor across model parallel group."""

    world_size = get_tensor_model_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Allocate output tensor.
    # TODO: Finish here
    raise NotImplementedError()

    return input_
