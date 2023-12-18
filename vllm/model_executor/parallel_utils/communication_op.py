import torch

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank
)
from vllm.model_executor.parallel_utils.fast_allreduce import FastAllreduce


fa_handle = None
is_capturing = False


def init_fast_ar():
    global fa_handle
    world_size = get_tensor_model_parallel_world_size()
    if world_size > 1:
        fa_handle = FastAllreduce(get_tensor_model_parallel_rank(), world_size)


def begin_capture():
    global is_capturing
    is_capturing = True


def end_capture():
    global is_capturing
    is_capturing = False


def tensor_model_parallel_all_reduce(input_: torch.Tensor):
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # fast allreduce only works with IPC pre-registered buffer.
    # This is only handled when captured with cuda graph
    if is_capturing and fa_handle is not None:
        if torch.cuda.is_current_stream_capturing():
            if fa_handle.should_fast_ar(input_):
                return fa_handle.all_reduce(input_)
        else:
            if fa_handle.should_fast_ar(input_):
                # if warm up, mimic the allocation pattern
                # since fast allreduce is out-of-place
                return torch.empty_like(input_)

    torch.distributed.all_reduce(input_,
                                group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_, dim=-1):
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
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
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor
