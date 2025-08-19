import torch

from vllm.distributed.device_communicators.pynccl_allocator import use_symmetric_memory
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.utils import  direct_register_custom_op


def all_reduce_symmetric_with_copy_impl(
    input_tensor: torch.Tensor, pynccl_comm: PyNcclCommunicator
) -> torch.Tensor:
    with use_symmetric_memory(pynccl_comm):
        symm_input = torch.empty_like(input_tensor)
        symm_output = torch.empty_like(input_tensor)
    symm_input.copy_(input_tensor)
    symm_output = pynccl_comm.all_reduce(symm_input, symm_output)
    return symm_output


def all_reduce_symmetric_with_copy_fake(
    input_tensor: torch.Tensor, pynccl_comm: PyNcclCommunicator
) -> torch.Tensor:
    return torch.empty_like(input_tensor)


direct_register_custom_op(
    op_name="all_reduce_symmetric_with_copy",
    op_func=all_reduce_symmetric_with_copy_impl,
    mutates_args=[],
    fake_impl=all_reduce_symmetric_with_copy_fake,
)