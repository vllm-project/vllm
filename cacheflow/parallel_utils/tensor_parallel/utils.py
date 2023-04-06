# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
from typing import List, Sequence

from cacheflow.parallel_utils.utils import divide
from cacheflow.parallel_utils import parallel_state

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False):
    """ Break a tensor into equal 1D chunks across tensor parallel ranks.

        Returns a Tensor or View with this rank's portion of the data.

        Arguments:
            tensor: The tensor to split

        Keyword Arguments:
            new_buffer (bool): If True, returns a new Tensor.
                               If False, returns a view into the existing Tensor.
                               Default is False

    """
    partition_size = torch.numel(tensor) // \
        parallel_state.get_tensor_model_parallel_world_size()
    start_index = partition_size * parallel_state.get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(partition_size, dtype=tensor.dtype,
                           device=torch.cuda.current_device(),
                           requires_grad=False)
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor):
    """ Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
        model parallel ranks.

        Returns a new Tensor with the gathered data.

        Arguments:
            tensor: A Tensor or view of this rank's portion of the data.
    """
    numel_gathered = torch.numel(tensor) * \
        parallel_state.get_tensor_model_parallel_world_size()
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype,
                           device=torch.cuda.current_device(),
                           requires_grad=False)
    # TODO: This API is experimental in pytorch (as of Feb 2022) and
    # this might break in future pytorch releases. We chose this API
    # as opposed to torch.distributed.all_gather for efficiency reasons.
    # This API calls directly NCCL all-gather versus the former does
    # internal copies and can potentially cause slow down.
    torch.distributed._all_gather_base(gathered, tensor,
                                       group=parallel_state.get_tensor_model_parallel_group())
    return gathered


class VocabUtility:
    """ Split the vocabulary into `world_size` chunks and return the first
        and last index of the vocabulary belonging to the `rank`
        partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
