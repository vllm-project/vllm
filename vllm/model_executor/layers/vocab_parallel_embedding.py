from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.utils import set_weight_attrs


def pad_vocab_size(vocab_size: int, pad_to: int = 64) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int,
                                              rank: int) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int,
                                       world_size: int) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                     rank)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 params_dtype: Optional[torch.dtype] = None):
        super().__init__()

        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.num_embeddings_padded = pad_vocab_size(num_embeddings)
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.tp_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = (
            vocab_range_from_global_vocab_size(
                self.num_embeddings_padded, get_tensor_model_parallel_rank(),
                self.tp_size))
        self.num_embeddings_per_partition = (self.vocab_end_index -
                                             self.vocab_start_index)
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition,
                        self.embedding_dim,
                        device=torch.cuda.current_device(),
                        dtype=params_dtype))
        set_weight_attrs(self.weight, {"parallel_dim": 0,
                                       "weight_loader": self.weight_loader})

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        assert loaded_weight.shape[0] == self.num_embeddings
        assert loaded_weight.shape[1] == self.embedding_dim
        loaded_weight = loaded_weight[self.vocab_start_index:self.
                                      vocab_end_index]
        param[:self.num_embeddings].data.copy_(loaded_weight)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            input_mask = ((input_ < self.vocab_start_index) |
                          (input_ >= self.vocab_end_index))
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output


class ParallelLMHead(VocabParallelEmbedding):
    # TODO: Add docstring
    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weight should be used in the sampler.")
