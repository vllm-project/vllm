from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)

from vllm.model_executor.parallel_utils.utils import VocabUtility


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
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = get_tensor_model_parallel_world_size()
        # TODO: Handle vocab padding here.
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tp_size))
        self.num_embeddings_per_partition = (self.vocab_end_index -
                                             self.vocab_start_index)

        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition,
                        self.embedding_dim,
                        device=torch.cuda.current_device(),
                        dtype=params_dtype))

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
