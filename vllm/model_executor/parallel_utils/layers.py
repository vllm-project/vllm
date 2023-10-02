# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)

from vllm.model_executor.parallel_utils.utils import (
    divide,
    VocabUtility,
    split_tensor_along_last_dim,
)


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


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.skip_bias_add = skip_bias_add
        self.quant_config = quant_config

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.create_weights(params_dtype)

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            device=torch.cuda.current_device(),
                            dtype=params_dtype))
        else:
            self.register_parameter('bias', None)

    def create_weights(self, dtype: torch.dtype) -> None:
        self.weight = Parameter(
            torch.empty(self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=dtype))

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.linear(x, self.weight, bias)

    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        input_parallel = input_
        # Matrix multiply.
        output_parallel = self.apply_weights(input_parallel, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Divide the weight matrix along the last dimension.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.skip_bias_add = skip_bias_add
        self.quant_config = quant_config

        self.create_weights(params_dtype)

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError('When not reduce the results, adding bias to the '
                             'results can lead to incorrect results')

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size,
                            device=torch.cuda.current_device(),
                            dtype=params_dtype))

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def create_weights(self, dtype: torch.dtype) -> None:
        self.weight = Parameter(
            torch.empty(self.output_size,
                        self.input_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=dtype))

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # TODO: simplify code below
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply_weights(input_parallel)
        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
