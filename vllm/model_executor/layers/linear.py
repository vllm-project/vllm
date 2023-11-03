from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)
from vllm.model_executor.parallel_utils.utils import (
    divide,
    split_tensor_along_last_dim,
)


class LinearMethodBase(ABC):

    @abstractmethod
    def create_weights(self, module: torch.nn.Module, input_size: int,
                       output_size: int, params_dtype: torch.dtype) -> None:
        del module
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      module: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        del module, x
        raise NotImplementedError


class FullPrecisionLinearMethod(LinearMethodBase):

    def __init__(self, separate_bias_add: bool = False):
        self.separate_bias_add = separate_bias_add

    def create_weights(self, module: torch.nn.Module, input_size: int,
                       output_size: int, params_dtype: torch.dtype) -> None:
        weight = Parameter(torch.empty(output_size,
                                       input_size,
                                       device=torch.cuda.current_device(),
                                       dtype=params_dtype),
                           requires_grad=False)
        module.register_parameter("weight", weight)

    def apply_weights(self,
                      module: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.separate_bias_add:
            if bias:
                return F.linear(x, module.weight) + bias
            return F.linear(x, module.weight)
        return F.linear(x, module.weight, bias)


class ReplicatedLinear(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if linear_method is None:
            linear_method = FullPrecisionLinearMethod()
        self.linear_method = linear_method
        self.linear_method.create_weights(self, self.input_size,
                                          self.output_size, self.params_dtype)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size,
                            device=torch.cuda.current_device(),
                            dtype=self.params_dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if not self.skip_bias_add else None
        output = self.linear_method.apply_weights(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


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
        linear_method: Optional[LinearMethodBase] = None,
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
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if linear_method is None:
            linear_method = FullPrecisionLinearMethod()
        self.linear_method = linear_method
        self.linear_method.create_weights(self, self.input_size,
                                          self.output_size_per_partition,
                                          self.params_dtype)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            device=torch.cuda.current_device(),
                            dtype=params_dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output_parallel = self.linear_method.apply_weights(self, input_, bias)
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
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Divide the weight matrix along the last dimension.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.skip_bias_add = skip_bias_add
        if linear_method is None:
            linear_method = FullPrecisionLinearMethod()
        self.linear_method = linear_method
        self.linear_method.create_weights(self, self.input_size_per_partition,
                                          self.output_size, self.params_dtype)

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError('When not reduce the results, adding bias to the '
                             'results can lead to incorrect results')

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size,
                            device=torch.cuda.current_device(),
                            dtype=params_dtype))
        else:
            self.register_parameter('bias', None)

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
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.linear_method.apply_weights(
            self, input_parallel)
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
