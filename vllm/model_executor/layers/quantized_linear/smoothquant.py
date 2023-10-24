from typing import Optional

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.tensor_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear)
from vllm.i8cugemm import I8CUGEMM
i8cugemm = I8CUGEMM()

class SQColumnParallelLinear(ColumnParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.input_size % self.quant_config.weight_bits == 0
        self.register_buffer('weight', 
                            torch.randint(-127, 
                                          127, 
                                          (self.output_size_per_partition,
                                           self.input_size), 
                                           dtype=torch.int8, 
                                           requires_grad=False))

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:


        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.output_size_per_partition), dtype=torch.int32, device=x.device)
        i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        y = y.view(*x_shape[:-1], -1)
        return y

class SQRowParallelLinear(RowParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert (self.input_size_per_partition %
                self.quant_config.weight_bits == 0)
        self.register_buffer('weight', 
                            torch.randint(-127, 
                                          127, 
                                          (self.output_size,
                                           self.input_size_per_partition), 
                                           dtype=torch.int8, 
                                           requires_grad=False))


    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.output_size), dtype=torch.int32, device=x.device)
        i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        y = y.view(*x_shape[:-1], -1)
        return y