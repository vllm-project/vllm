from typing import Optional

import torch
import threading
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)
from vllm.i8cugemm import I8CUGEMM

class Int8GEMM(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, "i8cugemm"):
            self.i8cugemm = I8CUGEMM()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Int8GEMM, "_instance"):
            with Int8GEMM._instance_lock:
                if not hasattr(Int8GEMM, "_instance"):
                    Int8GEMM._instance = object.__new__(cls)  
        return Int8GEMM._instance
    
    def get_i8cugemm(self):
        return self.i8cugemm

class SQColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GEMM = Int8GEMM()
        self.i8cugemm = GEMM.get_i8cugemm()

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.input_size % self.quant_config.weight_bits == 0
        self.register_buffer(
            'weight',
            torch.empty(self.output_size_per_partition,
                        self.input_size,
                        dtype=torch.int8,
                        requires_grad=False))

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert bias is None

        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.output_size_per_partition),
                        dtype=torch.int32,
                        device=x.device)
        self.i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        y = y.view(*x_shape[:-1], -1)
        return y


class SQRowParallelLinear(RowParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GEMM = Int8GEMM()
        self.i8cugemm = GEMM.get_i8cugemm()

    def create_weights(self, dtype: torch.dtype) -> None:
        assert (self.input_size_per_partition %
                self.quant_config.weight_bits == 0)
        self.register_buffer(
            'weight',
            torch.empty(self.output_size,
                        self.input_size_per_partition,
                        dtype=torch.int8,
                        requires_grad=False))

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.output_size),
                        dtype=torch.int32,
                        device=x.device)
        self.i8cugemm.linear_a8_w8_o32_(x, self.weight, y)
        y = y.view(*x_shape[:-1], -1)
        return y
