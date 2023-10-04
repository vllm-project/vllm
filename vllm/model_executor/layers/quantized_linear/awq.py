from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)


class AWQColumnParallelLinear(ColumnParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.input_size % self.quant_config.weight_bits == 0
        assert (self.output_size_per_partition %
                self.quant_config.pack_factor == 0)
        self.qweight = Parameter(
            torch.empty(
                self.input_size,
                self.output_size_per_partition //
                self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.qzeros = Parameter(
            torch.empty(
                self.input_size // self.quant_config.group_size,
                self.output_size_per_partition //
                self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size // self.quant_config.group_size,
                self.output_size_per_partition,
                device="cuda",
                dtype=dtype,
            ),
            requires_grad=False,
        )

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[-2], self.qweight.shape[-1] * pack_factor)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = quantization_ops.awq_gemm(reshaped_x, self.qweight, self.scales,
                                        self.qzeros, pack_factor)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)


class AWQRowParallelLinear(RowParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert (self.input_size_per_partition %
                self.quant_config.weight_bits == 0)
        assert self.output_size % self.quant_config.pack_factor == 0
        self.qweight = Parameter(
            torch.empty(
                self.input_size_per_partition,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.qzeros = Parameter(
            torch.empty(
                self.input_size_per_partition // self.quant_config.group_size,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size_per_partition // self.quant_config.group_size,
                self.output_size,
                device="cuda",
                dtype=dtype,
            ),
            requires_grad=False,
        )

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[-2], self.qweight.shape[-1] * pack_factor)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = quantization_ops.awq_gemm(reshaped_x, self.qweight, self.scales,
                                        self.qzeros, pack_factor)
        return out.reshape(out_shape)
