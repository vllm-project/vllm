from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantized_ops.awq import (awq_matmul,
                                                          get_shifter)
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)


class AWQColumnParallelLinear(ColumnParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.input_size % self.quant_config.group_size == 0
        if self.output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The tensor parallel size is not aligned with the quantized "
                "weight shape. Please use a different tensor parallel size.")
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

        self.shifter = get_shifter(self.quant_config.pack_factor,
                                   device="cuda")
        self.unpacked_qzeros = None

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        group_size = self.quant_config.group_size

        out_shape = (x.shape[:-1] + (self.qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = awq_matmul(reshaped_x,
                         self.qweight,
                         self.qzeros,
                         self.scales,
                         pack_factor,
                         group_size,
                         self.shifter)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)


class AWQRowParallelLinear(RowParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.output_size % self.quant_config.pack_factor == 0
        if self.input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The tensor parallel size is not aligned with the quantized "
                "weight shape. Please use a different tensor parallel size.")
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

        self.shifter = get_shifter(self.quant_config.pack_factor,
                                   device="cuda")
        self.unpacked_qzeros = None

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        group_size = self.quant_config.group_size
        out_shape = (x.shape[:-1] + (self.qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = awq_matmul(reshaped_x,
                         self.qweight,
                         self.qzeros,
                         self.scales,
                         pack_factor,
                         group_size,
                         self.shifter)
        return out.reshape(out_shape)
