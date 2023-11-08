from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)


class GPTQColumnParallelLinear(ColumnParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.input_size % self.quant_config.pack_factor == 0
        assert self.input_size % self.quant_config.group_size == 0
        assert (self.output_size_per_partition %
                self.quant_config.pack_factor == 0)

        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = self.input_size

        self.qweight = Parameter(
            torch.empty(
                self.input_size // self.quant_config.pack_factor,
                self.output_size_per_partition,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.qzeros = Parameter(
            torch.empty(
                self.input_size // group_size,
                self.output_size_per_partition //
                self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size // group_size,
                self.output_size_per_partition,
                device="cuda",
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.g_idx = Parameter(
            torch.tensor(
                [i // group_size for i in range(self.input_size)],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        output = torch.zeros((reshaped_x.shape[0], self.qweight.shape[-1]),
                             dtype=x.dtype,
                             device=x.device)
        quantization_ops.gptq_descact_matmul(reshaped_x,
                                             self.qweight, output,
                                             self.scales, self.qzeros,
                                             self.g_idx)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)


class GPTQRowParallelLinear(RowParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert (self.input_size_per_partition %
                self.quant_config.pack_factor == 0)
        assert self.output_size % self.quant_config.pack_factor == 0

        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = self.input_size_per_partition

        self.qweight = Parameter(
            torch.empty(
                self.input_size_per_partition // self.quant_config.pack_factor,
                self.output_size,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.qzeros = Parameter(
            torch.empty(
                self.input_size_per_partition // group_size,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size_per_partition // group_size,
                self.output_size,
                device="cuda",
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.g_idx = Parameter(
            torch.tensor(
                [
                    i // group_size
                    for i in range(self.input_size_per_partition)
                ],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        output = torch.zeros((reshaped_x.shape[0], self.qweight.shape[-1]),
                             dtype=x.dtype,
                             device=x.device)
        quantization_ops.gptq_descact_matmul(reshaped_x,
                                             self.qweight, output,
                                             self.scales, self.qzeros,
                                             self.g_idx)
        return output.reshape(out_shape)
