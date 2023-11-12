from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)


# FIXME(woosuk): Replace this and the CUDA kernel with a more optimized
# implementation.
def _gptq_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    shifter: torch.Tensor,
) -> torch.Tensor:
    """Matrix multiplication with GPTQ weights."""
    # qw: [input_size, output_size]
    qw = (qweight.unsqueeze(1) >> shifter.view(1, -1, 1)) & 0xf
    qw = qw.flatten(start_dim=0, end_dim=1)

    # qz: [input_size, output_size]
    qz = (qzeros[g_idx].unsqueeze(2) >> shifter.view(1, 1, -1)) & 0xf
    qz = qz + 1
    qz = qz.flatten(start_dim=1, end_dim=2)

    # qs: [input_size, output_size]
    qs = scales[g_idx]
    # w: [input_size, output_size]
    w = qs * (qw - qz).to(qs.dtype)

    # out: [batch_size, output_size]
    out = torch.matmul(x, w)
    return out


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
        # Initialize g_idx to be sequential.
        # This is required because old GPTQ models may not have g_idx.
        self.g_idx = Parameter(
            torch.tensor(
                [i // group_size for i in range(self.input_size)],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.shifter = torch.tensor(
            [0, 4, 8, 12, 16, 20, 24, 28],
            device="cuda",
            dtype=torch.int32,
        )

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        num_tokens = x.shape[:-1].numel()
        # FIXME(woosuk): The current GPTQ kernel performs poorly when the batch
        # size is large. As a temporary workaround, we use the PyTorch-based
        # GPTQ matmul implementation when the batch size is larger than 32.
        if num_tokens <= 32:
            output = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
            quantization_ops.gptq_descact_matmul(reshaped_x, self.qweight,
                                                 output, self.scales,
                                                 self.qzeros, self.g_idx)
        else:
            output = _gptq_matmul(reshaped_x, self.qweight, self.qzeros,
                                  self.scales, self.g_idx, self.shifter)
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
                self.input_size // group_size,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size // group_size,
                self.output_size,
                device="cuda",
                dtype=dtype,
            ),
            requires_grad=False,
        )
        # Initialize g_idx to be sequential.
        # This is required because old GPTQ models may not have g_idx.
        start_idx = self.tp_rank * self.input_size_per_partition
        self.g_idx = Parameter(
            torch.tensor(
                [(start_idx + i) // group_size
                 for i in range(self.input_size_per_partition)],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.shifter = torch.tensor(
            [0, 4, 8, 12, 16, 20, 24, 28],
            device="cuda",
            dtype=torch.int32,
        )

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        num_tokens = x.shape[:-1].numel()
        # FIXME(woosuk): The current GPTQ kernel performs poorly when the batch
        # size is large. As a temporary workaround, we use the PyTorch-based
        # GPTQ matmul implementation when the batch size is larger than 32.
        if num_tokens <= 32:
            output = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
            quantization_ops.gptq_descact_matmul(reshaped_x, self.qweight,
                                                 output, self.scales,
                                                 self.qzeros, self.g_idx)
        else:
            output = _gptq_matmul(reshaped_x, self.qweight, self.qzeros,
                                  self.scales, self.g_idx, self.shifter)
        return output.reshape(out_shape)
