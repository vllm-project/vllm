from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)


class GPTQLinear(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 *,
                 bias=True,
                 quant_config=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        self.use_exllama = True
        group_size = self.quant_config.group_size if (
            self.quant_config.group_size != -1) else self.input_size
        self.qweight = Parameter(
            torch.empty(
                self.input_size // self.quant_config.pack_factor,
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
                dtype=torch.float16,
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
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size,
                            device="cuda",
                            dtype=torch.float16))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def post_init(self):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        # make_q4 segfaults if g_idx is not on cpu in the act-order case.
        # In the non act-order case, None needs to be passed for g_idx.
        if not self.quant_config.desc_act:
            g_idx = torch.empty((1, 1), device="meta")
        else:
            g_idx = self.g_idx.to("cpu")
        self.q4 = quantization_ops.gptq_make_q4(self.qweight, self.qzeros,
                                                self.scales, g_idx,
                                                self.qweight.device.index)

    def forward(self, input_):
        out_shape = input_.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = input_.reshape(-1, input_.shape[-1])
        output = torch.empty((input_.shape[0], self.qweight.shape[-1]),
                             dtype=torch.float16,
                             device=input_.device)
        quantization_ops.gptq_q4_matmul(reshaped_x, self.q4, output)
        output = output.reshape(out_shape)

        output = output + self.bias if self.bias is not None else output
        return output


class GPTQColumnParallelLinear(ColumnParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.input_size % self.quant_config.pack_factor == 0
        assert (self.output_size_per_partition %
                self.quant_config.pack_factor == 0)
        self.use_exllama = True
        group_size = self.quant_config.group_size if (
            self.quant_config.group_size != -1) else self.input_size

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

    def post_init(self):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        # make_q4 segfaults if g_idx is not on cpu in the act-order case.
        # In the non act-order case, None needs to be passed for g_idx.
        if not self.quant_config.desc_act:
            g_idx = torch.empty((1, 1), device="meta")
        else:
            g_idx = self.g_idx.to("cpu")
        self.q4 = quantization_ops.gptq_make_q4(self.qweight, self.qzeros,
                                                self.scales, g_idx,
                                                self.qweight.device.index)

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        output = torch.empty((x.shape[0], self.qweight.shape[-1]),
                             dtype=torch.float16,
                             device=x.device)
        quantization_ops.gptq_q4_matmul(reshaped_x, self.q4, output)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)


class GPTQRowParallelLinear(RowParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert (self.input_size_per_partition %
                self.quant_config.pack_factor == 0)
        assert self.output_size % self.quant_config.pack_factor == 0
        group_size = self.quant_config.group_size if (
            self.quant_config.group_size != -1
        ) else self.input_size_per_partition
        if self.tp_size > 1 and (self.quant_config.desc_act
                                 and self.quant_config.group_size != -1):
            group_number = self.input_size // group_size
            self.use_exllama = False
        else:
            group_number = self.input_size_per_partition // group_size
            self.use_exllama = True
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
                group_number,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                group_number,
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

    def post_init(self):
        if not self.use_exllama:
            return
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        # make_q4 segfaults if g_idx is not on cpu in the act-order case.
        # In the non act-order case, None needs to be passed for g_idx.
        if not self.quant_config.desc_act:
            g_idx = torch.empty((1, 1), device="meta")
        else:
            g_idx = self.g_idx.to("cpu")
        self.q4 = quantization_ops.gptq_make_q4(self.qweight, self.qzeros,
                                                self.scales, g_idx,
                                                self.qweight.device.index)

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        if self.use_exllama:
            output = torch.empty((x.shape[0], self.qweight.shape[-1]),
                                 dtype=torch.float16,
                                 device=x.device)
            quantization_ops.gptq_q4_matmul(reshaped_x, self.q4, output)
        else:
            output = torch.zeros((x.shape[0], self.qweight.shape[-1]),
                                 dtype=torch.float32,
                                 device=x.device)
            quantization_ops.gptq_descact_matmul(reshaped_x.float(),
                                                 self.qweight, output,
                                                 self.scales.float(),
                                                 self.qzeros, self.g_idx)
            output = output.half()
        return output.reshape(out_shape)
