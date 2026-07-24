# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.hw_agnostic.custom_op import PluggableLayer
from vllm.model_executor.hw_agnostic.layers.linear import ReplicatedLinear


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        if force_fp32_compute:
            params_dtype = torch.float32

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

        # ``torch.mm(..., out_dtype=torch.float32)`` is the bf16-in / fp32-out
        # matmul fast path; the active backend (cuBLAS / hipBLAS / oneMKL)
        # picks the kernel.
        self.allow_router_gemm = (
            not bias
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if not self.allow_router_gemm and out_dtype == torch.float32:
            self.allow_router_gemm = self.weight.dtype == torch.bfloat16

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.allow_router_gemm and x.dtype == torch.bfloat16:
            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)
            return output, None

        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
