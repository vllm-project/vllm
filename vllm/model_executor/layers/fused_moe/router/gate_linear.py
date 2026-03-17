# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

if has_flashinfer():

    def flashinfer_tinygemm_router_gemm_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from flashinfer.gemm.routergemm import tinygemm_bf16

        tinygemm_bf16(x, weight, out, bias)

    def flashinfer_tinygemm_router_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        return

    direct_register_custom_op(
        op_name="flashinfer_tinygemm_router_gemm",
        op_func=flashinfer_tinygemm_router_gemm_impl,
        fake_impl=flashinfer_tinygemm_router_gemm_fake,
        mutates_args=["out"],
    )


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer with four-tier GEMM dispatch:

    1. DSV3 specialized kernel (SM90+, batch<=16, supported dims, no bias)
    2. cuBLAS bf16xbf16→fp32 (SM90+ + bf16 + fp32 out_dtype, no bias)
    3. Flashinfer tinygemm_bf16 kernel (SM90+, aligned dims, supports bias)
    4. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # Dimensions supported by the DSV3 specialized kernel
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168]

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
        is_hopper_or_blackwell = current_platform.is_device_capability(
            (9, 0)
        ) or current_platform.is_device_capability_family(100)
        can_use_specialized_kernels = (
            current_platform.is_cuda() and is_hopper_or_blackwell
        )

        # If fp32 compute is required and no specialized kernel is available,
        # store weights in fp32 so Tier 3 computes in fp32 natively.
        if force_fp32_compute and not can_use_specialized_kernels:
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

        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_specialized_router_gemm_no_bias = (
            can_use_specialized_kernels and not bias
        )

        # DSV3 specialized kernel eligibility (SM90+, exact dims, no bias)
        self.allow_dsv3_router_gemm = (
            self.allow_specialized_router_gemm_no_bias
            and output_size in self.DSV3_SUPPORTED_NUM_EXPERTS
            and input_size in self.DSV3_SUPPORTED_HIDDEN_SIZES
        )

        # cuBLAS bf16→fp32 eligibility (no bias)
        self.allow_cublas_router_gemm = (
            self.allow_specialized_router_gemm_no_bias
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

        # Flashinfer tinygemm_bf16 (SM90+, aligned dims, supports bias)
        self.allow_tinygemm_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype != torch.float32
            and input_size % 64 == 0
            and output_size % 16 == 0
            and has_flashinfer()
        )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        """Set output dtype for the router logits after init.

        Useful when the required dtype depends on the expert quantization
        method which is only known after the gate is constructed.
        """
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if (
            not self.allow_cublas_router_gemm
            and self.allow_specialized_router_gemm_no_bias
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16

        # tinygemm outputs bf16 — disable if fp32 output is now required
        if out_dtype == torch.float32:
            self.allow_tinygemm_router_gemm = False

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        import vllm._custom_ops as ops

        # Tier 1: DSV3 specialized kernel
        if self.allow_dsv3_router_gemm and x.shape[0] <= 16:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # Tier 2: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = ops.router_gemm_bf16_fp32(x, self.weight)
            return output, None

        # Tier 3: Flashinfer tinygemm_bf16
        if self.allow_tinygemm_router_gemm and x.dtype == torch.bfloat16:
            output = torch.empty(
                x.shape[0],
                self.weight.shape[0],
                dtype=torch.bfloat16,
                device=x.device,
            )
            torch.ops.vllm.flashinfer_tinygemm_router_gemm(
                x, self.weight, self.bias, output
            )
            return output, None

        # Tier 4: F.linear (ReplicatedLinear)
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
