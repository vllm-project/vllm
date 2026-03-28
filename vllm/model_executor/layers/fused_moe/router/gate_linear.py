# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

import vllm._custom_ops as ops
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer with multi-tier GEMM dispatch:

    1. DSV3 specialized kernel (SM90+, fp32 out, M<=16, H=7168, E=256/384)
    2. fp32 specialized kernel  (SM90+, fp32 in/out, M<=32, H=3072, E=256)
    3. gpt-oss specialized kernel (SM90+, bf16, M<=128, H=2880, E=32/128)
    4. cuBLAS bf16×bf16→fp32 (SM90+ + bf16 weight + fp32 out_dtype)
    5. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # Dimensions supported by the DSV3 specialized kernel
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168]

    # Dimensions supported by the gpt-oss specialized kernel
    GPT_OSS_SUPPORTED_NUM_EXPERTS = [32, 128]
    GPT_OSS_SUPPORTED_HIDDEN_SIZES = [2880]

    # Dimensions supported by the fp32 specialized kernel
    FP32_SUPPORTED_NUM_EXPERTS = [256]
    FP32_SUPPORTED_HIDDEN_SIZES = [3072]
    FP32_MAX_TOKENS = 32

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
            current_platform.is_cuda() and is_hopper_or_blackwell and not bias
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

        # DSV3 specialized kernel eligibility (SM90+, exact dims)
        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_dsv3_router_gemm = (
            self.allow_specialized_router_gemm
            and output_size in self.DSV3_SUPPORTED_NUM_EXPERTS
            and input_size in self.DSV3_SUPPORTED_HIDDEN_SIZES
        )

        # gpt-oss specialized kernel eligibility (SM90+, exact dims)
        self.allow_gpt_oss_router_gemm = (
            self.weight.dtype == torch.bfloat16
            and current_platform.is_cuda()
            and is_hopper_or_blackwell
            and output_size in self.GPT_OSS_SUPPORTED_NUM_EXPERTS
            and input_size in self.GPT_OSS_SUPPORTED_HIDDEN_SIZES
        )

        # fp32 specialized kernel eligibility (SM90+, exact dims, fp32 weight)
        self.allow_fp32_router_gemm = (
            self.weight.dtype == torch.float32
            and current_platform.is_cuda()
            and is_hopper_or_blackwell
            and output_size in self.FP32_SUPPORTED_NUM_EXPERTS
            and input_size in self.FP32_SUPPORTED_HIDDEN_SIZES
        )

        # cuBLAS bf16→fp32 eligibility
        self.allow_cublas_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
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
            and self.allow_specialized_router_gemm
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Tier 1: DSV3 specialized kernel
        if self.allow_dsv3_router_gemm and x.shape[0] <= 16:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # Tier 2: fp32 specialized kernel (H=3072, E=256, M<=32)
        # Accepts bf16 or fp32 activation; conversion to fp32 done in kernel.
        if self.allow_fp32_router_gemm and x.shape[0] <= self.FP32_MAX_TOKENS:
            output = ops.fp32_router_gemm(x, self.weight)
            return output, None

        # Tier 3: gpt-oss specialized kernel
        if self.allow_gpt_oss_router_gemm:
            output = torch.ops.vllm.gpt_oss_router_gemm(x, self.weight, self.bias)
            return output, None

        # Tier 4: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = ops.router_gemm_bf16_fp32(x, self.weight)
            return output, None

        # Tier 5: F.linear (ReplicatedLinear)
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias


def gpt_oss_router_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Dynamically run min-latency gemm if num_tokens <= 128.
    This must be wrapped in a custom op because our torch.compile integration
    does not support runtime dispatching on num_tokens.
    """
    if x.shape[0] <= 128:
        return ops.gpt_oss_router_gemm(x, weight, bias)
    else:
        return torch.nn.functional.linear(x, weight, bias)


def gpt_oss_router_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="gpt_oss_router_gemm",
    op_func=gpt_oss_router_gemm_impl,
    fake_impl=gpt_oss_router_gemm_fake,
)
