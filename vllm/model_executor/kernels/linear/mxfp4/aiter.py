# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

# NOTE: Do not import aiter at module scope. Importing aiter eagerly initializes HIP
# which can force the engine core to spawn instead of fork.
# is_aiter_found_and_supported() checks platform + arch + library availability via
# find_spec/amdsmi, so it stays HIP-free.
# Actual aiter imports are deferred to the functions/methods that need them,
# where HIP initialization is expected.
if is_aiter_found_and_supported():
    from vllm.utils.torch_utils import direct_register_custom_op

    def gemm_with_dynamic_quant(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
        x_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from aiter.ops.triton.gemm_afp4wfp4 import (
            gemm_afp4wfp4,
            gemm_afp4wfp4_preshuffled_weight_scales,
        )
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        if rocm_use_aiter_fp4_asm_gemm:
            from aiter import gemm_a4w4, per_1x32_f4_quant_hip

        M = x.shape[0]
        N = weight.shape[0]
        K = weight.shape[1]
        if rocm_use_aiter_fp4_asm_gemm:
            if M <= 64 and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K):
                if x_scales is None:
                    # use hip quant kernel for performance
                    if M >= 32:
                        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
                    else:
                        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=False)
                else:
                    x_q = x
                    x_s = x_scales

                if M >= 32:
                    x_s = x_s.view(torch.uint8).view(x_s.shape[0] // 32, -1)
                else:
                    x_s = x_s[:M, ...].view(torch.uint8)

                y = torch.empty(M, N, device=x_q.device, dtype=out_dtype)
                gemm_afp4wfp4_preshuffled_weight_scales(
                    x_q.view(torch.uint8),
                    weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
                    x_s,
                    weight_scale.view(torch.uint8).view(
                        weight_scale.shape[0] // 32, -1
                    ),
                    out_dtype,
                    y,
                )
            else:
                if x_scales is None:
                    # use hip quant kernel for performance
                    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
                else:
                    x_q = x
                    x_s = x_scales

                y = gemm_a4w4(
                    x_q,
                    weight.view(x_q.dtype),
                    x_s,
                    weight_scale.view(x_s.dtype),
                    dtype=out_dtype,
                    bpreshuffle=True,
                )
            return y[:M]
        else:
            if x_scales is None:
                x_q, x_s = dynamic_mxfp4_quant(x)
            else:
                x_q = x
                x_s = x_scales
            y = torch.empty(
                x_q.shape[0], weight.shape[0], device=x_q.device, dtype=out_dtype
            )

            gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, out_dtype, y)
            return y

    def gemm_with_dynamic_quant_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scales: torch.Tensor = None,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
    ) -> torch.Tensor:
        return torch.empty(
            (*x.shape[:-1], weight.shape[0]), dtype=out_dtype, device=x.device
        )

    direct_register_custom_op(
        op_name="gemm_with_dynamic_quant",
        op_func=gemm_with_dynamic_quant,
        mutates_args=[],
        fake_impl=gemm_with_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class AiterMxfp4LinearKernel(MxFp4LinearKernel):
    """AITER-based native MXFP4 GEMM kernel for ROCm."""

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.use_asm_gemm = rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()
        self.out_dtype = torch.get_default_dtype()

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.supports_mx():
            return False, "current platform does not support native MXFP4 computation"
        if is_aiter_found_and_supported():
            return True, None
        return False, "AITER not found or not supported on the current platform"

    @classmethod
    def can_implement(cls, c: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_asm_gemm:
            from aiter.ops.shuffle import shuffle_weight

            weight_scale = layer.weight_scale.data
            sm, sn = weight_scale.shape
            weight_scale = weight_scale.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
            weight_scale = weight_scale.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
            weight_scale = weight_scale.view(sm, sn)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

            layer.weight = Parameter(
                shuffle_weight(layer.weight.data, layout=(16, 16)),
                requires_grad=False,
            )
        else:
            layer.weight_scale = Parameter(
                layer.weight_scale.data.T.contiguous(), requires_grad=False
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = torch.ops.vllm.gemm_with_dynamic_quant(
            x,
            layer.weight,
            layer.weight_scale,
            self.use_asm_gemm,
            self.out_dtype,
        )
        if bias is not None:
            y = y + bias
        return y
