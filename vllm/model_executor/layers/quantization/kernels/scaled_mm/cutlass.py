# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
)


class CutlassInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = params.azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(),
            params.input_scale,
            params.input_zero_point,
            symmetric=symmetric,
        )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = params.input_zero_point is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(
                x_q,
                params.weight,
                scale_a=x_s,
                scale_b=params.weight_scale,
                out_dtype=x.dtype,
                azp_adj=params.azp_adj,
                azp=azp,
                bias=bias,
            )
        return ops.cutlass_scaled_mm(
            x_q,
            params.weight,
            scale_a=x_s,
            scale_b=params.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )


class CutlassFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # Fused GEMM_DQ
        output = ops.cutlass_scaled_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )
        return output.view(*output_shape)
