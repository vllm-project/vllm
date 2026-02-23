# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.kernels.linear.base.w8a8 as w8a8_linear
from vllm.platforms import current_platform


class FpKernel(w8a8_linear.FpKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU."
        return True, None

    @classmethod
    def can_implement(cls, c: w8a8_linear.FpKernelConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "supports Fp8 weight dtype only."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
        pass
