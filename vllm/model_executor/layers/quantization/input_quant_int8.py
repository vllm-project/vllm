# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
INT8 quantization CustomOp with native decomposition for Inductor fusion.

This follows the same pattern as QuantFP8 but provides INT8 quantization.
When compiled, Inductor fuses this with adjacent operations. Uses
inline_asm_elementwise when available to emit a single PTX
cvt.rni.sat.s8.f32 instruction for a faster saturating cast vs the
pure-torch round+clamp+cast fallback.
"""

import torch

from vllm.model_executor.custom_op import CustomOp

try:
    from torch._higher_order_ops.inline_asm_elementwise import (
        inline_asm_elementwise,
    )

    _HAS_INLINE_ASM = True
except ImportError:
    _HAS_INLINE_ASM = False


def _saturating_int8_cast(x: torch.Tensor) -> torch.Tensor:
    """Saturating float32 -> int8 cast with round-to-nearest and clamping.

    Uses PTX inline asm when available (1 instruction vs 3 for the fallback).
    Under torch.compile, Inductor fuses this into the surrounding Triton
    kernel (e.g., with divide/scale ops), so no extra kernel launch.
    """
    if _HAS_INLINE_ASM:
        return inline_asm_elementwise(
            x,
            asm_str="cvt.rni.sat.s8.f32 $0, $1;",
            constraints="=r,f",
            dtype=torch.int8,
            is_pure=True,
            pack=1,
        )
    return x.round().clamp(-128, 127).to(torch.int8)


@CustomOp.register("quant_int8")
class QuantInt8(CustomOp):
    """
    Quantize input tensor to INT8 with optional asymmetric zero point.
    """

    def __init__(
        self,
        static: bool,
        symmetric: bool = True,
        compile_native: bool = True,
    ):
        super().__init__(compile_native=compile_native)
        self.static = static
        self.symmetric = symmetric

    def forward_cuda(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        azp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        output = torch.empty_like(input, dtype=torch.int8)

        if scale is not None:
            assert self.symmetric == (azp is None), (
                "azp must only be provided for asymmetric quantization."
            )
            torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
            return output, scale, azp
        else:
            input_2d = input.view(-1, input.shape[-1])
            token_num = input_2d.shape[0]
            input_scales = torch.empty((token_num, 1), device=input.device, dtype=torch.float32)
            input_azp = None if self.symmetric else torch.empty_like(input_scales, dtype=torch.int32)
            torch.ops._C.dynamic_scaled_int8_quant(output, input_2d, input_scales, input_azp)
            return output, input_scales, input_azp

    def forward_native(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        azp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Native PyTorch decomposition. Under torch.compile, Inductor fuses
        this with adjacent ops into a single Triton kernel.
        """
        if scale is not None:
            return self._static_quant_native(input, scale, azp)
        else:
            return self._dynamic_quant_native(input, azp)

    def _static_quant_native(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        azp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        scaled = input.to(torch.float32) / scale.to(torch.float32)

        if azp is not None:
            scaled = scaled + azp.to(torch.float32)

        output = _saturating_int8_cast(scaled)
        return output, scale, azp

    def _dynamic_quant_native(
        self,
        input: torch.Tensor,
        azp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_2d = input.view(-1, input.shape[-1])

        input_f32 = input_2d.to(torch.float32)

        computed_azp = None
        if not self.symmetric:
            x_max = input_f32.max(dim=-1, keepdim=True)[0]
            x_min = input_f32.min(dim=-1, keepdim=True)[0]
            scale = (x_max - x_min) / 255.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            computed_azp = (
                (-128.0 - x_min / scale).round().clamp(-128, 127).to(torch.int32)
            )
            scaled = input_f32 / scale + computed_azp.to(torch.float32)
        else:
            x_max = input_f32.abs().max(dim=-1, keepdim=True)[0]
            x_max = torch.where(x_max == 0, torch.ones_like(x_max), x_max)
            scale = x_max / 127.0
            scaled = input_f32 / scale

        output = _saturating_int8_cast(scaled)
        return output.view(input.shape), scale, computed_azp
