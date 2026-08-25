# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _bf16_relu2_static_fp8_quant_kernel(
    x_ptr,
    scale_ptr,
    output_ptr,
    num_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    relu = tl.maximum(x, 0.0)
    # Preserve the standalone ReLU2 kernel's BF16 output boundary.
    activated = (relu * relu).to(tl.bfloat16).to(tl.float32)
    quantized = activated * (1.0 / tl.load(scale_ptr))
    quantized = tl.maximum(tl.minimum(quantized, FP8_MAX), FP8_MIN)
    tl.store(output_ptr + offsets, quantized, mask=mask)


@CustomOp.register("bf16_relu2_static_fp8_quant")
class Bf16ReLUSquaredStaticFp8Quant(CustomOp):
    """ReLU2 followed by static per-tensor FP8 quantization.

    The intermediate is rounded to BF16 to match separate ReLU2 and
    quantization kernels exactly.
    """

    def __init__(self) -> None:
        super().__init__(enforce_enable=True)
        self.fp8_min, self.fp8_max = get_fp8_min_max()

    def forward_native(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        assert scale.numel() == 1
        activated = torch.square(torch.clamp_min(x.to(torch.float32), 0.0)).to(
            torch.bfloat16
        )
        return (
            activated.to(torch.float32)
            .mul(scale.to(torch.float32).reciprocal())
            .clamp(self.fp8_min, self.fp8_max)
            .to(current_platform.fp8_dtype())
        )

    def forward_cuda(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        assert x.is_contiguous()
        assert scale.numel() == 1
        output = torch.empty_like(x, dtype=current_platform.fp8_dtype())
        block_size = 2048
        _bf16_relu2_static_fp8_quant_kernel[(triton.cdiv(x.numel(), block_size),)](
            x,
            scale,
            output,
            x.numel(),
            FP8_MIN=self.fp8_min,
            FP8_MAX=self.fp8_max,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return output
