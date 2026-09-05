# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import get_cached_compilation_config
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_FP8_MIN, _FP8_MAX = get_fp8_min_max()


@triton.jit
def _relu_squared_static_fp8_quant_kernel(
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
    input_is_nan = x != x
    relu = tl.maximum(x, 0.0)
    # Preserve the standalone ReLU2 kernel's BF16 output boundary.
    activated = (relu * relu).to(tl.bfloat16).to(tl.float32)
    # Match O2's Inductor-compiled QuantFP8 reciprocal-then-multiply.
    quantized = activated * (1.0 / tl.load(scale_ptr))
    quantized = tl.maximum(tl.minimum(quantized, FP8_MAX), FP8_MIN)
    # Match the O2 native chain by preserving NaN through the FP8 conversion.
    quantized = tl.where(input_is_nan, x, quantized)
    tl.store(output_ptr + offsets, quantized, mask=mask)


def is_relu_squared_static_fp8_quant_config_supported() -> bool:
    """Return whether compiled-Inductor execution is configured."""
    config = get_cached_compilation_config()
    return config.mode == CompilationMode.VLLM_COMPILE and config.backend == "inductor"


def relu_squared_static_fp8_quant(
    x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
    """BF16 ReLU2 followed by static per-tensor FP8 quantization."""
    assert x.dtype == torch.bfloat16
    assert x.is_contiguous()
    scale = linear.input_scale
    assert scale.dtype == torch.float32
    assert x.device == scale.device
    assert scale.numel() == 1

    output = torch.empty_like(x, dtype=current_platform.fp8_dtype())
    if x.numel() != 0:
        block_size = min(triton.next_power_of_2(x.shape[-1]), 2048)
        num_warps = min(max(block_size // 256, 1), 4)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        _relu_squared_static_fp8_quant_kernel[grid](
            x,
            scale,
            output,
            x.numel(),
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

    return QuantizedActivation(
        data=output,
        scale=scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kFp8StaticTensorSym,
    )
