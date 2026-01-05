# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

FP4_DTYPE = torch.uint8
FP8_DTYPE = current_platform.fp8_dtype()

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 256), (128, 128), (256, 256), (256, 128)]
BLOCK_SIZE = 16


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_silu_mul_nvfp4_quant(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    x = torch.randn(shape, dtype=dtype)

    # ref op
    ref_output = SiluAndMul().forward_native(x)
    ref_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(
        ref_output
    ).max().to(torch.float32)
    ref_output_quant, ref_block_scale = scaled_fp4_quant(ref_output, ref_global_scale)

    # fused op
    fused_output_quant = torch.empty_like(ref_output_quant)
    fused_block_scale = torch.empty_like(ref_block_scale)
    torch.ops._C.silu_and_mul_nvfp4_quant(
        fused_output_quant, fused_block_scale, x, ref_global_scale
    )

    # check dtype
    assert ref_output_quant.dtype == FP4_DTYPE
    assert fused_output_quant.dtype == FP4_DTYPE
    assert ref_output_quant.shape == fused_output_quant.shape

    assert ref_block_scale.dtype == FP8_DTYPE
    assert fused_block_scale.dtype == FP8_DTYPE
    assert ref_block_scale.shape == fused_block_scale.shape

    # check dequantized output
    ref_output_dequant = dequantize_nvfp4_to_dtype(
        ref_output_quant, ref_block_scale, ref_global_scale, dtype, device
    )
    fused_output_dequant = dequantize_nvfp4_to_dtype(
        fused_output_quant, fused_block_scale, ref_global_scale, dtype, device
    )

    atol, rtol = 3e-1, 3e-1
    torch.testing.assert_close(
        ref_output_dequant, fused_output_dequant, atol=atol, rtol=rtol
    )
