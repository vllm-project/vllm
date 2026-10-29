# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

FP4_DTYPE = torch.uint8
FP8_DTYPE = current_platform.fp8_dtype()

DTYPES = [torch.float16, torch.bfloat16]
# (M, 2 * K). K = 48, 80 and 3696 (Qwen2.5-72B down_proj at TP=8) are 16 mod 32, so
# the NVFP4 linear pads K; K = 80 and 96 leave padded columns in the 128x4 scale tile.
SHAPES = [
    (128, 256),
    (128, 128),
    (256, 256),
    (256, 128),
    (1, 256),
    (64, 256),
    (1, 96),
    (5, 96),
    (130, 96),
    (130, 160),
    (130, 192),
    (2, 7392),
]
BLOCK_SIZE = 16


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_silu_mul_nvfp4_quant(
    default_vllm_config,
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    x = torch.randn(shape, dtype=dtype)

    # ref op, rounded once from fp32 like the kernel, and quantized into the K-padded
    # buffer that the CUTLASS NVFP4 linear uses
    ref_output = SiluAndMul().forward_native(x.float()).to(dtype)
    m, k = ref_output.shape
    ref_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(
        ref_output
    ).max().to(torch.float32)
    ref_output_quant, ref_block_scale = scaled_fp4_quant(
        ref_output, ref_global_scale, padded_n=round_up(k, 32)
    )

    # fused op, into buffers holding stale bytes as under torch.compile buffer reuse
    fused_output_quant = torch.full_like(ref_output_quant, 0xFF)
    fused_block_scale = torch.empty_like(ref_block_scale)
    fused_block_scale.view(torch.uint8).fill_(0xFF)
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

    # check padding: zero data beyond K, zero scales outside the [M, K / 16] block
    assert not fused_output_quant[:, k // 2 :].any()
    sf_rows, sf_cols = fused_block_scale.shape
    fused_sf = convert_swizzled_to_linear(
        fused_block_scale.view(torch.uint8), sf_rows, sf_cols * BLOCK_SIZE, BLOCK_SIZE
    )
    outside = torch.ones_like(fused_sf, dtype=torch.bool)
    outside[:m, : k // BLOCK_SIZE] = False
    assert not fused_sf[outside].any()

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
