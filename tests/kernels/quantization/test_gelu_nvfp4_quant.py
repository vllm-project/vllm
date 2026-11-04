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
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# (M, K). K = 48, 80 and 3696 are 16 mod 32, so the NVFP4 linear pads K. M not a
# multiple of 128 or K not a multiple of 64 leaves padding in the 128x4 scale tiles.
SHAPES = [
    (128, 128),
    (256, 64),
    (1, 128),
    (64, 4096),
    (5, 48),
    (130, 80),
    (130, 96),
    (2, 3696),
]
BLOCK_SIZE = 16


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("has_bias", [True, False])
@torch.inference_mode()
def test_gelu_nvfp4_quant(
    default_vllm_config,
    dtype: torch.dtype,
    shape: tuple[int, int],
    has_bias: bool,
) -> None:
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    m, k = shape
    x = torch.randn(shape, dtype=dtype)
    bias = torch.randn(k, dtype=dtype) if has_bias else None

    # ref op, computed in fp32 and rounded once like the kernel, and quantized into
    # the K-padded buffer that the CUTLASS NVFP4 linear uses
    ref_input = x.float() if bias is None else x.float() + bias.float()
    ref_output = torch.nn.functional.gelu(ref_input).to(dtype)
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
    torch.ops._C.gelu_nvfp4_quant(
        fused_output_quant, fused_block_scale, x, ref_global_scale, bias
    )

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
    torch.testing.assert_close(
        ref_output_dequant, fused_output_dequant, atol=3e-1, rtol=3e-1
    )
