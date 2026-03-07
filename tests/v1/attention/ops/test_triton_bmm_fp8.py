# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused BMM + FP8 static quantization Triton kernel."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("N", [16])  # num_heads
@pytest.mark.parametrize("B", [1, 7, 32, 256])  # batch size
@pytest.mark.parametrize("L", [512])  # kv_lora_rank
@pytest.mark.parametrize("V", [128])  # v_head_dim
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_bmm_fp8_quant_correctness(N, B, L, V, dtype):
    """Test that fused BMM+FP8 matches separate BMM then FP8 quant."""
    from vllm.v1.attention.ops.triton_bmm_fp8 import bmm_fp8_quant

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # Create inputs
    input_tensor = torch.randn(N, B, L, dtype=dtype, device=device)
    weight = torch.randn(N, L, V, dtype=dtype, device=device)
    scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    # Reference: BMM then quantize separately
    ref_bmm = torch.bmm(input_tensor, weight)  # (N, B, V)
    # Transpose to (B, N*V) layout
    ref_output_bf16 = ref_bmm.transpose(0, 1).reshape(B, N * V)
    # Quantize
    fp8_dtype = current_platform.fp8_dtype()
    ref_fp8 = (
        (ref_output_bf16.float() * scale.item()).clamp(-448.0, 448.0).to(fp8_dtype)
    )

    # Fused kernel
    fused_output = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
    bmm_fp8_quant(input_tensor, weight, scale, fused_output)

    # Compare: convert both to float for comparison
    ref_float = ref_fp8.float()
    fused_float = fused_output.float()

    torch.testing.assert_close(fused_float, ref_float, atol=1.0, rtol=0.1)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
def test_bmm_fp8_quant_shapes():
    """Test various shapes work without errors."""
    from vllm.v1.attention.ops.triton_bmm_fp8 import bmm_fp8_quant

    device = torch.device("cuda:0")
    fp8_dtype = current_platform.fp8_dtype()
    dtype = torch.bfloat16
    scale = torch.tensor([0.005], dtype=torch.float32, device=device)

    shapes = [
        (16, 1, 512, 128),  # single token
        (16, 128, 512, 128),  # medium batch
        (128, 1, 512, 128),  # many heads, single token
        (16, 1, 256, 64),  # smaller dims
    ]

    for N, B, L, V in shapes:
        inp = torch.randn(N, B, L, dtype=dtype, device=device)
        w = torch.randn(N, L, V, dtype=dtype, device=device)
        out = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
        bmm_fp8_quant(inp, w, scale, out)
        assert out.shape == (B, N * V)
        assert out.dtype == fp8_dtype
