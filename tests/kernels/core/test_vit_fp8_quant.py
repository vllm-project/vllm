# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the stride-aware FP8 quantization kernel with head_dim padding."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.model_executor.layers.quantization.input_quant_fp8 import (
        quantize_fp8_pad_head_dim_triton,
    )

HEAD_DIMS = [72, 80, 128]
SEQ_LENS = [64, 256]
NUM_HEADS = [16]
SCALES = [0.01, 0.1, 1.0]


def _naive_fp8_quantize(
    tensor: torch.Tensor, scale: torch.Tensor, skip_scale: bool
) -> torch.Tensor:
    """Reference FP8 quantization in PyTorch."""
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = -fp8_max

    x = tensor.float()
    if not skip_scale:
        x = x / scale.item()
    x = x.clamp(fp8_min, fp8_max)
    return x.to(fp8_dtype)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("scale_val", SCALES)
def test_quantize_contiguous(
    head_dim: int, seq_len: int, num_heads: int, scale_val: float
) -> None:
    """Test quantization of contiguous 3D tensors."""
    torch.manual_seed(42)
    tensor = torch.randn(
        seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([scale_val], dtype=torch.float32, device="cuda").view(
        1, 1, 1, 1
    )

    result = quantize_fp8_pad_head_dim_triton(tensor, scale)

    padded_dim = (head_dim + 15) // 16 * 16
    assert result.shape == (seq_len, num_heads, padded_dim)
    assert result.is_contiguous()
    assert result.dtype == current_platform.fp8_dtype()

    # Compare unpadded portion against reference
    ref = _naive_fp8_quantize(tensor, scale, skip_scale=False)
    torch.testing.assert_close(
        result[:, :, :head_dim].float(), ref.float()
    )

    # Padded region should be zero
    if padded_dim > head_dim:
        assert (result[:, :, head_dim:].float() == 0).all()


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.parametrize("head_dim", [72, 80])
def test_quantize_non_contiguous(head_dim: int) -> None:
    """Test quantization from non-contiguous QKV views (interleaved buffer)."""
    seq_len, num_heads = 64, 16
    # Simulate interleaved QKV buffer: shape (seq_len, 3 * num_heads, head_dim)
    qkv = torch.randn(
        seq_len, 3 * num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    # Q is every 3rd head slice - non-contiguous view
    q = qkv[:, 0::3, :]
    assert not q.is_contiguous()

    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda").view(1, 1, 1, 1)
    result = quantize_fp8_pad_head_dim_triton(q, scale)

    padded_dim = (head_dim + 15) // 16 * 16
    assert result.shape == (seq_len, num_heads, padded_dim)
    assert result.is_contiguous()

    # Compare against contiguous reference
    ref = _naive_fp8_quantize(q.contiguous(), scale, skip_scale=False)
    torch.testing.assert_close(
        result[:, :, :head_dim].float(), ref.float()
    )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_skip_scale() -> None:
    """Test skip_scale=True produces cast-only output (no division)."""
    seq_len, num_heads, head_dim = 32, 8, 80
    tensor = torch.randn(
        seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda").view(1, 1, 1, 1)

    result_skip = quantize_fp8_pad_head_dim_triton(tensor, scale, skip_scale=True)
    result_noskip = quantize_fp8_pad_head_dim_triton(tensor, scale, skip_scale=False)

    # skip_scale should just cast, not divide
    ref_cast = _naive_fp8_quantize(tensor, scale, skip_scale=True)
    torch.testing.assert_close(
        result_skip[:, :, :head_dim].float(), ref_cast.float()
    )

    # With scale != 1.0, skip and no-skip should differ
    assert not torch.equal(result_skip.float(), result_noskip.float())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_4d_input() -> None:
    """Test that 4D input (B, S, H, D) is handled correctly."""
    B, S, H, D = 2, 32, 8, 72
    tensor = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda").view(1, 1, 1, 1)

    result = quantize_fp8_pad_head_dim_triton(tensor, scale)
    padded_dim = (D + 15) // 16 * 16
    assert result.shape == (B, S, H, padded_dim)
