# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy and parity tests for the fused Triton Gemma 4 2D RoPE kernel.

Compares ``fused_vision_2d_rope`` against the HuggingFace reference
``apply_multidimensional_rope`` across multiple Gemma 4 model shapes,
broadcasting patterns, non-contiguous inputs, and edge cases.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope

from vllm.model_executor.models.gemma4_mm import fused_vision_2d_rope
from vllm.triton_utils import HAS_TRITON

# Gemma 4 shapes:
# E2B: N_HEADS = 12, HEAD_DIM = 64
# 26B-A4B / 31B: N_HEADS = 16, HEAD_DIM = 72
SHAPES = [
    (1, 64, 12, 64),  # E2B single-image short
    (2, 256, 12, 64),  # E2B batched 256 tokens
    (1, 64, 16, 72),  # 26B/31B single-image short
    (2, 256, 16, 72),  # 26B/31B batched 256 tokens
    (1, 900, 16, 72),  # 26B/31B full-patch resolution
]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="CUDA and Triton required for fused 2D RoPE kernel tests",
)
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,head_dim",
    SHAPES,
    ids=[f"B{b}_L{s}_N{n}_D{d}" for b, s, n, d in SHAPES],
)
def test_fused_vision_2d_rope_parity_and_broadcasting(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> None:
    """Verify fused_vision_2d_rope parity against apply_multidimensional_rope,
    non-contiguous inputs, broadcasted cos/sin, and empty tensors."""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    cos = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)

    # 1. Reference HF implementation parity
    ref_out = apply_multidimensional_rope(
        x,
        cos,
        sin,
        position_ids=torch.zeros(
            batch_size, seq_len, 2, device=device, dtype=torch.long
        ),
        unsqueeze_dim=2,
    )
    fused_out = fused_vision_2d_rope(x, cos, sin)

    assert fused_out.shape == ref_out.shape
    diff = (fused_out.float() - ref_out.float()).abs()
    assert diff.max().item() <= 0.1

    # 1-ULP parity and cosine similarity
    a_i = fused_out.view(torch.int16)
    b_i = ref_out.view(torch.int16)
    same_sign = (a_i >= 0) == (b_i >= 0)
    ulp = torch.where(
        same_sign, (a_i.int() - b_i.int()).abs(), torch.tensor(999999, device=device)
    )
    ulp = torch.where(
        (a_i == b_i) | ((fused_out == 0) & (ref_out == 0)) | (diff <= 1e-6), 0, ulp
    )
    within_1ulp_pct = float((ulp <= 1).float().mean().item() * 100.0)
    assert within_1ulp_pct >= 95.0, f"within_1ulp_pct is {within_1ulp_pct}"

    cos_sim = torch.cosine_similarity(
        fused_out.float().flatten(), ref_out.float().flatten(), dim=0
    ).item()
    assert cos_sim >= 0.99999

    # 2. Broadcasted cos/sin (cos.shape[0] == 1)
    cos_bcast = torch.randn(1, seq_len, head_dim, device=device, dtype=dtype)
    sin_bcast = torch.randn(1, seq_len, head_dim, device=device, dtype=dtype)
    ref_bcast = apply_multidimensional_rope(
        x,
        cos_bcast.expand(batch_size, seq_len, head_dim),
        sin_bcast.expand(batch_size, seq_len, head_dim),
        position_ids=torch.zeros(
            batch_size, seq_len, 2, device=device, dtype=torch.long
        ),
        unsqueeze_dim=2,
    )
    fused_bcast = fused_vision_2d_rope(x, cos_bcast, sin_bcast)
    diff_bcast = (fused_bcast.float() - ref_bcast.float()).abs()
    assert diff_bcast.max().item() <= 0.1

    # 3. Non-contiguous input
    x_raw = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    x_transposed = x_raw.transpose(1, 2)
    assert not x_transposed.is_contiguous()
    out_transposed = fused_vision_2d_rope(x_transposed, cos, sin)
    assert out_transposed.shape == (batch_size, seq_len, num_heads, head_dim)

    # 4. Empty tensor
    x_empty = torch.randn(0, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    cos_empty = torch.randn(0, seq_len, head_dim, device=device, dtype=dtype)
    sin_empty = torch.randn(0, seq_len, head_dim, device=device, dtype=dtype)
    out_empty = fused_vision_2d_rope(x_empty, cos_empty, sin_empty)
    assert out_empty.numel() == 0
    assert out_empty.shape == (0, seq_len, num_heads, head_dim)
