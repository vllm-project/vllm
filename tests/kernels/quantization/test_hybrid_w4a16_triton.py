#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton path of the hybrid W4A16 kernel.

The hybrid kernel stores weights in ExLlama shuffle format [N, K//8] int32.
This test validates the Triton GEMM (triton_w4a16_skinny_fmt_gemm) that reads
from this layout.

Run `pytest tests/kernels/quantization/test_hybrid_w4a16_triton.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# This test module is ROCm/Triton specific. Avoid import-time failures on
# non-ROCm or environments without Triton by skipping early.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

device = "cuda"

hybrid_w4a16_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16"
)
triton_w4a16_skinny_fmt_gemm = hybrid_w4a16_module.triton_w4a16_skinny_fmt_gemm


pack_int4_exllama_shuffle = hybrid_w4a16_module.pack_int4_exllama_shuffle


def _pack_exllama_shuffle(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] int4 values into ExLlama shuffle format [N, K//8] int32."""
    return pack_int4_exllama_shuffle(w_int4_kn.t().contiguous())


def _w4a16_skinny_reference(
    a_mk: torch.Tensor,
    w_int4_kn: torch.Tensor,
    scales_nkg: torch.Tensor,
    *,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """Reference implementation for symmetric W4A16 with skinny layout.

    a_mk: [M, K] fp16/bf16
    w_int4_kn: [K, N] int4 values (unpacked, int32)
    scales_nkg: [N, K//G] scales (skinny layout)
    """
    M, K = a_mk.shape

    # Expand scales from [N, K//G] to [K, N]
    scales_kn = scales_nkg.t().contiguous()  # [K//G, N]
    s_full = scales_kn.repeat_interleave(group_size, dim=0).to(torch.float32)

    w_fp = (w_int4_kn - zp_bias).to(torch.float32) * s_full  # [K, N]
    out = a_mk.to(torch.float32) @ w_fp  # [M, N]
    return out.to(a_mk.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 32),
        (17, 256, 512, 32),
        (32, 512, 256, 64),
        (33, 512, 512, 128),
        (64, 1024, 256, 256),
    ],
)
def test_triton_w4a16_skinny_fmt_gemm_matches_reference(
    dtype, M, K, N, G, random_seed: int
):
    assert K % G == 0 and K % 8 == 0, (
        f"Invalid test shape: K={K} must be divisible by G={G} and 8"
    )

    set_random_seed(random_seed)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4 = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)

    # Pack into ExLlama shuffle format [N, K//8]
    b_packed = _pack_exllama_shuffle(w_int4)

    # Scales in skinny layout [N, K//G]
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    out = triton_w4a16_skinny_fmt_gemm(
        a=a,
        b_q=b_packed,
        scales=scales,
        group_size=G,
        zp_bias=8,
    )
    ref = _w4a16_skinny_reference(
        a,
        w_int4,
        scales,
        group_size=G,
        zp_bias=8,
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)


def _w4a16_skinny_reference_asymmetric(
    a_mk: torch.Tensor,
    w_int4_kn: torch.Tensor,
    scales_nkg: torch.Tensor,
    zp_raw_nkg: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Reference implementation for asymmetric W4A16 with skinny layout.

    a_mk: [M, K] fp16/bf16
    w_int4_kn: [K, N] int4 values (unpacked, int32)
    scales_nkg: [N, K//G] scales (skinny layout)
    zp_raw_nkg: [N, K//G] raw zero-points in activation dtype
    """
    # Expand scales and raw zp from [N, K//G] to [K, N]
    scales_kn = scales_nkg.t().contiguous()  # [K//G, N]
    s_full = scales_kn.repeat_interleave(group_size, dim=0).to(torch.float32)

    zp_raw_kn = zp_raw_nkg.t().contiguous()  # [K//G, N]
    zp_raw_full = zp_raw_kn.repeat_interleave(group_size, dim=0).to(torch.float32)

    # dequant: (nibble - zp_raw) * scale
    w_fp = (w_int4_kn.to(torch.float32) - zp_raw_full) * s_full  # [K, N]
    out = a_mk.to(torch.float32) @ w_fp  # [M, N]
    return out.to(a_mk.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G",
    [
        (1, 256, 256, 32),
        (17, 256, 512, 32),
        (32, 512, 256, 64),
        (33, 512, 512, 128),
        (64, 1024, 256, 128),
    ],
)
def test_triton_w4a16_skinny_fmt_gemm_asymmetric(dtype, M, K, N, G, random_seed: int):
    assert K % G == 0 and K % 8 == 0, (
        f"Invalid test shape: K={K} must be divisible by G={G} and 8"
    )

    set_random_seed(random_seed)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    w_int4 = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)

    # Pack into ExLlama shuffle format [N, K//8]
    b_packed = _pack_exllama_shuffle(w_int4)

    # Scales in skinny layout [N, K//G]
    scales = (0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)).to(
        dtype
    )

    # Raw per-group zero-points [N, K//G] in activation dtype
    zp_raw = torch.randint(0, 16, (N, K // G), device=device, dtype=torch.int32)
    zp = zp_raw.to(dtype)

    out = triton_w4a16_skinny_fmt_gemm(
        a=a,
        b_q=b_packed,
        scales=scales,
        group_size=G,
        zp=zp,
    )
    ref = _w4a16_skinny_reference_asymmetric(
        a,
        w_int4,
        scales,
        zp,
        group_size=G,
    )

    # bf16 accumulation at larger shapes needs slightly looser tolerance
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)
