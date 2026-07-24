# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the RDNA3 (gfx1100) MXFP4 weight-only WMMA GEMM.

Run inside the ROCm container after building _rocm_C:
    .venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_rdna3_wmma.py -v
"""

import pytest
import torch

from vllm.platforms import current_platform

E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="RDNA3 MXFP4 WMMA kernel is ROCm-only"
)


def _dequant_ref(w_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 (E2M1) + [N, K/32] uint8 (E8M0) -> [N, K] fp32."""
    N, k_half = w_packed.shape
    K = k_half * 2
    lo = (w_packed & 0xF).to(torch.long)  # even K
    hi = (w_packed >> 4).to(torch.long)  # odd K
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K)  # interleave even/odd
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    mag = E2M1_MAG.to(w_packed.device)[codes & 0x7]
    scale = torch.pow(2.0, scale_e8m0.float() - 127.0)  # [N, K/32]
    scale = scale.repeat_interleave(32, dim=1)  # [N, K]
    return sign * mag * scale


def _repack(w_packed, scale_e8m0):
    b_q = w_packed.contiguous().view(torch.int32).t().contiguous()  # [K/8, N]
    b_scale = scale_e8m0.t().contiguous()  # [K/32, N]
    return b_q, b_scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [1, 16, 64, 128])
@pytest.mark.parametrize("N,K", [(64, 256), (256, 512), (128, 4096)])
def test_mxfp4_wmma_matches_dequant_reference(dtype, M, N, K):
    if not hasattr(torch.ops._rocm_C, "mxfp4_gemm_rdna3"):
        pytest.skip("_rocm_C.mxfp4_gemm_rdna3 not built")

    torch.manual_seed(0)
    dev = "cuda"
    # Random E2M1 codes (full 0..255 byte = two 4-bit codes) and mild E8M0
    # scales centered on 127 (== 2^0) to stay well inside fp16/bf16 range.
    w_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    scale_e8m0 = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev)
    x = torch.randn(M, K, dtype=dtype, device=dev) * 0.1

    w_deq = _dequant_ref(w_packed, scale_e8m0)  # [N, K] fp32
    ref = (x.float() @ w_deq.t()).to(dtype)

    b_q, b_scale = _repack(w_packed, scale_e8m0)
    out = torch.ops._rocm_C.mxfp4_gemm_rdna3(x, b_q, b_scale)

    assert out.shape == (M, N)
    # bf16/fp16 accumulate-in-fp32; tolerate dtype rounding of the reduction.
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
