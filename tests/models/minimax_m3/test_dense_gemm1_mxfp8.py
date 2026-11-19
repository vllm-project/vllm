# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the MiniMax-M3.1 fused dense GEMM1 kernel
(vllm/models/minimax_m3/nvidia/ops/dense_gemm1_mxfp8.py).
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="dense GEMM1 MXFP8 kernel is SM100+ CUDA only",
)


def _mxfp8_quantize_f32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-quantize fp32 [M, N] with the kernel's integer-exact scale math:
    scale byte = E - 8 + (amax > 1.75 * 2^E) with E the biased exponent of the
    32-elem group amax (0 for amax=0), clamped to [0, 254]. Returns q and the
    unswizzled [M, N//32] scale bytes.
    """
    M, N = x.shape
    g = x.float().view(M, N // 32, 32)
    amax = g.abs().amax(-1)
    e_bias = amax.view(torch.int32) >> 23
    thr = torch.ldexp(torch.full_like(amax, 1.75), e_bias - 127)
    sb = (e_bias - 8 + (amax > thr).to(torch.int32)).clamp(0, 254)
    q = g * torch.ldexp(torch.ones_like(amax), 127 - sb)[..., None]
    q = q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(M, N), sb.to(torch.uint8)


# Per-rank I: TP4 shared expert, TP4 dense / TP1 shared, TP1 dense. 12288 is
# the only one whose weight exceeds half of L2 (grouped rasterization).
@pytest.mark.parametrize("num_tokens", [1, 37, 128, 1000, 4096])
@pytest.mark.parametrize("intermediate", [768, 3072, 12288])
def test_dense_gemm1_mxfp8_correctness(num_tokens: int, intermediate: int):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.models.minimax_m3.nvidia.ops.dense_gemm1_mxfp8 import run

    limit, alpha, beta = 7.0, 1.702, 1.0
    hidden = 6144
    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    w1 = (
        torch.randn(intermediate * 2, hidden, device="cuda", dtype=torch.bfloat16)
        * 0.05
    )

    # What Mxfp8OnlineLinearMethod.process_weights_after_loading produces.
    w1q, wsf_2d = mxfp8_e4m3_quantize(w1, is_sf_swizzled_layout=False)
    w1sf = swizzle_mxfp8_scale(wsf_2d, M=intermediate * 2, K=hidden)

    M, N = num_tokens, intermediate
    e8 = torch.float8_e8m0fnu
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.zeros(
        ((M + 127) // 128) * 128 * (N // 32), dtype=torch.uint8, device="cuda"
    )
    xq, xsf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    run(xq, w1q, xsf.view(-1).view(e8), w1sf.view(e8), q, sf, limit, alpha, beta)

    # fp32 mirror chain: same MXFP8 GEMM (fp32 out) -> fp32 clamped SwiGLU
    # -> MXFP8 quantize with the kernel's scale math.
    gemm = torch._scaled_mm(
        xq,
        w1q.t(),
        scale_a=xsf.view(e8),
        scale_b=w1sf.view(e8),
        out_dtype=torch.float32,
    )
    gate = gemm[:, :N].clamp(max=limit)
    up = gemm[:, N:].clamp(-limit, limit)
    act = gate * torch.sigmoid(gate * alpha) * (up + beta)
    q_ref, sf_ref_2d = _mxfp8_quantize_f32(act)
    sf_ref = swizzle_mxfp8_scale(sf_ref_2d, M=M, K=N)

    # Measured envelope: sf bytes identical; q bytes differ on <=0.007% of
    # elements, always by 1 fp8 ulp. Bounds below carry >10x margin.
    sf_mism = (sf != sf_ref).sum().item()
    assert sf_mism <= sf.numel() * 1e-4, f"sf byte mismatches: {sf_mism}"
    q_mism = q.view(torch.uint8) != q_ref.view(torch.uint8)
    assert q_mism.sum().item() <= q.numel() * 1e-3, (
        f"q byte mismatches: {q_mism.sum().item()}"
    )
