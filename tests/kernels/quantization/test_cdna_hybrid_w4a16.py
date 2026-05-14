#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CDNA hybrid W4A16 GEMM kernel (MI300, gfx942 / gfx950).

Skips cleanly on non-ROCm and on ROCm but non-CDNA hardware (e.g. RDNA),
so this file is a no-op on RDNA dev boxes and only exercises the kernel
where its tile sizes are tuned for.

Run: `pytest tests/kernels/quantization/test_cdna_hybrid_w4a16.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_mi3xx

if not on_mi3xx():
    pytest.skip("CDNA (gfx942 / gfx950) only", allow_module_level=True)

pytest.importorskip("triton")

cdna_mod = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.cdna_hybrid_w4a16"
)
_cdna_w4a16_gemm = cdna_mod._cdna_w4a16_gemm
_pack_int4_exllama_shuffle = cdna_mod._pack_int4_exllama_shuffle
CDNAHybridW4A16LinearKernel = cdna_mod.CDNAHybridW4A16LinearKernel
DECODE_M_THRESHOLD = cdna_mod.DECODE_M_THRESHOLD

device = "cuda"


def _w4a16_reference(
    a_mk: torch.Tensor,        # [M, K]
    w_int4_nk: torch.Tensor,   # [N, K] uint4 in int32
    scales_ng: torch.Tensor,   # [N, K//G]
    *,
    group_size: int,
    zp_ng: torch.Tensor | None,  # [N, K//G] raw zero-points in act dtype, or None
    zp_bias: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for (a @ dequant(W).T).

    The kernel stores weights in [N, K] orientation and computes
    out[m, n] = sum_k a[m, k] * (w[n, k] - zero) * scale[n, k//G].
    """
    assert a_mk.dtype in (torch.float16, torch.bfloat16)
    M, K = a_mk.shape
    N = w_int4_nk.shape[0]
    assert w_int4_nk.shape == (N, K)
    G = group_size
    assert K % G == 0
    num_groups = K // G
    assert scales_ng.shape == (N, num_groups)

    s_full = scales_ng.repeat_interleave(G, dim=1).to(torch.float32)  # [N, K]
    if zp_ng is None:
        z_full = torch.full(
            (N, K), zp_bias, dtype=torch.float32, device=a_mk.device
        )
    else:
        assert zp_ng.shape == (N, num_groups)
        z_full = zp_ng.repeat_interleave(G, dim=1).to(torch.float32)  # [N, K]

    w_fp = (w_int4_nk.to(torch.float32) - z_full) * s_full          # [N, K]
    out = a_mk.to(torch.float32) @ w_fp.t()                          # [M, N]
    return out.to(a_mk.dtype)


# Cover both decode (M <= DECODE_M_THRESHOLD) and prefill regimes, both
# group-size buckets that hit BLOCK_K-clamp paths, and both symmetric +
# asymmetric quantisation.
_SHAPES = [
    # (M, K, N, group_size, has_zp)
    (1,   256, 256, 32, False),    # decode, sym, small G
    (4,   512, 512, 64, False),    # decode, sym
    (16,  512, 512, 128, False),   # decode boundary, sym
    (17,  256, 512, 32, False),    # prefill entry, sym
    (32,  512, 256, 64, True),     # prefill, asym
    (64,  1024, 1024, 128, False), # prefill, square
    (128, 256, 1024, 32, True),    # prefill, wide-N asym
    (128, 1024, 256, 64, False),   # prefill, tall-K sym
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,K,N,G,has_zp", _SHAPES)
def test_cdna_hybrid_w4a16_gemm_matches_reference(dtype, M, K, N, G, has_zp):
    if not torch.cuda.is_available():
        pytest.skip("HIP device not available")

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)

    # Random uint4 weights in [N, K] orientation (kernel layout).
    w_int4_nk = torch.randint(
        0, 16, (N, K), device=device, dtype=torch.int32
    )
    b_packed = _pack_int4_exllama_shuffle(w_int4_nk)  # [N, K//8] int32

    scales = (
        0.05 * torch.rand((N, K // G), device=device, dtype=torch.float32)
    ).to(dtype)

    zp = None
    zp_bias = 8
    if has_zp:
        zp_int = torch.randint(
            0, 16, (N, K // G), device=device, dtype=torch.int32
        )
        zp = zp_int.to(dtype)
        zp_bias = 0  # zp tensor is the truth; no constant bias

    out = _cdna_w4a16_gemm(
        a=a,
        b_q_i32=b_packed,
        scales=scales,
        group_size=G,
        zp_bias=zp_bias,
        zp=zp,
    )
    ref = _w4a16_reference(
        a, w_int4_nk, scales,
        group_size=G,
        zp_ng=zp,
        zp_bias=zp_bias,
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_cdna_hybrid_w4a16_requires_contiguous_inputs():
    if not torch.cuda.is_available():
        pytest.skip("HIP device not available")

    set_random_seed(0)
    M, K, N, G = 32, 256, 256, 32
    # Non-contiguous activation (transposed view).
    a = torch.randn((K, M), device=device, dtype=torch.float16).t()
    w_int4_nk = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    b_packed = _pack_int4_exllama_shuffle(w_int4_nk)
    scales = torch.rand((N, K // G), device=device, dtype=torch.float16)

    with pytest.raises(AssertionError):
        _cdna_w4a16_gemm(
            a=a,
            b_q_i32=b_packed,
            scales=scales,
            group_size=G,
            zp_bias=8,
            zp=None,
        )
