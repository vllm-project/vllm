# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the RDNA3 (gfx1100) MXFP4 weight-only kernels: the
dense GEMM and the ``RDNA3Mxfp4Experts`` fused-MoE backend.

Run inside the ROCm container after building _rocm_C:
    .venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_rdna3.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 MXFP4 kernels are ROCm-only", allow_module_level=True)

from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_mxfp4_moe import (  # noqa: E402
    RDNA3Mxfp4Experts,
    repack_experts_rdna3,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402

E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

pytestmark = pytest.mark.skipif(
    not on_gfx1100(), reason="RDNA3 MXFP4 kernels require gfx1100"
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
    torch.manual_seed(0)
    dev = "cuda"
    # Random E2M1 codes (full 0..255 byte = two 4-bit codes) and mild E8M0
    # scales centered on 127 (== 2^0) to stay well inside fp16/bf16 range.
    w_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    scale_e8m0 = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev)
    x = torch.randn(M, K, dtype=dtype, device=dev) * 0.1

    w_deq = _dequant_ref(w_packed, scale_e8m0)  # [N, K] fp32
    ref = (x.double() @ w_deq.double().t()).float()

    b_q, b_scale = _repack(w_packed, scale_e8m0)
    out = torch.ops._rocm_C.mxfp4_gemm_rdna3(x, b_q, b_scale)

    assert out.shape == (M, N)
    # A K-term dot product rounds with the *accumulated* magnitude, not with the
    # result: these are full-range E2M1 codes, so at K=4096 sum|x||w| is ~2e3
    # while the sums themselves cancel down to ~1e2. Rounding the exact result
    # to bf16 already misses by ~0.5 there, so a fixed atol cannot work. Bound
    # the kernel by eps * sum|x||w| instead; measured slack is ~7x (the kernel
    # lands at 0.06-0.08 of this bound across dtypes and shapes, roughly twice
    # the unavoidable output-rounding floor).
    accum = x.abs().float() @ w_deq.abs().t()
    tol = 0.5 * torch.finfo(dtype).eps * accum + 1e-3
    err = (out.float() - ref).abs()
    worst = int(err.argmax())
    assert bool((err <= tol).all()), (
        f"error {err.flatten()[worst]:.4f} exceeds the rounding bound "
        f"{tol.flatten()[worst]:.4f} at flat index {worst}"
    )


def _swigluoai_interleaved(h: torch.Tensor) -> torch.Tensor:
    gate = h[..., ::2].clamp(max=7.0)
    up = h[..., 1::2].clamp(-7.0, 7.0)
    return (up + 1.0) * gate * torch.sigmoid(1.702 * gate)


@pytest.mark.parametrize("M", [1, 3, 16, 64])
@pytest.mark.parametrize(
    "activation, with_bias",
    [
        # No bias: the down GEMM reduces over top_k in its epilogue.
        (MoEActivation.SILU, False),
        # GPT-OSS: interleaved gate/up, per-expert bias, constants not passed
        # in the quant config (as on the Quark path); reduced in Python.
        (MoEActivation.SWIGLUOAI, True),
    ],
)
def test_rdna3_mxfp4_experts_match_dequant_reference(M, activation, with_bias):
    """``RDNA3Mxfp4Experts.apply`` against a dequantize-then-matmul MoE."""
    E, K, N, top_k = 8, 256, 128, 4
    dtype = torch.bfloat16
    dev = "cuda"
    torch.manual_seed(0)

    def rand_mxfp4(rows, cols):
        packed = torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8)
        scale = torch.randint(122, 126, (E, rows, cols // 32), dtype=torch.uint8)
        return packed.to(dev), scale.to(dev)

    w13, w13_s = rand_mxfp4(2 * N, K)
    w2, w2_s = rand_mxfp4(K, N)
    w13_ref = torch.stack([_dequant_ref(w13[e], w13_s[e]) for e in range(E)])
    w2_ref = torch.stack([_dequant_ref(w2[e], w2_s[e]) for e in range(E)])
    bias_init = torch.randn if with_bias else torch.zeros
    b13 = bias_init(E, 2 * N, device=dev)
    b2 = bias_init(E, K, device=dev)

    x = torch.randn(M, K, device=dev, dtype=dtype)
    topk_weights, topk_ids = torch.randn(M, E, device=dev).softmax(-1).topk(top_k)
    topk_ids = topk_ids.to(torch.int32)

    ref = torch.zeros(M, K, device=dev)
    for m in range(M):
        for j in range(top_k):
            e = topk_ids[m, j]
            h = x[m].float() @ w13_ref[e].t() + b13[e]
            if activation == MoEActivation.SWIGLUOAI:
                a = _swigluoai_interleaved(h)
            else:
                a = F.silu(h[:N]) * h[N:]
            y = a.to(dtype).float() @ w2_ref[e].t() + b2[e]
            ref[m] += topk_weights[m, j] * y

    deinterleave = activation == MoEActivation.SWIGLUOAI
    b_q13, b_s13 = repack_experts_rdna3(w13, w13_s, deinterleave=deinterleave)
    b_q2, b_s2 = repack_experts_rdna3(w2, w2_s, deinterleave=False)
    if deinterleave:
        b13 = torch.cat([b13[:, ::2], b13[:, 1::2]], dim=1)
    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_scale=b_s13,
        w2_scale=b_s2,
        w1_bias=b13.to(dtype) if with_bias else None,
        w2_bias=b2.to(dtype) if with_bias else None,
    )
    moe_config = make_dummy_moe_config(
        num_experts=E,
        experts_per_token=top_k,
        hidden_dim=K,
        intermediate_size=N,
        in_dtype=dtype,
        activation=activation,
    )
    experts = RDNA3Mxfp4Experts(moe_config, quant_config)
    ws13, ws2, out_shape = experts.workspace_shapes(
        M, N, K, top_k, E, E, None, activation
    )
    out = torch.empty(out_shape, device=dev, dtype=dtype)
    experts.apply(
        out,
        x,
        b_q13,
        b_q2,
        topk_weights,
        topk_ids,
        activation,
        E,
        None,
        None,
        None,
        torch.empty(ws13, device=dev, dtype=dtype),
        torch.empty(ws2, device=dev, dtype=dtype),
        None,
        False,
    )

    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 2e-2, f"relative L2 error {rel:.4f}"
