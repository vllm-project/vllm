# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused KDA/MLA MXFP4 o_proj epilogues match AITER's Triton quantizer.

The reference is ``dynamic_mxfp4_quant(epilogue_in_fp32)``. A 1-ulp disagreement
between Triton's and ATen's sigmoid/rsqrt is allowed at MXFP4 code boundaries.
"""

import importlib.util

import pytest
import torch

from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="fused o_proj quant requires ROCm + AITER",
)

H, D = 12, 128
K = H * D
EPS = 1e-5
MS = (1, 3, 8, 16, 17, 32, 64, 128)
MIN_AGREE = 0.999


def _ref_gated_rmsnorm(
    x: torch.Tensor, g: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (normed * w.float()) * torch.sigmoid(g.float())


def _agree(fused_q, fused_s, ref_q, ref_s, t: int) -> tuple[float, float, bool]:
    fq = fused_q.view(torch.uint8)[:t]
    rq = ref_q.view(torch.uint8)[:t]
    codes = (fq.cpu() == rq.cpu()).float().mean().item()
    scales = fused_s.view(torch.uint8)[:t].cpu() == ref_s.view(torch.uint8)[:t].cpu()
    scale_agree = scales.float().mean().item()
    layout_ok = (
        fused_q.shape == ref_q.shape
        and fused_q.dtype == ref_q.dtype
        and fused_q.stride() == ref_q.stride()
        and fused_s.shape == ref_s.shape
        and fused_s.dtype == ref_s.dtype
        and fused_s.stride() == ref_s.stride()
    )
    return codes, scale_agree, layout_ok


@pytest.mark.parametrize("m", MS)
def test_kda_fused_matches_fp32_then_quant(m: int) -> None:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.kimi_k3.amd.ops.oproj_quant import (
        fused_gated_rmsnorm_mxfp4_quant,
    )

    torch.manual_seed(0)
    x = torch.randn(m, H, D, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(m, H, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    fq, fs = fused_gated_rmsnorm_mxfp4_quant(x, g, w, eps=EPS)
    epi = _ref_gated_rmsnorm(x, g, w).reshape(m, K)
    tq, ts = dynamic_mxfp4_quant(epi)
    codes, scales, layout_ok = _agree(fq, fs, tq, ts, m)
    assert layout_ok, "fused KDA pair must match dynamic_mxfp4_quant layout"
    assert codes >= MIN_AGREE and scales >= MIN_AGREE, (
        f"KDA fused vs fp32-quant: codes={codes:.4f} scales={scales:.4f} at M={m}"
    )


@pytest.mark.parametrize("m", MS)
def test_mla_fused_matches_fp32_then_quant(m: int) -> None:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.kimi_k3.amd.ops.oproj_quant import (
        fused_sigmoid_gate_mxfp4_quant,
    )

    torch.manual_seed(0)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    fq, fs = fused_sigmoid_gate_mxfp4_quant(x, g)
    epi = x.float() * torch.sigmoid(g.float())
    tq, ts = dynamic_mxfp4_quant(epi)
    codes, scales, layout_ok = _agree(fq, fs, tq, ts, m)
    assert layout_ok, "fused MLA pair must match dynamic_mxfp4_quant layout"
    assert codes >= MIN_AGREE and scales >= MIN_AGREE, (
        f"MLA fused vs fp32-quant: codes={codes:.4f} scales={scales:.4f} at M={m}"
    )


def test_maybe_fused_declines_without_advertised_key() -> None:
    from vllm.models.kimi_k3.amd.ops.oproj_quant import (
        maybe_fused_kda_oproj_quant,
        maybe_fused_mla_oproj_quant,
    )

    o_proj = torch.nn.Linear(K, 8, bias=False)
    o_norm = torch.nn.Module()
    o_norm.weight = torch.ones(D, device="cuda", dtype=torch.bfloat16)
    o_norm.eps = EPS
    o_norm.activation = "sigmoid"
    x = torch.randn(1, H, D, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(1, H, D, device="cuda", dtype=torch.bfloat16)
    assert maybe_fused_kda_oproj_quant(x, g, o_norm, o_proj) is None
    attn = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    assert maybe_fused_mla_oproj_quant(attn, gate, o_proj) is None


def test_maybe_fused_returns_quantized_activation() -> None:
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
    from vllm.models.kimi_k3.amd.ops.oproj_quant import maybe_fused_mla_oproj_quant

    o_proj = torch.nn.Module()
    o_proj.input_quant_key = kMxfp4Dynamic
    attn = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
    qa = maybe_fused_mla_oproj_quant(attn, gate, o_proj)
    assert isinstance(qa, QuantizedActivation)
    assert qa.quant_key == kMxfp4Dynamic
    assert qa.orig_shape == attn.shape
    assert qa.scale.stride() == (1, 4)
