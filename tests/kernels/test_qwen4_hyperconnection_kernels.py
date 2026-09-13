# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness coverage for the self-developed Qwen4 hyperconnection kernels.

Torch references live in this test module only. The production wrappers are
required to fail closed instead of dispatching a Torch compute fallback.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")

from vllm.model_executor.layers.hyperconnection import (  # noqa: E402
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_gate_reduce,
    qwen4_hc_inject_combine,
)

DEVICE = current_platform.device_type
HC = 4
EPS = 1.0e-6

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Qwen4 hyperconnection Triton kernels require CUDA/ROCm.",
)


def _hc_refs(x, weight, logits, normed, injection, block, residual):
    hidden = weight.numel() // HC
    x3 = x.reshape(-1, HC, hidden).float()
    w2 = weight.reshape(HC, hidden).float()
    rms = torch.rsqrt(x3.square().mean(-1, keepdim=True) + EPS)
    norm_ref = (x3 * rms * (1.0 + w2)).to(x.dtype).reshape_as(x)
    gate_ref = (
        (
            torch.sigmoid(logits.float().reshape(-1, HC, hidden))
            * normed.float().reshape(-1, HC, hidden)
        )
        .mean(-2)
        .to(normed.dtype)
    )
    alpha = 2.0 * torch.sigmoid(injection.float() / HC)
    inject_ref = (
        (
            residual.float().reshape(-1, HC, hidden)
            + block.float().unsqueeze(-2) * alpha.unsqueeze(-1)
        )
        .to(residual.dtype)
        .reshape_as(residual)
    )
    return norm_ref, gate_ref, inject_ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,hidden", [(1, 257), (3, 513)])
def test_qwen4_hyperconnection_matches_torch(dtype, rows, hidden):
    torch.manual_seed(101 + rows + hidden)
    x = torch.randn((rows, HC * hidden), device=DEVICE, dtype=dtype)
    weight = torch.randn((HC * hidden,), device=DEVICE, dtype=dtype) * 0.01
    logits = torch.randn_like(x)
    normed = torch.randn_like(x)
    injection = torch.randn((rows, HC), device=DEVICE, dtype=dtype)
    block = torch.randn((rows, hidden), device=DEVICE, dtype=dtype)
    residual = torch.randn_like(x)
    norm_ref, gate_ref, inject_ref = _hc_refs(
        x, weight, logits, normed, injection, block, residual
    )

    norm_out = qwen4_grouped_gemma_rmsnorm(x, weight, HC, EPS)
    gate_out = qwen4_hc_gate_reduce(logits, normed, HC)
    inject_out = qwen4_hc_inject_combine(injection, block, residual, HC)

    atol = 3.0e-2 if dtype == torch.bfloat16 else 2.0e-2
    rtol = 2.0e-2
    torch.testing.assert_close(norm_out, norm_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(gate_out, gate_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(inject_out, inject_ref, atol=atol, rtol=rtol)

    snapshots = (norm_out.clone(), gate_out.clone(), inject_out.clone())
    for _ in range(10):
        torch.testing.assert_close(
            qwen4_grouped_gemma_rmsnorm(x, weight, HC, EPS),
            snapshots[0],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            qwen4_hc_gate_reduce(logits, normed, HC),
            snapshots[1],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            qwen4_hc_inject_combine(injection, block, residual, HC),
            snapshots[2],
            atol=0,
            rtol=0,
        )


def test_qwen4_hc_cpu_guards_fail_closed():
    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((16,), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        qwen4_grouped_gemma_rmsnorm(x, weight, 4, EPS)
