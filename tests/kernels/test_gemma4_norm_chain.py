# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.layers.fusion.gemma4_norm_chain import (
    attn_ff_norm,
    ple_norm_add_scale,
    post_ff_norm_add,
    supports_hidden_size,
)

EPS = 1e-6


def _rms(x, w):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    return out.to(torch.bfloat16)


def _rand(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 7, 256, 1031])
@pytest.mark.parametrize("hidden", [1536, 2560])
@pytest.mark.parametrize("quant", [True, False])
def test_norm_chain(tokens, hidden, quant):
    assert supports_hidden_size(hidden)
    torch.manual_seed(0)
    attn, x0, mlp, ple = (_rand(tokens, hidden, scale=s) for s in (1, 8, 2, 3))
    w_pa, w_pf, w_pff, w_ple, w_in = (_rand(hidden) * 0.5 + 1 for _ in range(5))
    scalar = torch.tensor([0.83], device="cuda", dtype=torch.bfloat16)

    x1, y, ys = attn_ff_norm(attn, x0, w_pa, w_pf, EPS, quant)
    x1_ref = _rms(attn, w_pa) + x0
    torch.testing.assert_close(x1, x1_ref, atol=0.07, rtol=1e-2)
    y_ref = _rms(x1, w_pf).float()
    if quant:
        torch.testing.assert_close(y.float() * ys, y_ref, atol=0.05 * y_ref.abs().max(), rtol=0.1)
    else:
        torch.testing.assert_close(y.float(), y_ref, atol=0.07, rtol=1e-2)

    x2 = post_ff_norm_add(mlp, x1, w_pff, EPS)
    torch.testing.assert_close(x2, _rms(mlp, w_pff) + x1, atol=0.07, rtol=1e-2)

    x3, nxt = ple_norm_add_scale(ple, x2, w_ple, scalar, w_in, EPS)
    x3_ref = (x2 + _rms(ple, w_ple)) * scalar
    torch.testing.assert_close(x3, x3_ref, atol=0.07, rtol=1e-2)
    torch.testing.assert_close(nxt, _rms(x3, w_in), atol=0.07, rtol=1e-2)
    x3b, nxt_b = ple_norm_add_scale(ple, x2, w_ple, scalar, None, EPS)
    assert nxt_b.numel() == 0
    torch.testing.assert_close(x3b, x3, atol=0, rtol=0)
