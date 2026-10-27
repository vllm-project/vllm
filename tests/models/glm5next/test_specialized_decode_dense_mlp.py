# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's dense MLP GEMVs against an FP64 PyTorch reference of
the stock rounding steps: their error must not exceed the stock DeepGEMM path's."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

from .test_specialized_decode_shared_expert import _packed, _quant, _weight

# One TP4 shard of GLM-5.3-Flash's dense MLP.
HIDDEN, INTER, LIMIT = 4096, 3072, 10.0

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)


def _dequant(w, e):
    s = (e.double() - 127).exp2()
    return w.double() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


def _reference(x, w13, e13, w2, e2, ue8m0):
    gate_up = (_quant(x, ue8m0).double() @ _dequant(w13, e13).T).bfloat16().double()
    gate, up = gate_up.split(INTER, dim=-1)
    act = (F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)).bfloat16()
    return _quant(act, ue8m0).double() @ _dequant(w2, e2).T


def _stock(x, w13, e13, w2, e2, ue8m0):
    """Input quant, DeepGEMM, clamped SwiGLU, quant, DeepGEMM, as the stock
    ``Glm5NextMLP`` runs a decode step."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt

    assert ue8m0

    def linear(a, w, e):
        q = per_token_group_quant_fp8_packed_for_deepgemm(a, 128, use_ue8m0=True)
        out = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device=a.device)
        fp8_gemm_nt(q, (w, _packed(e, w.shape[0])), out, is_deep_gemm_e8m0_used=True)
        return out

    gate_up = linear(x, w13, e13)
    act = torch.empty(x.shape[0], INTER, dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(act, gate_up, LIMIT, 1.0, 0.0)
    return linear(act, w2, e2)


def _errors(out, ref):
    error = (out.double() - ref).abs().flatten()
    return torch.stack([error.median(), error.quantile(0.99), error.max()])


def _run(tokens, ue8m0):
    from vllm.models.glm5next.nvidia.specialized_decode.dense_mlp import DenseMlp

    g = torch.Generator(device="cuda").manual_seed(tokens)
    w13, e13 = _weight(g, 2 * INTER, HIDDEN)  # [gate; up]
    w2, e2 = _weight(g, HIDDEN, INTER)
    dense = DenseMlp(
        w13, _packed(e13, 2 * INTER), w2, _packed(e2, HIDDEN), LIMIT, ue8m0
    )
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
    x[:, :128] = 0  # an all-zero group takes the 1e-10 amax floor
    return dense(x), (x, w13, e13, w2, e2, ue8m0)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_dense_mlp_no_worse_than_stock(tokens):
    """UE8M0 scales, the only ones DeepGEMM takes on SM100."""
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported():
        pytest.skip("The stock path needs DeepGEMM")
    out, args = _run(tokens, ue8m0=True)
    ref = _reference(*args)
    ours, stock = _errors(out, ref), _errors(_stock(*args), ref)
    # Median, p99 and max error, each no worse than stock's.
    assert (ours <= stock).all(), (ours.tolist(), stock.tolist())


@pytest.mark.parametrize("tokens", [1, 8])
def test_dense_mlp_fp32_scales(tokens):
    out, args = _run(tokens, ue8m0=False)
    ref = _reference(*args)
    torch.testing.assert_close(out.double(), ref, atol=5e-3 * ref.abs().max(), rtol=0)
