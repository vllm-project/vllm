# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper dense MLP (the `dense_mlp` GEMVs on SM90) against an FP64
reference of the stock rounding steps: its error must not exceed the stock
Hopper path's (Triton block-FP8 GEMMs)."""

import pytest
import torch

from vllm.platforms import current_platform

# One TP4 shard of GLM-5.3-Flash's dense MLP.
HIDDEN, INTER, LIMIT = 4096, 3072, 10.0
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Requires SM90"
)


def _scale_rule(amax, ue8m0):
    scale = (amax.float() * (1.0 / 448.0)).double()
    return torch.exp2(torch.ceil(torch.log2(scale))) if ue8m0 else scale


def _quant(x, ue8m0):
    """FP8 per 128 along the last dim, dequantized (the stock scale rule)."""
    blocks = x.double().unflatten(-1, (-1, 128))
    scale = _scale_rule(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10), ue8m0)
    return ((blocks / scale).float().to(FP8).double() * scale).flatten(-2)


def _dequant(w, s):
    return w.double() * s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)


def _reference(x, w13, s13, w2, s2, ue8m0):
    """FP64 with the stock rounding: BF16 gate/up and SwiGLU output."""
    gate_up = (_quant(x, ue8m0) @ _dequant(w13, s13).T).bfloat16().double()
    gate, up = gate_up.split(INTER, dim=-1)
    gate = gate.clamp(max=LIMIT)
    act = (gate * torch.sigmoid(gate) * up.clamp(-LIMIT, LIMIT)).bfloat16()
    return _quant(act, ue8m0) @ _dequant(w2, s2).T


def _stock(x, w13, s13, w2, s2, ue8m0):
    """Input quant, block-FP8 GEMM, clamped SwiGLU, quant, block-FP8 GEMM."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        w8a8_triton_block_scaled_mm,
    )

    def linear(a, w, s):
        q, qs = per_token_group_quant_fp8(a, 128, use_ue8m0=ue8m0)
        return w8a8_triton_block_scaled_mm(q, w, qs, s, [128, 128], torch.bfloat16)

    gate_up = linear(x, w13, s13)
    act = torch.empty(x.shape[0], INTER, dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(act, gate_up, LIMIT, 1.0, 0.0)
    return linear(act, w2, s2)


def _weights(g, n, k):
    w = (torch.randn(n, k, generator=g, device="cuda") * 8.0).to(FP8)
    s = (0.5 + torch.rand(n // 128, k // 128, generator=g, device="cuda")) * 2e-3
    return w, s


def _errors(out, ref):
    error = (out.double() - ref).abs().flatten()
    return torch.stack([error.median(), error.quantile(0.99), error.max()])


@pytest.mark.parametrize("tokens", [1, 4, 6, 8])
@pytest.mark.parametrize("ue8m0", [True, False])
def test_dense_mlp_no_worse_than_stock(tokens, ue8m0):
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.dense import (
        HopperDenseMlp,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    w13, s13 = _weights(g, 2 * INTER, HIDDEN)  # [gate; up]
    w2, s2 = _weights(g, HIDDEN, INTER)
    dense = HopperDenseMlp(w13, s13, w2, s2, LIMIT, ue8m0)
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
    x[:, :128] = 0  # an all-zero group takes the amax floor
    out = dense(x)
    args = (x, w13, s13, w2, s2, ue8m0)
    ref = _reference(*args)
    ours, stock = _errors(out, ref), _errors(_stock(*args), ref)
    # Median and p99 error no worse than stock's. The largest single-element
    # errors come from hidden values on an FP8 rounding tie, which any two
    # kernels round differently, so the max gets 25% slack.
    assert (ours[:2] <= stock[:2]).all(), (ours.tolist(), stock.tolist())
    assert ours[2] <= 1.25 * stock[2], (ours.tolist(), stock.tolist())
    assert torch.equal(out, dense(x))


def test_dense_mlp_cuda_graph():
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.dense import (
        HopperDenseMlp,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    w13, s13 = _weights(g, 2 * INTER, HIDDEN)
    w2, s2 = _weights(g, HIDDEN, INTER)
    dense = HopperDenseMlp(w13, s13, w2, s2, LIMIT, True)
    for tokens in (1, 8):
        x = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        dense.prepare([tokens])
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            y = dense(x)
        x.copy_(torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16())
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(y, dense(x))
