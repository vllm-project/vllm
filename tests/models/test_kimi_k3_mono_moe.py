# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 mono MoE launch (gfx950) against the kernels it replaces.

One launch runs routing, the routed experts and the shared expert. Routed
experts: AITER's ``biased_grouped_topk`` then ``fused_moe`` on the a4w4
SiTUv2 path, with weights as vLLM's AITER_MXFP4_BF16 loader leaves them
([gate; up] rows, ``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4``). Shared
expert: vLLM's KimiMLP math (bf16 gate_up GEMM, SiTU, bf16 down GEMM).

Routing is drawn from a pool of experts (decode batches touch few distinct
experts) or uniform (pool 0). The launch resets its own control words at exit,
so it is also replayed from a CUDA graph, twice.
"""

import os

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("the mono MoE launch is gfx950 only", allow_module_level=True)

os.environ.setdefault("AITER_SITUV2_A4W4", "1")
aiter = pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from aiter import ActivationType, QuantType, dtypes, get_torch_quant  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4  # noqa: E402

from vllm.models.kimi_k3.amd.mono.runner import M_MAX, mono_moe, supported  # noqa: E402

# Kimi-K3 at TP8: 896 experts, top-16, latent hidden 3584, 384 intermediate
# columns a rank; the shared expert runs at the full hidden 7168, 768 columns.
NE, TOPK, HIDDEN, INTER = 896, 16, 3584, 384
SH_HIDDEN, SH_INTER = 7168, 768
BETA, LINEAR_BETA = 4.0, 25.0
SH_BETA, SH_LINEAR_BETA = 4.0, 25.0


@pytest.fixture(scope="module")
def weights():
    torch.set_default_device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    quant = get_torch_quant(QuantType.per_1x32)

    def make(n, k, gate_up):
        qs, ss = [], []
        for e0 in range(0, NE, 64):
            w = torch.randn((min(64, NE - e0), n, k), generator=g, dtype=torch.bfloat16)
            q, s = quant(w * 0.05, quant_dtype=dtypes.fp4x2)
            qs.append(q.view(w.shape[0], n, k // 2))
            ss.append(s.view(w.shape[0], n, k // 32))
        q = shuffle_weight_a16w4(
            torch.cat(qs).view(torch.float4_e2m1fn_x2), 16, gate_up
        )
        s = torch.cat(ss)
        s = shuffle_scale_a16w4(s.view(-1, s.shape[-1]), NE, gate_up)
        q.is_shuffled = True
        return q, s

    w1, w1s = make(2 * INTER, HIDDEN, True)
    w2, w2s = make(HIDDEN, INTER, False)
    w_gu = torch.randn(2 * SH_INTER, SH_HIDDEN, generator=g, dtype=torch.bfloat16)
    w_dn = torch.randn(SH_HIDDEN, SH_INTER, generator=g, dtype=torch.bfloat16)
    w_gu, w_dn = w_gu * 0.02, w_dn * 0.02
    return w1, w2, w1s, w2s, w_gu, w_dn


def make_inputs(m, pool, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(m, NE, generator=g, device="cuda")
    if 0 < pool < NE:
        logits -= 8.0
        logits[:, torch.randperm(NE, generator=g, device="cuda")[:pool]] += 16.0
    x = torch.randn(m, HIDDEN, generator=g, dtype=torch.bfloat16, device="cuda")
    sx = torch.randn(m, SH_HIDDEN, generator=g, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NE, generator=g, device="cuda") * 0.01
    return logits, x, sx, bias


def routed_reference(logits, bias, x, w1, w2, w1s, w2s):
    m = x.shape[0]
    tw = torch.empty(m, TOPK, dtype=torch.float32, device="cuda")
    ti = torch.empty(m, TOPK, dtype=torch.int32, device="cuda")
    aiter.biased_grouped_topk(logits, bias, tw, ti, 1, 1, True, 1.0)
    out = fused_moe(
        x, w1, w2, tw, ti,
        quant_type=QuantType.per_1x32, activation=ActivationType.Situv2,
        w1_scale=w1s, w2_scale=w2s, gate_mode=GateMode.SEPARATED.value,
        swiglu_limit=0.0, beta=BETA, linear_beta=LINEAR_BETA,
    )  # fmt: skip
    return out, tw, ti


def shared_reference(x, w_gu, w_dn):
    gu = (x @ w_gu.t()).float()
    d = gu.shape[-1] // 2
    gate = SH_BETA * torch.tanh(gu[:, :d] / SH_BETA) * torch.sigmoid(gu[:, :d])
    up = SH_LINEAR_BETA * torch.tanh(gu[:, d:] / SH_LINEAR_BETA)
    return (gate * up).to(x.dtype) @ w_dn.t()


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def cos_min(a, b):
    return (
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1).min().item()
    )


def test_supported_sizes(weights):
    w1, _, _, _, w_gu, w_dn = weights
    for m in (1, M_MAX, M_MAX + 1):
        x = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        sx = torch.empty(m, SH_HIDDEN, dtype=torch.bfloat16, device="cuda")
        assert supported(x, NE, TOPK, INTER, sx, w_gu, w_dn) == (m <= M_MAX)
    assert not supported(x, NE, TOPK, INTER, x, w_gu, w_dn)


@pytest.mark.parametrize("pool", [64, 0])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 13, 16])
def test_mono_moe(weights, m, pool):
    w1, w2, w1s, w2s, w_gu, w_dn = weights
    logits, x, sx, bias = make_inputs(m, pool, seed=100 + m)
    ref, ref_w, ref_i = routed_reference(logits, bias, x, w1, w2, w1s, w2s)
    sh_ref = shared_reference(sx, w_gu, w_dn)

    tw = torch.empty(m, TOPK, dtype=torch.float32, device="cuda")
    ti = torch.empty(m, TOPK, dtype=torch.int32, device="cuda")
    out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    sh_out = torch.empty(m, SH_HIDDEN, dtype=torch.bfloat16, device="cuda")

    def run():
        return mono_moe(
            logits, bias, x, w1, w2, w1s, w2s, sx, w_gu, w_dn, topk=TOPK,
            situ_beta=BETA, situ_linear_beta=LINEAR_BETA, shared_beta=SH_BETA,
            shared_linear_beta=SH_LINEAR_BETA, out=out, shared_out=sh_out,
            topk_weights=tw, topk_ids=ti,
        )  # fmt: skip

    run()
    torch.accelerator.synchronize()
    assert torch.equal(torch.sort(ti, 1)[0], torch.sort(ref_i, 1)[0])
    w_err = (torch.sort(tw, 1)[0] - torch.sort(ref_w, 1)[0]).abs().max().item()
    assert w_err < 1e-6
    assert cos_min(out, ref) > 0.9999
    assert cos_min(sh_out, sh_ref) > 0.9999

    g = graph_of(run)
    for _ in range(2):
        out.fill_(7)
        sh_out.fill_(7)
        g.replay()
        torch.accelerator.synchronize()
        assert cos_min(out, ref) > 0.9999
        assert cos_min(sh_out, sh_ref) > 0.9999
