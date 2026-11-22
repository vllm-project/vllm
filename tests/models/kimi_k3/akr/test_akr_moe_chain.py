# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-level check of the amd_kernels moe/gemm routes (single GPU, real K3 TP8 shapes,
random weights): akr_moe.forward (router+sort aux -> down+quant -> own experts) vs the
stock sequence vLLM runs today (fp32 router GEMM + aiter biased_grouped_topk -> hgemm
down projection -> aiter fused_moe a4w4 SiTUv2)."""

import os
import types

import pytest
import torch

os.environ.setdefault("AITER_SITUV2_A4W4", "1")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("amd_kernels")

E, H, LAT, INTER, TOPK = 896, 7168, 3584, 384, 16
M_LIST = [1, 12, 24, 40, 65, 80, 128, 192]


@pytest.fixture(scope="module")
def weights():
    from amd_kernels.moe.latent_moe_common import moe_weights

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1)
    mk = lambda *s, lo=0, hi=256: torch.randint(
        lo, hi, s, dtype=torch.uint8, device=dev, generator=g
    )  # noqa: E731
    # vLLM's aiter-layout parameters: w13 [E, 2I, LAT/2], w2 [E, LAT, I/2] + E8M0 scales
    w = moe_weights(
        mk(E, 2 * INTER, LAT // 2),
        mk(E, 2 * INTER, LAT // 32, lo=118, hi=124),
        mk(E, LAT, INTER // 2),
        mk(E, LAT, 16, lo=118, hi=124),
    )
    gate = (torch.randn(E, H, device=dev, generator=g) * 0.02).bfloat16()
    bias = torch.randn(E, device=dev, generator=g) * 0.1
    down = (torch.randn(LAT, H, device=dev, generator=g) * 0.02).bfloat16()
    return w, gate, bias, down


def _stock(x, w, gate, bias, down):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401

    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = (
        w.w13.view(E, 2 * INTER, LAT // 2).view(fp4),
        w.w2.view(E, LAT, INTER // 2).view(fp4),
    )
    w13.is_shuffled = w2.is_shuffled = True
    s13 = w.s13.view(E, 2 * INTER, LAT // 32).view(torch.float8_e8m0fnu)
    s2 = w.s2.view(E, LAT, 16).view(torch.float8_e8m0fnu)
    m = x.shape[0]
    logits = torch.mm(x, gate.t(), out_dtype=torch.float32)
    from aiter import biased_grouped_topk

    tw = torch.empty(m, TOPK, dtype=torch.float32, device=x.device)
    ti = torch.empty(m, TOPK, dtype=torch.int32, device=x.device)
    biased_grouped_topk(logits, bias, tw, ti, 1, 1, True, 1.0)
    y = x @ down.t()
    out = fm.fused_moe(
        y,
        w13,
        w2,
        tw,
        ti,
        None,
        ActivationType.Situv2,
        QuantType.per_1x32,
        False,
        s13,
        s2,
        None,
        None,
        beta=4.0,
        linear_beta=25.0,
        gate_mode="separated",
        dtype=torch.bfloat16,
    )
    return out, tw, ti


def _fake_moe(w, gate, bias, down, captured):
    from vllm.models.kimi_k3.amd import akr_moe

    runner = types.SimpleNamespace(
        _shared_experts=None,
        _akr_defer_final=True,
        _up_proj_shard_size=896,
        routed_input_transform=types.SimpleNamespace(
            weight=types.SimpleNamespace(data=down)
        ),
        _shard_up_proj_tail=lambda lat, shared, trunc: (captured.append(lat), shared)[
            1
        ],
    )
    moe = types.SimpleNamespace(
        experts=runner,
        gate=types.SimpleNamespace(
            weight=types.SimpleNamespace(data=gate),
            e_score_correction_bias=types.SimpleNamespace(data=bias),
        ),
        shared_experts=lambda x: torch.zeros(
            x.shape[0], H, dtype=torch.bfloat16, device=x.device
        ),
        _akr_state=akr_moe.LayerState(w, True, True, False, None, None, 0.0, H, TOPK),
    )
    return moe


@pytest.mark.parametrize("m", M_LIST)
def test_chain_vs_stock(weights, m):
    from vllm.models.kimi_k3.amd import akr_moe

    w, gate, bias, down = weights
    x = (torch.randn(m, H, device="cuda") * 0.5).bfloat16()
    ref, tw, ti = _stock(x, w, gate, bias, down)
    cap: list = []
    moe = _fake_moe(w, gate, bias, down, cap)
    assert akr_moe.eligible(moe, x)
    akr_moe.forward(moe, x)
    torch.accelerator.synchronize()
    out = cap[0]
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 1.5e-2, rel.item()
    # chain vs the amd_kernels fp64 reference is covered by the package tests; here:
    # same routing
    from amd_kernels.moe.latent_moe_common import router_sigmoid_bias_topk

    aux = akr_moe._aux(moe, m, x.device, TOPK)
    w2, i2 = router_sigmoid_bias_topk(
        x, gate, bias, 1.0, topk=TOPK, shuffled=False, aux=aux
    )
    assert torch.equal(i2.sort(1).values, ti.sort(1).values)
    o2, o1 = i2.long().argsort(1), ti.long().argsort(1)
    assert torch.allclose(w2.gather(1, o2), tw.gather(1, o1), rtol=1e-4, atol=1e-6)
    print(f"M={m} chain vs stock rel L2 {rel.item():.2e}")
