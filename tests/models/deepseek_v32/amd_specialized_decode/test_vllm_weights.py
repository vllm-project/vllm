# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of runtime/vllm_weights.py and the kernel's in-place weight readers
(glm/kernel.py unit_mxfp4 / unit_bf16 / mma_units), at the TP8 shapes:

* aiter's (16, 16) shuffle and e8m0_shuffle, as vLLM's AITER_MXFP4_MXFP4 MoE backend
  leaves the experts: lane group g4 of a 16-row group reads 16 contiguous bytes ==
  native K 32 g4 .. + 32 of a 128-K tile, and ONE scale byte; the shared expert's
  native reader (row-major values, transposed scales) reads the same bytes;
* the permuted MFMA K order: an emulation of the kernel's A (MXFP4 weights) and B
  (FP8 activation staged linearly, BF16 mids) operands gives W @ x;
* the row-major BF16 reader is the old packed tile's operand;
* weights_from_model on a fake loaded layer: views of the parameters (only w_uk /
  w_uv copied) and the refusals (fused shared experts, MoE backend, quantized
  attention, shapes);
* FP8 checkpoints: the checkpoint's FP8 bytes / per-row scales back from every layout
  vLLM's FP8 linear kernels leave (aiter preshuffle, hipBLASLt, row-major, [K, N]
  view) vs a dequantized reference, MLA's BF16 W_UK_T for w_uk, the fused indexer's
  projections, and the refusals (other FP8 schemes / kernels, FNUZ, scales)."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.layout import (
    atom_mxfp4_scale_index,
)
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import vllm_weights as VW

E, INTER, HIDDEN = 256, 256, 6144
FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


def _u8(g, *shape):
    return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)


# aiter's layouts, in pure torch
def shuffle(x):
    *_, n, kb = x.shape
    return (
        x.reshape(-1, n // 16, 16, kb // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(x.shape)
    )


def e8m0_shuffle(s):
    m, n = s.shape
    return (
        s.view(m // 32, 2, 16, n // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(m, n)
    )


# the kernel's reader offsets (glm/kernel.py unit_mxfp4)
def atom_dword(rg, kc, g4, row, k):
    return (rg * (k // 64) + kc * 2 + g4 // 2) * 128 + (g4 % 2) * 64 + (row % 16) * 4


def native_dword(row, kc, g4, k):
    return row * (k // 8) + kc * 16 + g4 * 4


@pytest.mark.parametrize("n, k", [(512, 6144), (6144, 256)])
def test_expert_readers_index_exact(n, k):
    g = _g(n)
    nat, s = _u8(g, n, k // 2), _u8(g, n, k // 32)
    sh, ss, st = (
        shuffle(nat).view(-1),
        e8m0_shuffle(s).view(-1),
        s.t().contiguous().view(-1),
    )
    for row in list(range(0, n, n // 64)) + [n - 1]:
        for kc in range(k // 128):
            for g4 in range(4):
                want = nat[row, kc * 64 + g4 * 16 : kc * 64 + g4 * 16 + 16]
                w = atom_dword(row // 16, kc, g4, row, k)
                assert torch.equal(sh[4 * w : 4 * w + 16], want)
                w = native_dword(row, kc, g4, k)
                assert torch.equal(nat.view(-1)[4 * w : 4 * w + 16], want)
                col = kc * 4 + g4
                assert int(ss[atom_mxfp4_scale_index(row, col, k // 32)]) == int(
                    s[row, col]
                )
                assert int(st[col * n + row]) == int(s[row, col])


def _fp4(b):
    return torch.stack([FP4[(b & 15).long()], FP4[(b >> 4).long()]], -1).reshape(-1)


def _mfma(a, b):
    """16x16x32: lane l supplies row / column l % 16, K (l // 16) * 8 .. + 8."""
    A, B = torch.zeros(16, 32), torch.zeros(32, 16)
    for ln in range(64):
        A[ln % 16, (ln // 16) * 8 : (ln // 16) * 8 + 8] = a[ln]
        B[(ln // 16) * 8 : (ln // 16) * 8 + 8, ln % 16] = b[ln]
    return A @ B


def test_permuted_k_dot():
    g = _g(1)
    n, k = 32, 512
    nat = _u8(g, n, k // 2)
    s = torch.randint(118, 130, (n, k // 32), generator=g, dtype=torch.uint8)
    scale = torch.pow(2.0, s.float() - 127).repeat_interleave(32, 1)
    w = torch.stack([_fp4(nat[r]) for r in range(n)]) * scale
    x = torch.randn(k, generator=g)
    sh = shuffle(nat).view(-1)
    ss = e8m0_shuffle(
        torch.cat([s, torch.zeros(256 - n, k // 32, dtype=torch.uint8)])
    ).view(-1)
    for rg in range(n // 16):
        c = torch.zeros(16, 16)
        for kc in range(k // 128):
            xs = x[kc * 128 : (kc + 1) * 128]
            for sp in range(4):
                a, b_bf16, b_fp8 = [], [], []
                for ln in range(64):
                    g4, row = ln // 16, rg * 16 + ln % 16
                    o = atom_dword(rg, kc, g4, row, k)
                    code = int(ss[atom_mxfp4_scale_index(row, kc * 4 + g4, k // 32)])
                    a.append(
                        _fp4(sh[4 * o : 4 * o + 16])[sp * 8 : sp * 8 + 8]
                        * 2.0 ** (code - 127)
                    )
                    bw = g4 * 16 + sp * 4  # BF16 words (2 values)
                    b_bf16.append(xs[2 * bw : 2 * bw + 8])
                    fw = g4 * 8 + sp * 2  # linear FP8 words (4 values)
                    b_fp8.append(xs[4 * fw : 4 * fw + 8])
                assert all(torch.equal(p, q) for p, q in zip(b_bf16, b_fp8))
                c += _mfma(a, b_bf16)
        torch.testing.assert_close(
            c[:, 0], w[rg * 16 : rg * 16 + 16] @ x, rtol=1e-4, atol=1e-4
        )


def _p(x):
    return torch.nn.Parameter(x, requires_grad=False)


def _layer(g, shared_native=True, backend="AITER_MXFP4_MXFP4"):
    bf = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)
    m = lambda **kw: NS(**{k: _p(v) for k, v in kw.items()})
    fp4 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    w13, w2 = _u8(g, E, 2 * INTER, HIDDEN // 2), _u8(g, E, HIDDEN, INTER // 2)
    # 3-D scales, as vLLM keeps them
    s13, s2 = _u8(g, E, 2 * INTER, HIDDEN // 32), _u8(g, E, HIDDEN, INTER // 32)
    gs, ds = (HIDDEN // 32, 2 * INTER), (INTER // 32, HIDDEN)
    if not shared_native:
        gs, ds = gs[::-1], ds[::-1]
    shared = NS(
        gate_up_proj=m(
            weight=_u8(g, 2 * INTER, HIDDEN // 2).view(fp4), weight_scale=_u8(g, *gs)
        ),
        down_proj=m(
            weight=_u8(g, HIDDEN, INTER // 2).view(fp4), weight_scale=_u8(g, *ds)
        ),
    )
    experts = NS(
        w13_weight=_p(w13.view(fp4)),
        w13_weight_scale=_p(s13),
        w2_weight=_p(w2.view(fp4)),
        w2_weight_scale=_p(s2),
        quant_method=NS(mxfp4_backend=NS(value=backend)),
    )
    return NS(
        input_layernorm=m(weight=bf(HIDDEN)),
        post_attention_layernorm=m(weight=bf(HIDDEN)),
        self_attn=NS(
            q_a_layernorm=m(weight=bf(2048)),
            kv_a_layernorm=m(weight=bf(512)),
            fused_qkv_a_proj=m(weight=bf(2624, HIDDEN)),
            q_b_proj=m(weight=bf(2048, 2048)),
            o_proj=m(weight=bf(HIDDEN, 2048)),
            kv_b_proj=m(weight=bf(8 * 448, 512)),
        ),
        mlp=NS(
            gate=NS(
                weight=_p(bf(E, HIDDEN)),
                e_score_correction_bias=_p(torch.randn(E, generator=g)),
            ),
            experts=NS(routed_experts=experts),
            shared_experts=shared,
        ),
    )


def _params(ns):
    out = []
    for v in vars(ns).values():
        if isinstance(v, torch.nn.Parameter):
            out.append(v)
        elif isinstance(v, NS):
            out += _params(v)
    return out


@pytest.mark.parametrize("native", [True, False])
def test_weights_from_model_views(native):
    lay = _layer(_g(2), shared_native=native)
    t, kw = VW.weights_from_model(lay)
    assert kw == dict(shared_native=native)
    storages = {p.untyped_storage().data_ptr() for p in _params(lay)}
    copies = sorted(
        k for k, v in t.items() if v.untyped_storage().data_ptr() not in storages
    )
    assert copies == ["w_uk", "w_uv"]
    kv_b = lay.self_attn.kv_b_proj.weight.data.view(8, 448, 512)
    assert torch.equal(t["w_uk"].view(8, 512, 192), kv_b[:, :192].transpose(1, 2))
    assert torch.equal(t["w_uv"].view(8, 256, 512), kv_b[:, 192:])
    assert t["w_ug"].data_ptr() == lay.mlp.experts.routed_experts.w13_weight.data_ptr()
    assert t["s_ug_sh"].shape == (
        (HIDDEN // 32, 2 * INTER) if native else (2 * INTER, HIDDEN // 32)
    )


def test_weights_from_model_refusals():
    def refused(lay, what):
        with pytest.raises(VW.WeightViewError, match=what):
            VW.weights_from_model(lay)

    g = _g(3)
    refused(_layer(g, backend="AITER_MXFP4_BF16"), "AITER_MXFP4_BF16")
    lay = _layer(g)
    lay.mlp.shared_experts = None
    refused(lay, "shared")
    lay = _layer(g)
    lay.self_attn.o_proj.weight = _p(
        lay.self_attn.o_proj.weight.data.to(torch.float8_e4m3fn)
    )
    refused(lay, "o_proj: FP8 weights need Quark per-channel FP8")
    lay = _layer(g)
    ex = lay.mlp.experts.routed_experts
    ex.w2_weight = _p(ex.w2_weight.data[:200])
    refused(lay, "experts.w_dn")
    lay = _layer(g)
    lay.self_attn.q_b_proj.weight = _p(lay.self_attn.q_b_proj.weight.data.t())
    refused(lay, "contiguous")


# ---------------------------------------------------------------- FP8 checkpoints
def _fp8(g, *shape):
    """Random finite E4M3 bytes (0x7F / 0xFF are NaN)."""
    b = _u8(g, *shape)
    return torch.where((b & 0x7F) == 0x7F, b ^ 1, b).view(torch.float8_e4m3fn)


def _fp8_linear(g, rows, k, kernel="AiterPreshuffledPerTokenFp8ScaledMMLinearKernel"):
    """A Quark PTPC FP8 linear as process_weights_after_loading leaves it under
    ``kernel``; returns (module, checkpoint values [rows, k], scales [rows])."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    q, s = _fp8(g, rows, k), torch.rand(rows, generator=g) * 1e-2 + 1e-4
    how = VW.FP8_LINEAR_LAYOUTS.get(kernel, "rows")
    qb = q.view(torch.uint8)
    if how == "shuffled":
        w = shuffle(qb).view(torch.float8_e4m3fn)
        w = w.t() if "HipbMM" in kernel else w
        ws = s.view(1, rows) if "HipbMM" in kernel else s.view(rows, 1)
    elif "AiterPerToken" in kernel:
        w, ws = q, s.view(rows, 1)
    else:
        w, ws = q.t(), s.view(rows, 1)  # QuarkW8A8Fp8's weight.t() view
    scheme = type("QuarkW8A8Fp8", (), {})()
    scheme.weight_quant_key, scheme.activation_quant_key = (
        kFp8StaticChannelSym,
        kFp8DynamicTokenSym,
    )
    scheme.fp8_linear = type(kernel, (), {})()
    return NS(weight=_p(w), weight_scale=_p(ws), scheme=scheme), q, s


def test_unshuffle_16x16_inverts_aiter_shuffle():
    x = _u8(_g(10), 64, 256)
    assert torch.equal(VW.unshuffle_16x16(shuffle(x)), x)


@pytest.mark.parametrize("kernel", sorted(VW.FP8_LINEAR_LAYOUTS))
def test_fp8_linear_rows_every_layout(kernel):
    """The checkpoint's bytes and scales back, whatever layout vLLM's FP8 kernel left;
    the dequantized matrix is the reference's."""
    mod, q, s = _fp8_linear(_g(11), 2048, 512, kernel)
    w, sc = VW.fp8_linear_rows(mod, "x", (2048, 512))
    assert w.dtype is torch.float8_e4m3fn and w.is_contiguous()
    assert torch.equal(w.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(sc, s)
    assert torch.equal(w.float() * sc[:, None], q.float() * s[:, None])
    if VW.FP8_LINEAR_LAYOUTS[kernel] == "rows":  # a view, no copy
        assert w.untyped_storage().data_ptr() == mod.weight.untyped_storage().data_ptr()


def test_fp8_linear_rows_refusals():
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
    )

    def refused(mod, what):
        with pytest.raises(VW.WeightViewError, match=what):
            VW.fp8_linear_rows(mod, "o_proj", (2048, 512))

    g = _g(12)
    mod, _, _ = _fp8_linear(g, 2048, 512, "SomeOtherFp8Kernel")
    refused(mod, "unknown FP8 linear kernel SomeOtherFp8Kernel")
    mod, _, _ = _fp8_linear(g, 2048, 512)
    mod.scheme.weight_quant_key = kFp8StaticTensorSym
    refused(mod, "per-channel")
    mod, _, _ = _fp8_linear(g, 2048, 512)
    mod.scheme = type("QuarkW8A8Fp8PerBlock", (), {})()
    refused(mod, "QuarkW8A8Fp8PerBlock")
    mod, _, _ = _fp8_linear(g, 2048, 512)
    mod.weight = _p(mod.weight.data.view(torch.uint8).view(torch.float8_e4m3fnuz))
    refused(mod, "e4m3fnuz")
    mod, _, _ = _fp8_linear(g, 2048, 512)
    mod.weight_scale = _p(torch.ones(2048 // 128, 512 // 128))
    refused(mod, "per-row scales")
    mod, _, _ = _fp8_linear(g, 2048, 512)
    mod.weight_scale.data[3] = 0.0
    refused(mod, "non-positive")


def _fp8_layer(g, kernel="AiterPreshuffledPerTokenFp8ScaledMMLinearKernel"):
    """_layer with the FP8 attention of amd/GLM-5.2-Quark-MXFP4-AttnFP8 at TP8."""
    lay = _layer(g)
    a, ref = lay.self_attn, {}
    for name, shape in (
        ("fused_qkv_a_proj", (2624, HIDDEN)),
        ("q_b_proj", (2048, 2048)),
        ("o_proj", (HIDDEN, 2048)),
        ("kv_b_proj", (8 * 448, 512)),
    ):
        mod, q, s = _fp8_linear(g, *shape, kernel)
        setattr(a, name, mod)
        ref[name] = (q, s)
    # MLA's own BF16 dequantization (split_kv_b_proj): W_UK_T [N, P, L]
    q, s = ref["kv_b_proj"]
    deq = (q.float() * s[:, None]).to(torch.bfloat16).view(8, 448, 512)
    a.W_UK_T = _p(deq[:, :192].contiguous())  # (N, P, L) like W_UK.permute(1, 2, 0)
    return lay, ref


@pytest.mark.parametrize(
    "kernel",
    [
        "AiterPreshuffledPerTokenFp8ScaledMMLinearKernel",
        "ChannelWiseTorchFP8ScaledMMLinearKernel",
    ],
)
def test_weights_from_model_fp8_checkpoint(kernel):
    lay, ref = _fp8_layer(_g(13), kernel)
    t, _ = VW.weights_from_model(lay)
    assert sorted(
        k for k in t if k.startswith("s_") and "ug" not in k and "dn" not in k
    ) == [
        "s_o",
        "s_q_b",
        "s_qkv_a",
        "s_uv",
    ]
    for key, name in (
        ("qkv_a", "fused_qkv_a_proj"),
        ("q_b", "q_b_proj"),
        ("o", "o_proj"),
    ):
        q, s = ref[name]
        assert torch.equal(t[f"w_{key}"].view(torch.uint8), q.view(torch.uint8))
        assert torch.equal(t[f"s_{key}"], s)
    q, s = ref["kv_b_proj"]
    assert torch.equal(
        t["w_uv"].view(torch.uint8).view(8, 256, 512),
        q.view(torch.uint8).view(8, 448, 512)[:, 192:],
    )
    assert torch.equal(t["s_uv"].view(8, 256), s.view(8, 448)[:, 192:])
    # the dequantized W_UV rows equal the reference dequantization exactly
    deq = q.float().view(8, 448, 512)[:, 192:] * s.view(8, 448, 1)[:, 192:]
    assert torch.equal(
        t["w_uv"].float().view(8, 256, 512) * t["s_uv"].view(8, 256, 1), deq
    )
    assert t["w_uk"].dtype is torch.bfloat16
    assert torch.equal(
        t["w_uk"].view(8, 512, 192), lay.self_attn.W_UK_T.data.transpose(1, 2)
    )


def test_weights_from_model_fp8_refusals():
    lay, _ = _fp8_layer(_g(14))
    del lay.self_attn.W_UK_T  # VLLM_ROCM_USE_AITER_FP8BMM=1: W_K / W_V instead
    with pytest.raises(VW.WeightViewError, match="W_UK_T"):
        VW.weights_from_model(lay)


def test_index_weights_from_model():
    g = _g(15)
    wq_b, q, s = _fp8_linear(g, 4096, 2048)
    kw = torch.randn(160, HIDDEN, generator=g).to(torch.bfloat16)
    attn = NS(
        indexer=NS(
            wq_b=wq_b,
            wk_weights_proj=NS(weight=_p(kw)),
            k_norm=NS(weight=_p(torch.randn(128)), bias=_p(torch.randn(128))),
        )
    )
    t = VW.index_weights_from_model(attn)
    assert torch.equal(t["w_index_q"].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(t["s_index_q"], s)
    assert t["w_index_k"].data_ptr() == kw.data_ptr()
    assert torch.equal(t["w_index_w"], kw[128:])
    # a BF16 wq_b (BF16 checkpoint): no fused indexer
    attn.indexer.wq_b = NS(weight=_p(torch.zeros(4096, 2048, dtype=torch.bfloat16)))
    assert VW.index_weights_from_model(attn) is None
