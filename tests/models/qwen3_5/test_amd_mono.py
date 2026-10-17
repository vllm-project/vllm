# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Qwen3.8 mono decode kernels on MI355X against the ops vLLM
runs for the same decode step, random weights in the checkpoint's formats at
TP8 shapes.

- K1 (one GPU): input add + input_layernorm -> in_proj_qkvz / in_proj_ba ->
  AITER's conv update and fused gated delta rule -> RMSNormGated, the conv / SSM
  state each step leaves behind, pad rows.
- K2 (eight GPUs): o_proj + all-reduce + residual add + post_attention_layernorm
  + the sparse MoE block (softmax top-10 router, MXFP4 experts, the
  sigmoid-gated shared expert) + the final all-reduce. The MoE is compared on
  the kernel's own norm output, where routing cannot differ: against a torch
  emulation of the stock decode kernel's numerics, and loosely against aiter's
  ``fused_moe`` itself.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.platforms import current_platform

bf = torch.bfloat16
COS_MIN = 0.9999
STOCK_COS_MIN = 0.9995


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the Qwen3.8 mono kernels need gfx950 (MI355X)"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def _all_reduce_fp32(t: torch.Tensor) -> torch.Tensor:
    """The ranks' bf16 partials summed in fp32, as vLLM's custom all-reduce."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(bf)


# ---------------------------------------------------------------- K1


def _pairs(scratch, region, n):
    """A scratch mailbox region's first ``n`` values as f32."""
    off = region[0]
    return (
        scratch[off : off + n * 8]
        .view(torch.int32)
        .view(n, 2)[:, 0]
        .view(torch.float32)
    )


def _gdn_inputs(s: int, n_real: int, first: bool, slots: int = 16):
    from vllm.models.qwen3_5.amd.mono.layout import BA, CONV, HD, HIDDEN, NV, QKVZ

    g = torch.Generator(device="cuda").manual_seed(0)
    w = {
        "hidden": _randn(g, s, HIDDEN, scale=2.0),
        "residual": None if first else _randn(g, s, HIDDEN, scale=2.0),
        "ln_w": _randn(g, HIDDEN, scale=0.1),
        "w_qkvz": _randn(g, QKVZ, HIDDEN, scale=0.02),
        "w_ba": _randn(g, BA, HIDDEN, scale=0.05),
        "conv_w": _randn(g, CONV, 1, 4, scale=0.4),
        "a_log": _randn(g, NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, NV, scale=0.5),
        "norm_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    pad = _randn(g, slots, NV * HD * HD + 8192, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, : NV * HD * HD].view(slots, NV, HD, HD)
    conv = _randn(g, slots, CONV * 3 + 1024)[:, : CONV * 3]
    w["conv_state"] = conv.view(slots, 3, CONV).transpose(-1, -2)
    # live rows on slots 3 ..; pad rows on the null block 0
    idx = torch.zeros(s, dtype=torch.int32, device="cuda")
    idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device="cuda")
    w["st_idx"] = idx
    return w


def _gdn_core_ref(w, n: int, qkvz, ba, nk: int, nv: int, gate: str):
    """The GDN decode path on ROCm from in_proj's outputs (AITER's conv update
    and fused gated delta rule, then RMSNormGated with a ``gate`` output gate)
    on the first ``n`` rows; the rest are the cudagraph pad. -> (conv state,
    SSM state, core [S, NV HD])."""
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token,
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule,
    )

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.models.qwen3_5.amd.mono.layout import HD

    s = qkvz.size(0)
    with set_current_vllm_config(VllmConfig()):
        norm = RMSNormGated(
            HD, eps=1e-6, norm_before_gate=True, activation=gate, device="cuda"
        )
    norm.weight.data = w["norm_w"].clone()
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    z = torch.empty(s, nv, HD, dtype=bf, device="cuda")
    core = torch.empty(s, nv, HD, dtype=bf, device="cuda")
    mixed, b, a = fused_reshape_causal_conv1d_update_single_token(
        qkvz.contiguous(),
        n,
        nk,
        nv,
        HD,
        HD,
        ba.contiguous(),
        z,
        core,
        conv_state,
        w["conv_w"].view(-1, 4),
        None,
        "silu",
        conv_state_indices=w["st_idx"][:n],
        validate_data=True,
        qkvz_layout="flat",
    )
    fused_rearrange_sigmoid_gated_delta_rule(
        A_log=w["a_log"],
        a=a,
        b=b,
        dt_bias=w["dt_bias"],
        qkv=mixed,
        key_dim=nk * HD,
        value_dim=nv * HD,
        head_k_dim=HD,
        head_v_dim=HD,
        initial_state=rstate,
        inplace_final_state=True,
        cu_seqlens=torch.arange(n + 1, dtype=torch.int32, device="cuda"),
        ssm_state_indices=w["st_idx"][:n],
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core.reshape(-1),
    )
    return conv_state, rstate, norm.forward_native(core, z).reshape(s, nv * HD)


def _gdn_reference(w, n: int, proj: torch.Tensor | None = None):
    """Qwen3_5DecoderLayer's input norm and the GDN decode path on ROCm
    (``forward_hip`` -> ``_forward_core_decode_aiter`` -> ``_output_projection``
    without out_proj) on the first ``n`` rows; the rest are the cudagraph pad.
    ``proj``: in_proj's output to start from instead of computing it."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.models.qwen3_5.amd.mono.layout import HIDDEN, NK, NV

    with set_current_vllm_config(VllmConfig()):
        ln = GemmaRMSNorm(HIDDEN, eps=1e-6).cuda()
    ln.weight.data = w["ln_w"].clone()
    if w["residual"] is None:
        x, res = ln.forward_native(w["hidden"]), w["hidden"].clone()
    else:
        x, res = ln.forward_native(w["hidden"], w["residual"].clone())
    if proj is None:
        if "w_qkvzs" in w:  # online MXFP4: dynamic MXFP4 of x, fp32 accumulation
            qkvz = F.linear(_mxfp4_qdq(x).to(bf), w["w_qkvz_ref"])
        else:
            qkvz = F.linear(x, w["w_qkvz"])
        ba = F.linear(x, w["w_ba"])
    else:
        qkvz, ba = proj.split([w["w_qkvz"].size(0), w["w_ba"].size(0)], dim=-1)
    conv_state, rstate, core = _gdn_core_ref(w, n, qkvz, ba, NK, NV, "silu")
    return {
        "x": x,
        "residual": res,
        "proj": torch.cat([qkvz, ba], dim=-1),
        "conv_state": conv_state,
        "rstate": rstate,
        "core": core,
    }


@pytest.mark.parametrize("s", [1, 3, 8])
@torch.inference_mode()
def test_attention_front_matches_vllm_ops(s: int) -> None:
    """A full-attention layer's front, which the mono forward runs between K2
    launches: the compiled Gemma add + norm against GemmaRMSNorm.forward_native,
    and the fused split / q, k norm / RoPE / gate copy against the model's
    eager ops (``_project_qkv_gate`` off CUDA; K2 applies the sigmoid gate).
    TP8 shapes: 8 query heads and one KV head of 256 a rank, 64 rotary dims."""
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        _attention_front(s)


def _attention_front(s: int) -> None:
    from types import SimpleNamespace

    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    from vllm.models.qwen3_5.amd import mono_decode as md
    from vllm.models.qwen3_5.amd.mono.layout import HIDDEN

    hq, hkv, hd, rd = 8, 1, 256, 64
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    g = torch.Generator(device=dev).manual_seed(s)
    norm = GemmaRMSNorm(HIDDEN, eps=1e-6).to(dev, dtype=bf)
    norm.weight.copy_(_randn(g, HIDDEN, scale=0.1))
    x, res = _randn(g, s, HIDDEN), _randn(g, s, HIDDEN)
    got_x, got_res = md._gemma_norm(norm, x, res)
    want_x, want_res = norm.forward_native(x, res)
    assert torch.equal(got_res, want_res)
    assert _cos(got_x, want_x) > COS_MIN
    assert (got_x.float() - want_x.float()).abs().max().item() < 2e-2

    q_norm = GemmaRMSNorm(hd, eps=1e-6).to(dev, dtype=bf)
    k_norm = GemmaRMSNorm(hd, eps=1e-6).to(dev, dtype=bf)
    q_norm.weight.copy_(_randn(g, hd, scale=0.1))
    k_norm.weight.copy_(_randn(g, hd, scale=0.1))
    rope = RotaryEmbedding(hd, rd, 262144, 10000000.0, True, bf).to(dev)
    qkv = _randn(g, s, hq * 2 * hd + 2 * hkv * hd)
    positions = torch.randint(0, 200000, (s,), device=dev, generator=g)
    attn = SimpleNamespace(
        qkv_proj=lambda t: (t, None),
        q_size=hq * hd,
        kv_size=hkv * hd,
        rotary_emb=rope,
        q_norm=q_norm,
        k_norm=k_norm,
        num_heads=hq,
        num_kv_heads=hkv,
        head_dim=hd,
        attn=lambda q, k, v: q + k.repeat(1, hq) + v.repeat(1, hq),
    )
    got, got_gate = md._attention_core(attn, qkv, positions)

    q_gate, k, v = qkv.split([hq * hd * 2, hkv * hd, hkv * hd], dim=-1)
    q, gate = q_gate.view(s, hq, 2 * hd).chunk(2, dim=-1)
    q = q_norm.forward_native(q.reshape(s, hq, hd)).reshape(s, hq * hd)
    k = k_norm.forward_native(k.reshape(s, hkv, hd)).reshape(s, hkv * hd)
    q, k = rope.forward_native(positions, q, k)
    want = attn.attn(q, k, v)
    assert torch.equal(got_gate, gate.reshape(s, hq * hd))
    assert got_gate.is_contiguous() and got.is_contiguous()
    assert got.shape == want.shape
    assert _cos(got, want) > COS_MIN
    torch.testing.assert_close(got, want, atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("qq", [False, True])
@pytest.mark.parametrize(
    "s,n_real,first", [(1, 1, False), (5, 4, False), (8, 8, False), (8, 7, True)]
)
def test_k1_matches_vllm_ops(
    s: int,
    n_real: int,
    first: bool,
    qq: bool,
) -> None:
    """K1 against GemmaRMSNorm -> in_proj -> AITER's conv update + fused gated
    delta rule -> RMSNormGated on one decode step, the states it leaves behind,
    and the pad rows (null block: no state written, zero core). The conv state
    is in the (slots, 3, dim) layout AITER's conv update takes. ``qq``: an
    online-MXFP4 in_proj_qkvz on the ASM FP4 GEMM (in_proj_ba as K-split parts
    the gates sum)."""
    from vllm.models.qwen3_5.amd.mono import gdn
    from vllm.models.qwen3_5.amd.mono.layout import CORE, HIDDEN, QKVZ

    w = _gdn_inputs(s, n_real, first)
    if qq:
        g = torch.Generator(device="cuda").manual_seed(1)
        q, sc, w["w_qkvz"], w["w_qkvzs"] = _mxfp4_linear(g, QKVZ, HIDDEN)
        w["w_qkvz_ref"] = _mxfp4_dequant(q, sc).to(bf)
    ref = _gdn_reference(w, n_real)
    if qq:
        stock = _mxfp4_linear_stock(ref["x"], w["w_qkvz"], w["w_qkvzs"])
        assert _cos(stock, ref["proj"][:, :QKVZ]) > STOCK_COS_MIN
    key = gdn.K1Build(tokens=s, first=first, qq=qq)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    res_out = torch.empty(s, HIDDEN, dtype=bf, device="cuda")
    core = torch.full((s, CORE), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(gdn.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    gdn.gdn_pre(
        key,
        hidden=w["hidden"],
        residual=w["residual"],
        res_out=res_out,
        ln_w=w["ln_w"],
        w_qkvz=w["w_qkvz"],
        w_qkvzs=w.get("w_qkvzs"),
        w_ba=w["w_ba"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        norm_w=w["norm_w"],
        rstate=rstate,
        st_idx=w["st_idx"],
        core=core,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        layer=0,
    )
    torch.accelerator.synchronize()

    lay = gdn.scratch_layout(key)
    torch.testing.assert_close(res_out, ref["residual"], atol=0, rtol=0)
    # x stays in the CTAs' LDS: in_proj's output checks it
    proj = _pairs(scratch, lay["proj"], s * gdn.NPROJ).view(s, gdn.NPROJ)
    if qq:  # b and a: the parts summed in part order, as the gates do
        parts = _pairs(scratch, lay["bapart"], gdn.BA_TASKS * s * 16)
        parts = parts.view(2, gdn.BA_PARTS, s, 16)
        ba = parts[:, 0]
        for q in range(1, gdn.BA_PARTS):
            ba = ba + parts[:, q]
        proj = proj.clone()
        proj[:, gdn.B0 : gdn.B0 + 16] = ba[0].to(bf).float()
        proj[:, gdn.A0 : gdn.A0 + 16] = ba[1].to(bf).float()
    assert _cos(proj, ref["proj"]) > COS_MIN
    live = w["st_idx"][:n_real].long()

    def errs_vs(r):
        return {
            "conv_state": _rel_l2(conv_state[live], r["conv_state"][live]),
            "rstate": _rel_l2(rstate[live], r["rstate"][live]),
            "core": _rel_l2(core[:n_real], r["core"][:n_real]),
        }

    # from K1's own in_proj output: the conv, gates, recurrence and norm alone
    own = errs_vs(_gdn_reference(w, n_real, proj=proj.clone()))
    assert own["conv_state"] == 0 and own["rstate"] < 1e-6, own
    assert own["core"] < 1e-4, own
    # end to end: the GEMV's summation order flips a few bf16 roundings of
    # in_proj's output, which the recurrence carries into the state; fp4, an x
    # an ulp off the reference's can also land in another e2m1 bin
    errs = errs_vs(ref)
    tol = 5e-3 if qq else 5e-4
    assert errs["conv_state"] < tol and errs["rstate"] < tol, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    # the null block is left alone
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)


@pytest.mark.parametrize(
    "s,n_real,first", [(1, 1, False), (2, 2, False), (5, 4, False), (8, 7, True)]
)
@torch.inference_mode()
def test_k1_mxfp6_matches_vllm_ops(
    monkeypatch: pytest.MonkeyPatch, s: int, n_real: int, first: bool
) -> None:
    """K1 with the MXFP6 in_proj_qkvz (``q6``, moe.fp6_pack's layout) against
    the same ops on the weight rounded to MXFP6 (mxfp6.round_e2m3), as the
    model runs it with VLLM_QWEN38_FAKE_MXFP6_PROJ. At 2 rows the GDN tasks
    fill one round, so the task placement ``qp`` also moves the owner heads'
    tasks. The build without ``qp`` must give the same outputs bit for bit."""
    from vllm.models.qwen3_5.amd.mono import gdn
    from vllm.models.qwen3_5.amd.mono.layout import CORE, HIDDEN
    from vllm.models.qwen3_5.amd.mono.moe import fp6_pack
    from vllm.models.qwen3_5.amd.mxfp6 import round_e2m3

    w = _gdn_inputs(s, n_real, first)
    w6, w6s = fp6_pack(w["w_qkvz"])
    # The reference multiplies the bf16 x rows with the MXFP6 values, which
    # bf16 holds exactly.
    w["w_qkvz"] = round_e2m3(w["w_qkvz"])
    ref = _gdn_reference(w, n_real)
    key = gdn.K1Build(tokens=s, first=first, qq=True, q6=True)
    assert key.qp

    def run() -> tuple[torch.Tensor, ...]:
        conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
        res_out = torch.empty(s, HIDDEN, dtype=bf, device="cuda")
        core = torch.full((s, CORE), float("nan"), dtype=bf, device="cuda")
        scratch = torch.zeros(gdn.scratch_bytes(key), dtype=torch.uint8, device="cuda")
        gdn.gdn_pre(
            key,
            hidden=w["hidden"],
            residual=w["residual"],
            res_out=res_out,
            ln_w=w["ln_w"],
            w_qkvz=w6,
            w_qkvzs=w6s,
            w_ba=w["w_ba"],
            conv_w=w["conv_w"],
            conv_state=conv_state,
            a_log=w["a_log"],
            dt_bias=w["dt_bias"],
            norm_w=w["norm_w"],
            rstate=rstate,
            st_idx=w["st_idx"],
            core=core,
            scratch=scratch,
            epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
            layer=0,
        )
        torch.accelerator.synchronize()
        return res_out, core, conv_state, rstate

    got = run()
    res_out, core, conv_state, rstate = got
    torch.testing.assert_close(res_out, ref["residual"], atol=0, rtol=0)
    live = w["st_idx"][:n_real].long()
    errs = {
        "conv_state": _rel_l2(conv_state[live], ref["conv_state"][live]),
        "rstate": _rel_l2(rstate[live], ref["rstate"][live]),
        "core": _rel_l2(core[:n_real], ref["core"][:n_real]),
    }
    # Only the summation order of in_proj differs from the reference, as in
    # the bf16 build.
    assert errs["conv_state"] < 5e-4 and errs["rstate"] < 5e-4, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)

    # qp only moves tasks between CTAs. An empty build cache makes the build
    # without it compile instead of returning the qp kernel.
    monkeypatch.setattr(gdn, "_BUILDS", {})
    monkeypatch.setattr(gdn.K1Build, "qp", property(lambda self: False))
    for name, a, b in zip(("res_out", "core", "conv_state", "rstate"), run(), got):
        assert torch.equal(a, b), f"{name} differs without qp"


# ---------------------------------------------------------------- K2


def _u8(g, *shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device=g.device).to(torch.uint8)


def _moe_weights(rank: int, device, fse: bool = False):
    """One layer's MoE block at TP8 as vLLM holds it after loading: bf16 router
    and shared-expert gate; the routed experts' MXFP4 through the
    ``AITER_MXFP4_MXFP4`` load path (``shuffle_weights``, ``e8m0_shuffle``).
    The shared expert is bf16, or fused (``fse``) as MXFP4 expert 512."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.qwen3_5.amd.mono.layout import HIDDEN, RI, SI, E

    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards
    ne = E + int(fse)

    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w13q, w2q = _u8(gr, ne, 2 * RI, HIDDEN // 2), _u8(gr, ne, HIDDEN, RI // 2)
    w13, w2 = rocm_aiter_ops.shuffle_weights(
        w13q.clone().view(fp4), w2q.clone().view(fp4)
    )
    w13s = _u8(gr, ne * 2 * RI, HIDDEN // 32, lo=118, hi=123)
    w2s = _u8(gr, ne * HIDDEN, RI // 32, lo=118, hi=123)
    raw = {
        "w13": w13q,
        "w2": w2q,
        "w13s": w13s.view(ne, 2 * RI, -1),
        "w2s": w2s.view(ne, HIDDEN, -1),
    }
    w13s, w2s = w13s.clone().view(e8), w2s.clone().view(e8)
    mw = {
        "raw": raw,
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "w_sg": _randn(gc, 1, HIDDEN, scale=0.02),
        "w13": w13,
        "w2": w2,
        "w13s": e8m0_shuffle(w13s).view(ne, 2 * RI, -1),
        "w2s": e8m0_shuffle(w2s).view(ne, HIDDEN, -1),
    }
    if not fse:
        mw["w_sgu"] = _randn(gr, 2 * SI, HIDDEN, scale=0.02)
        mw["w_sd"] = _randn(gr, HIDDEN, SI, scale=0.03)
    return mw


def _mxfp4_linear(gr, n: int, k: int):
    """An online-MXFP4 linear (``Mxfp4OnlineLinearMethod``) as the AITER kernel
    holds it on the ASM FP4 GEMM: random fp4x2 (n, k / 2) and e8m0 (n, k / 32),
    the weight through ``shuffle_weight(layout=(16, 16))``, the scales through
    ``AiterMxfp4LinearKernel``'s swizzle. Returns (raw q, raw scales, w, ws)."""
    from aiter.ops.shuffle import shuffle_weight

    q = _u8(gr, n, k // 2)
    sc = _u8(gr, n, k // 32, lo=118, hi=123)
    w = shuffle_weight(q.clone().view(torch.float4_e2m1fn_x2), layout=(16, 16))
    sm, sn = sc.shape
    ws = sc.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6)
    return q, sc, w, ws.contiguous().view(sm, sn)


def _mxfp4_linear_stock(x, w, ws):
    """The ASM FP4 GEMM path of ``gemm_with_dynamic_quant`` at decode rows."""
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip

    xq, xs = per_1x32_f4_quant_hip(x, shuffle=True)
    return gemm_a4w4(
        xq, w.view(xq.dtype), xs, ws.view(xs.dtype), dtype=bf, bpreshuffle=True
    )[: x.shape[0]]


_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _mxfp4_dequant(q, scales):
    """Packed e2m1 bytes (low nibble first) times their e8m0 scales -> fp32."""
    lut = torch.tensor(_FP4 + tuple(-v for v in _FP4), device=q.device)
    v = lut[torch.stack([q & 0xF, q >> 4], -1).flatten(-2).long()]
    return v * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _mxfp4_qdq(v):
    """Dynamic MXFP4 of the last dim as AITER's activation quant: 1 x 32 blocks,
    e8m0 scale ceil_pow2(amax / 6), e2m1 nearest even saturating at 6."""
    b = v.float().unflatten(-1, (-1, 32))
    bits = (b.abs().amax(-1, keepdim=True) / 6.0).view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    sc = torch.exp2(e.clamp(1, 253).float() - 127)
    a = (b / sc).abs().clamp(max=6.0)
    q = torch.where(
        a < 2,
        torch.round(a * 2) / 2,
        torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
    )
    return (torch.sign(b) * q * sc).flatten(-2)


def _route_and_shared(x, mw, ids=None):
    """The bf16 router's softmax top-10 (or the experts ``ids``) renormalized;
    the shared expert (SiluAndMul) times sigmoid(shared_expert_gate). Fused (no
    ``w_sgu``), the shared expert is pick 10, expert 512, weight sigmoid (fp32)
    of the bf16 gate logit, and the shared output is zero."""
    from vllm.models.qwen3_5.amd.mono.layout import SI, TOPK, E

    logits = F.linear(x, mw["w_gate"])
    probs = torch.softmax(logits.float(), dim=-1)
    if ids is None:
        ids = torch.topk(probs, TOPK, dim=-1, sorted=True)[1]
    wts = probs.gather(-1, ids)
    wts = wts / wts.sum(-1, keepdim=True)
    if "w_sgu" not in mw:
        sg = torch.sigmoid(F.linear(x, mw["w_sg"]).float())
        wts = torch.cat([wts, sg], dim=-1)
        ids = torch.cat([ids, torch.full_like(ids[:, :1], E)], dim=-1)
        return wts, ids, torch.zeros_like(x)
    gu = F.linear(x, mw["w_sgu"])
    h = (F.silu(gu[:, :SI].float()) * gu[:, SI:].float()).to(bf)
    shared = torch.sigmoid(F.linear(x, mw["w_sg"])) * F.linear(h, mw["w_sd"])
    return wts, ids, shared


def _rm_logits(core, residual, w_rm, rank, rstd):
    """``rm``'s router logits as K2 computes them: this rank's w_rm times its
    core rows and its slice of the residual in fp32, summed over the ranks,
    times the post-attention norm's rstd and rounded to bf16."""
    from vllm.models.qwen3_5.amd.mono import layer as k2

    piece = slice(rank * k2.RM_K, (rank + 1) * k2.RM_K)
    part = torch.cat([core, residual[:, piece]], dim=1).float() @ w_rm.float().T
    dist.all_reduce(part)
    return (part * rstd).to(bf)


def _check_route(scratch, region, x, mw, rm_logits=None):
    """The kernel's ROUTE: each token's picks a top-10 of the bf16 logits up
    to ties (the kernel's logit sums round to bf16 in another order, and equal
    keys go to the lower expert), their weights the renormalized softmax.
    With ``rm`` the kernel's logits are ``rm_logits``. They are other fp32
    sums than the bf16 GEMM's, so a logit can round to the next bf16 value.
    At the test's top logits of 4 to 8 that step is 2^-5, which moves the
    weight by about 3%, more than the tolerance. So the picks are checked
    against the router's logits, where the slack allows two such steps, and
    the weights against the softmax of ``rm_logits``.
    Returns the picks, so the numerics compare on the same experts."""
    from vllm.models.qwen3_5.amd.mono.layout import TOPK

    s = x.shape[0]
    off = region[0]
    words = scratch[off : off + s * 2 * TOPK * 8].view(torch.int32)
    words = words.view(s, 2 * TOPK, 2)[:, :, 0]
    ids, wts = words[:, :TOPK].long(), words[:, TOPK:].contiguous().view(torch.float32)
    logits = F.linear(x, mw["w_gate"]).float()
    tenth = torch.topk(logits, TOPK, dim=-1)[0][:, -1:]
    slack = 2 * 2.0**-7 * tenth.abs()
    assert (logits.gather(-1, ids) >= tenth - slack).all(), "picks not a top-10"
    assert all(len(set(r.tolist())) == TOPK for r in ids), "repeated pick"
    if rm_logits is None:
        want = _route_and_shared(x, mw, ids)[0][:, :TOPK]
    else:
        want = torch.softmax(rm_logits.float(), dim=-1).gather(-1, ids)
        want = want / want.sum(-1, keepdim=True)
    torch.testing.assert_close(wts, want, rtol=1e-2, atol=1e-3)
    return ids


def _moe_emulation(x, mw, ids=None):
    """The MoE block with the routed experts as the stock decode kernel computes
    them (``fmoe_bf16_pertokenMXfp4_g1u1_flat``: x and silu(g) u through dynamic
    MXFP4, fp32 accumulation, the route weight after w2), in plain torch; then
    the all-reduce. Deterministic, unlike the kernel."""
    raw = mw["raw"]
    ri = raw["w2"].shape[-1] * 2  # unpadded
    wts, ids, shared = _route_and_shared(x, mw, ids)
    xq = _mxfp4_qdq(x)
    routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            gu = _mxfp4_dequant(raw["w13"][e], raw["w13s"][e]) @ xq[t]
            h = _mxfp4_qdq(F.silu(gu[:ri]) * gu[ri:])
            routed[t] += wts[t, k] * (_mxfp4_dequant(raw["w2"][e], raw["w2s"][e]) @ h)
    return _all_reduce_fp32((routed + shared.float()).to(bf))


def _moe_reference(x, mw, ids=None):
    """Qwen3NextSparseMoeBlock's decode path on ``x`` through vLLM's ops: the
    router and shared expert of ``_route_and_shared``, aiter's ``fused_moe`` for
    the MXFP4 experts, the all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    wts, ids, shared = _route_and_shared(x, mw, ids)
    routed = fused_moe(
        x,
        mw["w13"],
        mw["w2"],
        wts.float(),
        ids.to(torch.int32),
        None,
        ActivationType.Silu,
        QuantType.per_1x32,
        False,
        mw["w13s"],
        mw["w2s"],
        None,
        None,
        dtype=bf,
    )
    return _all_reduce_fp32(routed + shared)


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen3_5.amd.mono import layer as k2
    from vllm.models.qwen3_5.amd.mono.layout import CORE, HIDDEN, MAX_TOKENS
    from vllm.models.qwen3_5.amd.mono.moe import fp6_pack
    from vllm.models.qwen3_5.amd.mxfp6 import round_e2m3

    tp = get_tp_group()
    peers = PeerBuffer(k2.peer_bytes(MAX_TOKENS), tp.cpu_group, rank, tp_size, device)
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    step = 0
    for flags in K2_MODES:
        fse, oq = flags.get("fse", False), flags.get("oq", False)
        o6, rm = flags.get("o6", False), flags.get("rm", False)
        mw = _moe_weights(rank, device, fse=fse)
        gc = torch.Generator(device=device).manual_seed(11)
        gr = torch.Generator(device=device).manual_seed(200 + rank)
        ln_w = _randn(gc, HIDDEN, scale=0.1)
        if oq:
            oq_raw, os_raw, w_o, w_os = _mxfp4_linear(gr, HIDDEN, CORE)
            w_o_ref = _mxfp4_dequant(oq_raw, os_raw).to(bf)
        elif o6:
            # The MXFP6 values, which bf16 holds exactly, and their packing.
            w_o_ref = round_e2m3(_randn(gr, HIDDEN, CORE, scale=0.02))
            w_o, w_os = fp6_pack(w_o_ref)
        else:
            w_o, w_os = _randn(gr, HIDDEN, CORE, scale=0.02), None
        # rm_pack folds the router into the o_proj values that K2 multiplies.
        w_rm = None
        if rm:
            w_rm = k2.rm_pack(mw["w_gate"], ln_w, w_o_ref if o6 else w_o, rank)
        mode = str(flags)

        # both epoch parities, a layer slot each; 1 row (the picks as the union),
        # 2 and 3 (a union slot a pick), 4 and 8 (the fused shared expert's
        # tasks before the route). mono_decode takes rm below 6 rows only.
        for s in (1, 2, 3, 4, 8, 8):
            if rm and s >= 6:
                continue
            step += 1
            epoch.add_(1)
            key = k2.K2Build(tokens=s, **flags)
            core = _randn(gr, s, CORE)
            gate = _randn(gr, s, CORE, scale=2.0) if key.og else None
            residual = _randn(gc, s, HIDDEN, scale=2.0)
            scratch = torch.zeros(
                k2.scratch_bytes(key), dtype=torch.uint8, device=device
            )
            out = torch.empty(s, HIDDEN, dtype=bf, device=device)
            res_out = torch.empty_like(out)
            k2.layer_post(
                key,
                core=core,
                gate=gate,
                residual=residual,
                w_o=w_o,
                w_os=w_os,
                ln_w=ln_w,
                out=out,
                res_out=res_out,
                scratch=scratch,
                peers=peers.addresses,
                rank=rank,
                epoch=epoch,
                layer=step,
                w_rm=w_rm,
                **{k: v for k, v in mw.items() if k != "raw"},
            )
            torch.accelerator.synchronize()

            if gate is not None:
                core = (core.float() * torch.sigmoid(gate.float())).to(bf)
            if oq:
                part = F.linear(_mxfp4_qdq(core).to(bf), w_o_ref)
                part_stock = _mxfp4_linear_stock(core, w_o, w_os)
                assert _cos(part_stock, part) > STOCK_COS_MIN, (mode, s)
            else:
                part = F.linear(core, w_o_ref if o6 else w_o)
            res_ref = _all_reduce_fp32(part) + residual
            # the kernel sums the ranks in rank order, NCCL in its own order
            assert _rel_l2(res_out, res_ref) < 1e-2, (mode, s)
            v = res_ref.float()
            x_ref = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
            x_ref = (x_ref * (1.0 + ln_w.float())).to(bf)
            lay = k2.scratch_layout(key)["xrow"][0]
            x = scratch[lay : lay + s * HIDDEN * 2].view(bf).view(s, HIDDEN)
            # the AR sums: x is the consumers' (stage_norm's arithmetic)
            v = x.float() + residual.float()
            rstd = torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
            x = (v * rstd * (1.0 + ln_w.float())).to(bf)
            assert _cos(x, x_ref) > COS_MIN, (mode, s)
            rm_logits = _rm_logits(core, residual, w_rm, rank, rstd) if key.rm else None
            region = k2.scratch_layout(key)["route"]
            ids = _check_route(scratch, region, x, mw, rm_logits)
            emu = _cos(out, _moe_emulation(x.clone(), mw, ids))
            assert emu > COS_MIN, (mode, s, emu)
            # fused_moe's flat kernel varies run to run
            ref = _moe_reference(x.clone(), mw, ids)
            assert _cos(out, ref) > STOCK_COS_MIN, (mode, s)
            outs = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(outs, out)
            assert all(torch.equal(outs[0], o) for o in outs), (mode, "ranks disagree")
        del mw
        torch.accelerator.empty_cache()
    peers.close()


# the bf16 shared expert and o_proj; the stock weekly flags' fused shared
# expert, with the bf16 o_proj (attention, with and without its output gate)
# and the MXFP4 out_proj; with VLLM_QWEN38_MONO_FP6, the MXFP6 out_proj and
# o_proj (o6), with the router logits through them (rm)
K2_MODES: tuple[dict[str, bool], ...] = (
    {},
    {"fse": True},
    {"fse": True, "og": True},
    {"fse": True, "oq": True},
    {"fse": True, "o6": True},
    {"fse": True, "o6": True, "rm": True},
    {"fse": True, "og": True, "o6": True, "rm": True},
)


@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


REPLAY_WIDTHS = (8, 1, 3, 8, 2, 5)  # consecutive decode steps of mixed widths
REPLAYS = 300


@ray.remote(num_gpus=1, max_calls=1)
def _replay_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen3_5.amd.mono import gdn
    from vllm.models.qwen3_5.amd.mono import layer as k2
    from vllm.models.qwen3_5.amd.mono.layout import CORE, HIDDEN, MAX_TOKENS, QKVZ
    from vllm.models.qwen3_5.amd.mono.moe import fp6_pack
    from vllm.models.qwen3_5.amd.mxfp6 import round_e2m3

    tp = get_tp_group()
    peers = PeerBuffer(k2.peer_bytes(MAX_TOKENS), tp.cpu_group, rank, tp_size, device)
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    gr = torch.Generator(device=device).manual_seed(300 + rank)
    ln_w = _randn(gr, HIDDEN, scale=0.1)
    w = _gdn_inputs(MAX_TOKENS, MAX_TOKENS, False)
    # three GDN layers, each with its own state, as MonoDecode.forward runs
    # them: the bf16 weights, then the stock weekly flags' (MXFP4
    # in_proj_qkvz and out_proj, the fused shared expert), then
    # VLLM_QWEN38_MONO_FP6's (MXFP6 in_proj_qkvz and out_proj, the fused
    # shared expert, the router logits through out_proj below 6 rows)
    layers = []
    for kind in ("bf16", "fp4", "fp6"):
        fse = kind != "bf16"
        lw = {"mw": _moe_weights(rank, device, fse=fse)}
        lw["mw"].pop("raw")
        if kind == "fp4":
            _, _, lw["w_qkvz"], lw["w_qkvzs"] = _mxfp4_linear(gr, QKVZ, HIDDEN)
            _, _, lw["w_o"], lw["w_os"] = _mxfp4_linear(gr, HIDDEN, CORE)
        elif kind == "fp6":
            lw["w_qkvz"], lw["w_qkvzs"] = fp6_pack(w["w_qkvz"])
            w_o = round_e2m3(_randn(gr, HIDDEN, CORE, scale=0.02))
            lw["w_o"], lw["w_os"] = fp6_pack(w_o)
            lw["w_rm"] = k2.rm_pack(lw["mw"]["w_gate"], ln_w, w_o, rank)
        else:
            lw["w_qkvz"], lw["w_qkvzs"] = w["w_qkvz"], None
            lw["w_o"], lw["w_os"] = _randn(gr, HIDDEN, CORE, scale=0.02), None
        lw["k1"] = {
            "bf16": {},
            "fp4": {"qq": True},
            "fp6": {"qq": True, "q6": True},
        }[kind]
        lw["k2"] = {
            "bf16": lambda s: {},
            "fp4": lambda s: {"fse": True, "oq": True},
            "fp6": lambda s: {"fse": True, "o6": True, "rm": s < 6},
        }[kind]
        layers.append(lw)
    states0 = [(w["conv_state"].clone(), w["rstate"].clone()) for _ in layers]
    states = [(c.clone(), r.clone()) for c, r in states0]
    # The qq layout ends with the in_proj_ba parts, so it is the larger one.
    k1_scr = {
        s: torch.zeros(
            gdn.scratch_bytes(gdn.K1Build(tokens=s, qq=True)),
            dtype=torch.uint8,
            device=device,
        )
        for s in set(REPLAY_WIDTHS)
    }
    k2_scr = {
        s: torch.zeros(
            max(k2.scratch_bytes(k2.K2Build(tokens=s, fse=f)) for f in (False, True)),
            dtype=torch.uint8,
            device=device,
        )
        for s in set(REPLAY_WIDTHS)
    }
    outs = [torch.empty(s, HIDDEN, dtype=bf, device=device) for s in REPLAY_WIDTHS]
    # the last row of odd widths is a cudagraph pad row
    idxs = {}
    for s in set(REPLAY_WIDTHS):
        idxs[s] = w["st_idx"][:s].clone()
        if s % 2:
            idxs[s][-1] = 0

    def steps() -> None:
        for (c, r), (c0, r0) in zip(states, states0):
            c.copy_(c0)
            r.copy_(r0)
        for i, s in enumerate(REPLAY_WIDTHS):
            epoch.add_(1)
            h, res = w["hidden"][:s].contiguous(), w["residual"][:s].contiguous()
            idx = idxs[s]
            for li, ((conv, rstate), lw) in enumerate(zip(states, layers)):
                core = torch.empty(s, CORE, dtype=bf, device=device)
                res_attn = torch.empty_like(h)
                gdn.gdn_pre(
                    gdn.K1Build(tokens=s, **lw["k1"]),
                    hidden=h,
                    residual=res,
                    res_out=res_attn,
                    ln_w=w["ln_w"],
                    w_qkvz=lw["w_qkvz"],
                    w_qkvzs=lw["w_qkvzs"],
                    w_ba=w["w_ba"],
                    conv_w=w["conv_w"],
                    conv_state=conv,
                    a_log=w["a_log"],
                    dt_bias=w["dt_bias"],
                    norm_w=w["norm_w"],
                    rstate=rstate,
                    st_idx=idx,
                    core=core,
                    scratch=k1_scr[s],
                    epoch=epoch,
                    layer=li,
                )
                h = outs[i] if li == len(states) - 1 else torch.empty_like(h)
                res = torch.empty_like(h)
                key = k2.K2Build(tokens=s, **lw["k2"](s))
                k2.layer_post(
                    key,
                    core=core,
                    residual=res_attn,
                    w_o=lw["w_o"],
                    w_os=lw["w_os"],
                    ln_w=ln_w,
                    out=h,
                    res_out=res,
                    scratch=k2_scr[s],
                    peers=peers.addresses,
                    rank=rank,
                    epoch=epoch,
                    layer=k2.TAG_SLOTS // 2 + li,
                    w_rm=lw["w_rm"] if key.rm else None,
                    **lw["mw"],
                )

    def snapshot() -> list[torch.Tensor]:
        return [o.clone() for o in outs] + [t.clone() for st in states for t in st]

    steps()  # eager: builds every width before the capture
    torch.accelerator.synchronize()
    eager = snapshot()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        steps()
    graph.replay()
    torch.accelerator.synchronize()
    first = snapshot()
    assert all(torch.equal(a, b) for a, b in zip(eager, first)), "graph != eager"
    for n in range(REPLAYS - 1):
        graph.replay()
        torch.accelerator.synchronize()
        bad = [
            i
            for i, (a, b) in enumerate(zip(snapshot(), first))
            if not torch.equal(a, b)
        ]
        assert not bad, f"replay {n + 2}: tensors {bad} differ"
    for o in outs:
        gathered = [torch.empty_like(o) for _ in range(tp_size)]
        dist.all_gather(gathered, o)
        assert all(torch.equal(gathered[0], g) for g in gathered), "ranks disagree"
    peers.close()


@multi_gpu_test(num_gpus=8)
def test_mixed_width_graph_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """K1 and K2 of three layers at six consecutive decode steps of widths 1-8
    (pad rows included), captured in one HIP graph as a FULL decode graph holds
    them and replayed 300 times: bit-identical to the eager run every time, on
    every rank. Steps of different widths share the peer buffer and the epoch;
    a scratch race between widths shows up as a mismatch. The second layer
    runs the MXFP4 in_proj_qkvz / out_proj and fused-shared-expert variants,
    the third the MXFP6 in_proj_qkvz / out_proj with the router logits
    through out_proj below 6 rows."""
    multi_process_parallel(monkeypatch, 8, 1, _replay_worker)
