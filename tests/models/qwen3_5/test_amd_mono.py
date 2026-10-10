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


def _gdn_reference(w, n: int, proj: torch.Tensor | None = None):
    """Qwen3_5DecoderLayer's input norm and the GDN decode path on ROCm
    (``forward_hip`` -> ``_forward_core_decode_aiter`` -> ``_output_projection``
    without out_proj) on the first ``n`` rows; the rest are the cudagraph pad.
    ``proj``: in_proj's output to start from instead of computing it."""
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token,
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule,
    )

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
    from vllm.models.qwen3_5.amd.mono.layout import (
        CONV,
        CORE,
        HD,
        HIDDEN,
        KEY,
        NK,
        NV,
        VAL,
    )

    s = w["hidden"].size(0)
    with set_current_vllm_config(VllmConfig()):
        ln = GemmaRMSNorm(HIDDEN, eps=1e-6).cuda()
        norm = RMSNormGated(
            HD, eps=1e-6, norm_before_gate=True, activation="silu", device="cuda"
        )
    ln.weight.data = w["ln_w"].clone()
    norm.weight.data = w["norm_w"].clone()
    if w["residual"] is None:
        x, res = ln.forward_native(w["hidden"]), w["hidden"].clone()
    else:
        x, res = ln.forward_native(w["hidden"], w["residual"].clone())
    if proj is None:
        qkvz = F.linear(x, w["w_qkvz"])
        ba = F.linear(x, w["w_ba"])
    else:
        qkvz, ba = proj.split([w["w_qkvz"].size(0), w["w_ba"].size(0)], dim=-1)
        qkvz, ba = qkvz.contiguous(), ba.contiguous()
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    z = torch.empty(s, NV, HD, dtype=bf, device="cuda")
    core = torch.empty(s, NV, HD, dtype=bf, device="cuda")
    mixed, b, a = fused_reshape_causal_conv1d_update_single_token(
        qkvz,
        n,
        NK,
        NV,
        HD,
        HD,
        ba,
        z,
        core,
        conv_state,
        w["conv_w"].view(CONV, 4),
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
        key_dim=KEY,
        value_dim=VAL,
        head_k_dim=HD,
        head_v_dim=HD,
        initial_state=rstate,
        inplace_final_state=True,
        cu_seqlens=torch.arange(n + 1, dtype=torch.int32, device="cuda"),
        ssm_state_indices=w["st_idx"][:n],
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core.reshape(-1),
    )
    return {
        "x": x,
        "residual": res,
        "proj": torch.cat([qkvz, ba], dim=-1),
        "conv_state": conv_state,
        "rstate": rstate,
        "core": norm.forward_native(core, z).reshape(s, CORE),
    }


@pytest.mark.parametrize("s", [1, 3, 8])
@torch.inference_mode()
def test_attention_front_matches_vllm_ops(s: int) -> None:
    """A full-attention layer's front, which the mono forward runs between K2
    launches: the compiled Gemma add + norm against GemmaRMSNorm.forward_native,
    and the fused split / q, k norm / RoPE / sigmoid gate against the model's
    eager ops (``_project_qkv_gate`` off CUDA). TP8 shapes: 8 query heads and
    one KV head of 256 a rank, 64 rotary dims."""
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
    got = md._attention_core(attn, qkv, positions)

    q_gate, k, v = qkv.split([hq * hd * 2, hkv * hd, hkv * hd], dim=-1)
    q, gate = q_gate.view(s, hq, 2 * hd).chunk(2, dim=-1)
    q = q_norm.forward_native(q.reshape(s, hq, hd)).reshape(s, hq * hd)
    k = k_norm.forward_native(k.reshape(s, hkv, hd)).reshape(s, hkv * hd)
    q, k = rope.forward_native(positions, q, k)
    want = attn.attn(q, k, v) * torch.sigmoid(gate.reshape(s, hq * hd))
    assert got.is_contiguous() and got.shape == want.shape
    assert _cos(got, want) > COS_MIN
    torch.testing.assert_close(got, want, atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "s,n_real,first", [(1, 1, False), (5, 4, False), (8, 8, False), (8, 7, True)]
)
def test_k1_matches_vllm_ops(s: int, n_real: int, first: bool) -> None:
    """K1 against GemmaRMSNorm -> in_proj -> AITER's conv update + fused gated
    delta rule -> RMSNormGated on one decode step, the states it leaves behind,
    and the pad rows (null block: no state written, zero core). The conv state
    is in the (slots, 3, dim) layout AITER's conv update takes."""
    from vllm.models.qwen3_5.amd.mono import gdn
    from vllm.models.qwen3_5.amd.mono.layout import CORE, HIDDEN

    w = _gdn_inputs(s, n_real, first)
    ref = _gdn_reference(w, n_real)
    key = gdn.K1Build(tokens=s, first=first)
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
    x = scratch[lay["xrow"][0] : lay["xrow"][0] + s * HIDDEN * 2].view(bf)
    assert _cos(x, ref["x"]) > COS_MIN
    proj = _pairs(scratch, lay["proj"], s * gdn.NPROJ).view(s, gdn.NPROJ)
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
    # in_proj's output, which the recurrence carries into the state
    errs = errs_vs(ref)
    assert errs["conv_state"] < 5e-4 and errs["rstate"] < 5e-4, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    # the null block is left alone
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)


# ---------------------------------------------------------------- K2


def _moe_weights(rank: int, device):
    """One layer's MoE block at TP8 as vLLM holds it after loading: bf16 router,
    shared expert and its gate; the routed experts' MXFP4 through the
    ``AITER_MXFP4_MXFP4`` load path (``shuffle_weights``, ``e8m0_shuffle``)."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.qwen3_5.amd.mono.layout import HIDDEN, RI, SI, E

    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device).to(torch.uint8)

    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w13q, w2q = u8(gr, E, 2 * RI, HIDDEN // 2), u8(gr, E, HIDDEN, RI // 2)
    w13, w2 = rocm_aiter_ops.shuffle_weights(
        w13q.clone().view(fp4), w2q.clone().view(fp4)
    )
    w13s = u8(gr, E * 2 * RI, HIDDEN // 32, lo=118, hi=123)
    w2s = u8(gr, E * HIDDEN, RI // 32, lo=118, hi=123)
    raw = {
        "w13": w13q,
        "w2": w2q,
        "w13s": w13s.view(E, 2 * RI, -1),
        "w2s": w2s.view(E, HIDDEN, -1),
    }
    w13s, w2s = w13s.clone().view(e8), w2s.clone().view(e8)
    return {
        "raw": raw,
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "w_sg": _randn(gc, 1, HIDDEN, scale=0.02),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
        "w13": w13,
        "w2": w2,
        "w13s": e8m0_shuffle(w13s).view(E, 2 * RI, -1),
        "w2s": e8m0_shuffle(w2s).view(E, HIDDEN, -1),
    }


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


def _route_and_shared(x, mw):
    """The bf16 router's softmax top-10 renormalized; the shared expert
    (SiluAndMul) times sigmoid(shared_expert_gate)."""
    from vllm.models.qwen3_5.amd.mono.layout import SI, TOPK

    logits = F.linear(x, mw["w_gate"])
    probs = torch.softmax(logits.float(), dim=-1)
    wts, ids = torch.topk(probs, TOPK, dim=-1, sorted=True)
    wts = wts / wts.sum(-1, keepdim=True)
    gu = F.linear(x, mw["w_sgu"])
    h = (F.silu(gu[:, :SI].float()) * gu[:, SI:].float()).to(bf)
    shared = torch.sigmoid(F.linear(x, mw["w_sg"])) * F.linear(h, mw["w_sd"])
    return wts, ids, shared


def _moe_emulation(x, mw):
    """The MoE block with the routed experts as the stock decode kernel computes
    them (``fmoe_bf16_pertokenMXfp4_g1u1_flat``: x and silu(g) u through dynamic
    MXFP4, fp32 accumulation, the route weight after w2), in plain torch; then
    the all-reduce. Deterministic, unlike the kernel."""
    from vllm.models.qwen3_5.amd.mono.layout import RI

    raw = mw["raw"]
    wts, ids, shared = _route_and_shared(x, mw)
    xq = _mxfp4_qdq(x)
    routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            gu = _mxfp4_dequant(raw["w13"][e], raw["w13s"][e]) @ xq[t]
            h = _mxfp4_qdq(F.silu(gu[:RI]) * gu[RI:])
            routed[t] += wts[t, k] * (_mxfp4_dequant(raw["w2"][e], raw["w2s"][e]) @ h)
    return _all_reduce_fp32((routed + shared.float()).to(bf))


def _moe_reference(x, mw):
    """Qwen3NextSparseMoeBlock's decode path on ``x`` through vLLM's ops: the
    router and shared expert of ``_route_and_shared``, aiter's ``fused_moe`` for
    the MXFP4 experts, the all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    wts, ids, shared = _route_and_shared(x, mw)
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

    tp = get_tp_group()
    peers = PeerBuffer(k2.peer_bytes(MAX_TOKENS), tp.cpu_group, rank, tp_size, device)
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gc = torch.Generator(device=device).manual_seed(11)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    w_o = _randn(gr, HIDDEN, CORE, scale=0.02)
    ln_w = _randn(gc, HIDDEN, scale=0.1)

    # both epoch parities, a layer slot each; every decode width's build
    for step, s in enumerate((1, 3, 8, 8)):
        epoch.add_(1)
        key = k2.K2Build(tokens=s)
        core = _randn(gr, s, CORE)
        residual = _randn(gc, s, HIDDEN, scale=2.0)
        scratch = torch.zeros(k2.scratch_bytes(key), dtype=torch.uint8, device=device)
        out = torch.empty(s, HIDDEN, dtype=bf, device=device)
        res_out = torch.empty_like(out)
        k2.layer_post(
            key,
            core=core,
            residual=residual,
            w_o=w_o,
            ln_w=ln_w,
            out=out,
            res_out=res_out,
            scratch=scratch,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=step,
            **{k: v for k, v in mw.items() if k != "raw"},
        )
        torch.accelerator.synchronize()

        res_ref = _all_reduce_fp32(F.linear(core, w_o)) + residual
        # the kernel sums the ranks in rank order, NCCL in its own order
        assert _rel_l2(res_out, res_ref) < 1e-2, s
        v = res_ref.float()
        x_ref = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_ref = (x_ref * (1.0 + ln_w.float())).to(bf)
        lay = k2.scratch_layout(key)["xrow"][0]
        x = scratch[lay : lay + s * HIDDEN * 2].view(bf).view(s, HIDDEN)
        assert _cos(x, x_ref) > COS_MIN, s
        assert _cos(out, _moe_emulation(x.clone(), mw)) > COS_MIN, s
        # fused_moe's flat kernel varies run to run
        assert _cos(out, _moe_reference(x.clone(), mw)) > STOCK_COS_MIN, s
        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"
    peers.close()


@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)
