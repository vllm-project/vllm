# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Qwen3.8-Flash-Next mono decode kernels on MI355X against the
ops vLLM runs for the same decode step, random weights in the checkpoint's
formats at TP2 shapes.

- The HC front (one GPU): ``GatedResidual.mix`` (grouped Gemma norm, the
  down + inject GEMM, hc_silu, the up GEMM, the gate-mix) on a materialized
  state, from K2's sums of squares or its own.
- K1 (one GPU): the front -> in_proj_qkvz (online MXFP4) / in_proj_ba ->
  AITER's conv update and fused gated delta rule -> RMSNormGated, the states
  it leaves behind, pad rows.
- K2 (two GPUs): out_proj / o_proj + all-reduce + the attention HC combine,
  the MLP HC's mix, the sparse MoE block (softmax top-10, the MXFP4 experts
  with the intermediate padded to 128, the fused shared expert) + all-reduce +
  the MLP HC combine, and the sums of squares it hands the next front.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.models.qwen3_5.test_amd_mono import (
    COS_MIN,
    STOCK_COS_MIN,
    _all_reduce_fp32,
    _check_route,
    _cos,
    _gdn_core_ref,
    _moe_emulation,
    _moe_reference,
    _mxfp4_dequant,
    _mxfp4_linear,
    _mxfp4_linear_stock,
    _mxfp4_qdq,
    _on_gfx950,
    _pairs,
    _randn,
    _rel_l2,
    _route_and_shared,
    _u8,
)
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

bf = torch.bfloat16
TP = 2

pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the Flash-Next mono kernels need gfx950 (MI355X)"
)


# ---------------------------------------------------------------- HC
def _hc_weights(g):
    from vllm.models.qwen4_exp.amd.mono.layout import DOWN_ROWS, HH, LORA

    return {
        "hc_w": _randn(g, HH, scale=0.1),
        "w_down": _randn(g, DOWN_ROWS, HH, scale=0.01),
        "w_up": _randn(g, HH, LORA, scale=0.05),
    }


def _msq(state):
    """K2's ``msq_out`` of a state: [20, S, 4] fp32 sums of squares by 128
    columns of each stream."""
    from vllm.models.qwen4_exp.amd.mono import hc
    from vllm.models.qwen4_exp.amd.mono.layout import HC

    s = state.size(0)
    sq = state.float().view(s, HC, hc.NRED, hc.COLS).pow(2).sum(-1)
    return sq.permute(2, 0, 1).contiguous()


def _mix_ref(state, hw):
    """``GatedResidual.mix`` on ``state`` -> (block input, injection, xn)."""
    from vllm.models.qwen4_exp.amd.mono.layout import HC, LORA
    from vllm.models.qwen4_exp.amd.ops.hc import (
        grouped_gemma_rmsnorm,
        hc_gate_mix,
        hc_silu,
    )

    xn = grouped_gemma_rmsnorm(state, hw["hc_w"], 1e-6, HC)
    d = F.linear(xn, hw["w_down"])
    lora = hc_silu(d[:, :LORA].contiguous(), HC)
    gate = F.linear(lora, hw["w_up"])
    return hc_gate_mix(xn, gate, HC), d[:, LORA : LORA + HC].contiguous(), xn


@pytest.mark.parametrize("s", [1, 3, 8])
@pytest.mark.parametrize("self_msq", [False, True])
@torch.inference_mode()
def test_hc_front_matches_vllm_ops(s: int, self_msq: bool) -> None:
    """The QSA layers' HC front (``hc_front``) against ``GatedResidual.mix``:
    the block input and the injection logits."""
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN

    g = torch.Generator(device="cuda").manual_seed(s)
    state = _randn(g, s, HH, scale=2.0)
    hw = _hc_weights(g)
    want_x, want_inj, _ = _mix_ref(state, hw)
    key = k1.K1Build(tokens=s, tp=TP, self_msq=self_msq, front_only=True)
    x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    k1.hc_front(
        key,
        state=state,
        msq_in=None if self_msq else _msq(state),
        **hw,
        inj_out=inj,
        bin_out=x,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        layer=0,
    )
    torch.accelerator.synchronize()
    assert _cos(inj, want_inj) > COS_MIN
    torch.testing.assert_close(inj, want_inj, atol=2e-2, rtol=2e-2)
    assert _cos(x, want_x) > COS_MIN
    torch.testing.assert_close(x, want_x, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------- K1
def _gdn_inputs(s: int, n_real: int, slots: int = 16):
    from vllm.models.qwen4_exp.amd.mono.layout import HD, HH, HIDDEN, Layout

    lay = Layout(TP)
    g = torch.Generator(device="cuda").manual_seed(0)
    w = {
        "state": _randn(g, s, HH, scale=2.0),
        **_hc_weights(g),
        "w_ba": _randn(g, lay.BA, HIDDEN, scale=0.05),
        "conv_w": _randn(g, lay.CONV, 1, 4, scale=0.4),
        "a_log": _randn(g, lay.NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, lay.NV, scale=0.5),
        "norm_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    q, sc, w["w_qkvz"], w["w_qkvzs"] = _mxfp4_linear(g, lay.QKVZ, HIDDEN)
    w["w_qkvz_ref"] = _mxfp4_dequant(q, sc).to(bf)
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    n = lay.NV * HD * HD
    pad = _randn(g, slots, n + 8192, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, :n].view(slots, lay.NV, HD, HD)
    conv = _randn(g, slots, lay.CONV * 3 + 1024)[:, : lay.CONV * 3]
    w["conv_state"] = conv.view(slots, 3, lay.CONV).transpose(-1, -2)
    # live rows on slots 3 ..; pad rows on the null block 0
    idx = torch.zeros(s, dtype=torch.int32, device="cuda")
    idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device="cuda")
    w["st_idx"] = idx
    return w


def _gdn_reference(w, n: int, qkvz=None, ba=None):
    """The attention HC's mix, then the GDN decode path on ROCm (MXFP4
    in_proj_qkvz, AITER's conv update and fused gated delta rule,
    RMSNormGated) on the first ``n`` rows; the rest are the cudagraph pad.
    ``qkvz`` / ``ba``: in_proj's outputs to start from."""
    from vllm.models.qwen4_exp.amd.mono.layout import Layout

    lay = Layout(TP)
    x, inj, _ = _mix_ref(w["state"], w)
    if qkvz is None:
        qkvz = F.linear(_mxfp4_qdq(x).to(bf), w["w_qkvz_ref"])
        ba = F.linear(x, w["w_ba"])
    conv_state, rstate, core = _gdn_core_ref(w, n, qkvz, ba, lay.NK, lay.NV, "sigmoid")
    return {
        "x": x,
        "inj": inj,
        "qkvz": qkvz,
        "ba": ba,
        "conv_state": conv_state,
        "rstate": rstate,
        "core": core,
    }


@pytest.mark.parametrize(
    "s,n_real,self_msq", [(1, 1, False), (5, 4, False), (8, 8, False), (8, 7, True)]
)
@torch.inference_mode()
def test_k1_matches_vllm_ops(s: int, n_real: int, self_msq: bool) -> None:
    """K1 against the HC mix -> in_proj -> AITER's conv update + fused gated
    delta rule -> RMSNormGated on one decode step, the states it leaves behind,
    and the pad rows (null block: no state written, zero core)."""
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HIDDEN, ROWS, Layout

    lay = Layout(TP)
    w = _gdn_inputs(s, n_real)
    ref = _gdn_reference(w, n_real)
    stock = _mxfp4_linear_stock(ref["x"], w["w_qkvz"], w["w_qkvzs"])
    assert _cos(stock, ref["qkvz"]) > STOCK_COS_MIN
    key = k1.K1Build(tokens=s, tp=TP, self_msq=self_msq)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.full((s, lay.CORE), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    k1.gdn_pre(
        key,
        state=w["state"],
        msq_in=None if self_msq else _msq(w["state"]),
        hc_w=w["hc_w"],
        w_down=w["w_down"],
        w_up=w["w_up"],
        inj_out=inj,
        w_qkvz=w["w_qkvz"],
        w_qkvzs=w["w_qkvzs"],
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

    scr = k1.scratch_layout(key)
    off, n = scr["bin"]
    x = scratch[off : off + n].view(bf).view(s, HIDDEN)
    assert _cos(x, ref["x"]) > COS_MIN
    torch.testing.assert_close(inj, ref["inj"], atol=2e-2, rtol=2e-2)
    qkvz = _pairs(scratch, scr["proj"], s * lay.QKVZ).view(s, lay.QKVZ)
    # b and a: the K halves summed in order, as the gates do
    parts = _pairs(scratch, scr["bapart"], k1.ba_tasks(lay) * s * ROWS)
    parts = parts.view(lay.BA // ROWS, k1.BA_PARTS, s, ROWS)
    ba = parts[:, 0]
    for q in range(1, k1.BA_PARTS):
        ba = ba + parts[:, q]
    ba = ba.permute(1, 0, 2).reshape(s, lay.BA).to(bf)
    # in_proj on the kernel's own block input
    want_qkvz = F.linear(_mxfp4_qdq(x).to(bf), w["w_qkvz_ref"])
    assert _cos(qkvz, want_qkvz) > COS_MIN
    assert _cos(ba, F.linear(x, w["w_ba"])) > COS_MIN
    live = w["st_idx"][:n_real].long()

    def errs_vs(r):
        return {
            "conv_state": _rel_l2(conv_state[live], r["conv_state"][live]),
            "rstate": _rel_l2(rstate[live], r["rstate"][live]),
            "core": _rel_l2(core[:n_real], r["core"][:n_real]),
        }

    # from K1's own in_proj output: the conv, gates, recurrence and norm alone
    own = errs_vs(_gdn_reference(w, n_real, qkvz=qkvz.to(bf), ba=ba))
    assert own["conv_state"] == 0 and own["rstate"] < 1e-6, own
    assert own["core"] < 1e-4, own
    # end to end: the HC GEMVs' and in_proj's summation order flips a few bf16
    # roundings, an x an ulp off the reference's can land in another e2m1 bin
    errs = errs_vs(ref)
    assert errs["conv_state"] < 5e-3 and errs["rstate"] < 5e-3, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    # the null block is left alone
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)


# ---------------------------------------------------------------- K2
def _pad_halves(x, ri, rip, fill):
    """[ne, 2 ri, ...] [gate; up] rows -> [ne, 2 rip, ...], each half padded."""
    ne = x.shape[0]
    out = torch.full((ne, 2 * rip, *x.shape[2:]), fill, dtype=x.dtype, device=x.device)
    out[:, :ri] = x[:, :ri]
    out[:, rip : rip + ri] = x[:, ri:]
    return out


def _moe_weights(rank: int, device):
    """One layer's MoE block at TP2 as vLLM holds it after loading the fused
    shared expert as expert 512: bf16 router and shared-expert gate; the
    experts' MXFP4 with the intermediate zero-padded from RI to RIP (the pad
    scales e8m0 1.0), through ``shuffle_weights`` and ``e8m0_shuffle``."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN, E, Layout

    lay = Layout(TP)
    ri, rip, ne = lay.RI, lay.RIP, E + 1
    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards
    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w13q, w2q = _u8(gr, ne, 2 * ri, HIDDEN // 2), _u8(gr, ne, HIDDEN, ri // 2)
    w13s = _u8(gr, ne, 2 * ri, HIDDEN // 32, lo=118, hi=123)
    w2s = _u8(gr, ne, HIDDEN, ri // 32, lo=118, hi=123)
    raw = {"w13": w13q, "w2": w2q, "w13s": w13s, "w2s": w2s}
    w13p = _pad_halves(w13q, ri, rip, 0)
    w13sp = _pad_halves(w13s, ri, rip, 127)
    w2p = torch.zeros(ne, HIDDEN, rip // 2, dtype=torch.uint8, device=device)
    w2p[..., : ri // 2] = w2q
    w2sp = torch.full((ne, HIDDEN, rip // 32), 127, dtype=torch.uint8, device=device)
    w2sp[..., : ri // 32] = w2s
    w13, w2 = rocm_aiter_ops.shuffle_weights(w13p.view(fp4), w2p.view(fp4))
    return {
        "raw": raw,
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "w_sg": _randn(gc, 1, HIDDEN, scale=0.02),
        "w13": w13,
        "w2": w2,
        "w13s": e8m0_shuffle(w13sp.view(ne * 2 * rip, -1).view(e8)).view(
            ne, 2 * rip, -1
        ),
        "w2s": e8m0_shuffle(w2sp.view(ne * HIDDEN, -1).view(e8)).view(ne, HIDDEN, -1),
    }


def _moe_exact(x, mw, ids):
    """The routed experts in fp32 on the dequantized weights, x and the
    intermediate unquantized, then the all-reduce."""
    from vllm.models.qwen4_exp.amd.mono.layout import Layout

    ri = Layout(TP).RI
    raw = mw["raw"]
    wts, ids, _ = _route_and_shared(x, mw, ids)
    routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            gu = _mxfp4_dequant(raw["w13"][e], raw["w13s"][e]) @ x[t].float()
            h = F.silu(gu[:ri]) * gu[ri:]
            routed[t] += wts[t, k] * (_mxfp4_dequant(raw["w2"][e], raw["w2s"][e]) @ h)
    return _all_reduce_fp32(routed.to(bf))


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
    from vllm.models.qwen4_exp.amd.mono import k2
    from vllm.models.qwen4_exp.amd.mono.layout import (
        HC,
        HH,
        HIDDEN,
        MAX_TOKENS,
        Layout,
    )
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine

    lay = Layout(TP)
    tp = get_tp_group()
    peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gc = torch.Generator(device=device).manual_seed(11)
    hw = _hc_weights(gc)
    step = 0
    for flags in K2_MODES:
        gr = torch.Generator(device=device).manual_seed(200 + rank)
        oq = flags.get("oq", False)
        if oq:
            oq_raw, os_raw, w_o, w_os = _mxfp4_linear(gr, HIDDEN, lay.CORE)
            w_o_ref = _mxfp4_dequant(oq_raw, os_raw).to(bf)
        else:
            w_o, w_os = _randn(gr, HIDDEN, lay.CORE, scale=0.02), None
        mode = str(flags)
        # both epoch parities, a layer slot each; 1 row (the picks as the
        # union), 4 and 8 (the fused shared expert's tasks before the route)
        for s in (1, 3, 4, 8, 8):
            step += 1
            epoch.add_(1)
            key = k2.K2Build(tokens=s, tp=TP, **flags)
            core = _randn(gr, s, lay.CORE)
            gate = _randn(gr, s, lay.CORE, scale=2.0) if key.og else None
            state = _randn(gc, s, HH, scale=2.0)
            inj_a = _randn(gc, s, HC)
            scratch = torch.zeros(
                k2.scratch_bytes(key), dtype=torch.uint8, device=device
            )
            out = torch.full((s, HH), float("nan"), dtype=bf, device=device)
            msq = torch.full((20, s, HC), float("nan"), device=device)
            k2.layer_post(
                key,
                core=core,
                gate=gate,
                state=state,
                inj_a=inj_a,
                w_o=w_o,
                w_os=w_os,
                **hw,
                out=out,
                msq_out=msq,
                scratch=scratch,
                peers=peers.addresses,
                rank=rank,
                epoch=epoch,
                layer=step,
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
                part = F.linear(core, w_o)
            state1_ref = hc_combine(state, _all_reduce_fp32(part), inj_a, HC)
            scr = k2.scratch_layout(key)
            off, n = scr["state1"]
            state1 = scratch[off : off + n].view(bf).view(s, HH)
            # the kernel sums the ranks in rank order, NCCL in its own order
            assert _rel_l2(state1, state1_ref) < 1e-2, (mode, s)
            # the MLP HC mix on the kernel's own state
            x_ref, inj_ref, _ = _mix_ref(state1.clone(), hw)
            off, n = scr["bin"]
            x = scratch[off : off + n].view(bf).view(s, HIDDEN)
            assert _cos(x, x_ref) > COS_MIN, (mode, s)
            inj_m = _pairs(scratch, scr["inj"], s * HC).view(s, HC).to(bf)
            torch.testing.assert_close(inj_m, inj_ref, atol=2e-2, rtol=2e-2)
            # the MoE on the kernel's own block input, where routing cannot differ
            ids = _check_route(scratch, scr["route"], x.clone(), mw)
            emu = _moe_emulation(x.clone(), mw, ids)
            want = hc_combine(state1.clone(), emu, inj_m, HC)
            c = _cos(out - state1, want - state1)
            assert c > COS_MIN, (mode, s, c)
            # fused_moe rounds x and the intermediate into other e2m1 bins than
            # the emulation (cosine 0.999 between the two): measured against
            # the unquantized MoE, K2 must be no further off than fused_moe
            ref = hc_combine(
                state1.clone(), _moe_reference(x.clone(), mw, ids), inj_m, HC
            )
            exact = hc_combine(
                state1.clone(), _moe_exact(x.clone(), mw, ids), inj_m, HC
            )
            err = _rel_l2(out - state1, exact - state1)
            err_stock = _rel_l2(ref - state1, exact - state1)
            assert err < 1.02 * err_stock, (mode, s, err, err_stock)
            torch.testing.assert_close(msq, _msq(out), rtol=1e-5, atol=1e-3)
            outs = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(outs, out)
            assert all(torch.equal(outs[0], o) for o in outs), (mode, "ranks disagree")
    peers.close()


# the GDN layers' MXFP4 out_proj; the QSA layers' bf16 o_proj with the gate
K2_MODES: tuple[dict[str, bool], ...] = ({"oq": True}, {"og": True})


@multi_gpu_test(num_gpus=2)
def test_k2_tp2_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 2, 1, _k2_worker)
