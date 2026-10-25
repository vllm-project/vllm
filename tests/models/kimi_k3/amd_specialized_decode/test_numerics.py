# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's MonoKernels on MI355X, random weights in the checkpoint's formats
at TP8 shapes:

- ``*_matches_slow_reference``: every fused stage of K1 (one GPU) and K2 (eight
  GPUs) read from its mailbox / scratch / peer tap and checked against a
  float64 reference (``tests/models/mono_reference.py``) applied to the
  kernel's own input to that stage, with the kernel's rounding points
  reproduced. A bound is a priori (error over its bound <= 1: exact stages,
  bf16 hand-offs, fp32-accumulated dots of exact products) or measured (2x the
  maximum of a 20-seed sweep: chains through hardware exp2 / rcp / rsq).
- ``*_matches_vllm_ops``: K1 and K2 against the ops vLLM runs for the same
  spec-verify step (nightly).
- The protocol: a CUDA graph of K2 replayed 300 times, bit-identical, and a
  stalled rank (the fault hook): every rank's launch finishes and the output
  rank's flags fail the step.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from tests.models import mono_reference as R
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.platforms import current_platform

bf = torch.bfloat16
COS_MIN = 0.9999
K1_STEPS = 6
# measured bounds: 2x the maximum over 20 seeds of every case
BOUNDS = {
    "x": 1.01,
    "qk": 1.01,
    "state": 6.6e-7,
    "core": 1.01,
    "xrow": 1.01,
    "route_w": 9.3,
    "h": 2.0,
    "ln": 1.01,
}


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the K3 mono kernels need gfx950 (MI355X)"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def _tap(region: torch.Tensor, n: int, tag: int, dtype=torch.float32):
    """The ``n`` 32-bit mailbox values of ``region`` (tags checked) as
    ``dtype``: float32 / int32 words, or bf16 pairs (2 n values)."""
    return R.pairs(region, n, tag).view(dtype)


def _dot_err(got, a, b, add=None, out=bf):
    """|got - (a @ b^T + add)| over its a-priori bound: an fp32-accumulated dot
    of exact products (K 2^-24 sum |a||b|, an fp32 add of ``add``), then
    rounded to ``out``."""
    a, b = a.double(), b.double()
    ref = a @ b.T
    d = a.size(-1) * 2.0**-24 * (a.abs() @ b.abs().T)
    if add is not None:
        ref = ref + add.double()
        d = d + 2.0**-24 * ref.abs()
    bound = 0.5 * R.ulp(ref.abs() + d, out) + d
    return ((got.double() - ref).abs() / bound).max().item()


def _row_ulps(got, ref, rows: int = 1):
    """bf16 ulps of max(|ref|, the RMS of each run of ``rows`` x the last dim):
    a normalized output's error, at its row's scale where it cancels."""
    ref = ref.double()
    rms = ref.unflatten(-1, (rows, -1)).square().mean(-1, keepdim=True).sqrt()
    floor = rms.expand(*rms.shape[:-1], ref.size(-1) // rows).flatten(-2)
    return R.ulps(got, ref, bf, floor).max().item()


def _rank_sum(parts: torch.Tensor) -> torch.Tensor:
    """bf16 partials [ranks, ...] summed in fp32 in rank order, -> bf16."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(bf)


def _bf16_ends(x, d):
    """X's bf16 rounding (float64) and, where x +- d rounds otherwise, the
    other end (0 elsewhere)."""
    near, lo, hi = R.boundary(R.bf16_rne, x, d)
    base = R.bf16_rne(x)
    return base, torch.where(near, lo + hi - base, 0)


def _merge_max(m: dict, new: dict) -> None:
    for k, v in new.items():
        m[k] = max(m.get(k, 0.0), v)


# ---------------------------------------------------------------- K1


def _kda_inputs(
    nacc: list[int],
    dim_first: bool,
    q: int,
    slots: int = 16,
    seed: int = 0,
    pads: int = 0,
):
    """``len(nacc)`` requests of ``q`` tokens, request r's last step having
    accepted ``nacc[r]`` of them; the last ``pads`` requests pad a graph
    (NULL_BLOCK_ID 0 state, one accepted token)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        NPROJ,
        PROJ,
        QKV,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    S = q * len(nacc)
    sl = 3 + (q - 1)
    w = {
        "prefix": _randn(g, S, HIDDEN, scale=2.0),
        "delta": _randn(g, S, HIDDEN, scale=0.5),
        "blocks": _randn(g, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(g, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "w_in": _randn(g, NPROJ, HIDDEN, scale=0.02),
        "w_fb": _randn(g, PROJ, HD, scale=0.08),
        "conv_w": _randn(g, QKV, 4, scale=0.4, dtype=torch.float32),
        "a_log": _randn(g, NH, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, PROJ, scale=0.5, dtype=torch.float32),
        "on_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    w["w_in"][NPROJ - 4 :].zero_()
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    pad = _randn(g, slots, NH * HD * HD + 24576, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, : NH * HD * HD].view(slots, NH, HD, HD)
    conv = _randn(g, slots, QKV * sl + 4096)[:, : QKV * sl]
    if dim_first:
        w["conv_state"] = conv.view(slots, QKV, sl)
    else:
        w["conv_state"] = conv.view(slots, sl, QKV).transpose(-1, -2)
    w["st_idx"] = torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(-1, q)
    w["num_acc"] = torch.tensor(nacc, dtype=torch.int32, device="cuda")
    if pads:
        w["st_idx"][-pads:] = 0
        w["num_acc"][-pads:] = 1
    return w


def _kda_reference(w, nb: int, write_idx: int):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HD,
        NH,
        PROJ,
        QKV,
    )
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    x = attn_res(
        prefix,
        w["delta"],
        blocks,
        w["ares_nw"],
        w["ares_qk"],
        w["in_nw"],
        nb,
        write_idx,
        1e-5,
        1e-5,
    )
    proj = F.linear(x, w["w_in"])
    mixed_qkv, g2, f_a, beta = proj.split([QKV, PROJ, HD, NH + 4], dim=-1)[:4]
    g1 = F.linear(f_a, w["w_fb"])
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    qsl = torch.arange(0, S + 1, q, dtype=torch.int32, device="cuda")
    conv = causal_conv1d_update(
        mixed_qkv.contiguous(),
        conv_state,
        w["conv_w"],
        None,
        activation="silu",
        conv_state_indices=w["st_idx"][:, 0].contiguous(),
        num_accepted_tokens=w["num_acc"],
        query_start_loc=qsl,
        max_query_len=q,
        validate_data=False,
        out=torch.empty_like(mixed_qkv),
    )
    q_, k, v = (
        rearrange(t, "n (h d) -> 1 n h d", d=HD) for t in conv.split(PROJ, dim=-1)
    )
    core = torch.empty(1, S, NH, HD, dtype=bf, device="cuda")
    fused_recurrent_kda(
        q=q_,
        k=k,
        v=v,
        raw_g=rearrange(g1, "n (h d) -> 1 n h d", d=HD),
        raw_beta=beta[:, :NH].unsqueeze(0),
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        lower_bound=-5.0,
        initial_state=rstate,
        cu_seqlens=qsl,
        ssm_state_indices=w["st_idx"],
        num_accepted_tokens=w["num_acc"],
        uniform_sequence_length=q,
        out=core,
    )
    with set_current_vllm_config(VllmConfig()):
        norm = FusedRMSNormGated(HD, activation="sigmoid").cuda()
    norm.weight.data = w["on_w"].clone()
    out = norm.forward_cuda(core, rearrange(g2, "n (h d) -> 1 n h d", d=HD))
    return {
        "prefix": prefix,
        "blocks": blocks,
        "conv_state": conv_state,
        "rstate": rstate,
        "core": out.reshape(S, PROJ),
    }


def _strided_copy(t: torch.Tensor) -> torch.Tensor:
    """A copy with ``t``'s strides: vLLM's padded state pages stay padded."""
    out = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device)
    return out.copy_(t)


def _k1_key(w, nb: int, write_idx: int = -1):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        KdaPreBuild,
    )

    return KdaPreBuild(
        tokens=w["st_idx"].numel(),
        qlen=w["st_idx"].size(1),
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=write_idx,
    )


def _k1(w, nb: int, write_idx: int = -1, poison: bool = False):
    """K1 on copies of ``w``'s prefix, blocks and states: them updated, the core
    output and the scratch (its plain x rows NaN-filled first if ``poison``)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        PROJ,
        kda_pre,
        scratch_bytes,
        scratch_layout,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    S = w["st_idx"].numel()
    key = _k1_key(w, nb, write_idx)
    out = {k: _strided_copy(w[k]) for k in ("prefix", "blocks", "conv_state", "rstate")}
    out["core"] = torch.zeros(S, PROJ, dtype=bf, device="cuda")
    out["scratch"] = torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda")
    if poison:
        o, n = scratch_layout(key)["x"]
        out["scratch"][o : o + n] = 0xFF
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    kda_pre(
        key,
        prefix=out["prefix"],
        delta=w["delta"],
        blocks=out["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_in=w["w_in"],
        w_fb=w["w_fb"],
        conv_w=w["conv_w"],
        conv_state=out["conv_state"],
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        on_w=w["on_w"],
        rstate=out["rstate"],
        st_idx=w["st_idx"],
        num_acc=w["num_acc"],
        core_out=out["core"],
        scratch=out["scratch"],
        layer=0,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    return out


@needs_gfx950
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("nacc", [1, 3])
@pytest.mark.parametrize("write", [False, True])
def test_k1_matches_vllm_ops(dim_first: bool, nacc: int, write: bool) -> None:
    """K1 against attn_res -> in_proj -> f_b -> causal_conv1d_update ->
    fused_recurrent_kda -> FusedRMSNormGated on one spec-verify step (one
    request, 8 tokens), including the state each step leaves behind; a second
    launch is bit-identical."""
    nb = 4
    # a block-write layer appends block nb (block_write_idx == prev_valid_blocks)
    write_idx = nb if write else -1
    w = _kda_inputs([nacc], dim_first, q=8)
    ref = _kda_reference(w, nb, write_idx)
    out = _k1(w, nb, write_idx)

    torch.testing.assert_close(out["prefix"], ref["prefix"], atol=0, rtol=0)
    torch.testing.assert_close(out["blocks"], ref["blocks"], atol=0, rtol=0)
    # the conv output feeds the recurrence in fp32 here, bf16 in the reference
    errs = {k: _rel_l2(out[k], ref[k]) for k in ("conv_state", "rstate", "core")}
    assert errs["conv_state"] < 1e-3 and errs["rstate"] < 1e-3, errs
    assert _cos(out["core"], ref["core"]) > COS_MIN, errs


def _k1_check(w, out, nb: int, write_idx: int, tag: int = 257) -> dict:
    """Every K1 stage of one launch (inputs ``w``, results ``out``) against the
    float64 reference on the kernel's own input to it -> metric a stage."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        BETA_0,
        FA_0,
        G2_0,
        HD,
        HIDDEN,
        NH,
        NPROJ,
        NVCH,
        PROJ,
        QKV,
        scratch_layout,
    )

    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    lay = scratch_layout(_k1_key(w, nb, write_idx))
    sc = out["scratch"]
    m = {}

    prefix = R.bf16_rne(w["prefix"].double() + w["delta"].double())
    assert torch.equal(out["prefix"].double(), prefix), "prefix"
    blocks = w["blocks"].clone()
    if write_idx >= 0:
        blocks[:, write_idx] = out["prefix"]
    assert torch.equal(out["blocks"], blocks), "blocks"

    x = R.words(sc[lay["x"][0] :], S * HIDDEN, bf).view(S, HIDDEN)
    x_ref = R.attn_res(
        out["prefix"],
        blocks[:, :nb],
        w["ares_nw"],
        w["ares_qk"],
        w["in_nw"],
        1e-5,
        1e-5,
    )
    m["x"] = _row_ulps(x, x_ref)

    proj = _tap(sc[lay["proj"][0] :], S * NPROJ, tag).view(S, NPROJ)
    assert torch.equal(proj.double(), R.bf16_rne(proj)), "proj is not bf16"
    m["proj"] = _dot_err(proj, x, w["w_in"])

    qk = _tap(sc[lay["qk"][0] :], S * NH * 2 * HD, tag).view(-1, NH, q, 2, HD)
    onrm = _tap(sc[lay["onrm"][0] :], S * NH * NVCH, tag).view(-1, NH, q, NVCH)
    written: dict[str, set[int]] = {"conv_state": set(), "rstate": set()}
    pads = 0
    for r in range(S // q):
        rows = slice(r * q, (r + 1) * q)
        slots = w["st_idx"][r].tolist()
        acc = int(w["num_acc"][r]) - 1
        old = w["conv_state"][max(slots[0], 0)].double()
        win = old[:, acc : acc + 3]
        conv, _ = R.causal_conv1d(win, proj[rows, :QKV], w["conv_w"])
        cond, _ = R.causal_conv1d(
            win.abs(), proj[rows, :QKV].abs(), w["conv_w"].abs(), act=False
        )
        conv, cond = (t.view(q, 3, NH, HD) for t in (conv, cond))
        got = qk[r].permute(1, 2, 0, 3)
        m["qk"] = max(m.get("qk", 0), R.ulps(got, conv[:, :2], bf, cond[:, :2]).max())
        if slots[0] > 0:
            kept = old[:, acc + 1 : acc + 1 + old.size(1) - q]
            new = torch.cat([kept, proj[rows, :QKV].double().T], 1)
            assert torch.equal(out["conv_state"][slots[0]].double(), new), "conv state"
            written["conv_state"].add(slots[0])
        slot0 = slots[acc]
        if slot0 <= 0:
            assert not out["core"][rows].any(), "a pad request's core"
            pads += 1
            continue
        # bf16 hand-offs only in LDS (v, g1): an input within its error bound
        # of a rounding boundary may round either way; each such flip's effect
        # on the state is allowed
        fa = proj[rows, FA_0 : FA_0 + HD].double()
        g1 = fa @ w["w_fb"].double().T
        g_d = HD * 2.0**-24 * (fa.abs() @ w["w_fb"].double().abs().T)
        hand = [
            _bf16_ends(conv[:, 2], 2.0**-19 * cond[:, 2]),
            _bf16_ends(g1.view(q, NH, HD), g_d.view(q, NH, HD)),
        ]
        beta = proj[rows, BETA_0 : BETA_0 + NH]
        st = w["rstate"][slot0]
        rest = (w["a_log"], w["dt_bias"].view(NH, HD), -5.0, HD**-0.5)
        for j in range(q):
            qkj = (st, got[j, 0], got[j, 1])
            vg = [x[j] for x, _ in hand]
            ref = R.kda_step(*qkj, vg[0], vg[1], beta[j], *rest)[0]
            allow = torch.zeros_like(ref)
            for i, (_, alt) in enumerate(hand):
                for at in alt[j].nonzero():
                    flip = [x.clone() for x in vg]
                    flip[i][tuple(at)] = alt[j][tuple(at)]
                    allow += (
                        R.kda_step(*qkj, flip[0], flip[1], beta[j], *rest)[0] - ref
                    ).abs()
                m["n_flips"] = m.get("n_flips", 0) + int(alt[j].count_nonzero())
            st = out["rstate"][slots[j]]
            written["rstate"].add(slots[j])
            err = ((st.double() - ref).abs() - allow).clamp(min=0).amax((1, 2))
            err = err / ref.abs().amax((1, 2))
            m["state"] = max(m.get("state", 0), err.max().item())
            qn = got[j, 0].double()
            qn = qn * torch.rsqrt(qn.square().sum(-1, keepdim=True) + 1e-6) * HD**-0.5
            o = torch.einsum("hvk,hk->hv", st.double(), qn)
            o_d = 2.0**-20 * torch.einsum("hvk,hk->hv", st.double().abs(), qn.abs())
            o_lo, o_hi = R.bf16_rne(o - o_d), R.bf16_rne(o + o_d)
            hi = torch.maximum(o_lo.square(), o_hi.square()).view(NH, NVCH, -1).sum(-1)
            lo = torch.minimum(o_lo.square(), o_hi.square()) * (o_lo * o_hi > 0)
            lo = lo.view(NH, NVCH, -1).sum(-1)
            ss = onrm[r, :, j].double()
            out_of = torch.maximum(ss - hi, lo - ss).clamp(min=0) / (2.0**-20 * hi)
            m["onrm"] = max(m.get("onrm", 0), out_of.max().item())
            rstd = torch.rsqrt(onrm[r, :, j].double().sum(-1, keepdim=True) / HD + 1e-5)
            g2 = proj[r * q + j, G2_0 : G2_0 + PROJ].double().view(NH, HD)
            ends = torch.stack([o_lo, o_hi]) * rstd * w["on_w"] * torch.sigmoid(g2)
            got_core = out["core"][r * q + j].view(NH, HD).double()
            core = torch.maximum(torch.minimum(got_core, ends.amax(0)), ends.amin(0))
            m["core"] = max(
                m.get("core", 0), _row_ulps(got_core.flatten(), core.flatten(), NH)
            )
    for name, done in written.items():
        free = torch.ones(w[name].size(0), dtype=torch.bool)
        free[list(done)] = False
        assert torch.equal(out[name][free], w[name][free]), f"{name}: a free slot"
    return {k: float(v) for k, v in m.items()} | {"pad_requests": pads}


def _k1_io(q, requests, nb, steps, write, dim_first, pad, seed=0) -> dict:
    """``steps`` verify steps, each request accepting a random count a step, each
    step from the states the kernel left: every stage of each against the slow
    reference, a rerun without the poisoned scratch bit-identical."""
    gen = torch.Generator().manual_seed(seed * 64 + q * 16 + requests)
    write_idx = nb if write else -1
    w = _kda_inputs([1] * requests, dim_first, q=q, seed=seed * 64, pads=pad)
    m: dict = {}
    for step in range(steps):
        nacc = torch.randint(1, q + 1, (requests,), generator=gen).tolist()
        x = _kda_inputs(nacc, dim_first, q, seed=seed * 64 + 1 + step, pads=pad)
        w.update({k: x[k] for k in ("prefix", "delta", "blocks", "num_acc")})
        out = _k1(w, nb, write_idx, poison=True)
        _merge_max(m, _k1_check(w, out, nb, write_idx))
        again = _k1(w, nb, write_idx)
        assert all(torch.equal(out[k], again[k]) for k in out), "rerun differs"
        w.update({k: out[k] for k in ("conv_state", "rstate")})
    return m


@needs_gfx950
@pytest.mark.parametrize(
    "q, requests, nb, steps, write, dim_first, pad",
    [
        (8, 1, 4, 1, False, True, 0),
        (8, 1, 4, 1, True, False, 0),
        (2, 3, 3, 1, True, True, 1),
        (2, 4, 1, K1_STEPS, False, False, 1),
        (3, 2, 2, K1_STEPS, False, True, 0),
        (4, 2, 7, K1_STEPS, False, False, 0),
        (8, 1, 8, K1_STEPS, False, False, 0),
    ],
)
def test_k1_matches_slow_reference(q, requests, nb, steps, write, dim_first, pad):
    """K1 stage by stage at widths 2-8 (whole requests of 2-8 tokens), 1-8
    AttnRes blocks, both conv state layouts, block-write layers and graph pad
    requests."""
    m = _k1_io(q, requests, nb, steps, write, dim_first, pad)
    assert m["pad_requests"] == pad and m["n_flips"] > 0, m
    assert m["proj"] <= 1 and m["onrm"] <= 1, m
    assert all(m[k] <= BOUNDS[k] for k in ("x", "qk", "state", "core")), m


# ---------------------------------------------------------------- K2 / MoE


def _situ(g, u):
    g, u = g.float(), u.float()
    a = (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5))
    return (a * (25.0 * torch.tanh(u * 0.04))).to(bf)


def _all_reduce_fp32(t: torch.Tensor) -> torch.Tensor:
    """The ranks' bf16 partials summed in fp32, as vLLM's custom all-reduce."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(bf)


def _moe_reference(x, mw, rank: int):
    """KimiMoE's decode path on ``x``: sigmoid router + bias top-16, the latent
    down, aiter's MXFP4 experts, the latent all-reduce + RMSNorm, this rank's
    up_proj rows into the shared experts' output, the final all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        SI,
        TOPK,
        UP_N,
    )

    sig = torch.sigmoid(torch.mm(x.float(), mw["w_gate"].float().t()))
    ids = torch.topk(sig + mw["bias"], TOPK, dim=-1, sorted=True).indices
    wts = sig.gather(-1, ids)
    wts = wts / wts.sum(-1, keepdim=True)
    gu = F.linear(x, mw["w_sgu"])
    shared = F.linear(_situ(gu[:, :SI], gu[:, SI:]), mw["w_sd"])
    y = fused_moe(
        F.linear(x, mw["w_ld"]),
        mw["w13"],
        mw["w2"],
        wts.float(),
        ids.to(torch.int32),
        None,
        ActivationType.Situv2,
        QuantType.per_1x32,
        False,
        mw["w13s"],
        mw["w2s"],
        None,
        None,
        dtype=bf,
        gate_mode="separated",
        beta=4.0,
        linear_beta=25.0,
    )
    lat = _all_reduce_fp32(y).float()
    ln = lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-5) * mw["ln_w"]
    lo = rank * UP_N
    up = ln.to(bf).float() @ mw["w_up_shard"].float().t()
    shared[:, lo : lo + UP_N] = (shared[:, lo : lo + UP_N].float() + up).to(bf)
    return _all_reduce_fp32(shared)


def _moe_raw(rank: int, device, seed: int = 0, w13_codes=(118, 123)) -> dict:
    """The MoE weights as the checkpoint holds them: MXFP4 experts as fp4x2
    bytes (gate rows, then up) and E8M0 codes."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        RI,
        SI,
        UP_N,
        E,
    )

    gc = torch.Generator(device=device).manual_seed(7 + 1000 * seed)  # replicated
    gr = torch.Generator(device=device).manual_seed(100 + rank + 1000 * seed)  # TP

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device).to(torch.uint8)

    w_up = _randn(gc, HIDDEN, LAT, scale=0.02)
    return {
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "bias": _randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": _randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + _randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(bf),
        "w_up_shard": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
        "w13": u8(gr, E, 2 * RI, LAT // 2),
        "w2": u8(gr, E, LAT, RI // 2),
        "w13s": u8(gr, E, 2 * RI, LAT // 32, lo=w13_codes[0], hi=w13_codes[1]),
        "w2s": u8(gr, E, LAT, RI // 32, lo=118, hi=123),
    }


def _moe_weights(rank: int, device, raw: dict | None = None) -> dict:
    """``raw`` (``_moe_raw``'s, also kept as "raw") with the experts in aiter's
    shuffled layout, the layout the kernel reads."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import LAT, RI, E

    raw = _moe_raw(rank, device) if raw is None else raw
    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    return raw | {
        "raw": raw,
        "w13": rocm_aiter_ops.shuffle_weight_a16w4(raw["w13"].view(fp4), 16, False),
        "w2": rocm_aiter_ops.shuffle_weight_a16w4(raw["w2"].view(fp4), 16, False),
        "w13s": rocm_aiter_ops.shuffle_scale_a16w4(
            raw["w13s"].view(e8).view(-1, LAT // 32), E, False
        ),
        "w2s": e8m0_shuffle(raw["w2s"].view(e8).view(-1, RI // 32)),
    }


def _moe_args(mw):
    return {
        "w_gate": mw["w_gate"],
        "bias": mw["bias"],
        "w_ld": mw["w_ld"],
        "w_sgu": mw["w_sgu"],
        "w_sd": mw["w_sd"],
        "w13": mw["w13"],
        "w13s": mw["w13s"],
        "w2": mw["w2"],
        "w2s": mw["w2s"],
        "ln_w": mw["ln_w"],
        "w_up": mw["w_up_shard"],
    }


def _k2_inputs(rank: int, device, seed: int = 0):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HIDDEN,
        PROJ,
    )

    gc = torch.Generator(device=device).manual_seed(11 + 1000 * seed)
    gr = torch.Generator(device=device).manual_seed(200 + rank + 1000 * seed)
    return {
        "core": _randn(gr, 8, PROJ),
        "w_o": _randn(gr, HIDDEN, PROJ, scale=0.02),
        "prefix": _randn(gc, 8, HIDDEN, scale=2.0),
        "blocks": _randn(gc, 8, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(gc, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
    }


def _k2_launch(runner, w, mw, prefix, out, reset: bool, nb: int = 4):
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2

    key = k2.K2Build(
        tokens=w["core"].size(0), nblocks=nb + int(reset), reset=reset, **runner.fault
    )
    k2.k2_launch(
        key,
        core=w["core"],
        w_o=w["w_o"],
        prefix=prefix,
        blocks=w["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        out=out,
        layer=3,
        **_moe_args(mw),
        **runner.launch_args(),
        **runner.peer_args(),
    )
    return key


def _k2_x(runner, key) -> torch.Tensor:
    """The MLP AttnRes rows K2 left in scratch: the MoE's input."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN

    lay = k2.scratch_layout(key)["xrow"][0]
    s = key.tokens
    return runner.scratch[lay : lay + s * HIDDEN * 2].view(bf).view(s, HIDDEN)


def _assert_ranks_agree(out: torch.Tensor, tp_size: int) -> None:
    outs = [torch.empty_like(out) for _ in range(tp_size)]
    dist.all_gather(outs, out)
    assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"


def _assert_replays_bit_identical(runner, w, mw) -> None:
    """A graph of one step (the epoch bump captured with it), replayed: a
    block-write launch writes its prefix, so every replay must match the
    first."""
    prefix = torch.empty_like(w["prefix"])
    out = torch.empty_like(w["prefix"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.step_begin()
        _k2_launch(runner, w, mw, prefix, out, reset=True)
    graph.replay()
    torch.accelerator.synchronize()
    want_prefix, want_out = prefix.clone(), out.clone()
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(prefix, want_prefix) and torch.equal(out, want_out)
    assert not runner.health.any() and not runner.flags.any()


def _assert_mxfp4_equal(got, codes, v, d) -> float:
    """MXFP4 output (E2M1 values [..., K], E8M0 codes [..., K / 32]) of aiter's
    inline quant (``e8m0_roundup6``, RNE) against the reference values ``v``
    known to +-``d``: exact outside the boundary set, one of its two ends
    inside, the set under 1 % -> its fraction."""
    vb, db = v.unflatten(-1, (-1, 32)), d.unflatten(-1, (-1, 32))
    bc, clo, chi = R.boundary(R.e8m0_roundup6, vb.abs().amax(-1), db.amax(-1))
    codes = codes.long()
    assert torch.equal(codes[~bc], clo[~bc]), "codes differ"
    assert ((codes == clo) | (codes == chi))[bc].all(), "codes off by > 1"
    inv = torch.exp2(127 - codes.double())[..., None]
    bv, vlo, vhi = R.boundary(R.to_e2m1_rne, vb * inv, db * inv)
    gb = got.unflatten(-1, (-1, 32))
    assert torch.equal(gb[~bv], vlo[~bv]), "values differ"
    assert ((gb == vlo) | (gb == vhi))[bv].all(), "values off by > 1 step"
    frac = (bv.sum() + bc.sum() * 32).item() / v.numel()
    assert frac < 0.01, f"boundary set {frac:.2%} of the elements"
    return frac


def _peer(runner, s: int, name: str) -> torch.Tensor:
    """This rank's peer region ``name`` of the current step's parity."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import PEER_HEAD

    half = k2.peer_bytes(s) // 2
    at = PEER_HEAD + int(runner.epoch.item()) % 2 * half + k2.peer_layout(s)[name][0]
    return runner.peers.bytes[at:]


def _k2_check(runner, key, w, mw, prefix, out, seed: int) -> dict:
    """Every K2 stage of one launch (inputs ``w``, ``mw``; the prefix ``w``'s
    updated to ``prefix``; the output ``out``) against the float64 reference on
    the kernel's own input to it -> metric a stage."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        RI,
        SI,
        TOPK,
        TP,
        UP_N,
        E,
    )

    s, rank, device = key.tokens, runner.rank, w["core"].device
    tag = (int(runner.epoch.item()) % (1 << 22)) * 256 + 3 + 1
    lay = k2.scratch_layout(key)
    sc = runner.scratch

    def scratch(name, n, dtype=torch.float32):
        return _tap(sc[lay[name][0] :], n, tag, dtype)

    def peer(name, n, dtype=torch.float32):
        return _tap(_peer(runner, s, name), n, tag, dtype)

    def plain(name, n):
        return R.words(sc[lay[name][0] :], n, bf)

    m: dict = {}
    ar = peer("ar", TP * s * HIDDEN // 2, bf).view(TP, s, HIDDEN)
    for src in range(TP):
        x = _k2_inputs(src, device, seed)
        _merge_max(m, {"ar": _dot_err(ar[src], x["core"][:s], x["w_o"])})
    delta = _rank_sum(ar)
    want = delta if key.reset else R.bf16_rne(w["prefix"].double() + delta.double())
    assert torch.equal(prefix.double(), want.double()), "prefix"

    x = plain("xrow", s * HIDDEN).view(s, HIDDEN)
    blocks = w["blocks"][:, : key.nblocks]
    ares = (w["ares_nw"], w["ares_qk"], w["in_nw"], key.eps, key.out_eps)
    m["xrow"] = _row_ulps(x, R.attn_res(prefix, blocks, *ares))

    rlog = peer("rlog", TP * s * (E // TP)).view(TP, s, E // TP)
    logits = rlog.transpose(0, 1).reshape(s, E)
    m["rlog"] = _dot_err(logits, x, mw["w_gate"], out=torch.float32)
    lat = peer("lat", s * LAT // 2, bf).view(s, LAT)
    m["lat"] = _dot_err(lat, x, mw["w_ld"])
    sgu = scratch("sgu", s * SI, bf).view(s, 2 * SI)
    m["sgu"] = _dot_err(sgu, x, mw["w_sgu"])

    route = scratch("route", s * 2 * TOPK, torch.int32).view(s, 2, TOPK)
    ids, wts = route[:, 0].long(), route[:, 1].view(torch.float32)
    ref_ids, _, margin = R.route(logits, mw["bias"], TOPK, "sigmoid", 1.0)
    sig = torch.sigmoid(logits.double())
    top = (sig + mw["bias"].double()).sort(-1, descending=True).values[:, : TOPK + 1]
    gap = top[:, :-1] - top[:, 1:]
    clear = (gap == 0) | (gap > 2 * 2.0**-21 * top[:, :1].abs().clamp(min=1))
    rows = clear.all(-1)
    assert torch.equal(ids[rows], ref_ids[rows]), "top-k ids or their order"
    assert torch.equal(ids.sort(-1)[0][clear[:, -1]], ref_ids.sort(-1)[0][clear[:, -1]])
    m["n_route_rows"] = int(rows.sum())
    m["n_boundary_ties"] = int(((margin == 0) & rows).sum())
    ws = sig.gather(1, ids)
    m["route_w"] = R.ulps(wts, ws / ws.sum(-1, keepdim=True), torch.float32).max()

    h = scratch("h", s * SI // 2, bf).view(s, SI)
    g, u = sgu[:, :SI].double(), sgu[:, SI:].double()
    a, b = 4 * torch.tanh(g / 4) * torch.sigmoid(g), 25 * torch.tanh(u / 25)
    h_ref = R.situ(g, u)
    # tanh_libm's absolute error (<= 0.75 2^-23) through each of its 3 uses
    d = 2.0**-22 * (h_ref.abs() + 25 * a.abs() + 6 * b.abs())
    bound = 0.5 * R.ulp(h_ref.abs() + d, bf) + d
    m["h"] = ((h.double() - h_ref).abs() / bound).max()

    lat_codes = R.e8m0_roundup6(lat.double().abs().unflatten(-1, (-1, 32)).amax(-1))
    scale = R.e8m0(lat_codes).repeat_interleave(32, -1)
    latv = R.to_e2m1_rne(lat.double() / scale) * scale
    m["n_zero_lat_groups"] = int((lat_codes == 0).sum())
    union, slot = ids.unique(return_inverse=True)
    m["n_union"] = union.numel()
    raw = mw["raw"]
    w13 = R.dequant_mxfp4(raw["w13"][union], raw["w13s"][union])
    gu = torch.einsum("urk,tk->tur", w13, latv)
    # a wave chains 14 scaled MFMAs (a 128-product dot each: 129 2^-24, the
    # primitives' bound), then the two K halves are added
    d_gu = 144 * 2.0**-24 * torch.einsum("urk,tk->tur", w13.abs(), latv.abs())
    del w13
    picks = torch.arange(s, device=device)[:, None], slot
    g, u = gu[picks].requires_grad_().split(RI, -1)
    v = R.situ(g, u)
    dg, du = torch.autograd.grad(v.sum(), (g, u))
    g, u, v = g.detach(), u.detach(), v.detach()
    d_g, d_u = d_gu[picks].split(RI, -1)
    d = dg.abs() * d_g + du.abs() * d_u
    d = d + 18 * 2.0**-23 * (v.abs() + (g * dg).abs() + (u * du).abs())
    sat = (g.abs() > 4 * 2.65) | (u.abs() > 25 * 2.65)
    m["situ_saturated"] = sat.double().mean().item()
    inter = scratch("inter", s * TOPK * 48, torch.int32).view(s * TOPK, 48)
    inter = inter.view(torch.uint8)
    isc = scratch("isc", s * TOPK * RI // 32, torch.int32).view(s * TOPK, -1)
    vals = R.dequant_mxfp4(inter, torch.full_like(isc, 127))
    m["inter_boundary"] = _assert_mxfp4_equal(
        vals, isc, v.flatten(0, 1), d.flatten(0, 1)
    )
    m["n_zero_inter_groups"] = int((isc == 0).sum())

    iv = R.dequant_mxfp4(inter, isc).view(s, TOPK, RI)
    w2 = R.dequant_mxfp4(raw["w2"][union], raw["w2s"][union])
    y = e = torch.zeros(s, LAT, dtype=torch.float64, device=device)
    for k in range(TOPK):
        wt = wts[:, k : k + 1].double()
        acc = torch.einsum("tlr,tr->tl", w2[slot[:, k]], iv[:, k])
        cond = torch.einsum("tlr,tr->tl", w2[slot[:, k]].abs(), iv[:, k].abs())
        dk = wt * RI * 2.0**-24 * cond + 2.0**-24 * (wt * acc).abs()
        c = R.bf16_rne(wt * acc)
        ec = dk + R.ulp(c.abs() + dk, bf)
        y = R.bf16_rne(y + c)
        e = e + ec + R.ulp(y.abs() + e + ec, bf)
    del w2
    latr = peer("latr", TP * s * LAT // 2, bf).view(TP, s, LAT)
    m["latr"] = ((latr[rank].double() - y).abs() / e).max()

    ln = plain("ln", s * LAT).view(s, LAT)
    m["ln"] = _row_ulps(ln, R.rmsnorm(_rank_sum(latr), mw["ln_w"], key.ln_eps))
    shp = plain("shp", s * UP_N).view(s, UP_N)
    final = peer("final", TP * s * HIDDEN // 2, bf).view(TP, s, HIDDEN)
    mine = torch.zeros(HIDDEN, dtype=torch.bool, device=device)
    mine[rank * UP_N : (rank + 1) * UP_N] = True
    m["up"] = _dot_err(final[rank][:, mine], ln, mw["w_up_shard"], add=shp)
    m["sdown"] = max(
        _dot_err(shp, h, mw["w_sd"][mine]),
        _dot_err(final[rank][:, ~mine], h, mw["w_sd"][~mine]),
    )
    assert torch.equal(out, _rank_sum(final)), "out"
    return {k: float(v) for k, v in m.items()}


def _k2_case(runner, w, mw, nb, reset, seed, poison=True):
    """One K2 step of ``w``'s rows (scratch's plain rows NaN-filled first if
    ``poison``) -> (metrics, prefix, out, MLP input rows)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2

    s = w["core"].size(0)
    runner.step_begin()
    if poison:
        lay = k2.scratch_layout(k2.K2Build(tokens=s, nblocks=nb + int(reset)))
        for name in ("xrow", "ln", "shp"):
            runner.scratch[lay[name][0] : sum(lay[name])] = 0xFF
    prefix = w["prefix"].clone()
    out = torch.empty_like(prefix)
    key = _k2_launch(runner, w, mw, prefix, out, reset, nb)
    torch.accelerator.synchronize()
    m = _k2_check(runner, key, w, mw, prefix, out, seed)
    x = _k2_x(runner, key).clone()
    _assert_ranks_agree(out, TP_SIZE)
    dist.barrier()
    return m, prefix, out, x


TP_SIZE = 8
# (rows, AttnRes blocks, block-write layer)
K2_CASES = [(8, 4, False), (8, 4, True), (2, 1, False), (4, 7, True)]
TIES = ((5, 21), (84, 100))  # equal router rows 16 apart: equal logits
W13_CODES = (118, 123)


def _k2_io(runner, seed: int = 0) -> dict:
    """K2 stage by stage over ``K2_CASES``, then (8 rows) every token on the same
    16 experts with exact routing ties (the 16th pick between two equal
    experts, two equal experts inside the top 16) and every token on 16 of its
    own (the largest union); a rerun without the poisoned scratch
    bit-identical. A zero 32-column group of the latent down and of one picked
    expert's up rows makes all-zero MXFP4 groups."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import RI, TOPK

    device = runner.scratch.device
    raw = _moe_raw(runner.rank, device, seed, w13_codes=W13_CODES)
    raw["w_ld"][:32] = 0
    raw["w13"][TIES[1][0], RI : RI + 32] = 0
    mw = _moe_weights(runner.rank, device, raw)
    w = _k2_inputs(runner.rank, device, seed)
    m: dict = {}
    for s, nb, reset in K2_CASES:
        ws = w | {k: w[k][:s] for k in ("core", "prefix", "blocks")}
        got, prefix, out, x = _k2_case(runner, ws, mw, nb, reset, seed)
        _merge_max(m, got)
        if (s, nb, reset) == K2_CASES[0]:
            _, prefix2, out2, x2 = _k2_case(runner, ws, mw, nb, reset, seed, False)
            assert torch.equal(prefix, prefix2) and torch.equal(out, out2)
            assert torch.equal(x, x2), "rerun differs"
            own = 0.2 * x.float() / x.float().norm(dim=-1, keepdim=True)

    tie = mw | {"w_gate": mw["w_gate"].clone(), "bias": mw["bias"].clone()}
    tie["bias"][84:98] += 10.0
    for lo, hi in TIES:
        tie["w_gate"][hi] = tie["w_gate"][lo]
    tie["bias"][[5, 21]] = 5.0
    tie["bias"][100] = tie["bias"][84]
    disjoint = mw | {"w_gate": mw["w_gate"].clone()}
    disjoint["bias"] = torch.zeros_like(mw["bias"])
    disjoint["w_gate"][: 8 * TOPK] = own.repeat_interleave(TOPK, dim=0).to(bf)
    dist.broadcast(disjoint["w_gate"], src=0)
    for case, union in ((tie, TOPK), (disjoint, 8 * TOPK)):
        got, *_ = _k2_case(runner, w, case, 4, False, seed)
        _merge_max(m, got)
        assert got["n_union"] == union, got
    return m


def _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    return device


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    S, nb = 8, 4
    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)
    w = _k2_inputs(rank, device)

    for reset in (True, False):
        runner.step_begin()
        prefix = w["prefix"].clone()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        key = _k2_launch(runner, w, mw, prefix, out, reset)
        torch.accelerator.synchronize()

        attn = _all_reduce_fp32(F.linear(w["core"], w["w_o"]))
        # a block-write layer's prefix restarts at the attention output
        ref_prefix = attn if reset else w["prefix"].clone()
        delta = None if reset else attn
        x_ref = attn_res(
            ref_prefix,
            delta,
            w["blocks"],
            w["ares_nw"],
            w["ares_qk"],
            w["in_nw"],
            nb + int(reset),
            -1,
            1e-5,
            1e-5,
        )
        # the kernel sums the ranks in rank order, NCCL in its own order
        assert _rel_l2(prefix, ref_prefix) < 1e-2
        x = _k2_x(runner, key)
        assert _cos(x, x_ref) > COS_MIN
        assert _cos(out, _moe_reference(x, mw, rank)) > COS_MIN
        _assert_ranks_agree(out, tp_size)

    _assert_replays_bit_identical(runner, w, mw)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _k2_io_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    m = _k2_io(runner)
    assert m["n_boundary_ties"] > 0 and m["n_zero_lat_groups"] > 0, m
    assert m["n_zero_inter_groups"] > 0 and 0.01 <= m["situ_saturated"] <= 0.2, m
    a_priori = ("ar", "rlog", "lat", "sgu", "latr", "up", "sdown")
    assert all(m[k] <= 1 for k in a_priori), m
    assert all(m[k] <= BOUNDS[k] for k in ("xrow", "route_w", "h", "ln")), m
    _assert_replays_bit_identical(
        runner, _k2_inputs(rank, device), _moe_weights(rank, device)
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_slow_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """K2 stage by stage at 2, 4 and 8 rows, 1-8 AttnRes blocks, block-write
    layers, exact routing ties and the largest expert union; then a CUDA graph
    of it replayed."""
    multi_process_parallel(monkeypatch, 8, 1, _k2_io_worker)


STALLED = 3


@ray.remote(num_gpus=1, max_calls=1)
def _fault_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import WAIT_TICKS
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # rank STALLED's CTA 0 stalls past every wait's bound in the epoch 2 launch
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_layer": 3,
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    mw = _moe_weights(rank, device)
    w = _k2_inputs(rank, device)
    out = torch.empty(8, HIDDEN, dtype=bf, device=device)
    runner.step_begin()  # epoch 1: no stall
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    assert not runner.flags.any()

    runner.step_begin()  # epoch 2: rank STALLED stalls; every launch ends
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    waits = given_up_waits(runner.health.cpu(), k2.REGIONS)
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    assert gave_up.item() > 0, "no rank's wait gave up"
    if rank == 0:
        assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_stalled_rank_fails_the_step_without_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _fault_worker)


@needs_gfx950
def test_warmup_compiles_every_k3_build_without_launching() -> None:
    """The JIT warm-up compiles every build Kimi-K3 (93 layers, AttnRes blocks
    of 12) launches at 7 speculative tokens before capture, without running
    them: K1 and K2 at 1-8 blocks, and the 7 block-write KDA layers' K1 (writing
    block nb) and K2 resets. A launch on the null arguments would fault."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        CONV_W,
        KdaPreBuild,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.layer import K2Build
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import compile_build

    k1 = {"tokens": 8, "qlen": 8, "delta": True, "state_len": CONV_W - 1 + 7}
    for nb in range(1, 9):
        compile_build(KdaPreBuild(nblocks=nb, **k1))
        compile_build(K2Build(tokens=8, nblocks=nb))
        if nb < 8:
            compile_build(KdaPreBuild(nblocks=nb, write_idx=nb, **k1))
            compile_build(K2Build(tokens=8, nblocks=nb + 1, reset=True))
    torch.accelerator.synchronize()
