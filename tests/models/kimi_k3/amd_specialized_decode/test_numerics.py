# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's MonoKernels on MI355X against the ops vLLM runs for the same
spec-verify step, random weights in the checkpoint's formats at TP8 shapes:

- K1 (one GPU): AttnRes -> in_proj -> f_b -> conv -> KDA recurrence -> gated
  norm, the conv / SSM state updates and a block-write layer's block store.
- K2 (eight GPUs): o_proj + all-reduce + the MLP AttnRes + the latent MoE. The
  MoE is compared on the kernel's own MLP input, where routing cannot differ.
  Then a CUDA graph of K2 replayed 300 times, bit-identical, and a stalled rank
  (the fault hook): every rank's launch finishes and the output rank's flags
  fail the step.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.platforms import current_platform

bf = torch.bfloat16
COS_MIN = 0.9999
K1_STEPS = 6


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


# ---------------------------------------------------------------- K1


def _kda_inputs(
    nacc: list[int], dim_first: bool, q: int, slots: int = 16, seed: int = 0
):
    """``len(nacc)`` requests of ``q`` tokens, request r's last step having
    accepted ``nacc[r]`` of them."""
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


def _k1(w, nb: int, write_idx: int = -1):
    """K1 on copies of ``w``'s prefix, blocks and states: them updated, and the
    core output."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        PROJ,
        KdaPreBuild,
        kda_pre,
        scratch_bytes,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    S, q = w["st_idx"].numel(), w["st_idx"].size(1)
    key = KdaPreBuild(
        tokens=S,
        qlen=q,
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=write_idx,
    )
    out = {k: _strided_copy(w[k]) for k in ("prefix", "blocks", "conv_state", "rstate")}
    out["core"] = torch.zeros(S, PROJ, dtype=bf, device="cuda")
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
        scratch=torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda"),
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
    assert _cos(out["core"], ref["core"]) > COS_MIN and errs["core"] < 3e-3, errs
    again = _k1(w, nb, write_idx)
    assert all(torch.equal(out[k], again[k]) for k in out)


@needs_gfx950
@pytest.mark.parametrize(
    "q, requests, nb",
    [(2, 1, 1), (2, 3, 8), (2, 4, 2), (4, 1, 3), (4, 2, 7), (8, 1, 8)],
)
def test_k1_verify_steps_match_vllm_ops(q: int, requests: int, nb: int) -> None:
    """Consecutive verify steps of whole requests at every width K1 serves, each
    request accepting a varying count a step: the states each step leaves feed
    the next (K1 and the reference each on their own), and the slots no request
    owns are never written."""
    gen = torch.Generator().manual_seed(q * 16 + requests)
    w = _kda_inputs([1] * requests, dim_first=False, q=q)
    owned = w["st_idx"].flatten().long()
    state = ref_state = {k: w[k] for k in ("conv_state", "rstate")}
    for step in range(K1_STEPS):
        nacc = torch.randint(1, q + 1, (requests,), generator=gen).tolist()
        x = _kda_inputs(nacc, dim_first=False, q=q, seed=1 + step)
        step_in = {k: x[k] for k in ("prefix", "delta", "blocks", "num_acc")}
        out = _k1({**w, **step_in, **state}, nb)
        ref = _kda_reference({**w, **step_in, **ref_state}, nb, -1)
        state = {k: out[k] for k in ("conv_state", "rstate")}
        ref_state = {k: ref[k] for k in ("conv_state", "rstate")}
        torch.testing.assert_close(out["prefix"], ref["prefix"], atol=0, rtol=0)
        errs = {
            "core": _rel_l2(out["core"], ref["core"]),
            "conv_state": _rel_l2(
                state["conv_state"][owned[::q]], ref_state["conv_state"][owned[::q]]
            ),
            "rstate": _rel_l2(state["rstate"][owned], ref_state["rstate"][owned]),
        }
        assert _cos(out["core"], ref["core"]) > COS_MIN, (step, errs)
        assert errs["core"] < 3e-3, (step, errs)
        assert errs["conv_state"] < 1e-3 and errs["rstate"] < 1e-3, (step, errs)
    free = torch.ones(w["rstate"].size(0), dtype=torch.bool)
    free[owned.cpu()] = False
    assert torch.equal(state["rstate"][free], w["rstate"][free])
    conv_free = torch.ones_like(free)
    conv_free[owned[::q].cpu()] = False
    assert torch.equal(state["conv_state"][conv_free], w["conv_state"][conv_free])


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


def _moe_weights(rank: int, device):
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import (
        HIDDEN,
        LAT,
        RI,
        SI,
        UP_N,
        E,
    )

    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device).to(torch.uint8)

    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w_up = _randn(gc, HIDDEN, LAT, scale=0.02)
    return {
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "bias": _randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": _randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + _randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(bf),
        "w_up_shard": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
        "w13": rocm_aiter_ops.shuffle_weight_a16w4(
            u8(gr, E, 2 * RI, LAT // 2).view(fp4), 16, False
        ),
        "w2": rocm_aiter_ops.shuffle_weight_a16w4(
            u8(gr, E, LAT, RI // 2).view(fp4), 16, False
        ),
        "w13s": rocm_aiter_ops.shuffle_scale_a16w4(
            u8(gr, E, 2 * RI, LAT // 32, lo=118, hi=123).view(e8).view(-1, LAT // 32),
            E,
            False,
        ),
        "w2s": e8m0_shuffle(
            u8(gr, E, LAT, RI // 32, lo=118, hi=123).view(e8).view(-1, RI // 32)
        ),
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


def _k2_inputs(rank: int, device):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HIDDEN,
        PROJ,
    )

    gc = torch.Generator(device=device).manual_seed(11)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
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

    key = k2.K2Build(tokens=8, nblocks=nb + int(reset), reset=reset, **runner.fault)
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
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import TOPK
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

    # hostile routing: every token on the same TOPK experts, then each token on
    # TOPK of its own (the largest union)
    same = dict(mw, bias=mw["bias"].clone())
    same["bias"][:TOPK] += 10.0
    own = 0.2 * x.float() / x.float().norm(dim=-1, keepdim=True)
    disjoint = dict(mw, w_gate=mw["w_gate"].clone(), bias=torch.zeros_like(mw["bias"]))
    disjoint["w_gate"][: S * TOPK] = own.repeat_interleave(TOPK, dim=0).to(bf)
    dist.broadcast(disjoint["w_gate"], src=0)
    for case, union in ((same, TOPK), (disjoint, S * TOPK)):
        runner.step_begin()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        key = _k2_launch(runner, w, case, w["prefix"].clone(), out, reset=False)
        torch.accelerator.synchronize()
        x = _k2_x(runner, key)
        sig = torch.sigmoid(x.float() @ case["w_gate"].float().t())
        picks = torch.topk(sig + case["bias"], TOPK, dim=-1).indices
        assert picks.unique().numel() == union
        ref = _moe_reference(x, case, rank)
        assert _cos(out, ref) > COS_MIN and _rel_l2(out, ref) < 6e-3
        _assert_ranks_agree(out, tp_size)

    # A graph of one step (the epoch bump captured with it), replayed: a
    # block-write launch writes its prefix, so every replay must match the first.
    prefix = torch.empty_like(w["prefix"])
    out = torch.empty(S, HIDDEN, dtype=bf, device=device)
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


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


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
