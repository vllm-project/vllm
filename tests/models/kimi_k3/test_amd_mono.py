# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Kimi-K3 mono decode kernels (``VLLM_ROCM_MONO_DECODE``) on
MI355X and MI300X / MI325X against the ops vLLM runs for the same spec-verify
step, random weights in the formats vLLM holds at TP8 shapes (the routed
experts MXFP4 on gfx950, vLLM's packed-int4 requant on gfx942).

- K1 (one GPU): AttnRes -> in_proj -> f_b -> conv -> KDA recurrence -> gated
  norm, the conv / SSM state updates and a block-write layer's block store.
- K2 and the standalone MoE launch (eight GPUs): o_proj + all-reduce + the MLP
  AttnRes + the latent MoE. The MoE is compared on the kernel's own MLP input,
  where routing cannot differ.
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


def _on_arch(name: str) -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms import rocm

    return getattr(rocm, f"on_{name}")()


pytestmark = pytest.mark.skipif(
    not (_on_arch("gfx950") or _on_arch("gfx942")),
    reason="the K3 mono kernels need gfx950 (MI355X) or gfx942 (MI300X / MI325X)",
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


def _kda_inputs(nb: int, nacc: int, dim_first: bool, S: int, slots: int = 16):
    from vllm.models.kimi_k3.amd.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        NPROJ,
        PROJ,
        QKV,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    sl = 3 + (S - 1)
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
    w["st_idx"] = torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(1, S)
    w["num_acc"] = torch.full((1,), nacc, dtype=torch.int32, device="cuda")
    return w


def _kda_reference(w, nb: int, S: int, write_idx: int):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.mono.attention.kda import HD, NH, PROJ, QKV
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

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
    qsl = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    conv = causal_conv1d_update(
        mixed_qkv.contiguous(),
        conv_state,
        w["conv_w"],
        None,
        activation="silu",
        conv_state_indices=w["st_idx"][:, 0].contiguous(),
        num_accepted_tokens=w["num_acc"],
        query_start_loc=qsl,
        max_query_len=S,
        validate_data=False,
        out=torch.empty_like(mixed_qkv),
    )
    q, k, v = (
        rearrange(t, "n (h d) -> 1 n h d", d=HD) for t in conv.split(PROJ, dim=-1)
    )
    core = torch.empty(1, S, NH, HD, dtype=bf, device="cuda")
    fused_recurrent_kda(
        q=q,
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
        uniform_sequence_length=S,
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


@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("nacc", [1, 3])
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("S", [8, 3])
def test_k1_matches_vllm_ops(dim_first: bool, nacc: int, write: bool, S: int) -> None:
    """K1 against attn_res -> in_proj -> f_b -> causal_conv1d_update ->
    fused_recurrent_kda -> FusedRMSNormGated on one spec-verify step (one
    request, S tokens), including the state each step leaves behind."""
    from vllm.models.kimi_k3.amd.mono.attention.kda import (
        PROJ,
        KdaPreBuild,
        kda_pre,
        scratch_bytes,
    )

    nb = 4
    # a block-write layer appends block nb (block_write_idx == prev_valid_blocks)
    write_idx = nb if write else -1
    w = _kda_inputs(nb, nacc, dim_first, S)
    ref = _kda_reference(w, nb, S, write_idx)

    key = KdaPreBuild(
        tokens=S,
        qlen=S,
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=write_idx,
    )
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.zeros(S, PROJ, dtype=bf, device="cuda")
    kda_pre(
        key,
        prefix=prefix,
        delta=w["delta"],
        blocks=blocks,
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_in=w["w_in"],
        w_fb=w["w_fb"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        on_w=w["on_w"],
        rstate=rstate,
        st_idx=w["st_idx"],
        num_acc=w["num_acc"],
        core_out=core,
        scratch=torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda"),
        layer=0,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(prefix, ref["prefix"], atol=0, rtol=0)
    torch.testing.assert_close(blocks, ref["blocks"], atol=0, rtol=0)
    # the conv output feeds the recurrence in fp32 here, bf16 in the reference
    errs = {
        "conv_state": _rel_l2(conv_state, ref["conv_state"]),
        "rstate": _rel_l2(rstate, ref["rstate"]),
        "core": _rel_l2(core, ref["core"]),
    }
    assert errs["conv_state"] < 1e-3 and errs["rstate"] < 1e-3, errs
    assert _cos(core, ref["core"]) > COS_MIN, errs


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
    down, aiter's experts (MXFP4, or a16wi4 for i4x2 weights), the latent
    all-reduce + RMSNorm, this rank's up_proj rows into the shared experts'
    output, the final all-reduce."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from vllm.models.kimi_k3.amd.mono.stages.moe import SI, TOPK, UP_N

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


def _int4_experts(g, e: int, n: int, k: int, scale: float):
    """Random experts through vLLM's gfx942 requant (``mxfp4.py``):
    per_1x32_i4_quant, shuffle_weight(16, 16), pack_int8_to_packed_int4 and
    shuffle_scale_for_int4, a chunk of experts at a time."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_i4_quant
    from aiter.ops.shuffle import (
        pack_int8_to_packed_int4,
        shuffle_scale_for_int4,
        shuffle_weight,
    )

    chunk = 64
    w = torch.empty(e, n, k // 2, dtype=torch.uint8, device=g.device)
    s = torch.empty(e * n * (k // 32), dtype=bf, device=g.device)
    for lo in range(0, e, chunk):
        q, sc = per_1x32_i4_quant(_randn(g, chunk, n, k, scale=scale))
        q = q.view(dtypes.i4x2).view(chunk, n, k)
        w[lo : lo + chunk] = pack_int8_to_packed_int4(
            shuffle_weight(q.view(dtypes.i8), (16, 16))
        ).view(chunk, n, k // 2)
        sc = shuffle_scale_for_int4(sc, group_size=32).view(-1)
        s[lo * n * (k // 32) : (lo + chunk) * n * (k // 32)] = sc
    w = w.view(dtypes.i4x2)
    w.is_shuffled = True
    return w, s


def _moe_weights(rank: int, device):
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.mono.stages.moe import HIDDEN, LAT, RI, SI, UP_N, E

    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device).to(torch.uint8)

    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w_up = _randn(gc, HIDDEN, LAT, scale=0.02)
    mw = {
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "bias": _randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": _randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + _randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(bf),
        "w_up_shard": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
    }
    if _on_arch("gfx942"):
        mw["w13"], mw["w13s"] = _int4_experts(gr, E, 2 * RI, LAT, 0.03)
        mw["w2"], mw["w2s"] = _int4_experts(gr, E, LAT, RI, 0.05)
        return mw
    mw["w13"] = rocm_aiter_ops.shuffle_weight_a16w4(
        u8(gr, E, 2 * RI, LAT // 2).view(fp4), 16, False
    )
    mw["w2"] = rocm_aiter_ops.shuffle_weight_a16w4(
        u8(gr, E, LAT, RI // 2).view(fp4), 16, False
    )
    mw["w13s"] = rocm_aiter_ops.shuffle_scale_a16w4(
        u8(gr, E, 2 * RI, LAT // 32, lo=118, hi=123).view(e8).view(-1, LAT // 32),
        E,
        False,
    )
    mw["w2s"] = e8m0_shuffle(
        u8(gr, E, LAT, RI // 32, lo=118, hi=123).view(e8).view(-1, RI // 32)
    )
    return mw


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
    from vllm.models.kimi_k3.amd.mono import layer as k2
    from vllm.models.kimi_k3.amd.mono.attention.kda import HIDDEN, PROJ
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.kimi_k3.amd.mono.stages import moe
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    S, nb = 8, 4
    tp = get_tp_group()
    peers = PeerBuffer(
        max(k2.peer_bytes(S), moe.peer_bytes(S)), tp.cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    assert moe.experts_ok(mw["w13"], mw["w13s"], mw["w2"], mw["w2s"])
    gc = torch.Generator(device=device).manual_seed(11)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    core = _randn(gr, S, PROJ)
    w_o = _randn(gr, HIDDEN, PROJ, scale=0.02)
    prefix0 = _randn(gc, S, HIDDEN, scale=2.0)
    blocks = _randn(gc, S, 8, HIDDEN, scale=2.0)
    ares_nw = (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf)
    ares_qk = _randn(gc, HIDDEN, scale=0.05)
    in_nw = (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf)

    for reset in (False, True):
        epoch.add_(1)
        key = k2.K2Build(tokens=S, nblocks=nb + int(reset), reset=reset)
        scratch = torch.zeros(k2.scratch_bytes(key), dtype=torch.uint8, device=device)
        prefix = prefix0.clone()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        k2.k2_launch(
            key,
            core=core,
            w_o=w_o,
            prefix=prefix,
            blocks=blocks,
            ares_nw=ares_nw,
            ares_qk=ares_qk,
            in_nw=in_nw,
            out=out,
            scratch=scratch,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=3,
            **_moe_args(mw),
        )
        torch.accelerator.synchronize()

        attn = _all_reduce_fp32(F.linear(core, w_o))
        # a block-write layer's prefix restarts at the attention output
        ref_prefix = attn if reset else prefix0.clone()
        delta = None if reset else attn
        x_ref = attn_res(
            ref_prefix,
            delta,
            blocks,
            ares_nw,
            ares_qk,
            in_nw,
            nb + int(reset),
            -1,
            1e-5,
            1e-5,
        )
        # the kernel sums the ranks in rank order, NCCL in its own order
        assert _rel_l2(prefix, ref_prefix) < 1e-2
        lay = k2.scratch_layout(key)["xrow"][0]
        x = scratch[lay : lay + S * HIDDEN * 2].view(bf).view(S, HIDDEN)
        assert _cos(x, x_ref) > COS_MIN
        assert _cos(out, _moe_reference(x, mw, rank)) > COS_MIN
        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"

    # the standalone MoE launch (a layer K2 does not serve); fewer rows take
    # other LDS splits of the experts' stages on gfx942
    for s in (S, 5, 1):
        epoch.add_(1)
        x = _randn(gc, s, HIDDEN)
        out = torch.empty_like(x)
        queue = torch.zeros(moe.QUEUE_BYTES // 4, dtype=torch.int32, device=device)
        key = moe.MoeBuild(tokens=s)
        moe.moe(
            key,
            x=x,
            out=out,
            scratch=torch.zeros(
                moe.scratch_bytes(key), dtype=torch.uint8, device=device
            ),
            queue=queue,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=128 + 3,
            **_moe_args(mw),
        )
        torch.accelerator.synchronize()
        assert _cos(out, _moe_reference(x, mw, rank)) > COS_MIN, s
    peers.close()


@multi_gpu_test(num_gpus=8)
def test_k2_and_moe_tp8_match_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)
