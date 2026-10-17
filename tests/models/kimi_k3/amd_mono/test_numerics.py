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
- M1a (AttnRes -> qkv_a / g_proj -> q / kv RMSNorm + the fp8 KV write -> q_b
  -> fp4 W_UK -> the DCP query gather) and M1b (the a2a LSE combine -> fp4
  W_UV -> the gate): one GPU, then TP8 + DCP8 with a graph of two MLA layers
  (M1a -> M1b -> K2) replayed.
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


def _kda_inputs(nb: int, nacc: int, dim_first: bool, S: int, L: int):
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import (
        HD,
        HIDDEN,
        NH,
        NPROJ,
        PROJ,
        QKV,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    R = S // L
    slots = S + 4
    sl = 3 + (L - 1)
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
    w["st_idx"] = torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(R, L)
    # the requests' accepted counts differ, from nacc on
    w["num_acc"] = torch.tensor(
        [1 + (nacc - 1 + r) % L for r in range(R)], dtype=torch.int32, device="cuda"
    )
    return w


def _kda_reference(w, nb: int, S: int, L: int, write_idx: int):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import HD, NH, PROJ, QKV
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
    qsl = torch.arange(0, S + 1, L, dtype=torch.int32, device="cuda")
    conv = causal_conv1d_update(
        mixed_qkv.contiguous(),
        conv_state,
        w["conv_w"],
        None,
        activation="silu",
        conv_state_indices=w["st_idx"][:, 0].contiguous(),
        num_accepted_tokens=w["num_acc"],
        query_start_loc=qsl,
        max_query_len=L,
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
        uniform_sequence_length=L,
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


# (rows, tokens a request): the widths of concurrency 1 .. 16 at 1 + 3 or 1 + 7
# speculative tokens
K1_SHAPES = [(8, 8), (16, 8), (16, 4), (16, 16), (4, 4), (32, 8), (64, 4), (64, 16)]


@needs_gfx950
@pytest.mark.parametrize("S, L", K1_SHAPES)
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("nacc", [1, 3])
@pytest.mark.parametrize("write", [False, True])
def test_k1_matches_vllm_ops(
    S: int, L: int, dim_first: bool, nacc: int, write: bool
) -> None:
    """K1 against attn_res -> in_proj -> f_b -> causal_conv1d_update ->
    fused_recurrent_kda -> FusedRMSNormGated on one spec-verify step (S / L
    requests of L tokens), including the state each step leaves behind."""
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import (
        PROJ,
        KdaPreBuild,
        kda_pre,
        scratch_bytes,
    )
    from vllm.models.kimi_k3.amd.ops.mono.common.plan import HEALTH_WORDS

    nb = 4
    # a block-write layer appends block nb (block_write_idx == prev_valid_blocks)
    write_idx = nb if write else -1
    w = _kda_inputs(nb, nacc, dim_first, S, L)
    ref = _kda_reference(w, nb, S, L, write_idx)

    key = KdaPreBuild(
        tokens=S,
        qlen=L,
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=write_idx,
    )
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.zeros(S, PROJ, dtype=bf, device="cuda")
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
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
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()

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


# ---------------------------------------------------------------- M1a (MLA)


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(bf)


def _mla_inputs(S: int, nb: int, rank: int = 0, ranks: int = 1):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import (
        ENTRY,
        GROWS,
        KVL,
        NH,
        NOPE,
        QB,
        QKV_A,
        QL,
    )

    g = torch.Generator(device="cuda").manual_seed(0)  # replicated
    gr = torch.Generator(device="cuda").manual_seed(300 + rank)  # TP shards
    w = {
        "prefix": _randn(g, S, HIDDEN, scale=2.0),
        "delta": _randn(g, S, HIDDEN, scale=0.5),
        "blocks": _randn(g, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(g, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "w_qkv": _randn(g, QKV_A, HIDDEN, scale=0.02),
        "w_g": _randn(gr, GROWS, HIDDEN, scale=0.02),
        "q_nw": (1 + _randn(g, QL, scale=0.1, dtype=torch.float32)).to(bf),
        "kv_nw": (1 + _randn(g, KVL, scale=0.1, dtype=torch.float32)).to(bf),
        "w_qb": _randn(gr, QB, QL, scale=0.03),
    }
    # W_UK as vLLM's MLA keeps it: rows (n, h) quantized by aiter, the scales
    # column-major: W_K (12, 512, 64) strides (64, 768, 1), its scales (1, 12, 6144)
    uk = _randn(gr, KVL * NH, NOPE, scale=0.1)
    wq, ws = dynamic_mxfp4_quant(uk)
    w["w_k"] = wq.view(torch.uint8).unflatten(0, (KVL, NH)).transpose(0, 1)
    w["w_k_scale"] = ws.view(torch.uint8).unflatten(0, (KVL, NH)).transpose(0, 1)
    # hybrid page alignment: entries dense, blocks strided; a token another
    # DCP rank's (slot -1): every other one, or t % ranks != rank
    bs, nblk = 16, 2 * S // 16 + 2
    # finite E4M3 bytes (0x7f is NaN)
    pages = torch.randint(0, 0x70, (nblk, bs * ENTRY + 512), dtype=torch.uint8,
                          device="cuda", generator=gr)
    w["kv_cache"] = pages[:, : bs * ENTRY].view(nblk, bs, ENTRY)
    perm = torch.randperm(nblk * bs, device="cuda", generator=gr)[:S]
    if ranks == 1:
        perm[1::2] = -1
    else:
        perm[torch.arange(S, device="cuda") % ranks != rank] = -1
    w["slots"] = perm.to(torch.int64)
    return w


def _mla_reference(w, nb: int):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import KVL, NH, NOPE, QL, ROPE
    from vllm.platforms import current_platform

    fp8 = current_platform.fp8_dtype()
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    x = attn_res(prefix, w["delta"], blocks, w["ares_nw"], w["ares_qk"],
                 w["in_nw"], nb, -1, 1e-5, 1e-5)
    qkv = F.linear(x, w["w_qkv"])
    q_c, kv_c, k_pe = qkv.split([QL, KVL, ROPE], -1)
    q_c, kv_c = _rms(q_c, w["q_nw"], 1e-6), _rms(kv_c, w["kv_nw"], 1e-6)
    q = F.linear(q_c, w["w_qb"]).view(-1, NH, NOPE + ROPE)
    q_nope, q_pe = q.split([NOPE, ROPE], -1)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    ql = rocm_aiter_ops.batched_gemm_a16wfp4(
        q_nope.transpose(0, 1), w["w_k"], w["w_k_scale"],
        transpose_bm=True, prequant=True, y_scale=one,
    )
    q8 = torch.cat([ql, q_pe], -1).float().clamp(-448, 448).to(fp8)
    cache = w["kv_cache"].clone()
    entry = torch.cat([kv_c, k_pe], -1).float().clamp(-448, 448).to(fp8)
    flat = cache.view(-1, cache.size(1), cache.size(2))
    for t, s in enumerate(w["slots"].tolist()):
        if s >= 0:
            flat[s // cache.size(1), s % cache.size(1)] = entry[t].view(torch.uint8)
    gate = F.linear(x, w["w_g"]).sigmoid()
    return {"prefix": prefix, "blocks": blocks, "q": q8, "cache": cache, "gate": gate}


def _fp8_f(t):
    from vllm.platforms import current_platform

    return t.view(current_platform.fp8_dtype()).float()


@needs_gfx950
@pytest.mark.parametrize("S", [4, 8, 16, 32, 64])
def test_m1a_single_gpu_matches_vllm_ops(S: int) -> None:
    """M1a at DCP1 (ranks = 1: no gather, no peer memory) against attn_res ->
    qkv_a -> q / kv RMSNorm -> q_b -> aiter's fp4 W_UK -> the fp8 query, the
    fp8 cache rows and the gate."""
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import (
        ENTRY,
        GROWS,
        KVL,
        NH,
        MlaPreBuild,
        mla_pre,
        scratch_bytes,
    )
    from vllm.models.kimi_k3.amd.ops.mono.common.plan import HEALTH_WORDS

    nb = 4
    w = _mla_inputs(S, nb)
    ref = _mla_reference(w, nb)
    key = MlaPreBuild(tokens=S, nblocks=nb, delta=True, ranks=1)
    prefix, blocks, cache = w["prefix"].clone(), w["blocks"].clone(), w["kv_cache"].clone()
    q = torch.zeros(S, NH, ENTRY, dtype=torch.uint8, device="cuda")
    gate = torch.zeros(S, GROWS, dtype=bf, device="cuda")
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    mla_pre(
        key,
        prefix=prefix,
        delta=w["delta"],
        blocks=blocks,
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_qkv=w["w_qkv"],
        w_g=w["w_g"],
        q_nw=w["q_nw"],
        kv_nw=w["kv_nw"],
        w_qb=w["w_qb"],
        w_k=w["w_k"],
        w_k_scale=w["w_k_scale"],
        kv_cache=cache,
        slots=w["slots"],
        gate_out=gate,
        q_out=q,
        scratch=torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda"),
        peers=None,
        rank=0,
        layer=3,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()

    torch.testing.assert_close(prefix, ref["prefix"], atol=0, rtol=0)
    torch.testing.assert_close(blocks, ref["blocks"], atol=0, rtol=0)
    # GEMV sums differ from hipBLASLt's / aiter's in order only: an fp8 or bf16
    # value now and then lands one step off
    errs = {
        "gate": _rel_l2(gate, ref["gate"]),
        "q_nope": _rel_l2(_fp8_f(q[..., :KVL]), ref["q"][..., :KVL].float()),
        "q_pe": _rel_l2(_fp8_f(q[..., KVL:]), ref["q"][..., KVL:].float()),
        "cache": _rel_l2(_fp8_f(cache), _fp8_f(ref["cache"])),
        "q_same": (q == ref["q"].view(torch.uint8)).float().mean().item(),
        "cache_same": (cache == ref["cache"]).float().mean().item(),
    }
    assert errs["gate"] < 1e-2 and errs["cache"] < 2e-2, errs
    assert errs["q_nope"] < 3e-2 and errs["q_pe"] < 3e-2, errs
    assert _cos(_fp8_f(q), ref["q"].float()) > 0.999, errs
    # rows of unstored tokens (slot -1) untouched
    assert errs["cache_same"] > 0.99, errs


def _a2a_combine(o, lse, seq_lens, qsl, q: int):
    """``_dcp_a2a_pack_send`` + ``_dcp_a2a_unpack_combine`` for one rank's heads,
    every rank's partials stacked: o [R, S, 12, 512], lse [R, S, 12], each
    sender's local KV lengths seq_lens [R, requests]."""
    S = o.size(1)
    t = torch.arange(S, device=o.device)
    empty = (seq_lens[:, t // q] == 0) | (t >= qsl[S // q])
    lse = lse.masked_fill(empty.unsqueeze(-1), float("-inf"))
    lse = torch.where(lse.isnan() | (lse == float("inf")), float("-inf"), lse)
    m = lse.max(0).values
    m = torch.where(m == float("-inf"), 0.0, m)
    glse = (lse - m).exp().sum(0).log() + m
    w = (lse - glse).exp().nan_to_num(0.0)
    p = torch.where(w.unsqueeze(-1) == 0, 0.0, o.float())
    return (p * w.unsqueeze(-1)).sum(0).to(bf)


def _mla_back_inputs(S: int, q: int, R: int, rank: int = 0):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import GROWS, KVL, NH

    g = torch.Generator(device="cuda").manual_seed(1 + rank)
    nreq = S // q
    w = {
        # AITER's padded 128-head outputs, viewed to the gathered heads
        "o": _randn(g, S, 128, KVL, scale=0.5)[:, : R * NH],
        "lse": (torch.randn(S, 128, generator=g, device="cuda") * 3)[:, : R * NH],
        "gate": torch.rand(S, GROWS, generator=g, device="cuda").to(bf),
        # the first request with no local KV (when there are others)
        "seq_lens": torch.randint(1, 900, (nreq,), dtype=torch.int32,
                                  device="cuda", generator=g),
        "qsl": torch.arange(0, S + 1, q, dtype=torch.int32, device="cuda"),
    }
    # a request this rank holds none of, and with ranks, one no rank holds
    if nreq > 1:
        w["seq_lens"][rank % nreq] = 0
    if R > 1 and nreq > 2:
        w["seq_lens"][-1] = 0
    uv = _randn(g, 128 * NH, KVL, scale=0.05)
    wq, ws = dynamic_mxfp4_quant(uv)
    # (12, 128, 256) dense rows a head; the scales column-major as vLLM keeps them
    w["w_v"] = wq.view(torch.uint8).view(NH, 128, KVL // 2)
    w["w_v_scale"] = ws.view(torch.uint8).unflatten(0, (NH, 128))
    return w


def _uv_gate(lat, w):
    """vLLM's ``_v_up_proj`` (aiter's fp4 W_UV) times the gate."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import NH

    S = lat.size(0)
    out = torch.empty(S, NH, 128, dtype=bf, device=lat.device)
    rocm_aiter_ops.batched_gemm_a16wfp4(
        lat.transpose(0, 1), w["w_v"], w["w_v_scale"], out,
        transpose_bm=True, prequant=True, y_scale=None,
    )
    return out.view(S, -1) * w["gate"]


def _mla_back_reference(w):
    """DCP1: AITER's output is the latent (vLLM's ``_v_up_proj`` input)."""
    return _uv_gate(w["o"], w)


@needs_gfx950
@pytest.mark.parametrize("S, q", [(4, 4), (8, 4), (16, 4), (32, 4), (64, 4), (64, 8)])
def test_m1b_single_gpu_matches_vllm_ops(S: int, q: int) -> None:
    """M1b at DCP1 (ranks = 1: no LSE, no peer memory): AITER's strided
    output -> aiter's fp4 W_UV -> the gate."""
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import GROWS
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla_back import (
        MlaPostBuild,
        mla_post,
    )
    from vllm.models.kimi_k3.amd.ops.mono.common.plan import HEALTH_WORDS

    w = _mla_back_inputs(S, q, 1)
    ref = _mla_back_reference(w)
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    core = torch.zeros(S, GROWS, dtype=bf, device="cuda")
    mla_post(
        MlaPostBuild(tokens=S, qlen=q, ranks=1),
        o=w["o"],
        lse=None,
        seq_lens=w["seq_lens"],
        qsl=w["qsl"],
        w_v=w["w_v"],
        w_v_scale=w["w_v_scale"],
        gate=w["gate"],
        core=core,
        peers=None,
        rank=0,
        layer=5,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    err = _rel_l2(core, ref)
    assert err < 1e-2 and _cos(core, ref) > COS_MIN, err


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

    from vllm.models.kimi_k3.amd.ops.mono.stages.moe import SI, TOPK, UP_N

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
    from vllm.models.kimi_k3.amd.ops.mono.stages.moe import HIDDEN, LAT, RI, SI, UP_N, E

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


def _k2_inputs(rank: int, device, S: int = 8):
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import HIDDEN, PROJ

    gc = torch.Generator(device=device).manual_seed(11)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    return {
        "core": _randn(gr, S, PROJ),
        "w_o": _randn(gr, HIDDEN, PROJ, scale=0.02),
        "prefix": _randn(gc, S, HIDDEN, scale=2.0),
        "blocks": _randn(gc, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(gc, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
    }


def _k2_launch(runner, w, mw, prefix, out, reset: bool, nb: int = 4, layer: int = 3):
    from vllm.models.kimi_k3.amd.ops.mono import layer as k2

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
        layer=128 + layer,
        **_moe_args(mw),
        **runner.launch_args(),
        **runner.peer_args(),
    )
    return key


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

    from vllm.models.kimi_k3.amd.mono.runner import MonoRunner
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm.models.kimi_k3.amd.ops.mono import layer as k2
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import HIDDEN

    nb = 4
    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)
    for S, reset in K2_CASES:
        w = _k2_inputs(rank, device, S)
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
        lay = k2.scratch_layout(key)["xrow"][0]
        x = runner.scratch[lay : lay + S * HIDDEN * 2].view(bf).view(S, HIDDEN)
        cos_x = _cos(x, x_ref)
        assert cos_x > COS_MIN, (S, reset, cos_x)
        cos_out = _cos(out, _moe_reference(x, mw, rank))
        assert cos_out > COS_MIN, (S, reset, cos_out)
        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"

    # A graph of one step (the epoch bump captured with it), replayed: a
    # block-write launch writes its prefix, so every replay must match the first.
    S = K2_CASES[-1][0]
    w = _k2_inputs(rank, device, S)
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


# (rows, block-write layer): every width up to the runner's MAX_TOKENS
K2_CASES = [
    (8, False),
    (8, True),
    (4, False),
    (16, False),
    (16, True),
    (32, False),
    (64, False),
    (64, True),
]


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


def _gather(t: torch.Tensor) -> torch.Tensor:
    """Every rank's ``t`` stacked [ranks, ...]."""
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.stack(out)


@ray.remote(num_gpus=1, max_calls=1)
def _m1_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.mono.runner import MonoRunner
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import (
        ENTRY,
        GROWS,
        NH,
        MlaPreBuild,
        mla_pre,
    )
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla_back import (
        MlaPostBuild,
        mla_post,
    )

    R, nb = tp_size, 4
    # DCP8 over the TP group; the test's distributed init sets up no DCP group
    runner = MonoRunner(device, dcp=R)

    def pre(w, prefix, blocks, cache, q8, gate, layer=3):
        S = prefix.size(0)
        key = MlaPreBuild(
            tokens=S, nblocks=nb, delta=True, ranks=R, peer_off=runner.mla_peer_off
        )
        mla_pre(
            key,
            prefix=prefix,
            delta=w["delta"],
            blocks=blocks,
            ares_nw=w["ares_nw"],
            ares_qk=w["ares_qk"],
            in_nw=w["in_nw"],
            w_qkv=w["w_qkv"],
            w_g=w["w_g"],
            q_nw=w["q_nw"],
            kv_nw=w["kv_nw"],
            w_qb=w["w_qb"],
            w_k=w["w_k"],
            w_k_scale=w["w_k_scale"],
            kv_cache=cache,
            slots=w["slots"],
            gate_out=gate,
            q_out=q8,
            layer=layer,
            **runner.launch_args(),
            **runner.peer_args(),
        )

    def post(b, q, gate, core, layer=3):
        key = MlaPostBuild(
            tokens=core.size(0), qlen=q, ranks=R, peer_off=runner.mla_post_off
        )
        mla_post(
            key,
            o=b["o"],
            lse=b["lse"],
            seq_lens=b["seq_lens"],
            qsl=b["qsl"],
            w_v=b["w_v"],
            w_v_scale=b["w_v_scale"],
            gate=gate,
            core=core,
            layer=layer,
            **runner.launch_args(),
            **runner.peer_args(),
        )

    # M1a: this rank's query rows land at heads 12 * rank.. on every rank
    for S in (4, 8, 16, 32, 64):
        w = _mla_inputs(S, nb, rank, R)
        ref = _mla_reference(w, nb)
        runner.step_begin()
        prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
        cache = w["kv_cache"].clone()
        q8 = torch.zeros(S, R * NH, ENTRY, dtype=torch.uint8, device=device)
        gate = torch.zeros(S, GROWS, dtype=bf, device=device)
        pre(w, prefix, blocks, cache, q8, gate)
        torch.accelerator.synchronize()
        assert not runner.health.any() and not runner.flags.any(), S
        torch.testing.assert_close(prefix, ref["prefix"], atol=0, rtol=0)
        want = _gather(ref["q"].view(torch.uint8)).transpose(0, 1).reshape_as(q8)
        errs = {
            "q": _rel_l2(_fp8_f(q8), _fp8_f(want)),
            "gate": _rel_l2(gate, ref["gate"]),
            "cache_same": (cache == ref["cache"]).float().mean().item(),
        }
        assert errs["q"] < 3e-2 and errs["gate"] < 1e-2, (S, errs)
        assert _cos(_fp8_f(q8), _fp8_f(want)) > 0.999, (S, errs)
        assert errs["cache_same"] > 0.99, (S, errs)

    # M1b: every sender's partials for this rank's heads, each with its own
    # empty-shard mask, combined in rank order
    for S, q in ((4, 4), (16, 4), (64, 4), (64, 8)):
        b = _mla_back_inputs(S, q, R, rank)
        runner.step_begin()
        core = torch.zeros(S, GROWS, dtype=bf, device=device)
        post(b, q, b["gate"], core)
        torch.accelerator.synchronize()
        assert not runner.health.any() and not runner.flags.any(), (S, q)
        heads = slice(NH * rank, NH * (rank + 1))
        lat = _a2a_combine(
            _gather(b["o"])[:, :, heads],
            _gather(b["lse"])[:, :, heads],
            _gather(b["seq_lens"]),
            b["qsl"],
            q,
        )
        ref = _uv_gate(lat, b)
        if S // q > 2:  # the last request no rank holds
            assert not core[-q:].any(), (S, q)
        err = _rel_l2(core, ref)
        assert err < 1e-2 and _cos(core, ref) > COS_MIN, (S, q, err)

    # A graph of one step, two MLA layers of M1a -> M1b -> K2 (each layer its
    # own tags, the peer regions reused), replayed bit-identical
    S, q = 64, 4
    w = _mla_inputs(S, nb, rank, R)
    b = _mla_back_inputs(S, q, R, rank)
    k = _k2_inputs(rank, device, S)
    mw = _moe_weights(rank, device)
    prefix, blocks = torch.empty_like(w["prefix"]), torch.empty_like(w["blocks"])
    cache = w["kv_cache"].clone()
    q8 = torch.zeros(S, R * NH, ENTRY, dtype=torch.uint8, device=device)
    gate = torch.zeros(S, GROWS, dtype=bf, device=device)
    core = torch.zeros(S, GROWS, dtype=bf, device=device)
    out = torch.empty(S, HIDDEN, dtype=bf, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.step_begin()
        prefix.copy_(w["prefix"])
        blocks.copy_(w["blocks"])
        for layer in (3, 5):
            pre(w, prefix, blocks, cache, q8, gate, layer)
            post(b, q, gate, core, layer)
            k2w = dict(k, core=core, blocks=blocks)
            _k2_launch(runner, k2w, mw, prefix, out, False, layer=layer)
    graph.replay()
    torch.accelerator.synchronize()
    want = [t.clone() for t in (prefix, q8, core, out)]
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    for got, ref in zip((prefix, q8, core, out), want):
        assert torch.equal(got, ref)
    assert not runner.health.any() and not runner.flags.any()


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_m1_tp8_dcp8_matches_vllm_ops_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _m1_worker)


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

    from vllm.models.kimi_k3.amd.mono.runner import MonoRunner
    from vllm.models.kimi_k3.amd.ops.mono import layer as k2
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.ops.mono.common.debug import given_up_waits
    from vllm.models.kimi_k3.amd.ops.mono.common.sync import WAIT_TICKS

    runner = MonoRunner(device)
    # rank STALLED's CTA 0 stalls past every wait's bound in the epoch 2 launch
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_layer": 128 + 3,  # K2's slot of layer 3 (_k2_launch's default)
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
def test_warmup_compiles_without_launching() -> None:
    """The JIT warm-up compiles K1, K2, M1a and M1b before capture without running them:
    a launch on its null arguments would fault."""
    from vllm.models.kimi_k3.amd.mono.kernel import compile_build
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import KdaPreBuild
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla import MlaPreBuild
    from vllm.models.kimi_k3.amd.ops.mono.attention.mla_back import MlaPostBuild
    from vllm.models.kimi_k3.amd.ops.mono.layer import K2Build

    for s, q in ((8, 8), (16, 4), (64, 16)):
        compile_build(
            KdaPreBuild(tokens=s, qlen=q, nblocks=4, delta=True, state_len=3 + q - 1)
        )
        compile_build(K2Build(tokens=s, nblocks=4))
        compile_build(MlaPreBuild(tokens=s, nblocks=4, delta=True))
        compile_build(MlaPostBuild(tokens=s, qlen=q))
    torch.accelerator.synchronize()
