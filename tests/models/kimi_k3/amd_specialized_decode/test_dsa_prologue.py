# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1's DSA prologue as one launch (D1a, ``dsa_pre``,
``VLLM_KIMI_SD_DSA_PROLOGUE``) against the vLLM ops ``runtime.dsa.prologue``
runs, on one MI355X at TP8 + DCP1 shapes (12 heads a rank, 656 B fp8_ds_mla
records, the MXFP4 indexer's 32 x 128 query).

- P1: the fp8_ds_mla record, bit for bit: vLLM's ``concat_and_cache_mla``
  ("fp8_ds_mla") on D1a's own normed kv_c and k_pe writes the same cache bytes
  (padding slots untouched), and on scales at power-of-two boundaries.
- P2: AttnRes (prefix / blocks) bit-exact; every output (the bf16 query, the
  gate, the indexer's wq / LayerNorm(wk) / wv) against the vLLM ops on D1a's
  own intermediates (x, the normed q_c), then end to end from the inputs: GEMV
  sums differ from hipBLASLt's in order only.
- P3: a CUDA graph (the epoch bump + D1a) replayed over new inputs: every
  replay bit-equal to the eager launch on the same inputs.
"""

import pytest
import torch
import torch.nn.functional as F

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
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(bf)


def _inputs(S: int, nb: int, seed: int = 0):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre as d
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = {
        "prefix": _randn(g, S, HIDDEN, scale=2.0),
        "delta": _randn(g, S, HIDDEN, scale=0.5),
        "blocks": _randn(g, S, 8, HIDDEN, scale=2.0),
        "ares_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "ares_qk": _randn(g, HIDDEN, scale=0.05),
        "in_nw": (1 + _randn(g, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf),
        "w_qkv": _randn(g, d.QKV_A, HIDDEN, scale=0.02),
        "w_g": _randn(g, d.GROWS, HIDDEN, scale=0.02),
        "w_ik": _randn(g, d.IKD, HIDDEN, scale=0.02),
        "w_iv": _randn(g, d.IVN, HIDDEN, scale=0.02),
        "q_nw": (1 + _randn(g, d.QL, scale=0.1, dtype=torch.float32)).to(bf),
        "kv_nw": (1 + _randn(g, d.KVL, scale=0.1, dtype=torch.float32)).to(bf),
        "ik_nw": (1 + _randn(g, d.IKD, scale=0.1, dtype=torch.float32)).to(bf),
        "ik_nb": _randn(g, d.IKD, scale=0.1),
        "w_qb": _randn(g, d.NH * d.QHD, d.QL, scale=0.03),
        "w_iq": _randn(g, d.IQN, d.QL, scale=0.03),
    }
    # W_UK_T as vLLM's MLA keeps it: kv_b_proj's [12 x 256, 512] weight ->
    # [512, 12, 256] -> the nope half permuted to [12, 128, 512], latent
    # contiguous (strides (131072, 512, 1))
    kv_b = _randn(g, d.NH * 256, d.KVL, scale=0.05)
    w["w_uk"] = kv_b.T.view(d.KVL, d.NH, 256)[..., : d.NOPE].permute(1, 2, 0)
    # fp8_ds_mla pages, blocks strided (the merged indexer pages beside them);
    # random records; a padding token (slot -1) every third
    bs, nblk = 16, 2 * S // 16 + 2
    pages = torch.randint(
        0, 0x70, (nblk, bs * d.REC + 512), dtype=torch.uint8, device="cuda",
        generator=g,
    )  # fmt: skip
    w["kv_cache"] = pages[:, : bs * d.REC].view(nblk, bs, d.REC)
    perm = torch.randperm(nblk * bs, device="cuda", generator=g)[:S]
    perm[2::3] = -1
    w["slots"] = perm.to(torch.int64)
    return w


def _launch(key, w, scratch, epoch, health, flags):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre as d

    S = key.tokens
    out = {
        "q": torch.zeros(S, d.NH, d.ENTRY, dtype=bf, device="cuda"),
        "gate": torch.zeros(S, d.GROWS, dtype=bf, device="cuda"),
        "iq": torch.zeros(S, d.IQN, dtype=bf, device="cuda"),
        "ik": torch.zeros(S, d.IKD, dtype=bf, device="cuda"),
        "iv": torch.zeros(S, d.IVN, dtype=bf, device="cuda"),
    }
    names = (
        ["prefix", "delta", "blocks", "ares_nw", "ares_qk", "in_nw", "w_qkv"]
        + ["w_g", "w_ik", "w_iv", "q_nw", "kv_nw", "ik_nw", "ik_nb", "w_qb"]
        + ["w_iq", "w_uk", "kv_cache", "slots"]
    )
    d.dsa_pre(
        key,
        **{n: w[n] for n in names},
        gate_out=out["gate"],
        q_out=out["q"],
        iq_out=out["iq"],
        ik_out=out["ik"],
        iv_out=out["iv"],
        scratch=scratch,
        layer=3,
        epoch=epoch,
        health=health,
        alarm=flags.data_ptr(),
    )
    return out


def _scratch_rows(key, scratch):
    """D1a's intermediates from its scratch: x [S, 7168], the normed q_c
    [S, 1536] and kv_c [S, 512] (plain bf16), qkv_a's out [S, 2112] (pairs)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre as d
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN

    S = key.tokens
    lay = d.scratch_layout(key)

    def plain(name, cols):
        o, n = lay[name]
        return scratch[o : o + n].view(bf).view(S, cols)

    o, n = lay["qa"]
    qa = scratch[o : o + n].view(torch.int32).view(-1, 2)[:, 0].contiguous()
    return {
        "x": plain("x", HIDDEN),
        "qc": plain("qc", d.QL),
        "kvn": plain("kvn", d.KVL),
        "qa": qa.view(bf).view(S, d.QKV_A),
    }


def _attn_res(w, nb):
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    x = attn_res(
        prefix, w["delta"], blocks, w["ares_nw"], w["ares_qk"], w["in_nw"], nb, -1,
        1e-5, 1e-5,
    )  # fmt: skip
    return prefix, blocks, x


def _ops_from_x(w, x, qc=None, eps=1e-5):
    """The vLLM ops of ``runtime.dsa.prologue`` from x (and, given, the normed
    q_c, so the downstream compare is order-only)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre as d

    qkv = F.linear(x, w["w_qkv"])
    q_c, kv_c, k_pe = qkv.split([d.QL, d.KVL, d.ROPE], -1)
    if qc is None:
        qc = _rms(q_c, w["q_nw"], eps)
    q = F.linear(qc, w["w_qb"]).view(-1, d.NH, d.QHD)
    nope, rope = q.split([d.NOPE, d.ROPE], -1)
    lat = torch.bmm(nope.transpose(0, 1), w["w_uk"]).transpose(0, 1)
    return {
        "qkv": qkv,
        "qc": qc,
        "kvn": _rms(kv_c, w["kv_nw"], eps),
        "k_pe": k_pe,
        "q": torch.cat([lat, rope], -1),
        "gate": torch.sigmoid(F.linear(x, w["w_g"])),
        "iq": F.linear(qc, w["w_iq"]),
        "ik": F.layer_norm(
            F.linear(x, w["w_ik"]), (d.IKD,), w["ik_nw"], w["ik_nb"], eps
        ),
        "iv": F.linear(x, w["w_iv"]),
    }


def _ds_mla_write(cache, kvn, k_pe, slots):
    from vllm import _custom_ops as ops

    ref = cache.clone()
    one = torch.ones((), dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(kvn, k_pe, ref, slots, "fp8_ds_mla", one)
    return ref


def _state():
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    return (
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
    )


@needs_gfx950
@pytest.mark.parametrize("S", [1, 4, 8, 16])
@torch.inference_mode()
def test_d1a_matches_vllm_ops(S: int) -> None:
    """P1 + P2."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre as d

    nb = 4
    w = _inputs(S, nb)
    key = d.DsaPreBuild(tokens=S, nblocks=nb, delta=True, dump=True)
    ref_prefix, ref_blocks, ref_x = _attn_res(w, nb)
    cache0 = w["kv_cache"].clone()
    w = dict(w, prefix=w["prefix"].clone(), blocks=w["blocks"].clone())
    scratch = torch.zeros(d.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    epoch, health, flags = _state()
    out = _launch(key, w, scratch, epoch, health, flags)
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    mid = _scratch_rows(key, scratch)

    # AttnRes: the prefix (+ delta) and blocks bit-exact, x as attn_res's
    torch.testing.assert_close(w["prefix"], ref_prefix, atol=0, rtol=0)
    torch.testing.assert_close(w["blocks"], ref_blocks, atol=0, rtol=0)
    assert _rel_l2(mid["x"], ref_x) < 1e-2

    # P1: the record is vLLM's writer's on D1a's own normed kv_c and k_pe
    k_pe = mid["qa"][:, d.QL + d.KVL :]
    ref_cache = _ds_mla_write(cache0, mid["kvn"], k_pe, w["slots"])
    assert torch.equal(w["kv_cache"], ref_cache)
    assert not torch.equal(ref_cache, cache0)

    # P2a: from D1a's own x and normed q_c (the GEMVs' order only)
    r = _ops_from_x(w, mid["x"], qc=mid["qc"])
    errs = {
        "qkv": _rel_l2(mid["qa"], r["qkv"]),
        "kvn": _rel_l2(mid["kvn"], _rms(mid["qa"][:, d.QL : d.QL + d.KVL],
                                        w["kv_nw"], 1e-5)),
        "qc": _rel_l2(mid["qc"], _rms(mid["qa"][:, : d.QL], w["q_nw"], 1e-5)),
        "q_lat": _rel_l2(out["q"][..., : d.KVL], r["q"][..., : d.KVL]),
        "q_pe": _rel_l2(out["q"][..., d.KVL :], r["q"][..., d.KVL :]),
        "gate": _rel_l2(out["gate"], r["gate"]),
        "iq": _rel_l2(out["iq"], r["iq"]),
        "ik": _rel_l2(out["ik"], r["ik"]),
        "iv": _rel_l2(out["iv"], r["iv"]),
    }  # fmt: skip
    assert all(e < 1e-2 for e in errs.values()), errs
    for name in ("q", "gate", "iq", "ik", "iv"):
        assert _cos(out[name], r[name]) > COS_MIN, (name, errs)

    # P2b: end to end from the inputs
    e = _ops_from_x(w, ref_x)
    rc = _ds_mla_write(cache0, e["kvn"], e["k_pe"], w["slots"])
    same = (w["kv_cache"] == rc).float().mean().item()
    assert same > 0.99, same
    for name in ("q", "gate", "iq", "ik", "iv"):
        assert _cos(out[name], e[name]) > 0.999, (name, _rel_l2(out[name], e[name]))


@needs_gfx950
@torch.inference_mode()
def test_d1a_record_scale_boundaries() -> None:
    """P1 across binades: kv_a weights of powers of two (2^-14 .. 2^4) so the
    normed tiles' scales span many exponents, the smallest under the 1e-4
    floor."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre as d

    S, nb = 8, 2
    w = _inputs(S, nb, seed=5)
    key = d.DsaPreBuild(tokens=S, nblocks=nb, delta=True, dump=True)
    g = torch.Generator(device="cuda").manual_seed(11)
    tile_scale = torch.exp2(torch.randint(-12, 4, (4,), generator=g, device="cuda"))
    w["kv_nw"] = (
        tile_scale.repeat_interleave(128)
        * torch.exp2(torch.randint(-2, 2, (d.KVL,), generator=g, device="cuda"))
    ).to(bf)
    cache0 = w["kv_cache"].clone()
    scratch = torch.zeros(d.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    epoch, health, flags = _state()
    _launch(key, w, scratch, epoch, health, flags)
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    mid = _scratch_rows(key, scratch)
    ref = _ds_mla_write(cache0, mid["kvn"], mid["qa"][:, d.QL + d.KVL :], w["slots"])
    assert torch.equal(w["kv_cache"], ref)


@needs_gfx950
@torch.inference_mode()
def test_d1a_graph_replay() -> None:
    """P3: [epoch += 1, D1a] captured once, replayed over new inputs (prefix,
    delta, blocks, slots): each replay bit-equal to an eager launch on the same
    inputs."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre as d

    S, nb = 8, 4
    w = _inputs(S, nb)
    key = d.DsaPreBuild(tokens=S, nblocks=nb, delta=True)
    scratch = torch.zeros(d.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    epoch, health, flags = _state()
    live = {k: w[k] for k in ("prefix", "delta", "blocks", "slots", "kv_cache")}
    cache0 = w["kv_cache"].clone()

    def step():
        epoch.add_(1)
        return _launch(key, w, scratch, epoch, health, flags)

    step()  # compiled and warm
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = step()
    for seed in range(1, 6):
        fresh = _inputs(S, nb, seed=100 + seed)
        snap = {k: fresh[k].clone() for k in live}
        for k in live:
            live[k].copy_(fresh[k] if k != "kv_cache" else cache0)
        graph.replay()
        torch.accelerator.synchronize()
        got = {k: v.clone() for k, v in g_out.items()}
        got_state = {k: live[k].clone() for k in ("prefix", "blocks", "kv_cache")}
        for k in live:
            live[k].copy_(snap[k] if k != "kv_cache" else cache0)
        eager = step()
        torch.accelerator.synchronize()
        assert not health.any() and not flags.any()
        for k in got:
            assert torch.equal(got[k], eager[k]), (seed, k)
        for k in got_state:
            assert torch.equal(got_state[k], live[k]), (seed, k)
