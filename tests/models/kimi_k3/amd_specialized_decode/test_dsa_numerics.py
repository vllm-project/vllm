# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1's DSA layers on the specialized decode (``VLLM_KIMI_SD_DSA``):
the split sparse MLA (``dsa_attn``) and M1b's bf16 W_UV, on one MI355X, at
TP8 + DCP1 shapes (12 heads a rank, the fp8_ds_mla cache's 656 B records).

- U1: one slice a row (P = 1) is the stock FlyDSL sparse MLA, bit for bit.
- U2: P = 16 / 32 slices merged by their LSEs against an fp32 reference, as
  close as the stock kernel is.
- U3: rows shorter than the top-k (empty slices): o 0, LSE -inf, the merge
  unchanged.
- U4: -1 indices (``has_invalid``) masked.
- U10: M1b's combine + bf16 W_UV + the gate (``mla_post``, ``uv``) against
  torch on the same partials.
- U7: the in-kernel top-k fold against the converter's rows, bit for bit.
- U6 / U8 / U9 (eight GPUs): K2 with the DSA stage (``dsa`` 2, the fold or
  not) bit-equal to the split launches over graph replays, across the epoch
  tag's wrap, and a stalled DSA CTA failing the step without a hang.
"""

import pytest
import ray
import torch
import torch.nn.functional as F

from tests.utils import multi_gpu_test, multi_process_parallel
from vllm.platforms import current_platform

bf = torch.bfloat16
KVL, ROPE, TOPK, NH, S = 512, 64, 2048, 12, 8
RECORD = KVL + 16 + 2 * ROPE  # 656
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


# ---------------------------------------------------------------- inputs


def _cache(slots: int, g: torch.Generator) -> torch.Tensor:
    """[slots, 656] u8 fp8_ds_mla records: e4m3 latent, pow2 per-128 scales,
    bf16 rope."""
    latent = (torch.randn(slots, KVL, generator=g) * 0.3).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-6, 1, (slots, 4), generator=g).float())
    rope = (torch.randn(slots, ROPE, generator=g) * 0.5).to(bf)
    rec = torch.cat(
        [latent.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)],
        dim=1,
    )
    assert rec.shape[1] == RECORD
    return rec.cuda()


def _indices(lens, slots: int, g: torch.Generator, invalid: float = 0.0):
    """(indptr, idx): row t's ``lens[t]`` slots (distinct), ``invalid`` of them
    -1."""
    rows = []
    for n in lens:
        ix = torch.randperm(slots, generator=g)[:n].to(torch.int32)
        if invalid:
            ix[torch.rand(n, generator=g) < invalid] = -1
        rows.append(ix)
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(lens).cumsum(0)
    return indptr.cuda(), torch.cat(rows).cuda()


def _q(g: torch.Generator, s: int = S) -> torch.Tensor:
    return (torch.randn(s, NH, KVL + ROPE, generator=g) * 0.5).to("cuda", bf)


def _scale() -> float:
    return 192**-0.5


def _reference(q, rec, indptr, idx, scale):
    """fp32 sparse MLA: (o [S, 12, 512], natural-log LSE [S, 12]); a row with
    no valid key o 0, LSE -inf."""
    lat = rec[:, :KVL].view(torch.float8_e4m3fn).float()
    sc = rec[:, KVL : KVL + 16].contiguous().view(torch.float32)
    lat = (lat.view(-1, 4, 128) * sc[:, :, None]).view(-1, KVL)
    rope = rec[:, KVL + 16 :].contiguous().view(bf).float()
    k = torch.cat([lat, rope], 1)
    o = torch.zeros(q.shape[0], NH, KVL, device=q.device)
    lse = torch.full((q.shape[0], NH), float("-inf"), device=q.device)
    for t in range(q.shape[0]):
        ix = idx[int(indptr[t]) : int(indptr[t + 1])].long()
        ix = ix[ix >= 0]
        if ix.numel() == 0:
            continue
        s = q[t].float() @ k[ix].T * scale  # [12, n]
        lse[t] = torch.logsumexp(s, 1)
        o[t] = torch.softmax(s, 1) @ lat[ix]
    return o, lse


def _merge(part_o, part_lse):
    """The partials [P, S, 12, 512] / [P, S, 12] merged by their LSEs (what
    M1b's combine computes)."""
    m = part_lse.max(0).values
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w = torch.exp(part_lse - m_safe)  # empty slices: 0
    tot = w.sum(0)
    o = (w[..., None] * part_o.float()).sum(0) / tot.clamp_min(1e-30)[..., None]
    return o, m_safe + torch.log(tot)


def _split(key, q, rec, indptr, idx, scale):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    so, sl = dsa_attn.partial_shapes(key)
    dt = torch.float32 if key.part_fp32 else bf
    part_o = torch.full(so, float("nan"), dtype=dt, device="cuda")
    part_lse = torch.full(sl, float("nan"), dtype=torch.float32, device="cuda")
    dsa_attn.dsa_attn_launch(
        key,
        part_o=part_o,
        part_lse=part_lse,
        q=q,
        kv=rec,
        idx=idx,
        iptr=indptr,
        scale=scale,
    )
    return part_o, part_lse


def _stock(q, rec, indptr, idx, scale, has_invalid=False):
    from vllm.models.kimi_k3.amd.ops.flydsl_sparse_mla import (
        flydsl_sparse_mla_fp8ds_fwd,
    )

    return flydsl_sparse_mla_fp8ds_fwd(
        q,
        rec.unsqueeze(1),
        indptr,
        idx,
        scale,
        has_invalid=has_invalid,
        return_lse=True,
    )


# ---------------------------------------------------------------- U1-U4


@needs_gfx950
@pytest.mark.parametrize("n", [TOPK, 2000, 77])
@torch.inference_mode()
def test_dsa_attn_one_split_is_stock(n):
    """U1: P = 1 (C = 2048, the row a slice) runs the stock body over the
    whole row: o and LSE bit-exact."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )

    g = torch.Generator().manual_seed(n)
    rec = _cache(3 * TOPK, g)
    indptr, idx = _indices([n] * S, rec.shape[0], g)
    q, scale = _q(g), _scale()
    key = DsaAttnBuild(splits=1, chunk=TOPK, tile=128)
    # one slice of 16 tiles: the tile/chunk check is the split's, skip it here
    part_o, part_lse = _split_one(key, q, rec, indptr, idx, scale)
    out, lse = _stock(q, rec, indptr, idx, scale)
    assert torch.equal(part_o[0], out) and torch.equal(part_lse[0], lse)


def _split_one(key, q, rec, indptr, idx, scale):
    """``_split`` with ``check``'s one-tile-a-slice rule lifted (P = 1)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    real = dsa_attn.check
    dsa_attn.check = lambda key: None
    try:
        return _split(key, q, rec, indptr, idx, scale)
    finally:
        dsa_attn.check = real


@needs_gfx950
@pytest.mark.parametrize("splits", [16, 32])
@pytest.mark.parametrize("part_fp32", [False, True])
@torch.inference_mode()
def test_dsa_attn_split_matches_reference(splits, part_fp32):
    """U2: P slices of C = 2048 / P merged by their LSEs against fp32, as
    close as the stock (whole-row) kernel; every slice's LSE is its own."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )

    g = torch.Generator().manual_seed(splits)
    rec = _cache(3 * TOPK, g)
    indptr, idx = _indices([TOPK] * S, rec.shape[0], g)
    q, scale = _q(g), _scale()
    key = DsaAttnBuild(splits=splits, chunk=TOPK // splits, part_fp32=part_fp32)
    part_o, part_lse = _split(key, q, rec, indptr, idx, scale)
    assert not part_o.isnan().any() and not part_lse.isnan().any()
    ref_o, ref_lse = _reference(q, rec, indptr, idx, scale)
    o, lse = _merge(part_o, part_lse)
    st_o, st_lse = _stock(q, rec, indptr, idx, scale)
    err, st_err = _rel_l2(o, ref_o), _rel_l2(st_o, ref_o)
    print(f"P={splits} rel_l2 split {err:.2e} stock {st_err:.2e}")
    assert _cos(o, ref_o) > COS_MIN and err <= max(1.5 * st_err, 2e-3)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)
    # slice j's LSE alone
    c = TOPK // splits
    for j in (0, splits - 1):
        sub = idx.view(S, TOPK)[:, j * c : (j + 1) * c].reshape(-1).contiguous()
        sp = torch.arange(S + 1, dtype=torch.int32, device="cuda") * c
        _, ref_j = _reference(q, rec, sp, sub, scale)
        torch.testing.assert_close(part_lse[j], ref_j, atol=1e-3, rtol=1e-4)


@needs_gfx950
@torch.inference_mode()
def test_dsa_attn_empty_slices():
    """U3: rows of 0 .. 2048 indices: the slices past a row's end o 0, LSE
    -inf; the merge as the reference (a row of none: o 0)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )

    g = torch.Generator().manual_seed(3)
    rec = _cache(3 * TOPK, g)
    lens = [TOPK, 1, 0, 127, 128, 129, 1500, 2047]
    indptr, idx = _indices(lens, rec.shape[0], g)
    q, scale = _q(g), _scale()
    key = DsaAttnBuild()
    part_o, part_lse = _split(key, q, rec, indptr, idx, scale)
    for t, n in enumerate(lens):
        for j in range(key.splits):
            if j * key.chunk >= n:
                assert torch.isneginf(part_lse[j, t]).all(), (t, j)
                assert (part_o[j, t] == 0).all(), (t, j)
            else:
                assert torch.isfinite(part_lse[j, t]).all(), (t, j)
    ref_o, ref_lse = _reference(q, rec, indptr, idx, scale)
    o, lse = _merge(part_o, part_lse)
    assert _cos(o[[0, 1, 3, 4, 5, 6, 7]], ref_o[[0, 1, 3, 4, 5, 6, 7]]) > COS_MIN
    assert (o[2] == 0).all()
    ok = torch.isfinite(ref_lse)
    torch.testing.assert_close(lse[ok], ref_lse[ok], atol=1e-3, rtol=1e-4)


@needs_gfx950
@torch.inference_mode()
def test_dsa_attn_invalid_indices():
    """U4: -1 entries (``has_invalid``) masked, a whole slice of them empty."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )

    g = torch.Generator().manual_seed(4)
    rec = _cache(3 * TOPK, g)
    indptr, idx = _indices([TOPK] * S, rec.shape[0], g, invalid=0.3)
    idx.view(S, TOPK)[1, 128:256] = -1  # row 1's slice 1: none valid
    idx.view(S, TOPK)[2, 1024:] = -1  # row 2's second half
    q, scale = _q(g), _scale()
    key = DsaAttnBuild(has_invalid=True)
    part_o, part_lse = _split(key, q, rec, indptr, idx, scale)
    assert torch.isneginf(part_lse[1, 1]).all() and (part_o[1, 1] == 0).all()
    assert torch.isneginf(part_lse[8:, 2]).all()
    ref_o, ref_lse = _reference(q, rec, indptr, idx, scale)
    o, lse = _merge(part_o, part_lse)
    assert _cos(o, ref_o) > COS_MIN
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------- U10


def _w_uv(g: torch.Generator) -> torch.Tensor:
    """bf16 W_UV [12, 512 k, 128 v] laid out as the checkpoint's kv_b_proj
    hands it over (``kv_b_proj.weight.T.view(512, 12, 256)[..., 128:]``,
    heads first): k the stride-1 dim."""
    w = (torch.randn(NH * 256, KVL, generator=g) * KVL**-0.5).to("cuda", bf)
    return w.T.view(KVL, NH, 256)[..., 128:].transpose(0, 1)


@needs_gfx950
@pytest.mark.parametrize("splits,part_fp32", [(16, False), (32, False), (16, True)])
@torch.inference_mode()
def test_dsa_m1b_bf16_uv(splits, part_fp32):
    """U10: ``dsa_attn``'s partials through M1b (``src`` stream: the combine,
    the bf16 W_UV, the gate) against torch on the same partials: the merge,
    ``_v_up_proj``'s bmm, ``* sigmoid(g)``."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import GROWS
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla_back import (
        MlaPostBuild,
        mla_post,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    g = torch.Generator().manual_seed(10 + splits)
    rec = _cache(3 * TOPK, g)
    lens = [TOPK, 1500, 129, 1, TOPK, 700, 2047, TOPK]
    indptr, idx = _indices(lens, rec.shape[0], g)
    q, scale = _q(g), _scale()
    key = DsaAttnBuild(splits=splits, chunk=TOPK // splits, part_fp32=part_fp32)
    po, pl = _split(key, q, rec, indptr, idx, scale)
    w_uv = _w_uv(g)
    gate = torch.sigmoid(torch.randn(S, GROWS, generator=g)).to("cuda", bf)
    lat = _merge(po, pl)[0].to(bf)
    # the stock: _v_up_proj's bmm (bf16 out), then the gate mul
    ref = torch.bmm(lat.transpose(0, 1), w_uv).transpose(0, 1).reshape(S, GROWS)
    ref = ref * gate
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    core = torch.full((S, GROWS), float("nan"), dtype=bf, device="cuda")
    mla_post(
        MlaPostBuild(tokens=S, qlen=S, ranks=1, splits=splits, src="stream",
                     uv="bf16", part_fp32=part_fp32),
        o=po[0], lse=pl[0], seq_lens=torch.tensor([TOPK], dtype=torch.int32,
                                                  device="cuda"),
        qsl=torch.tensor([0, S], dtype=torch.int32, device="cuda"),
        w_v=w_uv.transpose(1, 2), w_v_scale=None, gate=gate, core=core,
        peers=None, rank=0, layer=5,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health, alarm=flags.data_ptr(),
    )  # fmt: skip
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    assert torch.isfinite(core.float()).all()
    err = _rel_l2(core, ref)
    # the latents' bf16 (merge order) and the bmm's accumulation order apart
    ulp = (core.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-6)
    print(f"P={splits} f32={part_fp32} rel_l2 {err:.2e} "
          f"frac>1ulp {(ulp > 2**-7).float().mean().item():.2e}")  # fmt: skip
    assert err < 4e-3 and _cos(core, ref) > COS_MIN, err


# ---------------------------------------------------------------- U9


@needs_gfx950
@pytest.mark.parametrize("splits", [16, 32])
@torch.inference_mode()
def test_dsa_graph_replay(splits):
    """U9 (one launch each): ``dsa_attn`` then M1b captured once in a CUDA
    graph, replayed over new queries, indices and row lengths (empty slices
    coming and going): every replay bit-equal to the eager launches on the
    same inputs, and to itself (no host value baked in, deterministic)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import GROWS
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla_back import (
        MlaPostBuild,
        mla_post,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    g = torch.Generator().manual_seed(90 + splits)
    rec = _cache(4 * TOPK, g)
    key = DsaAttnBuild(splits=splits, chunk=TOPK // splits)
    post = MlaPostBuild(
        tokens=S, qlen=S, ranks=1, splits=splits, src="stream", uv="bf16"
    )
    w_v = _w_uv(g).transpose(1, 2)
    # the persistent buffers a graph reads (the builder's, the runner's)
    q = _q(g)
    gate = torch.empty(S, GROWS, dtype=bf, device="cuda")
    idx = torch.zeros(S * TOPK, dtype=torch.int32, device="cuda")
    iptr = torch.zeros(S + 1, dtype=torch.int32, device="cuda")
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    epoch = torch.ones(1, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([TOPK], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, S], dtype=torch.int32, device="cuda")

    def step():
        po, pl = _split(key, q, rec, iptr, idx, _scale())
        core = torch.empty(S, GROWS, dtype=bf, device="cuda")
        mla_post(
            post, o=po[0], lse=pl[0], seq_lens=seq_lens, qsl=qsl, w_v=w_v,
            w_v_scale=None, gate=gate, core=core, peers=None, rank=0, layer=7,
            epoch=epoch, health=health, alarm=flags.data_ptr(),
        )  # fmt: skip
        return po, pl, core

    def fill(lens):
        ip, ix = _indices(lens, rec.shape[0], g)
        idx.zero_()
        idx[: ix.numel()].copy_(ix)
        iptr.copy_(ip)
        q.copy_(_q(g))
        gate.copy_(torch.sigmoid(torch.randn(S, GROWS, generator=g)).to(bf))

    fill([TOPK] * S)
    step()  # compiled and warm before the capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_po, g_pl, g_core = step()
    for lens in (
        [TOPK] * S,
        [1, 2, 3, 4, 5, 6, 7, 8],
        [TOPK, 100, 2047, 1, 129, TOPK, 64, 65],
    ):
        fill(lens)
        e_po, e_pl, e_core = step()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(g_pl, e_pl) and torch.equal(g_po, e_po), lens
        assert torch.equal(g_core, e_core), lens
        for _ in range(20):
            graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(g_core, e_core), lens
        ip, ix = iptr, idx
        ref_o, _ = _reference(q, rec, ip, ix, _scale())
        lat = _merge(e_po, e_pl)[0]
        assert _cos(lat, ref_o) > COS_MIN, lens
    assert not health.any() and not flags.any()


# ---------------------------------------------------------------- U7


BS = 64  # the KV cache's block size


def _paged(
    nblk: int, bsr: int, lens, g: torch.Generator, invalid: float = 0.0, bs: int = BS
):
    """A paged step: (rec, bt, topk, iptr). The flat cache ``rec`` of ``nblk``
    blocks of ``bsr`` rows (the first ``bs`` a block's tokens), the request's
    block table ``bt`` (a permutation), the indexer's logical top-k [S + 3,
    2048] (row t's first ``lens[t]`` distinct positions, then -1; ``invalid``
    of them -1 and one past the table)."""
    rec = _cache(nblk * bsr, g)
    bt = torch.randperm(nblk, generator=g).to(torch.int32)
    topk = torch.full((S + 3, TOPK), -1, dtype=torch.int32)
    for t, n in enumerate(lens):
        topk[t, :n] = torch.randperm(nblk * bs, generator=g)[:n].to(torch.int32)
        if invalid and n > 2:
            topk[t, :n][torch.rand(n, generator=g) < invalid] = -1
            topk[t, 1] = nblk * bs + 5  # block nblk: past the table
    iptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
    iptr[1:] = torch.tensor(lens).cumsum(0)
    return rec, bt.cuda(), topk.cuda(), iptr.cuda()


def _convert(bt, topk, iptr, bsr: int, invalid: int = 0, bs: int = BS):
    """The converter's rows in torch (an invalid position ``invalid``)."""
    rows = []
    for t in range(iptr.numel() - 1):
        tok = topk[t, : int(iptr[t + 1] - iptr[t])].long()
        blk = tok // bs
        ok = (tok >= 0) & (blk < bt.numel())
        row = bt[blk.clamp(0, bt.numel() - 1)].long() * bsr + tok % bs
        rows.append(torch.where(ok, row, invalid).to(torch.int32))
    return torch.cat(rows)


@needs_gfx950
@pytest.mark.parametrize("splits", [16, 32])
@pytest.mark.parametrize("has_invalid", [False, True])
@pytest.mark.parametrize("bs,nblk,bsr", [(BS, 80, 80), (2048, 4, 2100)])
@torch.inference_mode()
def test_dsa_fold_matches_converter(splits, has_invalid, bs, nblk, bsr):
    """U7: the in-kernel fold (the indexer's top-k, the block table, the flat
    pitch) against ``dsa_attn`` on the converter's rows, bit for bit: a pitch
    past the block size (64, and the hybrid engine's 2048), rows of every length, -1 and out-of-table positions
    (the converter's row 0, or with ``has_invalid`` masked)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        triton_convert_req_index_to_global_index,
    )

    g = torch.Generator().manual_seed(70 + splits + has_invalid)
    lens = [TOPK, 1, 2047, 129, 128, 700, TOPK, 3]
    rec, bt, topk, iptr = _paged(nblk, bsr, lens, g, invalid=0.2, bs=bs)
    q, scale = _q(g), _scale()
    idx = torch.full((S * TOPK,), 7, dtype=torch.int32, device="cuda")
    triton_convert_req_index_to_global_index(
        torch.zeros(S, dtype=torch.int32, device="cuda"), bt[None], topk[:S],
        iptr, idx, BLOCK_STRIDE_ROWS=bsr, BLOCK_SIZE=bs, NUM_TOPK_TOKENS=TOPK,
    )  # fmt: skip
    n = int(iptr[-1])
    assert torch.equal(idx[:n], _convert(bt, topk, iptr, bsr, bs=bs))
    if has_invalid:
        idx[:n] = _convert(bt, topk, iptr, bsr, invalid=-1, bs=bs)
    kw = dict(splits=splits, chunk=TOPK // splits, has_invalid=has_invalid)
    want = _split(DsaAttnBuild(**kw), q, rec, iptr, idx[:n].contiguous(), scale)
    key = DsaAttnBuild(fold=True, **kw)
    so, sl = want[0].shape, want[1].shape
    po = torch.full(so, float("nan"), dtype=bf, device="cuda")
    pl = torch.full(sl, float("nan"), dtype=torch.float32, device="cuda")
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    dsa_attn.dsa_attn_launch(
        key, part_o=po, part_lse=pl, q=q, kv=rec, iptr=iptr, scale=scale,
        topk=topk, bt=bt, bsr=bsr, bs=bs,
    )  # fmt: skip
    torch.accelerator.synchronize()
    assert torch.equal(po, want[0]) and torch.equal(pl, want[1])


@needs_gfx950
@pytest.mark.parametrize("splits", [16, 32])
@pytest.mark.parametrize("fold", [False, True])
@torch.inference_mode()
def test_dsa_stage_matches_standalone(splits, fold):
    """U6 (one GPU): K2's DSA stage (``standalone`` False: 512 threads, waves
    4 .. 7 the body's barriers only) in a launch of its own gives the
    standalone kernel's partials, bit for bit."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.dsa_attn import (
        DsaAttnBuild,
    )

    g = torch.Generator().manual_seed(60 + splits + fold)
    nblk, bsr = 80, 80
    rec, bt, topk, iptr = _paged(nblk, bsr, [TOPK, 1, 2047, 129, 128, 700, TOPK, 3], g)
    q, scale = _q(g), _scale()
    ins = dict(q=q, kv=rec, iptr=iptr, scale=scale)
    if fold:
        ins.update(topk=topk, bt=bt, bsr=bsr, bs=BS)
    else:
        ins.update(idx=_convert(bt, topk, iptr, bsr))
    got = []
    for standalone in (True, False):
        key = DsaAttnBuild(
            splits=splits, chunk=TOPK // splits, fold=fold, standalone=standalone
        )
        so, sl = dsa_attn.partial_shapes(key)
        po = torch.full(so, float("nan"), dtype=bf, device="cuda")
        pl = torch.full(sl, float("nan"), dtype=torch.float32, device="cuda")
        dsa_attn.dsa_attn_launch(key, part_o=po, part_lse=pl, **ins)
        torch.accelerator.synchronize()
        got.append((po, pl))
    assert torch.equal(got[0][0], got[1][0]) and torch.equal(got[0][1], got[1][1])


# ---------------------------------------------------------------- U6, U8, U9


DSA_LAYERS = (3, 5)
DSA_LENS = {3: [TOPK] * S, 5: [TOPK, 100, 2047, 1, 129, TOPK, 64, 65]}
DSA_STALLED = 3  # U8's stalled rank


def _dsa_k2(runner, P: int, dsa: int, fold: bool):
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2

    return k2.K2Build(
        tokens=S, nblocks=5, reset=True, m1b_qlen=S, m1b_ranks=1,
        m1b_off=runner.mla_post_off, m1b_splits=P, dsa=dsa, dsa_p=P,
        dsa_fold=fold, uv_bf16=True, **runner.fault,
    )  # fmt: skip


@ray.remote(num_gpus=1, max_calls=1)
def _dsa_k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    from tests.models.kimi_k3.amd_specialized_decode.test_numerics import (
        _init_worker,
        _k2_inputs,
        _moe_args,
        _moe_weights,
    )

    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    import torch.distributed as dist

    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import GROWS
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import WAIT_TICKS
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)
    g = torch.Generator().manual_seed(600 + rank)
    nblk, bsr = 80, 80
    qsl = torch.tensor([0, S], dtype=torch.int32, device=device)
    ins = {}
    for layer in DSA_LAYERS:
        rec, bt, topk, iptr = _paged(nblk, bsr, DSA_LENS[layer], g)
        ins[layer] = dict(
            w=_k2_inputs(rank, device, S, seed=layer),
            rec=rec, bt=bt, topk=topk, iptr=iptr,
            idx=_convert(bt, topk, iptr, bsr),
            q=_q(g),
            gate=torch.sigmoid(torch.randn(S, GROWS, generator=g)).to(device, bf),
            w_v=_w_uv(g).transpose(1, 2),
        )  # fmt: skip

    def launch(layer, P, dsa, fold, outs):
        x = ins[layer]
        key = _dsa_k2(runner, P, dsa, fold)
        dk = dsa_attn.DsaAttnBuild(tokens=S, splits=P, chunk=TOPK // P, fold=fold)
        so, sl = dsa_attn.partial_shapes(dk)
        po = torch.empty(so, dtype=bf, device=device)
        pl = torch.empty(sl, dtype=torch.float32, device=device)
        dsa_in = dict(q=x["q"], kv=x["rec"], iptr=x["iptr"], scale=_scale())
        if fold:
            dsa_in.update(topk=x["topk"], bt=x["bt"], bsr=bsr, bs=BS)
        else:
            dsa_in.update(idx=x["idx"])
        if dsa == 1:
            dsa_attn.dsa_attn_launch(dk, part_o=po, part_lse=pl, **dsa_in)
        core, prefix, out = outs
        m1b_in = dict(
            o=po[0], lse=pl[0], seq_lens=qsl, qsl=qsl, w_v=x["w_v"],
            w_v_scale=None, gate=x["gate"],
        )  # fmt: skip
        w = x["w"]
        k2.k2_launch(
            key, core=core, w_o=w["w_o"], prefix=prefix, blocks=w["blocks"],
            ares_nw=w["ares_nw"], ares_qk=w["ares_qk"], in_nw=w["in_nw"],
            out=out, layer=128 + layer, m1b_in=m1b_in,
            dsa_in=dsa_in if dsa == 2 else None,
            **_moe_args(mw), **runner.launch_args(), **runner.peer_args(),
        )  # fmt: skip
        return po, pl

    def buffers():
        return {
            layer: (
                torch.empty(S, GROWS, dtype=bf, device=device),
                torch.empty(S, HIDDEN, dtype=bf, device=device),
                torch.empty(S, HIDDEN, dtype=bf, device=device),
            )
            for layer in DSA_LAYERS
        }

    def capture(P, dsa, fold):
        bufs = buffers()
        runner.step_begin()
        for layer in DSA_LAYERS:  # compiled and warm before the capture
            launch(layer, P, dsa, fold, bufs[layer])
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.step_begin()
            for layer in DSA_LAYERS:
                launch(layer, P, dsa, fold, bufs[layer])
        return graph, bufs

    def clone(bufs):
        return {k: [t.clone() for t in v] for k, v in bufs.items()}

    def same(a, b):
        return all(torch.equal(x, y) for k in a for x, y in zip(a[k], b[k]))

    # U6: the split launches, the fused stage, the fused stage + the fold
    graphs = {}
    for P in (16, 32):
        got = {}
        for dsa, fold in ((1, False), (2, False), (2, True)):
            graph, bufs = capture(P, dsa, fold)
            graph.replay()
            torch.accelerator.synchronize()
            want = clone(bufs)
            for _ in range(200):
                graph.replay()
            torch.accelerator.synchronize()
            assert same(bufs, want), (P, dsa, fold)
            assert not runner.health.any() and not runner.flags.any(), (P, dsa, fold)
            got[dsa, fold] = want
            graphs[P, dsa, fold] = graph, bufs, want
        for v in ((2, False), (2, True)):
            assert same(got[v], got[1, False]), (P, v)
        core = got[1, False][3][0]
        assert torch.isfinite(core.float()).all() and core.float().abs().sum() > 0
        outs = [torch.empty_like(got[2, True][5][2]) for _ in range(tp_size)]
        dist.all_gather(outs, got[2, True][5][2])
        assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"

    # U9: the epoch's tag wraps (2^22) under the fused graphs' replays
    runner.epoch.fill_(2**22 - 2)
    for _ in range(4):
        for (P, dsa, fold), (graph, bufs, want) in graphs.items():
            if dsa == 2:
                graph.replay()
                torch.accelerator.synchronize()
                assert same(bufs, want), ("wrap", P, fold, runner.epoch.item())
    assert runner.epoch.item() > 2**22 + 2
    assert not runner.health.any() and not runner.flags.any()

    # U8: the last DSA CTA (an ``ardy`` flag) stalls past every wait's bound
    e = runner.epoch.item() + 2
    runner.fault = {
        "fault_rank": DSA_STALLED,
        "fault_epoch": e,
        "fault_layer": 128 + 3,
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    bufs = buffers()
    runner.step_begin()  # no stall
    launch(3, 16, 2, True, bufs[3])
    torch.accelerator.synchronize()
    assert not runner.flags.any()
    runner.step_begin()  # epoch e: rank DSA_STALLED's CTA 255 stalls
    assert runner.epoch.item() == e
    launch(3, 16, 2, True, bufs[3])
    torch.accelerator.synchronize()
    waits = given_up_waits(runner.health.cpu(), k2.REGIONS)
    if waits:
        print(f"rank {rank}: {waits[:3]}")
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    assert gave_up.item() > 0, "no rank's wait gave up"
    if rank == 0:
        assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_dsa_fused_tp8_bit_equal_replays_and_faults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U6: K2 with the DSA stage (``dsa`` 2, with and without the fold) gives
    the split launches' (``dsa`` 1) bits, P = 16 and 32, over 200 replays of a
    graph of two layers. U9: the fused graphs replayed across the epoch tag's
    wrap. U8: a DSA CTA stalled past the bounds: the combine's ``ardy`` waits
    give up, the step fails, no hang."""
    multi_process_parallel(monkeypatch, 8, 1, _dsa_k2_worker)
