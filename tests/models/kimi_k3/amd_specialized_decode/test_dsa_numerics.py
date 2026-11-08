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
"""

import pytest
import torch
import torch.nn.functional as F

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
