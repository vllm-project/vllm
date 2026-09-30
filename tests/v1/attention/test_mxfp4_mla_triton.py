# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton MXFP4 sparse-MLA kernel against a plain statement of its math.

Both PV modes are checked. Q in E4M3 (per-head absmax scale), K and V in
MXFP4 throughout; P is E4M3 with V's scales folded in per latent group in
"fp4" mode (the HIP kernel's arithmetic), bf16 in "bf16" mode (the FlyDSL
pathway). Each reference mirrors the kernel's structure -- chunks, a running
max within each split, the mode's P rounding, log-sum-exp combine -- so a
disagreement beyond fp32 summation order is a kernel bug.
"""

from __future__ import annotations

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops import mxfp4_mla as mx


def _on_gfx950() -> bool:
    from vllm.platforms import current_platform

    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950 scaled MFMA")

H, LATENT, ROW, GROUP = 16, 512, 272, 32
SCALE = LATENT**-0.5
SLOTS = 4096


def _case(lengths, seed=0, invalid_every=0):
    from vllm.v1.attention.ops.mxfp4_mla_store import store_mxfp4_mla

    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(SLOTS, LATENT, generator=g)
    latent[:, ::97] *= 8.0
    cache = torch.zeros(SLOTS, ROW, dtype=torch.uint8, device="cuda")
    store_mxfp4_mla(
        latent.to(torch.bfloat16).cuda(), torch.arange(SLOTS, device="cuda"), cache
    )
    q = (
        (torch.randn(len(lengths), H, LATENT, generator=g) * 2)
        .to(torch.bfloat16)
        .cuda()
    )
    idx = [
        torch.randint(0, SLOTS, (n,), generator=g, dtype=torch.int32) for n in lengths
    ]
    if invalid_every:
        for t in idx:
            t[::invalid_every] = -1
    indices = torch.cat(idx).cuda()
    indptr = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32
    ).cuda()
    return q, latent, cache, indices, indptr


def _reference(q, latent, indices, indptr, splits, block):
    from vllm.v1.attention.ops.mxfp4_mla_native import quantize_q

    kv = mx.quantize_dequantize(latent.to(torch.bfloat16), GROUP).float()
    q8, qs = quantize_q(q)
    qd = q8.view(torch.float8_e4m3fn).float().cpu()
    qs = qs.cpu()
    chunks_total = -(-2048 // block)
    cps = -(-chunks_total // splits)
    out = torch.zeros(q.shape[0], H, LATENT)
    for i in range(q.shape[0]):
        idx = indices[indptr[i] : indptr[i + 1]].cpu().long()
        parts = []
        for s in range(splits):
            m = torch.full((H,), float("-inf"))
            denom = torch.zeros(H)
            acc = torch.zeros(H, LATENT)
            for c in range(cps):
                t0 = (s * cps + c) * block
                blk = idx[t0 : t0 + block]
                if blk.numel() == 0:
                    break
                ok = blk >= 0
                safe = blk.clamp(min=0)
                sc = (kv[safe] @ qd[i].T) * (qs[i] * SCALE)[None, :]  # [T, H]
                sc = sc.masked_fill(~ok[:, None], float("-inf"))
                m_new = torch.maximum(m, sc.max(dim=0).values)
                alpha = torch.exp(m - m_new).nan_to_num(0.0)
                p = torch.exp(sc - m_new[None, :]).nan_to_num(0.0)
                denom = denom * alpha + p.sum(dim=0)
                acc = acc * alpha[:, None] + p.to(torch.bfloat16).float().T @ kv[safe]
                m = m_new
            if denom.max() > 0:
                parts.append((m, denom, acc))
        M = torch.stack([pm for pm, _, _ in parts]).max(dim=0).values
        num = sum(torch.exp(pm - M)[:, None] * pa for pm, _, pa in parts)
        den = sum(torch.exp(pm - M) * pl for pm, pl, _ in parts)
        out[i] = num / den[:, None]
    return out


@pytest.fixture(params=["fp4", "bf16"])
def mode(request, monkeypatch):
    import vllm.v1.attention.ops.mxfp4_mla_triton as tri

    monkeypatch.setattr(tri, "PV_MODE", request.param)
    return request.param


def _reference_for(mode, q, latent, cache, indices, indptr, splits):
    """fp4: the HIP kernel's reference (128-token chunks, E4M3 fold of P);
    bf16: the FlyDSL-pathway reference (64-token chunks, bf16 P)."""
    import vllm.v1.attention.ops.mxfp4_mla_triton as tri

    block = tri.BLOCK_N_BY_MODE[mode]
    cps = cdiv(cdiv(2048, block), splits)
    if mode == "fp4":
        import test_mxfp4_mla_native as nat_test

        # P is rounded against a running max per chunk, so the reference must
        # chunk exactly like the kernel; its chunk size is the module's TOK.
        old = nat_test.TOK
        nat_test.TOK = block
        try:
            return nat_test._reference(
                q, latent, cache, indices, indptr, chunks_per_split=cps
            )
        finally:
            nat_test.TOK = old
    return _reference(q, latent, indices, indptr, splits, block)


def _triton(q, cache, indices, indptr, num_splits=None, max_topk=2048):
    from vllm.v1.attention.ops.mxfp4_mla_triton import mxfp4_mla_triton

    out = torch.empty(q.shape[0], H, LATENT, dtype=torch.bfloat16, device="cuda")
    mxfp4_mla_triton(
        q, cache, indices, indptr, SCALE, max_topk, out, num_splits=num_splits
    )
    torch.accelerator.synchronize()
    return out.float().cpu()


def _assert_matches(got, want, tol):
    want = want.to(torch.bfloat16).float()
    err = (got - want).norm(dim=-1) / want.norm(dim=-1).clamp(min=1e-6)
    assert err.max() < tol, f"worst per-head relative error {err.max():.2e} (tol {tol})"


def test_q_quantisation_is_bit_identical_to_the_reference():
    from vllm.v1.attention.ops.mxfp4_mla_native import quantize_q
    from vllm.v1.attention.ops.mxfp4_mla_triton import HEADS, _quantize_q_kernel
    from vllm.v1.attention.ops.mxfp4_mla_triton import LATENT as L

    torch.manual_seed(0)
    q = (torch.randn(8, H, LATENT) * 2).to(torch.bfloat16).cuda()
    q[3] *= 1e-3
    q[5, 2] = 0
    q8 = torch.empty(8, H, LATENT, dtype=torch.uint8, device="cuda")
    qs = torch.empty(8, H, dtype=torch.float32, device="cuda")
    _quantize_q_kernel[(8,)](q, q.stride(0), q.stride(1), q8, qs, H=HEADS, LAT=L)
    a8, a_s = quantize_q(q)
    assert torch.equal(qs, a_s), "Q scales differ"
    assert torch.equal(q8, a8), f"{int((q8 != a8).sum())} Q codes differ"


@pytest.mark.parametrize(
    "lengths,invalid,splits",
    [
        ([2048, 2048], 0, None),  # decode-style: one chunk per split
        ([2048, 1000, 130, 5], 0, None),  # ragged, empty splits
        ([700, 64], 7, None),  # invalid slots
        ([2048, 1000, 130, 5], 11, 1),  # prefill-style: one split walks all chunks
        ([2048, 777], 0, 4),  # several chunks per split
    ],
)
def test_matches_its_arithmetic(mode, lengths, invalid, splits):
    import vllm.v1.attention.ops.mxfp4_mla_triton as tri

    q, latent, cache, indices, indptr = _case(
        lengths, seed=len(lengths), invalid_every=invalid
    )
    got = _triton(q, cache, indices, indptr, num_splits=splits)
    chunks = -(-2048 // tri.BLOCK_N_BY_MODE[mode])
    eff = tri.triton.next_power_of_2(splits or tri.choose_splits(len(lengths), chunks))
    want = _reference_for(mode, q, latent, cache, indices, indptr, eff)
    _assert_matches(got, want, 5e-3)


def test_rows_beyond_one_launch_slice(mode):
    import vllm.v1.attention.ops.mxfp4_mla_triton as tri

    chunks = -(-2048 // tri.BLOCK_N_BY_MODE[mode])
    old = tri.MAX_ROW_SPLITS_PER_LAUNCH
    tri.MAX_ROW_SPLITS_PER_LAUNCH = 2 * chunks  # 2 rows per launch
    try:
        q, latent, cache, indices, indptr = _case([300, 20, 129, 1, 256], seed=9)
        got = _triton(q, cache, indices, indptr)
    finally:
        tri.MAX_ROW_SPLITS_PER_LAUNCH = old
    eff = tri.triton.next_power_of_2(tri.choose_splits(5, chunks))
    want = _reference_for(mode, q, latent, cache, indices, indptr, eff)
    _assert_matches(got, want, 5e-3)
