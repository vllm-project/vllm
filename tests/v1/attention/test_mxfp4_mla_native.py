# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native FP8 x FP4 kernel against a plain statement of its arithmetic.

The reference is written in fp32 PyTorch with the same structure the kernel
uses -- 128-token splits, a split-local softmax, E4M3 weights formed after
folding V's E8M0 scale per latent group, then a log-sum-exp combine -- so any
disagreement beyond fp32 summation order is a kernel bug, not a numerics
difference. Ragged lengths and invalid slots exercise the partial-split and
empty-split paths.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops import mxfp4_mla as mx


def _on_gfx950() -> bool:
    from vllm.platforms import current_platform

    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950 scaled MFMA")

H, LATENT, ROW, GROUP, TOK = 16, 512, 272, 32, 128
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


def _reference(q, latent, cache, indices, indptr, chunks_per_split=1):
    """chunks_per_split > 1 mirrors a workgroup that walks several 128-token
    chunks with an online softmax: weights are rounded against the running
    max, and the accumulator is rescaled as the max grows."""
    from vllm.v1.attention.ops.mxfp4_mla_native import quantize_q

    kv = mx.quantize_dequantize(latent.to(torch.bfloat16), GROUP).float()
    s = torch.exp2(cache[:, mx.scale_region_offset(LATENT) :].cpu().float() - 127.0)
    codes = kv / s.repeat_interleave(GROUP, dim=1)
    q8, qs = quantize_q(q)
    qd = q8.view(torch.float8_e4m3fn).float().cpu()  # dequantised without its scale
    qs = qs.cpu()
    out = torch.zeros(q.shape[0], H, LATENT)
    for i in range(q.shape[0]):
        idx = indices[indptr[i] : indptr[i + 1]].cpu().long()
        parts = []
        span = TOK * chunks_per_split
        for s0 in range(0, idx.numel(), span):
            m = torch.full((H,), float("-inf"))
            denom = torch.zeros(H)
            acc = torch.zeros(H, LATENT)
            for t0 in range(s0, min(s0 + span, idx.numel()), TOK):
                blk = idx[t0 : t0 + TOK]
                ok = blk >= 0
                safe = blk.clamp(min=0)
                sc = (qd[i] @ kv[safe].T) * (qs[i][:, None] * SCALE)  # [H, T]
                sc = sc.masked_fill(~ok[None, :], float("-inf"))
                m_new = torch.maximum(m, sc.max(dim=1).values)
                alpha = torch.exp(m - m_new).nan_to_num(0.0)
                p = torch.exp(sc - m_new[:, None]).nan_to_num(0.0)
                denom = denom * alpha + p.sum(dim=1)
                pv = torch.zeros(H, LATENT)
                for g in range(LATENT // GROUP):
                    w = (p * s[safe, g][None, :]).clamp(max=448.0)
                    w = w.to(torch.float8_e4m3fn).float()
                    cols = slice(g * GROUP, (g + 1) * GROUP)
                    pv[:, cols] = w @ codes[safe][:, cols]
                acc = acc * alpha[:, None] + pv
                m = m_new
            parts.append((m, denom, acc))
        M = torch.stack([pm for pm, _, _ in parts]).max(dim=0).values
        num = sum(torch.exp(pm - M)[:, None] * pa for pm, _, pa in parts)
        den = sum(torch.exp(pm - M) * pl for pm, pl, _ in parts)
        out[i] = num / den[:, None]
    return out


def _native(q, cache, indices, indptr, max_topk=2048, num_splits=None):
    from vllm.v1.attention.ops.mxfp4_mla_native import mxfp4_mla_native

    out = torch.empty(q.shape[0], H, LATENT, dtype=torch.bfloat16, device="cuda")
    mxfp4_mla_native(
        q, cache, indices, indptr, SCALE, max_topk, out, num_splits=num_splits
    )
    torch.accelerator.synchronize()
    return out.float().cpu()


def _assert_matches(got, want, tol):
    """Per-(row, head) relative L2 error, against a bf16-rounded reference.

    The kernel stores bf16, so the reference is rounded the same way; an
    elementwise tolerance would flag near-zero outputs where one bf16 ulp is a
    large relative change.
    """
    want = want.to(torch.bfloat16).float()
    err = (got - want).norm(dim=-1) / want.norm(dim=-1).clamp(min=1e-6)
    assert err.max() < tol, f"worst per-head relative error {err.max():.2e} (tol {tol})"


def test_device_q_quantisation_is_bit_identical_to_the_reference():
    """The kernel quantises Q itself; it must agree with quantize_q exactly.

    An earlier version used the default device division, which is not
    correctly rounded: 21 of 32768 codes came out one E4M3 step off, and in a
    head attending over 5 tokens that alone moved the output by 2.6%.
    """
    from vllm.v1.attention.ops.mxfp4_mla_native import device_quantize_q, quantize_q

    torch.manual_seed(0)
    q = (torch.randn(8, H, LATENT) * 2).to(torch.bfloat16).cuda()
    q[3] *= 1e-3
    q[5, 2] = 0  # an all-zero head takes the scale = 1 branch
    a8, a_s = quantize_q(q)
    b8, b_s = device_quantize_q(q)
    torch.accelerator.synchronize()
    assert torch.equal(a_s, b_s), "Q scales differ"
    assert torch.equal(a8, b8), f"{int((a8 != b8).sum())} Q codes differ"


@pytest.mark.parametrize(
    "lengths,invalid",
    [([2048, 2048], 0), ([2048, 1000, 130, 5], 0), ([700, 64], 7)],
)
def test_native_matches_its_arithmetic(lengths, invalid):
    q, latent, cache, indices, indptr = _case(
        lengths, seed=len(lengths), invalid_every=invalid
    )
    got = _native(q, cache, indices, indptr)
    want = _reference(q, latent, cache, indices, indptr)
    # bf16 output rounding alone is ~1.6e-3 per head.
    _assert_matches(got, want, 5e-3)


def _triton(q, cache, indices, indptr):
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as ops

    out = torch.empty(q.shape[0], H, LATENT, dtype=torch.bfloat16, device="cuda")
    ops.rocm_sparse_attn_prefill(
        q=q,
        kv=cache.unsqueeze(1),
        indices=None,
        topk_length=None,
        scale=SCALE,
        head_dim=LATENT,
        nope_head_dim=LATENT,
        rope_head_dim=0,
        attn_sink=None,
        output=out,
        ragged_indices=indices,
        ragged_indptr=indptr,
    )
    return out.float().cpu()


def test_native_vs_bf16_weights_is_the_e4m3_cost():
    """Informational bound: E4M3 Q and P against the bf16-Q/P unpack path."""
    q, latent, cache, indices, indptr = _case([2048, 1500, 256], seed=3)
    got = _native(q, cache, indices, indptr)
    plain = _triton(q, cache, indices, indptr)
    rel = (got - plain).norm() / plain.norm()
    # E4M3 carries 3 mantissa bits (step 6.25%); on random data with a flat
    # softmax this lands near 4%. GSM8K, not this number, decides acceptability.
    assert rel < 0.06, f"native vs bf16-Q/P path differ by {rel:.3%}"


def test_rows_beyond_one_launch_slice():
    """A row-split budget smaller than the batch exercises the slicing loop."""
    import vllm.v1.attention.ops.mxfp4_mla_native as nat

    old = nat.MAX_ROW_SPLITS_PER_LAUNCH
    nat.MAX_ROW_SPLITS_PER_LAUNCH = 8  # 4 splits at max_topk 512 -> 2 rows a launch
    try:
        q, latent, cache, indices, indptr = _case([300, 20, 129, 1, 256], seed=9)
        got = _native(q, cache, indices, indptr, max_topk=512)
        want = _reference(q, latent, cache, indices, indptr)
    finally:
        nat.MAX_ROW_SPLITS_PER_LAUNCH = old
    _assert_matches(got, want, 5e-3)


@pytest.mark.parametrize("num_splits", [1, 2, 3])
def test_multi_chunk_workgroups(num_splits):
    """Fewer splits than chunks: each workgroup walks several chunks with an
    online softmax, the prefill-size configuration. 3 does not divide 16, so
    the last split is short."""
    q, latent, cache, indices, indptr = _case(
        [2048, 1000, 130, 5], seed=4, invalid_every=11
    )
    got = _native(q, cache, indices, indptr, num_splits=num_splits)
    want = _reference(
        q, latent, cache, indices, indptr, chunks_per_split=-(-16 // num_splits)
    )
    _assert_matches(got, want, 5e-3)


def test_choose_splits():
    from vllm.v1.attention.ops.mxfp4_mla_native import choose_splits

    assert [choose_splits(n, 16) for n in (1, 32, 128, 256, 512, 1024, 2048, 8192)] == [
        16,
        16,
        16,
        8,
        4,
        2,
        1,
        1,
    ]
