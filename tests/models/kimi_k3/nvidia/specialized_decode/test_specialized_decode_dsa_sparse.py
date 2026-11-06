# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: the DCP sparse MLA decode (dsa/attn_sparse.py)
against an FP32 emulation of its stated precision (E4M3 query / cache, FP32 logits,
E4M3 P at 448 of the running max per chunk, FP32 sums, one BF16 output rounding)
and against exact attention, where its error must not exceed FlashInfer's prims_ts
sparse decode of the same rows (the kernel it replaces). One GPU."""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100 (Blackwell) only",
)

ROW, LATENT, PADDED, HEADS, WIDTH, CAP = 576, 512, 128, 96, 2048, 64
SOFTMAX = 1.0 / math.sqrt(192)
Q_SCALE, K_SCALE = 0.05, 0.03


def _inputs(counts, rows_total=40_000, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(counts)
    q = (torch.randn(CAP, PADDED, ROW, generator=g, device="cuda") * 2).clamp(-8, 8)
    q[:, HEADS:] = 0
    query = q.to(torch.float8_e4m3fn)
    kv = (torch.randn(rows_total, ROW, generator=g, device="cuda") * 3).clamp(-12, 12)
    kv = kv.to(torch.float8_e4m3fn)
    rows = torch.full((batch, WIDTH), -1, dtype=torch.int32, device="cuda")
    for b, n in enumerate(counts):
        perm = torch.randperm(rows_total, generator=g, device="cuda")[:n]
        rows[b, :n] = perm.to(torch.int32)
    valid = torch.tensor(counts, dtype=torch.int32, device="cuda")
    return query, kv, rows, valid


def _reference(query, kv, rows, valid, quantize_p):
    """FP32 attention of the first HEADS heads; ``quantize_p``: the kernel's
    schedule (each of SPLIT CTAs takes a contiguous 1 / SPLIT of the keys in
    chunks of CHUNK, E4M3 P at 448 of its running max, FP32 sums; the CTAs'
    partials merge in FP32)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_sparse import (
        CHUNK,
        SPLIT,
    )

    batch = rows.shape[0]
    scale_log2 = SOFTMAX * Q_SCALE * K_SCALE * math.log2(math.e)
    out = torch.zeros(batch, HEADS, LATENT, device="cuda")
    lse = torch.full((batch, HEADS), -math.inf, device="cuda")
    for b in range(batch):
        n = int(valid[b])
        if n == 0:
            continue
        q = query[b, :HEADS].float()
        k = kv[rows[b, :n].long()].float()
        s = (q @ k.T) * scale_log2  # [H, n] log2 units
        if not quantize_p:
            p = torch.exp2(s - s.max(dim=1, keepdim=True).values)
            out[b] = (p @ k[:, :LATENT]) / p.sum(1, keepdim=True) * K_SCALE
            m = s.max(dim=1).values
            lse[b] = (m + torch.log2(p.sum(1))) * math.log(2)
            continue
        per = -(-n // SPLIT)
        parts = []
        for first in range(0, n, per):
            m_run = torch.full((HEADS,), -math.inf, device="cuda")
            l_run = torch.zeros(HEADS, device="cuda")
            acc = torch.zeros(HEADS, LATENT, device="cuda")
            for c0 in range(first, min(n, first + per), CHUNK):
                c1 = min(c0 + CHUNK, first + per, n)
                sc = s[:, c0:c1]
                m_new = torch.maximum(m_run, sc.max(dim=1).values)
                alpha = torch.where(
                    torch.isinf(m_run),
                    torch.ones_like(m_run),
                    torch.exp2(m_run - m_new),
                )
                p = torch.exp2(sc - (m_new - math.log2(448))[:, None])
                pq = p.to(torch.float8_e4m3fn).float()
                l_run = l_run * alpha + p.sum(1)
                acc = acc * alpha[:, None] + pq @ k[c0:c1, :LATENT]
                m_run = m_new
            parts.append((m_run, l_run, acc))
        top = torch.stack([m for m, _, _ in parts]).max(0).values
        total = sum(torch.exp2(m - top) * part for m, part, _ in parts)
        merged = sum(torch.exp2(m - top)[:, None] * a for m, _, a in parts)
        out[b] = merged / total[:, None] * K_SCALE
        lse[b] = (top - math.log2(448) + torch.log2(total)) * math.log(2)
    return out, lse


def _run_sparse(query, kv, rows, valid):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_sparse import (
        compile_sparse,
        run_sparse,
    )

    batch = rows.shape[0]
    kernel = compile_sparse(batch, HEADS, PADDED, WIDTH, CAP, 0)
    latent = torch.full((batch, PADDED, LATENT), float("nan"), device="cuda")
    latent = latent.bfloat16()
    lse = torch.full((batch, PADDED), float("nan"), device="cuda")
    scale_log2 = SOFTMAX * Q_SCALE * K_SCALE * math.log2(math.e)
    run_sparse(kernel, query, kv, rows, valid, latent, lse, scale_log2, K_SCALE)
    torch.accelerator.synchronize()
    return latent, lse


_PLANS: dict = {}


def _run_prims_ts(query, kv, rows, valid):
    latent, lse = _run_prims_ts_nosync(query, kv, rows, valid)
    torch.accelerator.synchronize()
    return latent, lse


def _run_prims_ts_nosync(query, kv, rows, valid):
    """FlashInfer prims_ts on the plan of the batch's rows (planned once)."""
    from flashinfer.attention.prims_ts import (
        BatchSparseMLADecodePagedTSWrapper,
        get_prims_ts_sparse_mla_decode_workspace_size,
    )

    batch, device = rows.shape[0], torch.device("cuda")
    if batch in _PLANS:
        plan, latent, lse, indptr = _PLANS[batch]
        plan.run(
            query[:batch],
            kv,
            rows,
            valid,
            qo_indptr=indptr,
            softmax_scale=SOFTMAX * Q_SCALE * K_SCALE,
            output_scale=K_SCALE,
            out=latent,
            lse=lse,
            validate=False,
        )
        return latent, lse
    kwargs = dict(
        max_q_len=1,
        q_dtype=torch.float8_e4m3fn,
        max_topk=WIDTH,
        packed_query=True,
        qk_rope_head_dim=64,
        kv_cache_format="native",
    )
    size = get_prims_ts_sparse_mla_decode_workspace_size(
        device, batch, PADDED, return_lse=True, **kwargs
    )
    plan = BatchSparseMLADecodePagedTSWrapper(
        torch.empty(size, dtype=torch.uint8, device=device)
    )
    plan.plan(device, batch, PADDED, return_lse=True, **kwargs)
    latent = torch.empty(batch, PADDED, LATENT, dtype=torch.bfloat16, device=device)
    lse = torch.empty(batch, PADDED, dtype=torch.float32, device=device)
    indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    _PLANS[batch] = plan, latent, lse, indptr
    return _run_prims_ts_nosync(query, kv, rows, valid)


@pytest.mark.parametrize(
    "counts",
    [
        [256, 255, 0, 1, 31],
        [300, 2048, 257, 513, 7],
        [240, 251, 262, 258, 247, 256, 230, 271, 266, 249],
    ],
    ids=["dcp8", "saturated", "ten"],
)
def test_sparse_matches_emulation(counts):
    query, kv, rows, valid = _inputs(counts)
    latent, lse = _run_sparse(query, kv, rows, valid)
    ref, ref_lse = _reference(query, kv, rows, valid, quantize_p=True)
    got = latent[:, :HEADS].float()
    for b, n in enumerate(counts):
        if n == 0:
            assert torch.all(got[b] == 0)
            assert torch.all(torch.isneginf(lse[b, :HEADS]))
            continue
        scale = ref[b].abs().max()
        err = (got[b] - ref[b]).abs().max() / scale
        # One BF16 rounding plus E4M3 P flips at rounding ties (exp2 approx).
        assert err < 2e-2, (b, n, float(err))
        torch.testing.assert_close(lse[b, :HEADS], ref_lse[b], atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("counts", [[256, 255, 1, 31, 200], [300, 2048, 257, 513, 7]])
def test_sparse_error_within_flashinfer(counts):
    if not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    query, kv, rows, valid = _inputs(counts, seed=1)
    exact, exact_lse = _reference(query, kv, rows, valid, quantize_p=False)
    ours, ours_lse = _run_sparse(query, kv, rows, valid)
    theirs, theirs_lse = _run_prims_ts(query, kv, rows, valid)
    for b in range(len(counts)):
        scale = exact[b].abs().max()
        e_ours = ((ours[b, :HEADS].float() - exact[b]) / scale).pow(2).mean().sqrt()
        e_fi = ((theirs[b, :HEADS].float() - exact[b]) / scale).pow(2).mean().sqrt()
        assert e_ours <= 1.25 * e_fi + 1e-4, (b, float(e_ours), float(e_fi))
        torch.testing.assert_close(
            ours_lse[b, :HEADS], theirs_lse[b, :HEADS], atol=2e-3, rtol=1e-4
        )


def test_sparse_push_into_value_mailbox():
    """``push``: the merged rows and base-2 LSEs land in DcpValue's "bf16" mailbox
    slot [parity, rank, row, head % H] of owner head // H (as DcpValue's own push
    of the unpushed rows: no negative zeros; a row with nothing to merge sends
    only EMPTY_LSE), and the q mailbox rows read are re-armed."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.attn_sparse import (
        compile_sparse,
        run_sparse,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
        EMPTY_LSE,
        PAIR_SENT,
    )

    counts = [256, 0, 37, 300, 2048]
    owners, dcp, parity, rank = 12, 8, 1, 3
    query, kv, rows, valid = _inputs(counts, seed=2)
    latent, lse = _run_sparse(query, kv, rows, valid)
    batch = len(counts)
    shape = (dcp, 2, dcp, CAP, owners, LATENT)
    boxes = torch.empty(shape, dtype=torch.int16, device="cuda")
    boxes.view(torch.int32).fill_(PAIR_SENT - (1 << 32))
    lboxes = torch.full((dcp, 2, dcp, CAP, owners), -1, dtype=torch.int32).cuda()
    i64 = dict(dtype=torch.int64, device="cuda")
    mail = SimpleNamespace(
        peers=torch.tensor([boxes[o].data_ptr() for o in range(dcp)], **i64),
        lse_peers=torch.tensor([lboxes[o].data_ptr() for o in range(dcp)], **i64),
    )
    kernel = compile_sparse(batch, HEADS, PADDED, WIDTH, CAP, 0, push=(owners, dcp))
    pushed_q = query.clone()
    out = torch.zeros(batch, PADDED, LATENT, dtype=torch.bfloat16, device="cuda")
    out_lse = torch.zeros(batch, PADDED, device="cuda")
    scale_log2 = SOFTMAX * Q_SCALE * K_SCALE * math.log2(math.e)
    args = (pushed_q, kv, rows, valid, out, out_lse, scale_log2, K_SCALE)
    run_sparse(kernel, *args, push=(mail, parity, rank))
    torch.accelerator.synchronize()
    want = latent.view(torch.int16).clone()
    want[want == -32768] = 0  # no negative zeros
    for b, n in enumerate(counts):
        for gh in range(HEADS):
            owner, h = gh // owners, gh % owners
            got_lse = lboxes[owner, parity, rank, b, h].view(torch.float32)
            row = boxes[owner, parity, rank, b, h]
            if n == 0:
                empty = torch.tensor(EMPTY_LSE, dtype=torch.float32)
                assert float(got_lse) == float(empty)
                assert torch.all(row.view(torch.int32) == PAIR_SENT - (1 << 32))
                continue
            assert torch.equal(row, want[b, gh]), (b, gh)
            expect = float(lse[b, gh]) * math.log2(math.e)
            assert abs(float(got_lse) - expect) <= 1e-5 * abs(expect) + 1e-5
    rearmed = pushed_q[:batch, :HEADS].view(torch.uint8)
    assert torch.all(rearmed == 0xFF)
    assert torch.equal(pushed_q[:batch, HEADS:], query[:batch, HEADS:])
    assert torch.equal(pushed_q[batch:], query[batch:])
