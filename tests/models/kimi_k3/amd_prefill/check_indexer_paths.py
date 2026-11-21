# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Which implementation each sub-switch variant calls, DCP8 row-sharded tile loop vs
stock, and the per-step JIT of the unbucketed aiter scorer (wall time of the first
prefill_indexer call per new total key count)."""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import aiter.ops.topk as AT
import amd_kernels.indexer.mxfp4_indexer_common as IX
import check_indexer_prefill as c

from vllm.models.kimi_k3.amd.ops import fp4_indexer_prefill as fp

calls = {"score": 0, "topk": 0, "aiter_topk": 0}
S_orig, T_orig, A_orig = (
    IX.mxfp4_prefill_score,
    IX.mxfp4_prefill_topk,
    AT.top_k_per_row_prefill,
)
IX.mxfp4_prefill_score = lambda *a, **k: (
    calls.__setitem__("score", calls["score"] + 1),  # type: ignore[func-returns-value]
    S_orig(*a, **k),
)[1]
IX.mxfp4_prefill_topk = lambda *a, **k: (
    calls.__setitem__("topk", calls["topk"] + 1),  # type: ignore[func-returns-value]
    T_orig(*a, **k),
)[1]
AT.top_k_per_row_prefill = lambda *a, **k: (
    calls.__setitem__("aiter_topk", calls["aiter_topk"] + 1),  # type: ignore[func-returns-value]
    A_orig(*a, **k),
)[1]

torch.manual_seed(0)
chunk, cache, rows = c.make([200_000], [4096])
q = (torch.randn(rows, 32, 128, device="cuda") * 0.5).to(torch.bfloat16)
w = torch.randn(rows, 32, device="cuda").to(torch.bfloat16)
for v in ["stock", "bucket", "score", "topk", "score+topk", "all"]:
    for k in calls:
        calls[k] = 0
    c.run(v, chunk, cache, q, w, rows)
    print(f"{v:11s} calls {calls}", flush=True)

# gate: default mean-window threshold (60K) with short windows must use aiter top-k
fp.akr.TOPK_MIN_MEAN_WINDOW = 60_000
short, scache, srows = c.make([20_000], [2048])
sq = q[:srows]
sw = w[:srows]
c.setv((1, 1, 1))
fp.akr.TOPK_MIN_MEAN_WINDOW = 60_000
for k in calls:
    calls[k] = 0
out = torch.empty(srows, 2048, dtype=torch.int32, device="cuda")
plan = fp.prepare_prefill_metadata(c.Meta([short]), 1, 0, 1).plans[0]
fp.prefill_indexer(sq, sw, scache, out, short, 1, 1, plan=plan)
torch.accelerator.synchronize()
print("short windows (mean 21K) all-on:", calls, flush=True)
del short, scache

# DCP8-style: rank 3 of 8 owns rows [3*per, 4*per) of the chunk; global ks/ke; keys
# staged locally (world=1 stage_keys on the full cache); score_rows+topk_rows vs aiter
# on identical staged keys.
chunk2, cache2, rows2 = c.make([150_000, 40_000], [3000, 1000])
per = -(-rows2 // 8)
payload, scales = fp.stage_keys(cache2, chunk2, 1, 1)
ks, ke = chunk2.cu_seqlen_ks, chunk2.cu_seqlen_ke
q2 = (torch.randn(rows2, 32, 128, device="cuda") * 0.5).to(torch.bfloat16)
w2 = torch.randn(rows2, 32, device="cuda").to(torch.bfloat16)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (  # noqa: E402
    compute_prefill_schedule,
    flydsl_pa_mqa_logits_fp4_prefill,
)

from vllm.models.kimi_k3.amd.ops.fp4_indexer import quantize_queries  # noqa: E402

bad = 0
for rank in range(8):
    lo, hi = rank * per, min((rank + 1) * per, rows2)
    st, en = ks[lo:hi].contiguous(), ke[lo:hi].contiguous()
    qp, qs = quantize_queries(q2[lo:hi])
    width = payload.shape[0] * 64
    n = hi - lo
    lg0 = torch.empty(n, width, device="cuda")
    lg1 = torch.empty(n, width, device="cuda")
    pages = payload.shape[0]
    table = fp.akr.identity_page_table(
        torch.device("cuda"), fp.akr.page_table_width(pages)
    )
    batches = torch.zeros(n, device="cuda", dtype=torch.int32)
    _, sched, nct = compute_prefill_schedule(batches, st, en, 256, max(4096, n), width)
    flydsl_pa_mqa_logits_fp4_prefill(
        qp,
        qs,
        payload,
        scales,
        table,
        w2[lo:hi],
        batches,
        st,
        en,
        width,
        weight_scale=1.0,
        cta_info=sched,
        n_ctas=nct,
        out=lg0,
    )
    S_orig(qp, qs, w2[lo:hi], payload, scales, st, en, lg1, 1.0)
    pos = torch.arange(width, device="cuda")[None]
    m = (pos >= st[:, None]) & (pos < en[:, None])
    diff = ((lg0 - lg1).abs() * m).max().item()
    ref = torch.full((n, 2048), -1, dtype=torch.int32, device="cuda")
    A_orig(
        logits=lg0,
        rowStarts=st,
        rowEnds=en,
        indices=ref,
        values=None,
        numRows=n,
        stride0=lg0.stride(0),
        stride1=1,
        stable=False,
        k=2048,
    )
    ref = torch.where(ref >= 0, ref - st[:, None], -1)
    got = torch.full((n, 2048), -2, dtype=torch.int32, device="cuda")
    T_orig(lg1, st, en, got)
    # compare as sets
    a, b = ref.sort(1).values, got.sort(1).values
    mism = (a != b).any(1).sum().item()
    bad += mism
    print(
        f"DCP8-style rank {rank}: rows {n} max|logit diff| in window {diff:.2e}, rows "
        f"with different top-k sets {mism}",
        flush=True,
    )


# JIT per new width: aiter scorer, bucketed vs unbucketed, wall time of the first call
# at each new total
def first_call(bucket, lens):
    c.setv((1 if bucket else 0, 0, 0))
    res = []
    for L in lens:
        ch, ca, r = c.make([L], [2048])
        qq = q[:r]
        ww = w[:r]
        o = torch.empty(r, 2048, dtype=torch.int32, device="cuda")
        torch.accelerator.synchronize()
        t = time.time()
        pl = fp.prepare_prefill_metadata(c.Meta([ch]), 1, 0, 1).plans[0]
        fp.prefill_indexer(qq, ww, ca, o, ch, 1, 1, plan=pl)
        torch.accelerator.synchronize()
        res.append((time.time() - t) * 1e3)
        del ch, ca
    return res


lens = [90_001, 91_000, 93_333, 95_017, 98_500, 101_111]
print(
    "unbucketed first-call ms per new total:",
    [f"{x:.0f}" for x in first_call(False, lens)],
    flush=True,
)
lens2 = [90_050, 91_500, 93_800, 95_517, 98_900, 101_511]
print(
    "bucketed   first-call ms per new total:",
    [f"{x:.0f}" for x in first_call(True, lens2)],
    flush=True,
)
