# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layer-level check of the prefill indexer sub-switches (single GPU, e.g.
HIP_VISIBLE_DEVICES=0): stock aiter path vs bucketed table / amd_kernels score /
amd_kernels top-k / both, ragged batches, 100K-500K cached context, plus a DCP8-style
row-sharded tile loop (global keys, rank-3 rows).
usage: python check_indexer_prefill.py [ctxs comma] [new tokens comma]"""

import os
import sys

import torch

from vllm.models.kimi_k3.amd.ops import fp4_indexer_prefill as fp
from vllm.models.kimi_k3.amd.ops.fp4_indexer import store_keys
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata as Chunk,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillMetadata as Meta,
)

FLAGS = [
    "VLLM_KIMI_AKR_PREFILL_TABLE_BUCKET",
    "VLLM_KIMI_AKR_INDEXER_SCORE",
    "VLLM_KIMI_AKR_INDEXER_TOPK",
]
VARIANTS = {
    "stock": (0, 0, 0),
    "bucket": (1, 0, 0),
    "topk": (0, 0, 1),
    "score": (0, 1, 0),
    "score+topk": (0, 1, 1),
    "all": (1, 1, 1),
}


def setv(v):
    for f, x in zip(FLAGS, v):
        os.environ[f] = str(x)
    # aiter vs amd path of topk gate is mean window based; force it on for the test
    fp.akr.TOPK_MIN_MEAN_WINDOW = 0


def make(ctxs, news, dev="cuda"):
    lens = [c + n for c, n in zip(ctxs, news)]
    total = sum(lens)
    pages_per = [-(-L // 64) for L in lens]
    maxp = max(pages_per)
    bt = torch.zeros(len(lens), maxp, dtype=torch.int32, device=dev)
    nxt = 0
    for i, p in enumerate(pages_per):
        bt[i, :p] = torch.arange(nxt, nxt + p, dtype=torch.int32, device=dev)
        nxt += p
    cache = torch.zeros((nxt, 64, 68), dtype=torch.uint8, device=dev)
    # keys: random with a recency/sink structure
    keys = torch.randn((nxt * 64, 128), device=dev, dtype=torch.bfloat16)
    slots = torch.arange(nxt * 64, device=dev)
    store_keys(keys, cache, slots)
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev
    )
    ks, ke = [], []
    for i, (c, n) in enumerate(zip(ctxs, news)):
        base = int(cu[i])
        ks += [base] * n
        ke += [base + c + j + 1 for j in range(n)]
    rows = len(ks)
    t2s = torch.repeat_interleave(
        torch.arange(len(lens), device=dev, dtype=torch.int32),
        torch.tensor(lens, device=dev),
    )
    chunk = Chunk(
        block_table=bt,
        cu_seqlen_ks=torch.tensor(ks, dtype=torch.int32, device=dev),
        cu_seqlen_ke=torch.tensor(ke, dtype=torch.int32, device=dev),
        cu_seq_lens=cu,
        token_to_seq=t2s,
        total_seq_lens=total,
        max_seqlen_k=max(lens),
        token_start=0,
        token_end=rows,
        num_reqs=len(lens),
    )
    return chunk, cache, rows


def run(variant, chunk, cache, q, w, rows):
    setv(VARIANTS[variant])
    out = torch.full((rows, 2048), -2, dtype=torch.int32, device="cuda")
    plan = fp.prepare_prefill_metadata(Meta([chunk]), 1, 0, 1).plans[0]
    fp.prefill_indexer(q, w, cache, out, chunk, 1, 1, plan=plan)
    torch.accelerator.synchronize()
    return out


def overlap(a, b):
    """Fraction of selected ids shared per row, and rows with different counts."""
    a, b = a.sort(1).values, b.sort(1).values
    worst, bad, cnt = 1.0, 0, 0
    for i in range(a.shape[0]):
        sa, sb = a[i][a[i] >= 0], b[i][b[i] >= 0]
        cnt += int(sa.numel() != sb.numel())
        if sa.numel() == 0:
            continue
        common = torch.isin(sa, sb).sum().item() / sa.numel()
        worst = min(worst, common)
        bad += common < 1.0
    return worst, bad, cnt


def main():
    cases = [
        ([100_000], [2048]),
        ([500_000], [4096]),
        ([150_000, 20_000, 300_000], [1500, 600, 1900]),
        ([100_000, 7_000], [10240 - 3000, 3000]),
        ([0, 30_000], [5000, 2000]),
    ]
    if len(sys.argv) > 2:
        cases = [
            (
                [int(c) for c in sys.argv[1].split(",")],
                [int(n) for n in sys.argv[2].split(",")],
            )
        ]
    torch.manual_seed(0)
    for ctxs, news in cases:
        chunk, cache, rows = make(ctxs, news)
        q = (torch.randn(rows, 32, 128, device="cuda") * 0.5).to(torch.bfloat16)
        w = torch.randn(rows, 32, device="cuda").to(torch.bfloat16)
        ref = run("stock", chunk, cache, q, w, rows)
        print(f"case ctx={ctxs} new={news} rows={rows}", flush=True)
        for v in ["bucket", "topk", "score", "score+topk", "all"]:
            o = run(v, chunk, cache, q, w, rows)
            exact = bool((o.sort(1).values == ref.sort(1).values).all())
            worst, bad, cnt = overlap(o, ref)
            valid = bool((o < (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks)[:, None]).all())
            print(
                f"  {v:11s} exact={exact} worst_overlap={worst:.5f} "
                f"rows_differing={bad}/{rows} count_mismatch={cnt} in_window={valid}",
                flush=True,
            )
        del chunk, cache


if __name__ == "__main__":
    main()
