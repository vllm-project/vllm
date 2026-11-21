# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The timed closures are built and run within one loop iteration (no late binding).
# ruff: noqa: B023
"""CUDA-graph replay microbenchmarks (min of 15, L2 flushed before every replay) of the
prefill routes, stock vs integrated.
  KDA: KimiK3DeltaAttention._forward (conv + gates + chunk_kda + o_norm) with
  VLLM_KIMI_AKR_KDA_PREFILL 0 / 1 INDEXER: one DCP-rank tile loop (score + top-k +
  epilogue) with the aiter path vs score_rows / topk_rows
usage: python bench_prefill_routes.py kda|indexer"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
dev = "cuda"
flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)


def graph_min(fn, n=15):
    fn()
    fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(n):
        flush.zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.accelerator.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return min(ts)


def bench_kda():
    from types import SimpleNamespace

    import check_kda_prefill as c

    import vllm.models.kimi_k3.amd.kda as kda
    from vllm.models.kimi_k3.amd import akr_kda_prefill as ak
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention

    layer = c.make_layer()
    print("case,T,stock_us,akr_us,speedup")
    for name, nd, pl, cx in [
        ("1x2048", 0, [2048], [0]),
        ("1x4096 c100K", 0, [4096], [100_000]),
        ("3 ragged 5260", 0, [3000, 1500, 700, 60], [0, 120_000, 40, 300_000]),
        ("1x10240 c200K", 0, [10240], [200_000]),
        ("1x16384 c500K", 0, [16384], [500_000]),
        ("mixed 24 dec + 4096", 24, [4096], [100_000]),
    ]:
        b = c.make_batch(nd, pl, cx)
        res = []
        for akr in (False, True):
            os.environ["VLLM_KIMI_AKR_KDA_PREFILL"] = "1" if akr else "0"
            rec, conv = b["rec"].clone(), b["conv"].clone()
            layer.kv_cache = (conv, rec)
            out = torch.empty(1, b["T"], 12, 128, device=dev, dtype=torch.bfloat16)
            kda.get_forward_context = lambda: SimpleNamespace(  # type: ignore[return-value]
                attn_metadata={layer.prefix: b["m"]}
            )

            def f():
                KimiK3DeltaAttention._forward(
                    layer, b["mixed"], b["g1"], b["g2"], b["beta"], out
                )

            res.append(graph_min(f))
            assert ak.try_prefill is not None
        print(
            f"{name},{b['T']},{res[0]:.0f},{res[1]:.0f},{res[0] / res[1]:.2f}x",
            flush=True,
        )
        b = None  # free the batch before the next case


def bench_indexer():
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
        flydsl_pa_mqa_logits_fp4_prefill,
    )
    from aiter.ops.topk import top_k_per_row_prefill
    from amd_kernels.indexer.mxfp4_indexer_common import (
        mxfp4_prefill_score as score_rows,
    )
    from amd_kernels.indexer.mxfp4_indexer_common import (
        mxfp4_prefill_topk as topk_rows,
    )

    from vllm.models.kimi_k3.amd.ops import fp4_indexer_prefill as fp
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import quantize_queries

    print(
        "case,rows,stock_us(score+topk+epilogue),akr_score_us,akr_topk_us,akr_total_us,speedup"
    )
    for ctx, M in [
        (100_000, 4096),
        (500_000, 4096),
        (150_000, 5720),
        (500_000, 5720),
        (100_000, 10240),
        (150_000, 16384),
    ]:
        total = ctx + M
        pages = (total + 63) // 64
        width = pages * 64
        payload = torch.randint(
            0, 255, (pages, 1, 4, 64, 16), dtype=torch.uint8, device=dev
        )
        scales = torch.randint(
            110, 125, (pages, 1, 4, 64), dtype=torch.uint8, device=dev
        )
        per = -(-M // 8)
        rows = min(per, max(128, (1024**3 - 1) // (4 * width) // 128 * 128))
        base = 3 * per
        en = (
            torch.arange(rows, device=dev, dtype=torch.int32) + base + ctx + 1
        ).contiguous()
        st = torch.zeros(rows, device=dev, dtype=torch.int32)
        q = (torch.randn(rows, 32, 128, device=dev) * 0.5).to(torch.bfloat16)
        wts = torch.randn(rows, 32, device=dev).to(torch.bfloat16)
        qp, qs = quantize_queries(q)
        table = fp.akr.identity_page_table(
            torch.device(dev), fp.akr.page_table_width(pages)
        )
        batches = torch.zeros(rows, device=dev, dtype=torch.int32)
        _, sched, nct = compute_prefill_schedule(
            batches, st, en, 256, max(4096, rows), width
        )
        logits = torch.empty(rows, width, device=dev, dtype=torch.float32)
        sel = torch.full((rows, 2048), -1, device=dev, dtype=torch.int32)

        def stock():
            flydsl_pa_mqa_logits_fp4_prefill(
                qp,
                qs,
                payload,
                scales,
                table,
                wts,
                batches,
                st,
                en,
                width,
                weight_scale=1.0,
                cta_info=sched,
                n_ctas=nct,
                out=logits,
            )
            top_k_per_row_prefill(
                logits=logits,
                rowStarts=st,
                rowEnds=en,
                indices=sel,
                values=None,
                numRows=rows,
                stride0=logits.stride(0),
                stride1=1,
                stable=False,
                k=2048,
            )
            sel.copy_(torch.where(sel >= 0, sel - st[:, None], -1))

        def new_s():
            score_rows(qp, qs, wts, payload, scales, st, en, logits, 1.0)

        def new_t():
            topk_rows(logits, st, en, sel)

        a = graph_min(stock)
        s_ = graph_min(new_s)
        t_ = graph_min(new_t)
        print(
            f"ctx{ctx // 1000}K "
            f"M{M},{rows},{a:.0f},{s_:.0f},{t_:.0f},{s_ + t_:.0f},{a / (s_ + t_):.2f}x",
            flush=True,
        )
        payload = scales = logits = None  # free before the next case


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        (bench_kda if sys.argv[1] == "kda" else bench_indexer)()
