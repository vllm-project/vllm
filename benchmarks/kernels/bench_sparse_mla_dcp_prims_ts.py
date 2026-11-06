# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 sparse MLA under DCP8: the whole attention op of one layer on one
rank, FLASHINFER_MLA_SPARSE on prims_ts (per-tensor fp8, nvfp4_ds_mla) vs
FLASHMLA_SPARSE (fp8_ds_mla, nvfp4_ds_mla).

The op is the backend's forward_mqa on the DCP-gathered 96-head query: the
global top-k -> local physical index conversion, head padding, the kernel, and
the output/LSE fix-up the DCP combine consumes. FlashMLA's per-step scheduler
metadata is built once outside the timed region (the builder's job; layers
share it). FlashMLA reads a bf16 query, prims_ts an E4M3 one quantized before
the DCP all-gather (its per-rank cost is printed as "qquant").

Decode is timed as CUDA graph replays, prefill (16K-token chunks) eagerly, each
after a 256 MiB L2 flush; median of --iters. prims_ts plans row capacities as
the backend does: powers of two up to 4096 plus every CUDA graph size, here the
decode row count. --pow2-only drops the latter, so decode runs on the next
larger power-of-two capacity (a row count that is not a graph size).

  python benchmarks/kernels/bench_sparse_mla_dcp_prims_ts.py \
      [--prefill] [--pow2-only]
"""

import argparse
import math
from types import SimpleNamespace

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims
from vllm.v1.attention.backends.mla import flashmla_sparse as fms
from vllm.v1.attention.ops.flashmla import get_mla_metadata

DCP, HEADS, TOPK, BLOCK = 8, 96, 2048, 64
# Packed layout: other layers' pages between this layer's blocks.
STRIDE_PAGES = 2
SCALE = 192**-0.5
ROW_BYTES = {"fp8": 576, "fp8_ds_mla": 656, "nvfp4_ds_mla": 352}
POW2_ONLY = False


def agentx_contexts(n: int, seed: int) -> list[int]:
    """AgentX-like decode contexts: log-normal (median 72K, p90 165K), 15% short."""
    rng = np.random.default_rng(seed)
    ctx = np.exp(rng.normal(np.log(72_000), np.log(165_000 / 72_000) / 1.2816, n))
    short = rng.random(n) < 0.15
    ctx[short] = rng.integers(256, 8_192, int(short.sum()))
    return [int(c) for c in np.clip(ctx, 256, 1 << 20)]


def prefill_chunks() -> dict[str, list[tuple[int, int]]]:
    """16K-token chunks: (new tokens, prior context) per request."""
    rng = np.random.default_rng(7)
    mixed, budget = [], 16_384
    while budget > 0:
        new = int(min(budget, rng.integers(512, 8_192)))
        mixed.append((new, agentx_contexts(1, int(rng.integers(1 << 30)))[0]))
        budget -= new
    return {
        "fresh16k": [(16_384, 0)],
        "ctx32k": [(16_384, 32_768)],
        "ctx128k": [(16_384, 131_072)],
        "mixed": mixed,
    }


class Problem:
    """One rank's (rank 0) local cache of every request, written by the
    production writer, and the rows' global top-k (logical token ids)."""

    def __init__(self, rows: list[tuple[int, int]], q_len: int, dtype: str, seed=0):
        """``rows``: (request, position) of every query row; ``q_len`` only
        names the request layout. Request r has context ``self.lens[r]``."""
        self.dtype = dtype
        dev = self.dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(seed)
        reqs = max(r for r, _ in rows) + 1
        lens = [0] * reqs
        for r, p in rows:
            lens[r] = max(lens[r], p + 1)
        self.lens = lens
        local = [math.ceil(n / DCP) for n in lens]
        bpr = [math.ceil(n / BLOCK) + 1 for n in local]
        nblocks = 1 + sum(bpr)
        raw = torch.zeros(
            (nblocks, STRIDE_PAGES * BLOCK, ROW_BYTES[dtype]),
            dtype=torch.uint8,
            device=dev,
        )
        self.cache = raw[:, :BLOCK]
        width = max(bpr)
        table = torch.zeros((reqs, width), dtype=torch.int32, device=dev)
        start = 1
        for r in range(reqs):
            table[r, : bpr[r]] = torch.arange(start, start + bpr[r], device=dev)
            start += bpr[r]
        self.block_table = table
        # Write every local token (rank 0 owns global tokens 0, 8, 16, ...).
        scale = torch.ones(1, dtype=torch.float32, device=dev)
        for r in range(reqs):
            n = local[r]
            pos = torch.arange(n, device=dev)
            slots = table[r, pos // BLOCK].long() * BLOCK + pos % BLOCK
            for s in range(0, n, 1 << 16):
                e = min(n, s + (1 << 16))
                kv_c = torch.randn(e - s, 512, device=dev, generator=g) * 0.17
                k_pe = torch.randn(e - s, 64, device=dev, generator=g) * 0.5
                ops.concat_and_cache_mla(
                    kv_c.bfloat16(),
                    k_pe.bfloat16(),
                    self.cache,
                    slots[s:e],
                    kv_cache_dtype=dtype,
                    scale=scale,
                )
        t = len(rows)
        self.req_id = torch.tensor([r for r, _ in rows], dtype=torch.int32, device=dev)
        topk = torch.full((t, TOPK), -1, dtype=torch.int32, device=dev)
        for i, (_, p) in enumerate(rows):
            c = min(TOPK, p + 1)
            if c == p + 1:
                topk[i, :c] = torch.arange(c, device=dev, dtype=torch.int32)
            else:
                sel = torch.randint(0, p + 1, (c,), device=dev, generator=g)
                topk[i, :c] = sel.int()
        self.topk = topk
        q = torch.randn(t, HEADS, 576, device=dev, generator=g) * 2
        self.q_bf16 = q.bfloat16()
        self.q_fp8 = self.q_bf16.to(torch.float8_e4m3fn)

    def metadata(self, **extra):
        return SimpleNamespace(
            num_decode_tokens=self.topk.shape[0],
            num_actual_tokens=self.topk.shape[0],
            block_size=BLOCK,
            req_id_per_token=self.req_id,
            block_table=self.block_table,
            cp_kv_cache_interleave_size=1,
            **extra,
        )


def _common(impl, p: Problem):
    impl._indexer = None
    impl._topk_indices_buffer = p.topk
    impl.index_group = None
    impl.dcp_world_size, impl.dcp_rank = DCP, 0
    impl.need_to_return_lse_for_decode = True
    impl.kv_lora_rank, impl.qk_rope_head_dim = 512, 64
    impl.num_heads = HEADS // DCP


def flashinfer_op(p: Problem):
    impl = object.__new__(fims.FlashInferMLASparseImpl)
    _common(impl, p)
    impl.group_conversion_decode_only = False
    impl.scale = SCALE
    impl.use_prims_ts = True
    impl.prims_ts_format = fims._PRIMS_TS_FORMATS[p.dtype]
    buckets = {1 << i for i in range(13)}
    if not POW2_ONLY:
        buckets.add(min(p.topk.shape[0], 4096))
    impl.prims_ts_buckets = tuple(sorted(buckets))
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    md = p.metadata()
    cache = p.cache.view(torch.float8_e4m3fn) if p.dtype == "fp8" else p.cache
    return lambda: impl.forward_mqa(p.q_fp8, cache, md, layer)


def flashmla_op(p: Problem):
    impl = object.__new__(fms.FlashMLASparseImpl)
    _common(impl, p)
    impl.kv_cache_dtype = p.dtype
    impl.softmax_scale = SCALE
    impl.q_head_size = 576
    sched = fms.FlashMLASparseMetadata.FP8KernelMetadata(
        scheduler_metadata=get_mla_metadata()[0],
        dummy_block_table=torch.zeros((1, 1), dtype=torch.int32, device=p.dev),
        cache_lens=torch.full((1,), 1 << 20, dtype=torch.int32, device=p.dev),
    )
    md = p.metadata(fp8_use_mixed_batch=True, fp8_extra_metadata=sched)
    return lambda: impl.forward_mqa(p.q_bf16, p.cache, md, None)


class Timer:
    def __init__(self):
        self.flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")

    def __call__(self, fn, iters: int, graph: bool) -> float:
        fn()
        torch.accelerator.synchronize()
        if graph:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.accelerator.synchronize()
            with torch.cuda.graph(g):
                fn()
            fn = g.replay
        times = []
        for _ in range(iters):
            self.flush.zero_()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            fn()
            b.record()
            torch.accelerator.synchronize()
            times.append(a.elapsed_time(b) * 1e3)
        return float(np.median(times))


def qquant_us(timer: Timer, rows: int, iters: int) -> float:
    """The layer's per-rank static E4M3 query quantization (12 local heads)."""
    nope = torch.randn(rows, HEADS // DCP, 512, device="cuda").bfloat16()
    pe = torch.randn(rows, HEADS // DCP, 64, device="cuda").bfloat16()
    scale = torch.ones(1, dtype=torch.float32, device="cuda")

    def fn():
        q = torch.cat((nope, pe), -1).view(rows, -1)
        ops.scaled_fp8_quant(q, scale)

    return timer(fn, iters, graph=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--pow2-only", action="store_true")
    args = ap.parse_args()
    global POW2_ONLY
    POW2_ONLY = args.pow2_only
    timer = Timer()
    paths = [
        ("FI fp8", "fp8", flashinfer_op),
        ("FI nvfp4", "nvfp4_ds_mla", flashinfer_op),
        ("FM fp8_ds_mla", "fp8_ds_mla", flashmla_op),
        ("FM nvfp4", "nvfp4_ds_mla", flashmla_op),
    ]
    if not args.prefill:
        print("decode DCP8 rank 0, us per layer (CUDA graph, L2 flushed)")
        print(f"{'shape':22s} " + " ".join(f"{n:>14s}" for n, _, _ in paths))
        for b in (1, 16, 64):
            for q in (1, 5):
                for name, ctx in (
                    ("agentx", agentx_contexts(b, seed=b)),
                    ("uniform128k", [131_072] * b),
                ):
                    rows = [(r, ctx[r] - q + j) for r in range(b) for j in range(q)]
                    res = []
                    for _, dtype, op in paths:
                        p = Problem(rows, q, dtype)
                        res.append(timer(op(p), args.iters, graph=True))
                        del p
                    qq = qquant_us(timer, b * q, args.iters)
                    print(
                        f"B{b:<3d} q{q} {name:13s} "
                        + " ".join(f"{t:14.1f}" for t in res)
                        + f"   qquant {qq:.1f}",
                        flush=True,
                    )
        return
    print("prefill DCP8 rank 0, 16K-token chunk, us per layer (eager, L2 flushed)")
    print(f"{'chunk':12s} " + " ".join(f"{n:>14s}" for n, _, _ in paths))
    for name, reqs in prefill_chunks().items():
        rows = [
            (r, prior + i) for r, (new, prior) in enumerate(reqs) for i in range(new)
        ]
        res = []
        for _, dtype, op in paths:
            p = Problem(rows, 1, dtype)
            res.append(timer(op(p), max(5, args.iters // 5), graph=False))
            del p
            torch.accelerator.empty_cache()
        print(f"{name:12s} " + " ".join(f"{t:14.1f}" for t in res), flush=True)


if __name__ == "__main__":
    main()
