# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The timed closures are built and run within one loop iteration (no late binding).
# ruff: noqa: B023
"""Stock vs amd_kernels microbenchmarks of the 8-rank non-layer pieces of Kimi-K3.1:
embedding, lm_head GEMM + logits all-gather / owner exchange, drafter context_proj.

CUDA-graph replay, two-graph differencing (a graph of N1 calls and one of N2 calls: the
per-call time is the difference of their minima over N2 - N1, which removes the fixed
replay skew between the eight processes), 15 rounds, max over ranks, us per call.

Run (spawns 8 ranks, needs 8 GPUs):
    python benchmarks/kernels/benchmark_akr_nonlayer_8rank.py [embed lmhead ctx]
"""

import sys
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

HIDDEN, VOCAB = 7168, 163840


def _graph(fn, n):
    from vllm.distributed.parallel_state import graph_capture

    fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    # vLLM's collectives register their buffers inside graph_capture
    with (
        graph_capture(
            device=torch.device("cuda", torch.accelerator.current_device_index())
        ) as gc,
        torch.cuda.graph(g, stream=gc.stream),
    ):
        for _ in range(n):
            fn()
    g.replay()
    torch.accelerator.synchronize()
    return g


def per_call(fn, n1=4, n2=20, rounds=15):
    """Us per call of fn: (min over rounds of the N2-call graph - min over rounds of the
    N1-call graph) / (N2 - N1), max over ranks."""
    g1, g2 = _graph(fn, n1), _graph(fn, n2)
    best = [float("inf")] * 2
    for _ in range(rounds):
        for i, g in enumerate((g1, g2)):
            torch.distributed.barrier()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            g.replay()
            e.record()
            torch.accelerator.synchronize()
            best[i] = min(best[i], s.elapsed_time(e) * 1e3)
    t = torch.tensor([(best[1] - best[0]) / (n2 - n1)], device="cuda")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return float(t)


def show(rank, name, stock, new):
    if rank == 0:
        print(
            f"  {name:34s} stock {stock:8.1f}   akr {new:8.1f}   x{stock / new:.2f}  "
            "us/call",
            flush=True,
        )


def bench_embed(rank):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.kimi_k3.amd import akr_embed

    if rank == 0:
        print("embedding (VocabParallelEmbedding.forward vs VocabGather)")
    emb = VocabParallelEmbedding(VOCAB, HIDDEN)
    emb.weight.data.normal_()
    ids = {m: torch.randint(0, VOCAB, (m,), device="cuda") for m in (12, 40, 80, 192)}
    stock = {m: per_call(lambda m=m: emb(ids[m])) for m in ids}
    assert akr_embed.install(emb)
    for m in ids:
        show(rank, f"{m} rows", stock[m], per_call(lambda m=m: emb(ids[m])))


def bench_lmhead(rank):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.kimi_k3.amd import akr_lmhead
    from vllm.v1.worker.gpu.sample.batch_shard import all_to_all_logits

    lm_head = ParallelLMHead(VOCAB, HIDDEN)
    lm_head.weight.data.normal_(0, 0.02)
    lp = LogitsProcessor(VOCAB)
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=48),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    assert akr_lmhead.setup("target", lm_head, lp, cfg)
    norm = RMSNorm(HIDDEN, eps=1e-5).to(torch.bfloat16).cuda()
    if rank == 0:
        print("lm_head: GEMM + all-gather (target compute_logits after the norm)")
    for m in (12, 24, 40, 80, 128, 192):
        x = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        stock = per_call(lambda: lp(lm_head, x))
        new = per_call(lambda: akr_lmhead.logits("target", lm_head, x))
        show(rank, f"{m} rows all-gather", stock, new)
    if rank == 0:
        print(
            "lm_head: batch-sharded owner exchange (vLLM compute_logits_local + "
            "all_to_all_logits vs owned)"
        )
    for m in (12, 40, 80, 128, 192):
        counts = [m // 8 + int(r < m % 8) for r in range(8)]
        md = SimpleNamespace(
            tp_size=8, num_logits_per_rank=counts, num_local_logits=counts[rank]
        )
        hidden = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        perm = torch.randperm(m, device="cuda")

        def stock_fn():
            h = norm(hidden[perm], None)
            local = lp(lm_head, h, skip_gather=True)
            return all_to_all_logits(local, md)

        stock = per_call(stock_fn)
        ex = akr_lmhead._state["target"]
        counts_dev = torch.tensor(counts, dtype=torch.int32, device="cuda")

        # owned_logits minus its eager host-to-device counts copy (not capturable)
        def new_fn():
            x = norm(hidden, None)
            return ex.owned(x, lm_head.weight, perm, counts_dev, counts[rank])

        new = per_call(new_fn)
        show(rank, f"{m} rows owner", stock, new)


def bench_ctx(rank):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.amd import akr_draft

    if rank == 0:
        print(
            "drafter context_proj + context_norm (replicated GEMM + norm vs sharded + "
            "gather; includes the packer)"
        )
    w = (torch.randn(HIDDEN, 35840, device="cuda") * 0.01).to(torch.bfloat16)
    norm = RMSNorm(HIDDEN, eps=1e-5).to(torch.bfloat16).cuda()
    model = SimpleNamespace(
        model=SimpleNamespace(
            context_proj=SimpleNamespace(weight=SimpleNamespace(data=w)),
            context_norm=norm,
        )
    )
    ctx = akr_draft.ContextProj(model, torch.device("cuda"))
    ctx.warm_up(torch.device("cuda"))
    for rows in (40, 65, 80, 192, 240):
        aux = [
            torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
            for _ in range(5)
        ]
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        stage = torch.empty(rows, 35840, dtype=torch.bfloat16, device="cuda")

        def stock_fn():
            torch.cat(aux, dim=-1, out=stage)
            out.copy_(norm(stage @ w.t()))

        def new_fn():
            ctx.pack_graph(aux, rows)
            ctx.combine_graph(rows, out)

        show(
            rank, f"R={rows} (cat+GEMM+norm+copy)", per_call(stock_fn), per_call(new_fn)
        )


BENCH = {"embed": bench_embed, "lmhead": bench_lmhead, "ctx": bench_ctx}


def worker(rank, port, names):
    torch.accelerator.set_device_index(rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{rank}")
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        init_test_distributed_environment(8, 1, rank, str(port))
        for n in names:
            BENCH[n](rank)
            torch.accelerator.synchronize()
            torch.distributed.barrier()


if __name__ == "__main__":
    names = sys.argv[1:] or list(BENCH)
    mp.spawn(worker, args=(29700 + hash(tuple(names)) % 200, names), nprocs=8)
