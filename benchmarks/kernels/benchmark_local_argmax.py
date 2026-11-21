#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the vocab-parallel argmax used by draft-token selection.

Three arms at TP=N: a full-vocab argmax over gathered logits, the elementwise
chain that `LogitsProcessor.get_top_tokens` used before the Triton kernels, and
the current implementation. All arms run in one process, so they share a head,
a GEMM and a clock state.

`--breakdown` additionally times the post-GEMM tail on precomputed logits,
which separates the part this code owns from the GEMM in front of it.

Usage:
    torchrun --nproc-per-node=<N> benchmark_local_argmax.py [options]

Example:
    torchrun --nproc-per-node=8 benchmark_local_argmax.py --breakdown

"""

import argparse
import json
import os

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.model_executor.layers.local_argmax import pack_top, select_top
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=129280)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--sizes", default="1,8,24,64,128,256,512,1024,2048")
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--inner", type=int, default=30)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument(
        "--burn",
        type=float,
        default=15.0,
        help="seconds of GEMM before timing, to settle the clocks",
    )
    p.add_argument("--breakdown", action="store_true")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def burn(device, seconds):
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    start = torch.cuda.Event(enable_timing=True)
    now = torch.cuda.Event(enable_timing=True)
    start.record()
    while True:
        for _ in range(200):
            a = (a @ a).div_(100.0)
        now.record()
        torch.cuda.synchronize()
        if start.elapsed_time(now) / 1000.0 >= seconds:
            return


def timed(fn, reps, inner):
    for _ in range(3 * inner):
        fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        per.append(start.elapsed_time(end) * 1000.0 / inner)
    per.sort()
    return per[len(per) // 2]


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    config = VllmConfig()
    ctx = set_current_vllm_config(config)
    ctx.__enter__()
    init_distributed_environment(
        world_size=world,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    head = ParallelLMHead(args.vocab, args.hidden, params_dtype=torch.bfloat16).to(
        device
    )
    processor = LogitsProcessor(args.vocab)
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(4242 + rank)
        head.weight.copy_(
            torch.randn(*head.weight.shape, generator=gen).mul_(0.02).to(torch.bfloat16)
        )
    vocab_start = head.shard_indices.org_vocab_start_index

    def before_tail(logits, rows):
        """The elementwise chain that the Triton kernels replaced."""
        values, indices = logits.max(dim=-1)
        global_indices = indices + vocab_start
        if world == 1:
            return global_indices
        pair = torch.stack([values.float(), global_indices.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        gathered = gathered.view(rows, world, 2)
        winner = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top = gathered[:, :, 1].gather(dim=-1, index=winner)
        return top.squeeze(-1).to(torch.int64)

    def after_tail(logits, rows):
        pair = pack_top(logits, vocab_start)
        if world == 1:
            return pair[:, 1].to(torch.int64)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        return select_top(gathered.view(rows, world, 2), world)

    def before(hidden, rows):
        logits = processor._apply_head(head, hidden, None)
        processor._mask_outside_vocab(logits, head)
        return before_tail(logits, rows)

    def make_hidden(rows):
        gen = torch.Generator(device="cpu").manual_seed(100 + rows)
        return torch.randn(rows, args.hidden, generator=gen).to(device, torch.bfloat16)

    burn(device, args.burn)
    dist.barrier()

    sizes = [int(x) for x in args.sizes.split(",")]
    results = {
        "world": world,
        "vocab": args.vocab,
        "hidden": args.hidden,
        "sweep": {},
        "breakdown": {},
    }

    for p in range(1, args.passes + 1):
        log(f"\n--- pass {p}/{args.passes} ---")
        log(
            f"{'rows':>6} {'full us':>9} {'before us':>10} {'after us':>9} "
            f"{'after/before':>13} {'after/full':>11} {'identical':>10}"
        )
        for rows in sizes:
            hidden = make_hidden(rows)
            with torch.no_grad():
                same = bool(
                    torch.equal(
                        before(hidden, rows),
                        processor.get_top_tokens(head, hidden),
                    )
                )
                t_full = timed(
                    lambda h=hidden: processor(head, h).argmax(-1),
                    args.reps,
                    args.inner,
                )
                t_before = timed(
                    lambda h=hidden, r=rows: before(h, r),
                    args.reps,
                    args.inner,
                )
                t_after = timed(
                    lambda h=hidden: processor.get_top_tokens(head, h),
                    args.reps,
                    args.inner,
                )
            if p == args.passes:
                results["sweep"][rows] = {
                    "full_us": t_full,
                    "before_us": t_before,
                    "after_us": t_after,
                    "identical": same,
                }
            log(
                f"{rows:>6} {t_full:>9.1f} {t_before:>10.1f} {t_after:>9.1f} "
                f"{t_before / t_after:>12.2f}x {t_full / t_after:>10.2f}x "
                f"{str(same):>10}"
            )

    if args.breakdown:
        log("\n--- post-GEMM tail on precomputed logits ---")
        log(
            f"{'rows':>6} {'prefix us':>10} {'before us':>10} {'after us':>9} "
            f"{'pack us':>8} {'gather us':>10} {'select us':>10}"
        )
        for rows in sizes:
            hidden = make_hidden(rows)
            with torch.no_grad():
                logits = processor._apply_head(head, hidden, None)
                processor._mask_outside_vocab(logits, head)

                def prefix(h=hidden):
                    lg = processor._apply_head(head, h, None)
                    processor._mask_outside_vocab(lg, head)

                t_prefix = timed(prefix, args.reps, args.inner)
                t_before = timed(
                    lambda lg=logits, r=rows: before_tail(lg, r),
                    args.reps,
                    args.inner,
                )
                t_after = timed(
                    lambda lg=logits, r=rows: after_tail(lg, r),
                    args.reps,
                    args.inner,
                )
                t_pack = timed(
                    lambda lg=logits: pack_top(lg, vocab_start),
                    args.reps,
                    args.inner,
                )
                pair = pack_top(logits, vocab_start)
                t_gather = timed(
                    lambda pr=pair: tensor_model_parallel_all_gather(pr, dim=-1),
                    args.reps,
                    args.inner,
                )
                gathered = tensor_model_parallel_all_gather(pair, dim=-1).view(
                    rows, world, 2
                )
                t_select = timed(
                    lambda g=gathered: select_top(g, world),
                    args.reps,
                    args.inner,
                )
            results["breakdown"][rows] = {
                "prefix_us": t_prefix,
                "before_tail_us": t_before,
                "after_tail_us": t_after,
                "pack_us": t_pack,
                "allgather_us": t_gather,
                "select_us": t_select,
            }
            log(
                f"{rows:>6} {t_prefix:>10.1f} {t_before:>10.1f} "
                f"{t_after:>9.1f} {t_pack:>8.1f} {t_gather:>10.1f} "
                f"{t_select:>10.1f}"
            )

    if rank == 0:
        log(
            "\nidentical everywhere:",
            all(v["identical"] for v in results["sweep"].values()),
        )
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            log("wrote", args.output_json)


if __name__ == "__main__":
    main()
