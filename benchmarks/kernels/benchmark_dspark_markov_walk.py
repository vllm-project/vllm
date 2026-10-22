# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark Markov walk: the eager per-step loop vs dspark_markov_walk (sm_90a).

Both arms walk the same base logits. Times are per call (all draft steps),
replayed from a CUDA graph with a cold L2 cache.
"""

from functools import partial
from statistics import median

import torch
import torch.nn.functional as F

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.markov_walk import DSparkMarkovWalker


def eager_walk(
    w1, w2, base, anchor, tokens, idx, temperature, seeds, pos, draft_logits, cols
):
    """The loop of DSparkSpeculator._sample_sequential (full vocabulary)."""
    prev = anchor
    for i in range(base.shape[1]):
        logits = base[:, i] + F.linear(F.embedding(prev, w1), w2)
        if draft_logits is None:
            tok = logits.argmax(dim=-1)
        else:
            tok = gumbel_sample(
                logits,
                idx[:, i],
                temperature,
                seeds,
                pos[:, i] - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=cols[i],
            )
        tokens[:, i] = tok
        prev = tok


def timer(method: str):
    if method == "cupti":
        from flashinfer.testing import bench_gpu_time_with_cupti

        return partial(
            bench_gpu_time_with_cupti, use_cuda_graph=True, cold_l2_cache=True
        )
    from flashinfer.testing import bench_gpu_time_with_cudagraph

    return partial(bench_gpu_time_with_cudagraph, cold_l2_cache=True)


@torch.inference_mode()
def main(args) -> None:
    device = torch.device("cuda")
    bench = timer(args.timer)
    print("vocab steps batch mode | eager_us fused_us speedup | eager W2 GB/s")
    for vocab in args.vocab:
        g = torch.Generator(device=device).manual_seed(0)
        w1 = (torch.randn(vocab, 256, generator=g, device=device) * 0.05).bfloat16()
        w2 = (torch.randn(vocab, 256, generator=g, device=device) * 0.02).bfloat16()
        for steps in args.steps:
            walker = DSparkMarkovWalker(w1, w2, steps)
            walker.warmup()
            max_reqs = max(args.batch)
            draft_logits = torch.zeros(
                max_reqs, steps, vocab, dtype=torch.bfloat16, device=device
            )
            temperature = torch.ones(max_reqs, device=device)
            seeds = torch.arange(max_reqs, dtype=torch.int64, device=device)
            cols = torch.arange(steps, dtype=torch.int32, device=device)
            for batch in args.batch:
                base = (
                    torch.randn(batch, steps, vocab, generator=g, device=device) * 3
                ).bfloat16()
                anchor = torch.randint(0, vocab, (batch,), generator=g, device=device)
                anchor_ids = anchor.int()
                idx = (
                    torch.arange(batch, device=device)[:, None]
                    .expand(batch, steps)
                    .contiguous()
                )
                pos = torch.arange(batch * steps, device=device).view(batch, steps) + 1
                tokens = torch.empty(batch, steps, dtype=torch.int64, device=device)
                for mode in args.mode:
                    cache = draft_logits if mode == "probabilistic" else None
                    temp = temperature if mode == "probabilistic" else None

                    run_eager = partial(
                        eager_walk,
                        w1,
                        w2,
                        base,
                        anchor,
                        tokens,
                        idx,
                        temperature,
                        seeds,
                        pos,
                        cache,
                        cols,
                    )
                    run_fused = partial(
                        walker.walk,
                        base,
                        anchor_ids,
                        1,
                        idx.view(-1),
                        temp,
                        seeds,
                        pos.view(-1),
                        tokens,
                        cache,
                    )
                    eager_us = median(bench(run_eager)) * 1e3
                    fused_us = median(bench(run_fused)) * 1e3
                    gbps = steps * vocab * 256 * 2 / (eager_us * 1e3)
                    print(
                        f"{vocab} {steps} {batch} {mode} | {eager_us:.1f} "
                        f"{fused_us:.1f} {eager_us / fused_us:.2f}x | "
                        f"{gbps:.0f}"
                    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab", type=int, nargs="+", default=[151936, 129280, 248320]
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 7, 16])
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["greedy", "probabilistic"],
        choices=["greedy", "probabilistic"],
    )
    parser.add_argument("--timer", default="cupti", choices=["cupti", "cudagraph"])
    main(parser.parse_args())
