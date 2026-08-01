# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DSpark's sequential Markov-head sampling loop.

The baseline materializes the Markov bias before adding it to request-major
draft backbone logits. The two addmm variants model DSpark's step-major
sampling layout and use either out-of-place ``addmm`` or in-place ``addmm_``.
All variants include the embedding and argmax dependency between speculative
positions. Logit-buffer resets model a fresh LM-head output and occur outside
the measured interval.

Example:

.. code-block:: console

    .venv/bin/python benchmarks/kernels/benchmark_dspark_markov_head.py \
        --batch-sizes 1 2 4 8 16 32 64 --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class BenchmarkRunner:
    run: Callable[[], torch.Tensor]
    reset: Callable[[], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--vocab-size", type=int, default=163840)
    parser.add_argument("--markov-rank", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--mode", choices=("eager", "cudagraph", "both"), default="both"
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--replays-per-sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def summarize(samples_us: Sequence[float]) -> dict[str, Any]:
    mean_us = statistics.mean(samples_us)
    return {
        "median_us": statistics.median(samples_us),
        "p10_us": percentile(samples_us, 0.1),
        "p90_us": percentile(samples_us, 0.9),
        "mean_us": mean_us,
        "cv_pct": statistics.pstdev(samples_us) / mean_us * 100.0,
        "samples_us": list(samples_us),
    }


def make_inputs(
    batch_size: int,
    vocab_size: int,
    markov_rank: int,
    num_speculative_tokens: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    padded_vocab_size = math.ceil(vocab_size / 64) * 64
    base_logits = torch.randn(
        (batch_size, num_speculative_tokens, vocab_size),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    markov_w1 = torch.randn(
        (vocab_size, markov_rank),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    markov_w2 = torch.randn(
        (padded_vocab_size, markov_rank),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    anchors = torch.randint(
        vocab_size,
        (batch_size,),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    return base_logits, markov_w1, markov_w2, anchors


def make_runner(
    implementation: str,
    base_logits: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    anchors: torch.Tensor,
) -> BenchmarkRunner:
    vocab_size = base_logits.shape[-1]
    num_speculative_tokens = base_logits.shape[1]

    if implementation == "baseline":
        pristine_logits = base_logits
        working_logits = torch.empty_like(pristine_logits)

        def runner() -> torch.Tensor:
            previous = anchors
            for step in range(num_speculative_tokens):
                markov_embed = F.embedding(previous, markov_w1)
                bias = F.linear(markov_embed, markov_w2)[..., :vocab_size]
                logits = working_logits[:, step] + bias
                previous = logits.argmax(dim=-1)
            return previous

    elif implementation in ("addmm", "addmm_inplace"):
        # Production obtains this layout by writing DSpark's sample indices in
        # step-major order before the LM head. The one-time conversion here is
        # benchmark setup and deliberately outside the timed sequential loop.
        pristine_logits = base_logits.transpose(0, 1).contiguous()
        working_logits = torch.empty_like(pristine_logits)

        def runner() -> torch.Tensor:
            previous = anchors
            for step in range(num_speculative_tokens):
                markov_embed = F.embedding(previous, markov_w1)
                if implementation == "addmm":
                    logits = torch.addmm(
                        working_logits[step],
                        markov_embed,
                        markov_w2[:vocab_size].t(),
                    )
                else:
                    logits = working_logits[step].addmm_(
                        markov_embed,
                        markov_w2[:vocab_size].t(),
                    )
                previous = logits.argmax(dim=-1)
            return previous

    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    def reset() -> None:
        working_logits.copy_(pristine_logits)

    return BenchmarkRunner(runner, reset)


def capture_runner(
    runner: BenchmarkRunner,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        runner.reset()
        output = runner.run()
    torch.accelerator.synchronize()

    runner.reset()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.run()
    torch.accelerator.synchronize()
    return graph, output


def measure_replays(
    run: Callable[[], object],
    reset: Callable[[], None],
    samples: int,
    replays_per_sample: int,
) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays_per_sample)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays_per_sample)]
    samples_us = []
    for _ in range(samples):
        for replay, (start, end) in enumerate(zip(starts, ends)):
            reset()
            start.record()
            run()
            end.record()
        ends[-1].synchronize()
        elapsed_us = [
            starts[replay].elapsed_time(ends[replay]) * 1000.0
            for replay in range(replays_per_sample)
        ]
        samples_us.append(statistics.mean(elapsed_us))
    return samples_us


def benchmark_eager(
    runner: BenchmarkRunner,
    warmups: int,
    samples: int,
    replays_per_sample: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        runner.reset()
        runner.run()
    torch.accelerator.synchronize()

    samples_us = measure_replays(
        runner.run,
        runner.reset,
        samples,
        replays_per_sample,
    )
    return summarize(samples_us)


def benchmark_cudagraph(
    runner: BenchmarkRunner,
    warmups: int,
    samples: int,
    replays_per_sample: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    graph, output = capture_runner(runner)
    for _ in range(warmups):
        runner.reset()
        graph.replay()
    torch.accelerator.synchronize()

    samples_us = measure_replays(
        graph.replay,
        runner.reset,
        samples,
        replays_per_sample,
    )
    return summarize(samples_us), output


def check_outputs(
    runners: dict[str, BenchmarkRunner],
) -> dict[str, Any]:
    outputs = {}
    for implementation, runner in runners.items():
        runner.reset()
        outputs[implementation] = runner.run()
    torch.accelerator.synchronize()
    baseline_tokens = outputs["baseline"]
    addmm_matches = baseline_tokens == outputs["addmm"]
    inplace_matches = baseline_tokens == outputs["addmm_inplace"]
    return {
        "tokens": baseline_tokens.numel(),
        "matching_tokens": inplace_matches.sum().item(),
        "match_rate": inplace_matches.float().mean().item(),
        "out_of_place_matching_tokens": addmm_matches.sum().item(),
        "out_of_place_match_rate": addmm_matches.float().mean().item(),
    }


def benchmark_case(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    inputs = make_inputs(
        batch_size=batch_size,
        vocab_size=args.vocab_size,
        markov_rank=args.markov_rank,
        num_speculative_tokens=args.num_speculative_tokens,
        dtype=DTYPES[args.dtype],
        seed=args.seed + batch_size,
    )
    runners = {
        implementation: make_runner(
            implementation,
            *inputs,
        )
        for implementation in ("baseline", "addmm", "addmm_inplace")
    }
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "correctness": check_outputs(runners),
        "timings": {},
    }

    modes = ("eager", "cudagraph") if args.mode == "both" else (args.mode,)
    for mode in modes:
        mode_result = {}
        for implementation, runner in runners.items():
            if mode == "eager":
                timing = benchmark_eager(
                    runner,
                    args.warmups,
                    args.samples,
                    args.replays_per_sample,
                )
            else:
                timing, _ = benchmark_cudagraph(
                    runner,
                    args.warmups,
                    args.samples,
                    args.replays_per_sample,
                )
            mode_result[implementation] = timing
        baseline_us = mode_result["baseline"]["median_us"]
        addmm_us = mode_result["addmm"]["median_us"]
        inplace_us = mode_result["addmm_inplace"]["median_us"]
        mode_result["out_of_place_speedup"] = baseline_us / addmm_us
        mode_result["out_of_place_latency_reduction_pct"] = (
            (baseline_us - addmm_us) / baseline_us * 100.0
        )
        mode_result["speedup"] = baseline_us / inplace_us
        mode_result["latency_reduction_pct"] = (
            (baseline_us - inplace_us) / baseline_us * 100.0
        )
        mode_result["inplace_vs_out_of_place_reduction_pct"] = (
            (addmm_us - inplace_us) / addmm_us * 100.0
        )
        result["timings"][mode] = mode_result

    del inputs, runners
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    results = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "config": {
            "batch_sizes": args.batch_sizes,
            "vocab_size": args.vocab_size,
            "markov_rank": args.markov_rank,
            "num_speculative_tokens": args.num_speculative_tokens,
            "dtype": args.dtype,
            "logit_scale": 1.0,
            "mode": args.mode,
            "warmups": args.warmups,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "seed": args.seed,
        },
        "cases": [benchmark_case(args, batch_size) for batch_size in args.batch_sizes],
    }
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
