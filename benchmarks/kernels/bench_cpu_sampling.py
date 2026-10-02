# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare CPU sampling paths, including RNG and per-call bucket construction.

Usage:
    .venv/bin/python benchmarks/kernels/bench_cpu_sampling.py --compiled
    .venv/bin/python benchmarks/kernels/bench_cpu_sampling.py \
        --vocab 151936 --batch 16 --threads 4 --output sampling.json

Logits generation and warmup/compilation are excluded from latency. Each
observation averages --iters calls rotating through four distinct input buffers.
"""

import argparse
import json
import platform
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

SampleFn = Callable[[torch.Tensor], torch.Tensor]


def baseline_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


def baseline_seeded_sample(
    logits: torch.Tensor, generators: dict[int, torch.Generator]
) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    # Match the old forward_cpu path, including its redundant global fill.
    q.exponential_()
    for row, generator in generators.items():
        q[row].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def make_inputs(distribution: str, batch: int, vocab: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20261003 + batch * 17 + vocab)
    ring = []
    for index in range(4):
        if distribution == "zipf":
            ranks = torch.arange(1, vocab + 1, dtype=torch.float64)
            base = (-1.1 * ranks.log()).float()
            logits = torch.stack(
                [base[torch.randperm(vocab, generator=generator)] for _ in range(batch)]
            )
        elif distribution == "uniform":
            shifts = torch.arange(batch, dtype=torch.float32) / 8 + index / 4
            logits = shifts[:, None].expand(batch, vocab).contiguous()
        else:
            logits = torch.randn(batch, vocab, generator=generator) * 2
            if distribution == "topgap10":
                positions = torch.randint(vocab, (batch,), generator=generator)
                logits[torch.arange(batch), positions] = logits.amax(dim=1) + 10
        ring.append(logits)
    return ring


def make_methods(
    sampler: TopKTopPSampler, batch: int, compiled: SampleFn | None
) -> dict[str, SampleFn]:
    old_generators = {
        row: torch.Generator().manual_seed(91001 + row) for row in range(batch)
    }
    new_generators = {
        row: torch.Generator().manual_seed(91001 + row) for row in range(batch)
    }
    methods: dict[str, SampleFn] = {
        "torch_eager": baseline_random_sample,
        "torch_request_seeded": lambda logits: baseline_seeded_sample(
            logits, old_generators
        ),
        "bucket_global": lambda logits: sampler.forward_cpu(logits, {}, None, None)[0],
        "bucket_request_seeded": lambda logits: sampler.forward_cpu(
            logits, new_generators, None, None
        )[0],
    }
    if compiled is not None:
        methods["torch_compiled"] = compiled
    return methods


def run_profile(methods: dict[str, SampleFn], ring: list[torch.Tensor], iters: int):
    for name, fn in methods.items():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            for index in range(iters):
                with record_function(name):
                    fn(ring[index % len(ring)])
        print(f"\n{name}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, nargs="+", default=[32000, 128256, 151936])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["zipf", "normal", "uniform", "topgap10"],
        default=["zipf", "normal", "uniform", "topgap10"],
    )
    parser.add_argument("--iters", type=int, default=10, help="Calls per observation")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--output", type=Path, help="Save metadata and timings as JSON")
    parser.add_argument("--profile", action="store_true", help="Profile the final case")
    args = parser.parse_args()
    if (
        min(args.vocab + args.batch + args.threads + [args.iters, args.warmup]) < 1
        or args.repeats < 3
    ):
        parser.error("Sizes, threads, iters and warmup must be positive; repeats >= 3")
    if not current_platform.is_cpu():
        parser.error("This benchmark requires the vLLM CPU backend")
    current_platform.import_kernels()
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261003)
    sampler = TopKTopPSampler()
    order_rng = random.Random(38321)
    report: dict = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_parallel_info": torch.__config__.parallel_info(),
        },
        "settings": {**vars(args), "output": str(args.output) if args.output else None},
        "timing_contract": [
            "FP32 logits are pre-generated in four buffers, rotated on every call.",
            "Every path advances RNG and allocates its outputs inside timing.",
            "Bucket methods call forward_cpu, including torch seed generation, "
            "allocation and rebuilding all buckets on every call.",
            "Per-request generator construction is outside timing for both methods.",
            "The old seeded CPU baseline globally fills noise before overwriting "
            "request-seeded rows; the unseeded baseline uses out-of-place division.",
            "Warmup/compilation is recorded separately; methods are interleaved "
            "in a shuffled order for every observation.",
            "Percentiles describe per-observation mean latency, not individual calls.",
            "No top-k/top-p filtering is requested. These are sampling "
            "microbenchmarks, not end-to-end model throughput.",
        ],
        "warmups": [],
        "unavailable": [],
        "results": [],
    }

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    for threads in args.threads:
        torch.set_num_threads(threads)
        compiled = (
            torch.compile(baseline_random_sample, dynamic=True)
            if args.compiled
            else None
        )
        for batch in args.batch:
            for vocab in args.vocab:
                for distribution in args.distributions:
                    case = {
                        "threads": threads,
                        "batch": batch,
                        "vocab": vocab,
                        "distribution": distribution,
                    }
                    print(f"\n{case}", flush=True)
                    ring = make_inputs(distribution, batch, vocab)
                    methods = make_methods(sampler, batch, compiled)
                    active = {}
                    for name, fn in methods.items():
                        started = time.perf_counter()
                        try:
                            for index in range(args.warmup):
                                output = fn(ring[index % len(ring)])
                            assert output.shape == (batch,)
                        except Exception as error:
                            if name != "torch_compiled":
                                raise
                            report["unavailable"].append(
                                {**case, "method": name, "error": repr(error)}
                            )
                            print(f"  {name}: unavailable: {error}", flush=True)
                            continue
                        report["warmups"].append(
                            {
                                **case,
                                "method": name,
                                "seconds": time.perf_counter() - started,
                            }
                        )
                        active[name] = fn
                    timings: dict[str, list[float]] = {name: [] for name in active}
                    for repeat in range(args.repeats):
                        order = list(active)
                        order_rng.shuffle(order)
                        for name in order:
                            fn = active[name]
                            started_ns = time.perf_counter_ns()
                            for index in range(args.iters):
                                fn(ring[(repeat * args.iters + index) % len(ring)])
                            elapsed_us = (time.perf_counter_ns() - started_ns) / 1000
                            timings[name].append(elapsed_us / args.iters)
                    medians = {
                        name: statistics.median(values)
                        for name, values in timings.items()
                    }
                    for name, values in timings.items():
                        percentiles = statistics.quantiles(
                            values, n=10, method="inclusive"
                        )
                        row = {
                            **case,
                            "method": name,
                            "median_us": medians[name],
                            "p10_us": percentiles[0],
                            "p90_us": percentiles[-1],
                            "observations_us": values,
                        }
                        report["results"].append(row)
                        print(
                            f"  {name:24s} median={medians[name]:10.2f} us "
                            f"p10={percentiles[0]:.2f} p90={percentiles[-1]:.2f}",
                            flush=True,
                        )
                    eager_speedup = medians["torch_eager"] / medians["bucket_global"]
                    seeded_speedup = (
                        medians["torch_request_seeded"]
                        / medians["bucket_request_seeded"]
                    )
                    print(
                        f"  eager/bucket={eager_speedup:.2f}x "
                        f"seeded old/new={seeded_speedup:.2f}x",
                        flush=True,
                    )
                    save()
    if args.profile:
        run_profile(active, ring, args.iters)
    if args.output:
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
