# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

r"""Benchmark watermark PRF latency and peak memory on an accelerator.

Examples:
    python benchmarks/kernels/benchmark_watermark_prf.py --output-json baseline.json
    python benchmarks/kernels/benchmark_watermark_prf.py \
        --implementation baseline=vllm.v1.watermarking.prfs:PhiloxPRF \
        --implementation fused=my_module:FusedPhiloxPRF
"""

import argparse
import gc
import importlib
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF


@dataclass(frozen=True)
class BenchmarkResult:
    implementation: str
    stage: str
    batch_size: int
    vocab_size: int
    context_width: int
    latency_median_ms: float
    latency_p20_ms: float
    latency_p80_ms: float
    peak_memory_mib: float


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_stages(value: str) -> list[str]:
    stages = value.split(",")
    unknown = set(stages) - {"prf", "watermarker"}
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stages: {', '.join(sorted(unknown))}"
        )
    return stages


def _load_implementation(value: str) -> tuple[str, Callable[[int], WatermarkPRF]]:
    name, separator, target = value.partition("=")
    if not separator:
        target = name
        name = target.rsplit(":", 1)[-1]
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError(
            "implementation must be [NAME=]MODULE:ATTRIBUTE"
        )
    factory = getattr(importlib.import_module(module_name), attribute)
    return name, factory


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


def _measure_accelerator(
    operation: Callable[[], Any], warmup: int, iterations: int
) -> tuple[float, float, float, float]:
    for _ in range(warmup):
        output = operation()
    del output
    torch.accelerator.synchronize()

    starts = [torch.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        output = operation()
        end.record()
    del output
    torch.accelerator.synchronize()
    elapsed_ms = [
        start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
    ]

    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()
    baseline_bytes = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    output = operation()
    torch.accelerator.synchronize()
    peak_bytes = torch.accelerator.max_memory_allocated()
    del output

    return (
        statistics.median(elapsed_ms),
        _quantile(elapsed_ms, 0.2),
        _quantile(elapsed_ms, 0.8),
        (peak_bytes - baseline_bytes) / 2**20,
    )


def _check_compatibility(
    implementations: list[tuple[str, Callable[[int], WatermarkPRF]]],
    key: int,
    context_width: int,
    device: torch.device,
) -> None:
    contexts = torch.arange(
        2 * context_width, dtype=torch.int64, device=device
    ).reshape(2, context_width)
    token_ids = torch.arange(1024, dtype=torch.int64, device=device)
    expected = PhiloxPRF(key).uniform(contexts, token_ids)
    for name, factory in implementations:
        actual = factory(key).uniform(contexts, token_ids)
        if not torch.equal(actual, expected):
            raise ValueError(f"{name} does not match PhiloxPRF compatibility output")


def _benchmark_shape(
    name: str,
    factory: Callable[[int], WatermarkPRF],
    stage: str,
    batch_size: int,
    vocab_size: int,
    context_width: int,
    dtype: torch.dtype,
    key: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> BenchmarkResult:
    contexts = torch.randint(
        0,
        vocab_size,
        (batch_size, context_width),
        dtype=torch.int64,
        device=device,
    )
    token_ids = torch.arange(vocab_size, dtype=torch.int64, device=device)
    prf = factory(key)

    if stage == "prf":
        operation = lambda: prf.uniform(contexts, token_ids)
    else:
        logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=device)
        watermarker = GumbelWatermarker(key, context_width, prf)
        operation = lambda: watermarker.sample(
            logits, contexts, lambda values: values.argmax(dim=-1)
        )

    median_ms, p20_ms, p80_ms, peak_memory_mib = _measure_accelerator(
        operation, warmup, iterations
    )
    return BenchmarkResult(
        implementation=name,
        stage=stage,
        batch_size=batch_size,
        vocab_size=vocab_size,
        context_width=context_width,
        latency_median_ms=median_ms,
        latency_p20_ms=p20_ms,
        latency_p80_ms=p80_ms,
        peak_memory_mib=peak_memory_mib,
    )


def _print_results(results: list[BenchmarkResult]) -> None:
    print("implementation\tstage\tbatch\tlatency_ms(p20/median/p80)\tpeak_memory_mib")
    for result in results:
        latency = (
            f"{result.latency_p20_ms:.3f}/{result.latency_median_ms:.3f}/"
            f"{result.latency_p80_ms:.3f}"
        )
        print(
            f"{result.implementation}\t{result.stage}\t{result.batch_size}\t"
            f"{latency}\t{result.peak_memory_mib:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--implementation",
        action="append",
        type=_load_implementation,
        help=(
            "PRF constructor as [NAME=]MODULE:ATTRIBUTE; repeat to compare candidates. "
            "Defaults to the in-tree PhiloxPRF."
        ),
    )
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 8, 32, 256])
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=["prf", "watermarker"],
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--skip-compatibility-check", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.accelerator.is_available():
        parser.error("an accelerator is required")
    if args.vocab_size <= 0 or args.context_width <= 0:
        parser.error("vocab size and context width must be positive")
    if args.warmup <= 0 or args.iterations <= 0:
        parser.error("warmup and iterations must be positive")

    implementations = args.implementation or [("philox", PhiloxPRF)]
    device = torch.device(current_platform.device_type)
    if not args.skip_compatibility_check:
        _check_compatibility(implementations, args.key, args.context_width, device)

    results = []
    for name, factory in implementations:
        for stage in args.stages:
            for batch_size in args.batch_sizes:
                try:
                    result = _benchmark_shape(
                        name,
                        factory,
                        stage,
                        batch_size,
                        args.vocab_size,
                        args.context_width,
                        getattr(torch, args.dtype),
                        args.key,
                        args.warmup,
                        args.iterations,
                        device,
                    )
                except torch.OutOfMemoryError as error:
                    raise RuntimeError(
                        f"out of memory for {name}, {stage}, batch size {batch_size}"
                    ) from error
                results.append(result)

    _print_results(results)
    if args.output_json is not None:
        report = {
            "device": current_platform.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "parameters": {
                "batch_sizes": args.batch_sizes,
                "vocab_size": args.vocab_size,
                "context_width": args.context_width,
                "stages": args.stages,
                "dtype": args.dtype,
                "key": args.key,
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            "results": [asdict(result) for result in results],
        }
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
