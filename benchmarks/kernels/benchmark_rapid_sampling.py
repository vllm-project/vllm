#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Benchmark the rapid-sampling CUDA sampler.

The sweep mirrors the upstream rapid-sampling benchmark: batch size, vocab
size, top-p, top-k, and a small set of logit distributions. The measured
interval contains only the sampler call; logits and parameter tensors are
created outside the timing loop.
"""

import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402
from vllm.v1.sample.ops.topk_topp_sampler import (
    flashinfer_sample,
    rapid_sample,
    rapid_sample_input_supported,
)  # noqa: E402


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    vocab_size: int
    top_p: float
    top_k: int
    logit_type: int


def create_logits(config: BenchmarkConfig, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (config.batch_size, config.vocab_size)

    if config.logit_type == 0:
        return torch.zeros(shape, dtype=torch.float32, device="cuda")
    if config.logit_type == 1:
        return torch.rand(
            shape, dtype=torch.float32, device="cuda", generator=generator
        )
    if config.logit_type == 2:
        return (
            torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
            / 2
        )
    if config.logit_type == 3:
        return (
            torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
            * 3
        )
    if config.logit_type == 4:
        return (
            torch.rand(shape, dtype=torch.float32, device="cuda", generator=generator)
            * 4
        )
    raise ValueError(f"Unsupported logit_type: {config.logit_type}")


def make_rapid_args(
    config: BenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    top_k = torch.full(
        (config.batch_size,), config.top_k, dtype=torch.int32, device="cuda"
    )
    top_p = torch.full(
        (config.batch_size,), config.top_p, dtype=torch.float32, device="cuda"
    )
    return top_k, top_p


def make_flashinfer_args(
    config: BenchmarkConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    top_k = None
    if 0 < config.top_k < config.vocab_size:
        top_k = torch.full(
            (config.batch_size,), config.top_k, dtype=torch.int32, device="cuda"
        )

    top_p = None
    if config.top_p < 1.0:
        top_p = torch.full(
            (config.batch_size,), config.top_p, dtype=torch.float32, device="cuda"
        )
    return top_k, top_p


def benchmark_cuda_call(fn, warmup_iters: int, benchmark_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.accelerator.synchronize()

    times: list[float] = []
    for _ in range(benchmark_iters):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


def run_config(
    config: BenchmarkConfig,
    providers: list[str],
    warmup_iters: int,
    benchmark_iters: int,
) -> dict:
    logits = create_logits(config, seed=2026 + config.logit_type)
    if not rapid_sample_input_supported(logits):
        raise ValueError(
            "rapid-sampling requires CUDA float32 logits with vocab size "
            "in (0, 1048576] and divisible by 4."
        )

    result: dict[str, object] = asdict(config)

    if "rapid" in providers:
        rapid_top_k, rapid_top_p = make_rapid_args(config)
        result["rapid_ms"] = benchmark_cuda_call(
            lambda: rapid_sample(logits, rapid_top_k, rapid_top_p),
            warmup_iters,
            benchmark_iters,
        )

    if "rapid_penalty" in providers:
        rapid_top_k, rapid_top_p = make_rapid_args(config)
        penalties = torch.zeros_like(logits)
        presence_penalties = torch.full(
            (config.batch_size,), 0.1, dtype=torch.float32, device="cuda"
        )
        repetition_penalties = torch.full(
            (config.batch_size,), 0.1, dtype=torch.float32, device="cuda"
        )
        penalty_decays = torch.full(
            (config.batch_size,), 0.996, dtype=torch.float32, device="cuda"
        )
        result["rapid_penalty_ms"] = benchmark_cuda_call(
            lambda: rapid_sample(
                logits,
                rapid_top_k,
                rapid_top_p,
                penalties=penalties,
                presence_penalties=presence_penalties,
                repetition_penalties=repetition_penalties,
                penalty_decays=penalty_decays,
            ),
            warmup_iters,
            benchmark_iters,
        )

    if "flashinfer" in providers:
        flashinfer_top_k, flashinfer_top_p = make_flashinfer_args(config)
        if flashinfer_top_k is None and flashinfer_top_p is None:
            result["flashinfer_ms"] = None
        else:
            result["flashinfer_ms"] = benchmark_cuda_call(
                lambda: flashinfer_sample(
                    logits,
                    flashinfer_top_k,
                    flashinfer_top_p,
                ),
                warmup_iters,
                benchmark_iters,
            )

    rapid_ms = result.get("rapid_ms")
    flashinfer_ms = result.get("flashinfer_ms")
    if isinstance(rapid_ms, float) and isinstance(flashinfer_ms, float):
        result["flashinfer_over_rapid"] = flashinfer_ms / rapid_ms

    return result


def create_configs(
    batch_sizes: list[int],
    vocab_sizes: list[int],
    top_ps: list[float],
    top_ks: list[int],
    logit_types: list[int],
) -> list[BenchmarkConfig]:
    configs = []
    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            for top_p in top_ps:
                for top_k in top_ks:
                    for logit_type in logit_types:
                        configs.append(
                            BenchmarkConfig(
                                batch_size=batch_size,
                                vocab_size=vocab_size,
                                top_p=top_p,
                                top_k=top_k,
                                logit_type=logit_type,
                            )
                        )
    return configs


def print_result(result: dict) -> None:
    parts = [
        f"batch={result['batch_size']}",
        f"vocab={result['vocab_size']}",
        f"top_p={result['top_p']}",
        f"top_k={result['top_k']}",
        f"logit_type={result['logit_type']}",
    ]
    for key in ("rapid_ms", "rapid_penalty_ms", "flashinfer_ms"):
        if key in result:
            value = result[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}=skipped")
    if "flashinfer_over_rapid" in result:
        parts.append(f"flashinfer/rapid={result['flashinfer_over_rapid']:.2f}x")
    print("  " + ", ".join(parts))


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark rapid-sampling against FlashInfer."
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--top-ps", type=float, nargs="+", default=[0.5, 0.9, 1.0])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[-1, 50, 1000])
    parser.add_argument("--logit-types", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["rapid", "flashinfer", "rapid_penalty"],
        default=["rapid", "flashinfer"],
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument(
        "--reference-sweep",
        action="store_true",
        help="Use the full upstream rapid-sampling sweep.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    providers = list(dict.fromkeys(args.providers))
    if "flashinfer" in providers and not flashinfer_available():
        providers.remove("flashinfer")
        print("FlashInfer is not installed; skipping FlashInfer provider.")

    if args.reference_sweep:
        args.batch_sizes = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        args.vocab_sizes = [262144, 151936, 131072, 100256, 65536, 50304, 32768, 256]
        args.top_ps = [0.1, 0.3, 0.7, 0.9, 1.0]
        args.top_ks = [-1, 1, 10, 100, 1000, 10000]
        args.logit_types = [0, 1, 2, 3, 4]

    configs = create_configs(
        args.batch_sizes,
        args.vocab_sizes,
        args.top_ps,
        args.top_ks,
        args.logit_types,
    )

    if not args.quiet:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Providers: {providers}")
        print(f"Total configurations: {len(configs)}")

    results = []
    for idx, config in enumerate(configs, start=1):
        if not args.quiet:
            print(f"[{idx}/{len(configs)}]")
        result = run_config(
            config,
            providers,
            args.warmup_iters,
            args.benchmark_iters,
        )
        results.append(result)
        if not args.quiet:
            print_result(result)

    if args.save_json is not None:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "providers": providers,
            "warmup_iters": args.warmup_iters,
            "benchmark_iters": args.benchmark_iters,
            "results": results,
        }
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved results to {args.save_json}")

    if args.quiet:
        for result in results:
            print_result(result)


if __name__ == "__main__":
    main()
