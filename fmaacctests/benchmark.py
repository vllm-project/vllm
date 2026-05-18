# SPDX-License-Identifier: Apache-2.0
"""Benchmark old mul/add RoPE arithmetic against explicit tl.fma."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fma_variants import HEAD_DIM, MAX_POS, N_HEAD, ROPE_DIM, launch_variant


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def make_inputs(
    num_tokens: int,
    cache_dtype: torch.dtype,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(
        num_tokens,
        N_HEAD,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.randint(
        0,
        MAX_POS,
        (num_tokens,),
        dtype=torch.int64,
        device=device,
    )
    cos_sin_cache = torch.randn(
        MAX_POS,
        ROPE_DIM,
        dtype=cache_dtype,
        device=device,
    )
    weights = torch.randn(
        num_tokens,
        N_HEAD,
        dtype=torch.bfloat16,
        device=device,
    )
    return positions, q, cos_sin_cache, weights


def time_variant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    *,
    use_fma: bool,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, float]:
    softmax_scale = HEAD_DIM**-0.5
    head_scale = N_HEAD**-0.5
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    weights_out = torch.empty_like(weights, dtype=torch.float32)

    for _ in range(warmup):
        launch_variant(
            positions,
            q,
            cos_sin_cache,
            weights,
            softmax_scale,
            head_scale,
            q_fp8,
            weights_out,
            use_fma=use_fma,
        )
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            launch_variant(
                positions,
                q,
                cos_sin_cache,
                weights,
                softmax_scale,
                head_scale,
                q_fp8,
                weights_out,
                use_fma=use_fma,
            )
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)

    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_case(
    num_tokens: int,
    cache_dtype_name: str,
    device: str,
    seed: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, object]:
    cache_dtype = _dtype(cache_dtype_name)
    positions, q, cos_sin_cache, weights = make_inputs(
        num_tokens,
        cache_dtype,
        device,
        seed,
    )

    # Compile both specializations before timing either one.
    for use_fma in (False, True):
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        weights_out = torch.empty_like(weights, dtype=torch.float32)
        launch_variant(
            positions,
            q,
            cos_sin_cache,
            weights,
            HEAD_DIM**-0.5,
            N_HEAD**-0.5,
            q_fp8,
            weights_out,
            use_fma=use_fma,
        )
    torch.cuda.synchronize()

    muladd = time_variant(
        positions,
        q,
        cos_sin_cache,
        weights,
        use_fma=False,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )
    fma = time_variant(
        positions,
        q,
        cos_sin_cache,
        weights,
        use_fma=True,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )
    delta_pct = (fma["median_ms"] / muladd["median_ms"] - 1.0) * 100.0
    return {
        "num_tokens": num_tokens,
        "cache_dtype": cache_dtype_name,
        "muladd": muladd,
        "fma": fma,
        "fma_vs_muladd_median_delta_pct": delta_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[257, 1023, 4096])
    parser.add_argument(
        "--cache-dtypes",
        nargs="+",
        default=["float32", "bfloat16"],
        choices=["float32", "bfloat16"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for num_tokens in args.tokens:
        for cache_dtype_name in args.cache_dtypes:
            rows.append(
                run_case(
                    num_tokens,
                    cache_dtype_name,
                    args.device,
                    args.seed,
                    args.warmup,
                    args.iters,
                    args.repeats,
                )
            )

    print("tokens cache_dtype muladd_median_ms fma_median_ms delta_pct")
    for row in rows:
        muladd = row["muladd"]
        fma = row["fma"]
        print(
            f"{row['num_tokens']:>6} {row['cache_dtype']:<8} "
            f"{muladd['median_ms']:.6f} {fma['median_ms']:.6f} "
            f"{row['fma_vs_muladd_median_delta_pct']:+.3f}"
        )

    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
