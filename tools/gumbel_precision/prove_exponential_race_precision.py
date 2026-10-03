# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA proof for fp32 exponential-race tail truncation.

This script is intentionally not a unit test. It is a reproducible, GPU-only
statistical proof for the hidden Gumbel-max idiom:

    q.exponential_()
    sample = (probs / q).argmax()

For q ~ Exp(1), this is equivalent to argmax(log(probs) + Gumbel). On CUDA,
fp32 exponential samples inherit a 24-bit uniform lower-tail cutoff, so very
small q values are impossible. The many-tail experiment below chooses a case
where a correct sampler should select a low-probability tail token dozens of
times, while fp32 q cannot select one.
"""

from __future__ import annotations

import argparse
import math
import time

import torch


def _seed(seed: int) -> None:
    torch.manual_seed(seed)


def measure_exponential_lower_tail(
    *,
    device: torch.device,
    samples: int,
    chunk_size: int,
    seed: int,
) -> None:
    threshold = 2.0**-24
    print(f"lower-tail threshold: {threshold:.18e}")
    for dtype in (torch.float32, torch.float64):
        _seed(seed)
        count_below = 0
        min_q = float("inf")
        max_q = 0.0
        start = time.perf_counter()
        remaining = samples
        while remaining > 0:
            n = min(chunk_size, remaining)
            q = torch.empty((n,), dtype=dtype, device=device)
            q.exponential_()
            count_below += int((q < threshold).sum().item())
            min_q = min(min_q, float(q.min().item()))
            max_q = max(max_q, float(q.max().item()))
            remaining -= n
        torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
        print(
            f"{dtype}: samples={samples} count(q < 2^-24)={count_below} "
            f"min={min_q:.18e} max={max_q:.6f} elapsed={elapsed:.2f}s"
        )


def run_many_tail_race(
    *,
    device: torch.device,
    trials: int,
    num_tail_tokens: int,
    gap: float,
    chunk_trials: int,
    seed: int,
) -> None:
    p_tail = math.exp(-gap)
    expected_tail_hits = (
        trials * (num_tail_tokens * p_tail) / (1.0 + num_tail_tokens * p_tail)
    )
    print(
        "many-tail race: "
        f"trials={trials} num_tail_tokens={num_tail_tokens} gap={gap} "
        f"expected_tail_hits={expected_tail_hits:.4f}"
    )

    for dtype in (torch.float32, torch.float64):
        _seed(seed)
        hits = 0
        p0 = torch.tensor(1.0, dtype=dtype, device=device)
        pt = torch.tensor(p_tail, dtype=dtype, device=device)
        start = time.perf_counter()
        remaining = trials
        while remaining > 0:
            batch = min(chunk_trials, remaining)
            q0 = torch.empty((batch,), dtype=dtype, device=device)
            q0.exponential_()
            qt = torch.empty((batch, num_tail_tokens), dtype=dtype, device=device)
            qt.exponential_()
            head_score = p0 / q0
            tail_score = (pt / qt).amax(dim=-1)
            hits += int((tail_score > head_score).sum().item())
            remaining -= batch
        torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
        print(f"{dtype}: tail_hits={hits} elapsed={elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower-tail-samples", type=int, default=200_000_000)
    parser.add_argument("--lower-tail-chunk-size", type=int, default=10_000_000)
    parser.add_argument("--race-trials", type=int, default=100_000)
    parser.add_argument("--race-tail-tokens", type=int, default=262_144)
    parser.add_argument("--race-gap", type=float, default=20.5)
    parser.add_argument("--race-chunk-trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required for this proof.")

    device = torch.accelerator.current_accelerator()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this proof.")

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={device}")
    measure_exponential_lower_tail(
        device=device,
        samples=args.lower_tail_samples,
        chunk_size=args.lower_tail_chunk_size,
        seed=args.seed,
    )
    run_many_tail_race(
        device=device,
        trials=args.race_trials,
        num_tail_tokens=args.race_tail_tokens,
        gap=args.race_gap,
        chunk_trials=args.race_chunk_trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
