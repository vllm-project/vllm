# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: per-step latency of repetition detection algorithms.

Three configurations are compared on a simulated decode loop where one
output token is appended per step and ``check_sequence_repetition`` is
invoked:

  * naive (current in-tree)        — bounded ``max_pattern_size`` only
  * rolling_hash (no state)        — recomputes prefix hashes per step
  * rolling_hash + state           — incremental, prefix hashes amortized

Two scenarios are exercised:

  * "random"  — pseudo-random tokens (no detection should fire)
  * "loop"    — random prefix then a repeating tail (detection fires)

Run::

    .venv/bin/python benchmarks/benchmark_repetition_detection.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import types

# The local CUDA build can fail to import vllm._C on hosts that lack the
# matching driver symbols. The detection logic is pure-Python, so stub
# the C extension before any vllm import.
sys.modules.setdefault("vllm._C", types.ModuleType("vllm._C"))

from vllm.sampling_params import RepetitionDetectionParams  # noqa: E402
from vllm.v1.core.sched.utils import (  # noqa: E402
    RollingHashState,
    check_sequence_repetition,
)


def _gen_stream(n: int, vocab: int, scenario: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    if scenario == "random":
        return [rng.randrange(vocab) for _ in range(n)]
    if scenario == "loop":
        prefix_len = max(0, n // 2)
        prefix = [rng.randrange(vocab) for _ in range(prefix_len)]
        # 64-token pattern repeated until total length n.
        pattern = [rng.randrange(vocab) for _ in range(64)]
        tail_len = n - prefix_len
        tail = (pattern * ((tail_len // len(pattern)) + 1))[:tail_len]
        return prefix + tail
    raise ValueError(f"unknown scenario: {scenario}")


def _bench_one(
    label: str,
    stream: list[int],
    params: RepetitionDetectionParams,
    use_state: bool,
) -> tuple[float, int]:
    """Simulate per-step ``check_sequence_repetition`` over the stream.

    Returns ``(total_seconds, last_step_with_hit_or_n)``.
    """
    state: RollingHashState | None = RollingHashState() if use_state else None
    token_ids: list[int] = []
    hit_step = -1
    t0 = time.perf_counter()
    for tok in stream:
        token_ids.append(tok)
        # Mirror check_stop's call signature.
        if check_sequence_repetition(token_ids, params, state=state):
            hit_step = len(token_ids)
            break
    t1 = time.perf_counter()
    return (t1 - t0, hit_step if hit_step >= 0 else len(token_ids))


def _fmt_us(seconds: float, steps: int) -> str:
    if steps == 0:
        return "       n/a"
    return f"{(seconds / steps) * 1e6:8.2f} us"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8192, help="stream length")
    p.add_argument("--vocab", type=int, default=1000, help="effective vocab")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument(
        "--max-pattern-size",
        type=int,
        default=128,
        help=(
            "bounded mode cap. Only used by configs that respect it. The "
            "rolling_hash unbounded config ignores this and uses 0."
        ),
    )
    p.add_argument("--min-count", type=int, default=3)
    args = p.parse_args()

    print(
        f"# stream length n={args.n}  vocab={args.vocab}  "
        f"min_count={args.min_count}  max_pattern_size={args.max_pattern_size}"
    )
    print()
    header = (
        f"{'scenario':<10}  {'config':<32}  {'wall':>9}  "
        f"{'per-step':>10}  {'steps':>7}  {'hit?':>5}"
    )
    print(header)
    print("-" * len(header))

    for scenario in ("random", "loop"):
        stream = _gen_stream(args.n, args.vocab, scenario, args.seed)

        configs = [
            (
                "naive (bounded)",
                RepetitionDetectionParams(
                    max_pattern_size=args.max_pattern_size,
                    min_pattern_size=2,
                    min_count=args.min_count,
                    algorithm="naive",
                ),
                False,
            ),
            (
                "rolling_hash (bounded, no state)",
                RepetitionDetectionParams(
                    max_pattern_size=args.max_pattern_size,
                    min_pattern_size=2,
                    min_count=args.min_count,
                    algorithm="rolling_hash",
                ),
                False,
            ),
            (
                "rolling_hash (bounded, +state)",
                RepetitionDetectionParams(
                    max_pattern_size=args.max_pattern_size,
                    min_pattern_size=2,
                    min_count=args.min_count,
                    algorithm="rolling_hash",
                ),
                True,
            ),
            (
                "rolling_hash (unbounded, no state)",
                RepetitionDetectionParams(
                    max_pattern_size=0,
                    min_pattern_size=2,
                    min_count=args.min_count,
                    algorithm="rolling_hash",
                ),
                False,
            ),
            (
                "rolling_hash (unbounded, +state)",
                RepetitionDetectionParams(
                    max_pattern_size=0,
                    min_pattern_size=2,
                    min_count=args.min_count,
                    algorithm="rolling_hash",
                ),
                True,
            ),
        ]

        for label, params, use_state in configs:
            wall, hit_step = _bench_one(label, stream, params, use_state)
            steps = hit_step  # number of decode steps actually executed
            hit = "yes" if hit_step < len(stream) else "no"
            print(
                f"{scenario:<10}  {label:<32}  "
                f"{wall:>7.3f}s  {_fmt_us(wall, steps):>10}  "
                f"{steps:>7d}  {hit:>5}"
            )
        print()


if __name__ == "__main__":
    main()
