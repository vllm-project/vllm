"""Micro-benchmark for the weight-only-quantization community model (sprint note 101/120).

Runs without a GPU: it times the analytic model itself so results stay comparable across
CI machines. Use it to check that a change did not make the model pathologically slow.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_community import weight_only_quantization as model  # noqa: E402


def run(requests: int, iters: int) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        stats = model.simulate(requests)
        samples.append((time.perf_counter() - start) * 1e6)
        assert not model.check_invariants(stats)
    samples.sort()
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": samples[len(samples) // 2],
        "p99_us": samples[min(len(samples) - 1, int(len(samples) * 0.99))],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = run(args.requests, args.iters)
    if args.json:
        print({
            "topic": model.TOPIC,
            "requests": args.requests,
            "iters": args.iters,
            "python": platform.python_version(),
            **results,
        })
        return 0

    print(f"{model.TOPIC} | requests={args.requests} | iters={args.iters} | python={platform.python_version()}")
    for key, value in results.items():
        print(f"  {key:<8} {value:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
