# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build ``synthetic_acceptance_rates`` from a drafter's measured acceptance.

``synthetic_acceptance_length`` resolves a target mean to the minimum-variance
schedule, so the drafter commits a nearly constant number of tokens per step. A
benchmark reading a latency distribution rather than a throughput number wants
the measured shape instead.

Input is either ``acceptance_histogram`` from
``--per-request-spec-decode-metrics summary`` (index ``j`` counts steps accepting
exactly ``j`` draft tokens, so the survival function is a suffix sum), or the
``Per-position acceptance rate`` field in a server log, which is already the
unconditional survival function ``synthetic_acceptance_rates`` expects.

Usage:
    python spec_acceptance_rates.py --log server.log
    python spec_acceptance_rates.py --histogram 39 1 0 3 --json
"""

import argparse
import json
import re
import statistics
import sys

# Matches the SpecDecoding metrics line emitted by SpecDecodingLogging.log().
_LINE = re.compile(
    r"Mean acceptance length:\s*([\d.]+).*?"
    r"Per-position acceptance rate:\s*([0-9.,\s]+?),\s*Avg Draft acceptance rate"
)


def moments(rates: list[float]) -> tuple[float, float]:
    """Mean and variance of the accepted-draft count implied by a survival function.

    For a non-negative integer variable, E[A] = sum of the survival function and
    E[A^2] = sum of (2i - 1) * P(A >= i).
    """
    ea = sum(rates)
    ea2 = sum((2 * i - 1) * rates[i - 1] for i in range(1, len(rates) + 1))
    return ea, ea2 - ea * ea


def min_variance_schedule(mean_accepted: float, n: int) -> list[float]:
    """Mirror of SpeculativeConfig._acceptance_length_to_rates, for comparison."""
    num_full = int(mean_accepted)
    frac = mean_accepted - num_full
    return ([1.0] * num_full + [frac] + [0.0] * (n - num_full - 1))[:n]


def rates_from_histogram(hist: list[float]) -> list[float]:
    """acceptance_histogram -> unconditional survival function.

    hist[j] counts steps accepting exactly j draft tokens, so
    rates[i] = P(A >= i + 1) = sum(hist[i + 1:]) / sum(hist).
    """
    total = sum(hist)
    if total <= 0:
        raise ValueError("acceptance_histogram is empty")
    return [sum(hist[i + 1 :]) / total for i in range(len(hist) - 1)]


def rates_from_log(text: str, skip: int) -> list[float]:
    """Pool the logged per-position survival function across stats windows."""
    hits = _LINE.findall(text)
    if not hits:
        raise ValueError(
            "no 'SpecDecoding metrics' lines found; run the server at INFO "
            "logging with stats enabled (do not pass --disable-log-stats)"
        )
    windows = [
        [float(x) for x in r.split(",") if x.strip()] for _, r in hits[skip:] or hits
    ]
    k = len(windows[0])
    pooled = [statistics.fmean([w[i] for w in windows]) for i in range(k)]
    # Averaging across windows can introduce a tiny inversion; the validator
    # requires a non-increasing vector.
    for i in range(1, k):
        pooled[i] = min(pooled[i], pooled[i - 1])
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", help="server log at INFO with stats enabled")
    source.add_argument(
        "--histogram",
        nargs="+",
        type=float,
        help="metrics.speculative_decoding.acceptance_histogram",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=2,
        help="log mode: drop the first N stats windows as warmup (default: 2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a --speculative-config fragment instead of a report",
    )
    args = parser.parse_args()

    try:
        if args.histogram is not None:
            rates = rates_from_histogram(args.histogram)
            source_desc = "acceptance_histogram"
        else:
            with open(args.log, errors="ignore") as handle:
                rates = rates_from_log(handle.read(), args.skip)
            source_desc = args.log
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    rates = [round(min(1.0, max(0.0, x)), 6) for x in rates]
    mean_accepted, variance = moments(rates)
    schedule = min_variance_schedule(mean_accepted, len(rates))
    _, schedule_variance = moments(schedule)

    if args.json:
        print(
            json.dumps(
                {
                    "rejection_sample_method": "synthetic",
                    "synthetic_acceptance_rates": rates,
                },
                indent=2,
            )
        )
        return

    print(f"source                 {source_desc}")
    print(f"measured rates         {rates}")
    print(f"mean acceptance length {1 + mean_accepted:.4f}")
    print(f"variance of accepted   {variance:.4f}")
    print(f"min-variance schedule  {[round(x, 6) for x in schedule]}")
    print(f"its variance           {schedule_variance:.4f}")
    if schedule_variance > 1e-9:
        print(f"ratio measured/minvar  {variance / schedule_variance:.1f}x")
    print()
    print("Pass to --speculative-config to replay this drafter's shape:")
    print('  "rejection_sample_method": "synthetic",')
    print(f'  "synthetic_acceptance_rates": {rates}')


if __name__ == "__main__":
    main()
