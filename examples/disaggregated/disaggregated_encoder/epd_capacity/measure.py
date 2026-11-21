# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure one E or PD worker's saturation rate under an E+PD replay."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from urllib.request import urlopen

from prometheus_client.parser import text_string_to_metric_families


def snapshot(url):
    start = perf_counter()
    with urlopen(url, timeout=5) as response:
        data = response.read().decode("utf-8")
    timestamp = (start + perf_counter()) / 2
    counts = {}
    waiting = []
    for family in text_string_to_metric_families(data):
        for sample in family.samples:
            if sample.name == "vllm:request_success_total" and sample.labels.get(
                "finished_reason"
            ) in {"stop", "length", "repetition"}:
                counts[tuple(sorted(sample.labels.items()))] = sample.value
            elif sample.name == "vllm:num_requests_waiting":
                waiting.append(sample.value)
    if not counts or not waiting:
        raise ValueError(f"Missing vLLM request counters or waiting gauge at {url}")
    return timestamp, counts, sum(waiting)


def summarize_window(samples, start, end):
    """Measure successful completions between scrapes in the steady window."""
    window = [sample for sample in samples if start <= sample[0] <= end]
    if len(window) < 3:
        raise ValueError("too few steady-window scrapes, increase the request count")
    deltas = 0.0
    for (_, before, _), (_, after, _) in zip(window, window[1:]):
        if any(key not in after or after[key] < count for key, count in before.items()):
            raise ValueError("vLLM completion counters reset during measurement")
        deltas += sum(count - before.get(key, 0.0) for key, count in after.items())
    seconds = window[-1][0] - window[0][0]
    return {
        "rps": deltas / seconds,
        "mean_waiting": fmean(row[2] for row in window),
        "backlogged_fraction": fmean(row[2] >= 1 for row in window),
        "window_seconds": seconds,
    }


def probe(benchmark_args, concurrency, urls, output, seed):
    command = [
        "vllm",
        "bench",
        "serve",
        *benchmark_args,
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(seed),
        "--save-result",
        "--save-detailed",
        "--result-filename",
        str(output),
    ]
    samples = [[] for _ in urls]
    with (
        ThreadPoolExecutor(max_workers=len(urls)) as pool,
        subprocess.Popen(command) as child,
    ):
        deadline = perf_counter() + 15 * 60
        try:
            while child.poll() is None:
                if perf_counter() >= deadline:
                    raise TimeoutError(
                        f"benchmark timed out at concurrency {concurrency}"
                    )
                for values, sample in zip(
                    samples, pool.map(snapshot, urls), strict=True
                ):
                    values.append(sample)
                sleep(1)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)

    result = json.loads(output.read_text(encoding="utf-8"))
    arrivals = result["start_times"]
    if result["failed"] or len(arrivals) != result["completed"] or not arrivals:
        raise ValueError("benchmark must complete every replay request")
    first, last = min(arrivals), max(arrivals)
    span = last - first
    measurements = [
        summarize_window(series, first + 0.2 * span, first + 0.8 * span)
        for series in samples
    ]
    return {
        "concurrency": concurrency,
        "target": measurements[0],
        "peers": measurements[1:],
    }


def isolated(probe_result):
    target = probe_result["target"]
    return (
        target["rps"] > 0
        and target["backlogged_fraction"] >= 0.8
        and all(peer["mean_waiting"] < 1 for peer in probe_result["peers"])
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Pass workload arguments to vllm bench serve after --.",
    )
    parser.add_argument("--stage", choices=("encoder", "pd"), required=True)
    parser.add_argument("--target-metrics", required=True)
    parser.add_argument("--peer-metrics", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--concurrency-levels", nargs="+", type=int, default=[8, 16, 32, 64, 128]
    )
    parser.add_argument("--seed", type=int, default=117)
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    benchmark = arguments[separator + 1 :]
    if not benchmark:
        parser.error("pass workload arguments after --")
    controlled = {
        "--request-rate",
        "--max-concurrency",
        "--seed",
        "--result-filename",
        "--result-dir",
        "--save-result",
        "--save-detailed",
        "--ramp-up-strategy",
        "--ramp-up-start-rps",
        "--ramp-up-end-rps",
    }
    if controlled.intersection(arg.split("=", 1)[0] for arg in benchmark):
        parser.error(
            "the tool controls benchmark concurrency, request rate, seed, and output"
        )
    levels = sorted(set(args.concurrency_levels))
    if len(levels) < 2 or levels[0] < 1:
        parser.error("provide at least two distinct positive concurrency levels")
    urls = [args.target_metrics, *args.peer_metrics]
    if len(set(urls)) != len(urls):
        parser.error("target and peer metrics endpoints must be distinct")

    try:
        attempts = []
        with TemporaryDirectory() as temp:
            for index, concurrency in enumerate(levels):
                result = probe(
                    benchmark,
                    concurrency,
                    urls,
                    Path(temp) / f"bench-{index}.json",
                    args.seed + index,
                )
                previous = attempts[-1] if attempts else None
                attempts.append(result)
                print(json.dumps(result), flush=True)
                if previous and isolated(previous) and isolated(result):
                    rates = (previous["target"]["rps"], result["target"]["rps"])
                    mean_rate = fmean(rates)
                    if abs(rates[0] - rates[1]) <= 0.15 * mean_rate:
                        args.output.write_text(
                            json.dumps(
                                {
                                    "stage": args.stage,
                                    "saturation_rps": mean_rate,
                                    "probes": attempts,
                                },
                                indent=2,
                                allow_nan=False,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        return
        parser.error(
            "no isolated saturation plateau; check peer headroom or concurrency"
        )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
