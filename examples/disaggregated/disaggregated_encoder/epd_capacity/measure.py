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

REPLAY_OPTIONS = {
    "--backend": "openai-chat",
    "--endpoint": "/v1/chat/completions",
    "--num-warmups": 0,
    "--ready-check-timeout-sec": 0,
}
RANDOM_IMAGE_OPTIONS = {
    "--random-mm-base-items-per-request": 1,
    "--random-mm-num-mm-items-range-ratio": 0.0,
    "--random-mm-limit-mm-per-prompt": {"image": 1, "video": 0},
}


def copy_single_image_trace(source, destination):
    """Stage the validated JSONL so every probe uses the same request records."""
    if source.suffix != ".jsonl":
        raise ValueError("custom_image replay requires a JSONL dataset")
    rows = 0
    with (
        source.open(encoding="utf-8") as data,
        destination.open("w", encoding="utf-8") as replay,
    ):
        for line_number, line in enumerate(data, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"custom_image line {line_number}: {error.msg}"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(f"custom_image line {line_number} must be an object")
            if "content" in item:
                parts = item["content"]
                single_image = (
                    isinstance(parts, list)
                    and all(
                        isinstance(part, dict)
                        and part.get("type") in ("text", "image", "image_url")
                        for part in parts
                    )
                    and sum(part.get("type") != "text" for part in parts) == 1
                )
            else:
                images = item.get("image_files")
                single_image = isinstance(images, list) and len(images) == 1
            if not single_image:
                raise ValueError(
                    f"custom_image line {line_number} must contain exactly one image"
                )
            replay.write(line)
            rows += 1
    if not rows:
        raise ValueError("custom_image dataset must contain at least one request")


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


def completion_delta(before, after):
    if any(after.get(key, -1) < count for key, count in before.items()):
        raise ValueError("vLLM completion counters reset during measurement")
    return sum(count - before.get(key, 0.0) for key, count in after.items())


def summarize_window(samples, start, end):
    """Measure successful completions between scrapes in the steady window."""
    window = [sample for sample in samples if start <= sample[0] <= end]
    if len(window) < 3:
        raise ValueError("too few steady-window scrapes, increase the request count")
    deltas = sum(
        completion_delta(before, after)
        for (_, before, _), (_, after, _) in zip(window, window[1:])
    )
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
    before = [snapshot(url) for url in urls]
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
        after = list(pool.map(snapshot, urls))

    result = json.loads(output.read_text(encoding="utf-8"))
    arrivals = result["start_times"]
    completed = result["completed"]
    if result["failed"] or len(arrivals) != completed or not arrivals:
        raise ValueError("benchmark must complete every replay request")
    finished = [
        completion_delta(previous[1], current[1])
        for previous, current in zip(before, after, strict=True)
    ]
    if finished[0] != completed or sum(finished[1:]) != completed:
        raise ValueError(
            f"expected {completed} completions in each stage, got "
            f"{finished[0]:g} on the target and {sum(finished[1:]):g} on the peers. "
            "Use dedicated workers and start the proxy with --decode-retries 0."
        )
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
        epilog="Pass single-image random-mm or custom_image replay options after --.",
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
    replay_parser = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False, exit_on_error=False
    )
    replay_parser.add_argument(
        "--dataset-name", choices=("random-mm", "custom_image"), default="random-mm"
    )
    replay_parser.add_argument("--dataset-path", type=Path)
    known_replay_options = REPLAY_OPTIONS | RANDOM_IMAGE_OPTIONS
    for option, value in known_replay_options.items():
        convert = json.loads if isinstance(value, dict) else type(value)
        replay_parser.add_argument(option, choices=[value], type=convert)
    normalized = []
    for arg in benchmark:
        option, sep, value = arg.partition("=")
        if option.startswith("--"):
            option = option.replace("_", "-")
        normalized.append(option + sep + value)
    try:
        replay, benchmark = replay_parser.parse_known_args(normalized)
    except argparse.ArgumentError as error:
        parser.error(f"single-image replay: {error}")

    controlled = {
        *known_replay_options,
        "--dataset-name",
        "--dataset-path",
        "--config",
        "--extra-body",
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
    options = {
        arg.split("=", 1)[0].partition(".")[0]
        for arg in benchmark
        if arg.startswith("--")
    }
    if any(flag.startswith(option) for flag in controlled for option in options):
        parser.error(
            "the tool controls the single-image workload, concurrency, seed, "
            "and benchmark output"
        )
    replay_options = (
        known_replay_options if replay.dataset_name == "random-mm" else REPLAY_OPTIONS
    )
    benchmark.append(f"--dataset-name={replay.dataset_name}")
    benchmark.extend(
        f"{key}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in replay_options.items()
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
            dataset_path = replay.dataset_path
            if replay.dataset_name == "custom_image":
                if dataset_path is None:
                    parser.error("custom_image requires --dataset-path")
                verified_path = Path(temp) / "single-image.jsonl"
                copy_single_image_trace(dataset_path, verified_path)
                dataset_path = verified_path
            if dataset_path is not None:
                benchmark.extend(("--dataset-path", str(dataset_path)))
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
