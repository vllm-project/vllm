# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the cold and warm startup time of vLLM models.

This script measures total startup time (including model loading, compilation,
and cache operations) for both cold and warm scenarios:
- Cold startup: Fresh start with no caches (temporary cache directories)
- Warm startup: Using cached compilation and model info
"""

import argparse
import json
import multiprocessing
import os
import shlex
import shutil
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any, NamedTuple

import numpy as np
import requests
from tqdm import tqdm

from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)
from vllm.benchmarks.sweep.server import ServerProcess
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

PERCENTAGES = [10, 25, 50, 75, 90, 99]


class MetricDesc(NamedTuple):
    """Descriptor for a metric to collect from each iteration."""

    iter_key: str  # key in the iteration result dict
    suffix: str  # result key suffix, e.g. "startup", "compilation"
    display_name: str


class MetricStats(NamedTuple):
    """Aggregated statistics for a single benchmark metric."""

    key: str  # e.g. "cold_startup", "warm_encoder_compilation"
    display_name: str
    values: list[float]
    avg: float
    percentiles: dict[int, float]


_BASE_METRICS = [
    MetricDesc("total_startup_time", "startup", "Startup time"),
    MetricDesc("compilation_time", "compilation", "Compilation time"),
]
_SERVE_METRICS = [
    MetricDesc("ready_time", "ready", "Launch to ready"),
    MetricDesc("first_token_time", "first_token", "Launch to first token"),
]
_ENCODER_METRIC = MetricDesc(
    "encoder_compilation_time",
    "encoder_compilation",
    "Encoder compilation time",
)


def _compute_metric(
    phase: str,
    desc: MetricDesc,
    iterations: list[dict[str, float]],
) -> MetricStats:
    values = [m[desc.iter_key] for m in iterations]
    arr = np.array(values)
    return MetricStats(
        key=f"{phase}_{desc.suffix}",
        display_name=desc.display_name,
        values=values,
        avg=float(np.mean(arr)),
        percentiles=dict(zip(PERCENTAGES, np.percentile(arr, PERCENTAGES).tolist())),
    )


def _collect_phase_metrics(
    phase: str,
    iterations: list[dict[str, float]],
    has_encoder: bool,
) -> list[MetricStats]:
    metrics = [_compute_metric(phase, desc, iterations) for desc in _BASE_METRICS]
    if has_encoder:
        metrics.append(_compute_metric(phase, _ENCODER_METRIC, iterations))
    return metrics


def _print_phase(phase_name: str, metrics: list[MetricStats]) -> None:
    print(f"\n{phase_name}:")
    for m in metrics:
        print(f"Avg {m.display_name.lower()}: {m.avg:.2f} seconds")
    for m in metrics:
        print(f"{m.display_name} percentiles:")
        for pct, val in m.percentiles.items():
            print(f"  {pct}%: {val:.2f} seconds")


def _metric_to_json(m: MetricStats) -> dict[str, Any]:
    return {
        f"avg_{m.key}_time": m.avg,
        f"{m.key}_times": m.values,
        f"{m.key}_percentiles": m.percentiles,
    }


@contextmanager
def cold_startup():
    """Context manager to measure cold startup time:
    1. Uses a temporary directory for vLLM cache to avoid any pollution
       between cold startup iterations.
    2. Uses inductor's fresh_cache to clear torch.compile caches.
    """
    from torch._inductor.utils import fresh_cache

    # Use temporary directory for caching to avoid any pollution between cold startups
    original_cache_root = os.environ.get("VLLM_CACHE_ROOT")
    temp_cache_dir = tempfile.mkdtemp(prefix="vllm_startup_bench_cold_")
    try:
        os.environ["VLLM_CACHE_ROOT"] = temp_cache_dir
        with fresh_cache():
            yield
    finally:
        # Clean up temporary cache directory
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
        if original_cache_root:
            os.environ["VLLM_CACHE_ROOT"] = original_cache_root
        else:
            os.environ.pop("VLLM_CACHE_ROOT", None)


def run_startup_in_subprocess(engine_args, result_queue):
    """Run LLM startup in a subprocess and return timing metrics via a queue.
    This ensures complete isolation between iterations.
    """
    try:
        # Import inside the subprocess to avoid issues with forking
        from vllm import LLM

        # Measure total startup time
        start_time = time.perf_counter()

        llm = LLM.from_engine_args(engine_args)

        total_startup_time = time.perf_counter() - start_time

        # Extract compilation time if available
        compilation_time = 0.0
        encoder_compilation_time = 0.0
        if hasattr(llm.llm_engine, "vllm_config"):
            vllm_config = llm.llm_engine.vllm_config
            if (
                hasattr(vllm_config, "compilation_config")
                and vllm_config.compilation_config is not None
            ):
                compilation_time = vllm_config.compilation_config.compilation_time
                encoder_compilation_time = (
                    vllm_config.compilation_config.encoder_compilation_time
                )

        result_queue.put(
            {
                "total_startup_time": total_startup_time,
                "compilation_time": compilation_time,
                "encoder_compilation_time": encoder_compilation_time,
            }
        )

    except Exception as e:
        result_queue.put(None)
        result_queue.put(str(e))


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: list[MetricStats]
) -> None:
    base_name = os.path.splitext(args.output_json)[0]
    for m in metrics:
        records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={f"avg_{m.key}_time": [m.avg]},
            extra_info={
                f"{m.key}_times": m.values,
                f"{m.key}_percentiles": m.percentiles,
            },
        )
        if records:
            write_to_json(f"{base_name}.{m.key}.pytorch.json", records)


def _read_completion(
    server: ServerProcess,
    row: dict[str, Any],
    reference: tuple[str, str, list[int] | None] | None,
) -> tuple[str, str, list[int] | None]:
    session = server.http
    url = server.server_address
    with session.get(url + "/v1/models", timeout=30) as response:
        response.raise_for_status()
        try:
            model = response.json()["data"][0]["id"]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("Expected /v1/models to list a model ID") from exc
        if not isinstance(model, str):
            raise ValueError("Expected a string model ID")

    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "stream": True,
        "temperature": 0,
        "max_tokens": 8,
        "return_token_ids": True,
    }
    text: list[str] = []
    ids: list[int] | None = None
    finish = None
    with session.post(
        url + "/v1/completions", json=payload, stream=True, timeout=300
    ) as response:
        response.raise_for_status()
        # Do not wait for a full buffer before timestamping a short SSE frame.
        for line in response.iter_lines(chunk_size=1):
            arrival = time.monotonic()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                if not text or finish not in ("stop", "length"):
                    break
                output = model, "".join(text), ids
                if reference is not None and output != reference:
                    raise ValueError("Completion differs from the reference")
                row["verified"] = True
                return output
            frame = json.loads(data)
            if not isinstance(frame, dict):
                raise ValueError("Expected a JSON object in completion stream")
            if "error" in frame:
                raise ValueError(f"Completion failed: {frame['error']}")
            choices = frame.get("choices")
            if not isinstance(choices, list):
                raise ValueError("Expected a list of completion choices")

            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError(
                        "Expected a JSON object for each completion choice"
                    )
                index = choice.get("index")
                if type(index) is not int or index != 0 or finish is not None:
                    raise ValueError("Unexpected completion choice")
                chunk = choice.get("text")
                if not isinstance(chunk, str):
                    raise ValueError("Expected text in completion choice")
                if chunk:
                    row.setdefault("first_token_time", arrival - server.launch_time)
                    text.append(chunk)

                token_ids = choice.get("token_ids")
                if token_ids is not None:
                    if not isinstance(token_ids, list) or not all(
                        type(token) is int for token in token_ids
                    ):
                        raise ValueError("Expected integer completion token IDs")
                    if ids is None:
                        ids = []
                    ids.extend(token_ids)
                finish = choice.get("finish_reason")
    raise ValueError("Incomplete completion stream")


def _bench_server(args: argparse.Namespace) -> None:
    defaults = EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args([])
    overrides = [
        name
        for name, default in vars(defaults).items()
        if getattr(args, name) != default
    ]
    if overrides:
        flags = ", ".join("--" + name.replace("_", "-") for name in overrides)
        raise ValueError(f"Put {flags} inside --serve-cmd")
    cmd = shlex.split(args.serve_cmd)
    timeout = args.server_ready_timeout
    if not cmd or timeout <= 0:
        raise ValueError("Expected a command and a positive readiness timeout")
    log_dir = None
    results: dict[str, Any] = {}
    metrics = []
    reference = None
    failed = False
    stop_launching = False
    for phase, count in (
        ("cold", args.num_iters_cold),
        ("warmup", args.num_iters_warmup),
        ("warm", args.num_iters_warm),
    ):
        rows = []
        for _ in range(count):
            if stop_launching:
                break
            row: dict[str, Any] = {"verified": False}
            rows.append(row)
            try:
                if args.output_json and log_dir is None:
                    output_path = Path(args.output_json)
                    log_dir = Path(
                        tempfile.mkdtemp(
                            prefix=f"{output_path.stem}-logs-", dir=output_path.parent
                        )
                    )
                log_path = log_dir / f"{phase}-{len(rows)}.log" if log_dir else None
                with (
                    cold_startup() if phase == "cold" else nullcontext(),
                    log_path.open("wb") if log_path else nullcontext() as log,
                    requests.Session() as session,
                ):
                    if log_path:
                        row["server_log"] = str(log_path)
                    session.trust_env = False
                    server = ServerProcess(cmd, [], show_stdout=True, session=session)
                    if server.is_server_ready(timeout=1.0):
                        raise RuntimeError(f"{server.server_address} already answers")
                    server.start(log=log, dev_mode=False)
                    try:
                        server.wait_until_ready(
                            timeout, poll_interval=0.05, request_timeout=1.0
                        )
                        row["ready_time"] = time.monotonic() - server.launch_time
                        try:
                            reference = _read_completion(server, row, reference)
                        except (requests.RequestException, ValueError, KeyError) as exc:
                            row["error"] = str(exc)
                    finally:
                        if server.stop(grace_period=30):
                            row["forced_kill"] = True
                            raise RuntimeError("Server required SIGKILL")
                        if server.is_server_ready(timeout=1.0):
                            raise RuntimeError("Address still answers after shutdown")
            except (
                OSError,
                RuntimeError,
                ValueError,
                TimeoutExpired,
                KeyboardInterrupt,
            ) as exc:
                row["verified"] = False
                row["error"] = str(exc) or type(exc).__name__
                stop_launching = True
            if not row["verified"]:
                failed = True
                print(f"{phase} {len(rows)} unverified: {row}")

        verified = [row for row in rows if row["verified"]]
        results[f"{phase}_iterations"] = rows
        results[f"{phase}_verified_starts"] = len(verified)
        if verified and phase != "warmup":
            phase_metrics = [
                _compute_metric(phase, m, verified) for m in _SERVE_METRICS
            ]
            metrics.extend(phase_metrics)
            _print_phase(f"{phase.upper()} STARTUP", phase_metrics)
        print(f"{phase.capitalize()} verified starts: {len(verified)}/{len(rows)}")

    if args.output_json:
        for metric in metrics:
            results.update(_metric_to_json(metric))
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        if reference:
            export_args = argparse.Namespace(**vars(args))
            export_args.model = reference[0]
            save_to_pytorch_benchmark_format(export_args, metrics)
    if failed:
        raise SystemExit(1)


def add_cli_args(parser: FlexibleArgumentParser):
    parser.add_argument(
        "--num-iters-cold",
        type=int,
        default=3,
        help="Number of cold startup iterations.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=1,
        help="Number of warmup iterations before benchmarking warm startups.",
    )
    parser.add_argument(
        "--num-iters-warm",
        type=int,
        default=3,
        help="Number of warm startup iterations.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the startup time results in JSON format.",
    )

    parser.add_argument(
        "--serve-cmd",
        help="Time a server command from launch to readiness and to the first "
        "non-empty text in a streamed completion. Put all engine arguments inside "
        "the quoted command. A start verifies when its completed response matches "
        "the first completed start in text and token IDs (when returned).",
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for the server started by --serve-cmd to become ready. "
        "A server that exits is reported at once.",
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser


def main(args: argparse.Namespace):
    if args.serve_cmd:
        return _bench_server(args)

    # Set multiprocessing start method to 'spawn' for clean process isolation
    # This ensures each subprocess starts fresh without inheriting state
    multiprocessing.set_start_method("spawn", force=True)

    engine_args = EngineArgs.from_cli_args(args)

    def create_llm_and_measure_startup():
        """Create LLM instance in a subprocess and measure startup time.
        Returns timing metrics, using subprocess for complete isolation.
        """
        # Create a queue for inter-process communication
        result_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_startup_in_subprocess,
            args=(
                engine_args,
                result_queue,
            ),
        )
        process.start()
        process.join()

        if not result_queue.empty():
            result = result_queue.get()
            if result is None:
                if not result_queue.empty():
                    error_msg = result_queue.get()
                    raise RuntimeError(f"Subprocess failed: {error_msg}")
                else:
                    raise RuntimeError("Subprocess failed with unknown error")
            return result
        else:
            raise RuntimeError("Subprocess did not return a result")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print("Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 to collect startup metrics.\n")

    # Collect cold startup iterations
    print("Measuring cold startup time...\n")
    cold_iterations = []
    for _ in tqdm(range(args.num_iters_cold), desc="Cold startup iterations"):
        with cold_startup():
            cold_iterations.append(create_llm_and_measure_startup())

    # Warmup for warm startup
    print("\nWarming up for warm startup measurement...\n")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        create_llm_and_measure_startup()

    # Collect warm startup iterations
    print("\nMeasuring warm startup time...\n")
    warm_iterations = []
    for _ in tqdm(range(args.num_iters_warm), desc="Warm startup iterations"):
        warm_iterations.append(create_llm_and_measure_startup())

    # Determine if encoder compilation occurred in any iteration
    has_encoder = any(
        m["encoder_compilation_time"] > 0 for m in cold_iterations + warm_iterations
    )

    cold_metrics = _collect_phase_metrics("cold", cold_iterations, has_encoder)
    warm_metrics = _collect_phase_metrics("warm", warm_iterations, has_encoder)
    all_metrics = cold_metrics + warm_metrics

    # Print results
    print("\n" + "=" * 60)
    print("STARTUP TIME BENCHMARK RESULTS")
    print("=" * 60)
    _print_phase("COLD STARTUP", cold_metrics)
    _print_phase("WARM STARTUP", warm_metrics)
    print("=" * 60)

    # Output JSON results if specified
    if args.output_json:
        results: dict[str, Any] = {}
        for m in all_metrics:
            results.update(_metric_to_json(m))
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, all_metrics)
