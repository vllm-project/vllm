# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect fixed-SHA serving no-regression evidence for prefix routing."""

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    num_prompts: int
    request_rate: str
    max_concurrency: int
    seed: int


@dataclass(frozen=True)
class MetricRule:
    scenario: str
    field: str
    direction: Literal["higher", "lower"]
    max_regression_pct: float


@dataclass
class ServerProcess:
    command: list[str]
    environment: dict[str, str]
    log_path: Path
    process: subprocess.Popen | None = None
    log_file: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            self.command,
            env=self.environment,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            **kwargs,
        )

    def assert_running(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {self.process.returncode}\n"
                f"{_tail(self.log_path)}"
            )

    def stop(self, timeout: float = 30.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _vllm_cli() -> str:
    executable = Path(sys.executable).with_name("vllm")
    if not executable.is_file():
        raise RuntimeError(f"vLLM CLI not found next to Python: {executable}")
    return str(executable)


def _free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _wait_for_health(server: ServerProcess, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        server.assert_running()
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(
        f"server did not become healthy within {timeout}s\n{_tail(server.log_path)}"
    )


def _server_command(
    args: argparse.Namespace,
    mode: Literal["baseline", "candidate"],
    port: int,
    event_port: int,
    replay_port: int,
    extra_server_args: list[str],
) -> list[str]:
    kv_events_config = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://*:{event_port}",
        "replay_endpoint": f"tcp://*:{replay_port}",
    }
    command = [
        _vllm_cli(),
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(args.max_model_len),
        "--enable-prefix-caching",
        "--kv-events-config",
        json.dumps(kv_events_config, separators=(",", ":")),
    ]
    if mode == "candidate":
        routing_config = {
            "nodes": [
                {
                    "id": "node0",
                    "url": "local",
                    "local": True,
                    "event_endpoint": f"tcp://127.0.0.1:{event_port}",
                    "replay_endpoint": f"tcp://127.0.0.1:{replay_port}",
                    "data_parallel_rank": 0,
                }
            ],
            "routing_token": "prefix-routing-performance-evidence",
            "event_sync_interval": 1.0,
            "event_replay_timeout": 2.0,
        }
        command.extend(
            [
                "--enable-prefix-routing",
                "--prefix-routing-config",
                json.dumps(routing_config, separators=(",", ":")),
            ]
        )
    command.extend(extra_server_args)
    return command


def _benchmark_command(
    args: argparse.Namespace,
    scenario: BenchmarkScenario,
    server_url: str,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    return [
        _vllm_cli(),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        server_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(scenario.num_prompts),
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0.0",
        "--random-prefix-len",
        "0",
        "--request-rate",
        scenario.request_rate,
        "--max-concurrency",
        str(scenario.max_concurrency),
        "--num-warmups",
        str(args.num_warmups),
        "--seed",
        str(scenario.seed),
        "--ignore-eos",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "95,99",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]


def _run_benchmark(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    result_path: Path,
    expected_prompts: int,
) -> dict[str, Any]:
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark exited with code {completed.returncode}\n{_tail(log_path)}"
        )
    if not result_path.exists():
        raise RuntimeError(f"benchmark result was not created: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("completed") != expected_prompts or result.get("failed") != 0:
        raise RuntimeError(
            "benchmark did not complete the expected workload: "
            f"completed={result.get('completed')} failed={result.get('failed')}"
        )
    return result


def _validate_workload_equivalence(
    results: dict[str, dict[str, list[dict[str, Any]]]],
    scenarios: list[BenchmarkScenario],
) -> None:
    for scenario in scenarios:
        totals = {
            (
                result["total_input_tokens"],
                result["total_output_tokens"],
                result["completed"],
            )
            for mode in ("baseline", "candidate")
            for result in results[mode][scenario.name]
        }
        if len(totals) != 1:
            raise RuntimeError(
                f"{scenario.name} workload differs between benchmark phases: {totals}"
            )


def compare_performance(
    results: dict[str, dict[str, list[dict[str, Any]]]],
    rules: list[MetricRule],
) -> tuple[bool, list[dict[str, Any]]]:
    """Compare median candidate metrics against median baseline metrics."""
    comparisons: list[dict[str, Any]] = []
    for rule in rules:
        for mode in ("baseline", "candidate"):
            if any(rule.field not in result for result in results[mode][rule.scenario]):
                raise RuntimeError(
                    f"{mode} {rule.scenario} result is missing metric {rule.field!r}"
                )
        baseline = statistics.median(
            float(result[rule.field]) for result in results["baseline"][rule.scenario]
        )
        candidate = statistics.median(
            float(result[rule.field]) for result in results["candidate"][rule.scenario]
        )
        if baseline <= 0:
            raise ValueError(
                f"baseline metric {rule.scenario}.{rule.field} must be positive"
            )
        ratio = candidate / baseline
        if rule.direction == "higher":
            regression_pct = max(0.0, (1.0 - ratio) * 100.0)
        else:
            regression_pct = max(0.0, (ratio - 1.0) * 100.0)
        passed = regression_pct <= rule.max_regression_pct
        comparisons.append(
            {
                **asdict(rule),
                "baseline_median": baseline,
                "candidate_median": candidate,
                "candidate_to_baseline_ratio": ratio,
                "regression_pct": regression_pct,
                "passed": passed,
            }
        )
    return all(item["passed"] for item in comparisons), comparisons


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="prefix-perf-model")
    parser.add_argument("--device-env-var", default="ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--latency-prompts", type=int, default=32)
    parser.add_argument("--throughput-prompts", type=int, default=128)
    parser.add_argument("--throughput-concurrency", type=int, default=16)
    parser.add_argument("--num-warmups", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--max-throughput-regression-pct", type=float, default=5.0)
    parser.add_argument("--max-latency-regression-pct", type=float, default=10.0)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--extra-server-args-json", default="[]")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> list[str]:
    if len(args.expected_sha) != 40:
        raise SystemExit("--expected-sha must be an exact 40-character Git SHA")
    positive_values = {
        "max-model-len": args.max_model_len,
        "input-len": args.input_len,
        "output-len": args.output_len,
        "latency-prompts": args.latency_prompts,
        "throughput-prompts": args.throughput_prompts,
        "throughput-concurrency": args.throughput_concurrency,
        "startup-timeout": args.startup_timeout,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise SystemExit(f"--{name} must be positive")
    if args.num_warmups < 0:
        raise SystemExit("--num-warmups must be non-negative")
    if args.repetitions < 3:
        raise SystemExit("--repetitions must be at least 3 for fixed-SHA evidence")
    if args.cooldown_seconds < 0:
        raise SystemExit("--cooldown-seconds must be non-negative")
    for name in (
        "max_throughput_regression_pct",
        "max_latency_regression_pct",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    extra_server_args = json.loads(args.extra_server_args_json)
    if not isinstance(extra_server_args, list) or not all(
        isinstance(value, str) for value in extra_server_args
    ):
        raise SystemExit("--extra-server-args-json must be a JSON string list")
    return extra_server_args


def main() -> None:
    args = _parse_args()
    extra_server_args = _validate_args(args)
    actual_sha = _git("rev-parse", "HEAD")
    if actual_sha != args.expected_sha:
        raise SystemExit(
            f"expected exact Git SHA {args.expected_sha!r}, found {actual_sha}"
        )
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("fixed-SHA performance evidence requires a clean worktree")

    result_dir = args.result_dir or Path("/tmp") / f"prefix-routing-perf-{actual_sha}"
    result_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = result_dir / "performance-evidence.json"
    server_port, event_port, replay_port = _free_ports(3)
    server_url = f"http://127.0.0.1:{server_port}"
    scenarios = [
        BenchmarkScenario("latency", args.latency_prompts, "1.0", 1, 0),
        BenchmarkScenario(
            "throughput",
            args.throughput_prompts,
            "inf",
            args.throughput_concurrency,
            1,
        ),
    ]
    rules = [
        MetricRule(
            "throughput",
            "request_throughput",
            "higher",
            args.max_throughput_regression_pct,
        ),
        MetricRule(
            "throughput",
            "total_token_throughput",
            "higher",
            args.max_throughput_regression_pct,
        ),
        MetricRule(
            "latency",
            "mean_ttft_ms",
            "lower",
            args.max_latency_regression_pct,
        ),
        MetricRule(
            "latency",
            "p95_ttft_ms",
            "lower",
            args.max_latency_regression_pct,
        ),
        MetricRule(
            "latency",
            "mean_tpot_ms",
            "lower",
            args.max_latency_regression_pct,
        ),
        MetricRule(
            "latency",
            "p95_e2el_ms",
            "lower",
            args.max_latency_regression_pct,
        ),
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment[args.device_env_var] = args.device
    results: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {scenario.name: [] for scenario in scenarios}
        for mode in ("baseline", "candidate")
    }
    evidence: dict[str, Any] = {
        "git_sha": actual_sha,
        "pythonhashseed": "0",
        "model": args.model,
        "served_model_name": args.served_model_name,
        "device_env_var": args.device_env_var,
        "device": args.device,
        "server_config": {
            "max_model_len": args.max_model_len,
            "extra_server_args": extra_server_args,
        },
        "benchmark_config": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_warmups": args.num_warmups,
            "repetitions": args.repetitions,
            "scenarios": [asdict(scenario) for scenario in scenarios],
        },
        "rules": [asdict(rule) for rule in rules],
        "phase_order": [],
        "raw_results": results,
        "started_at_unix": time.time(),
        "success": False,
    }

    try:
        for repetition in range(args.repetitions):
            phase_order = (
                ("baseline", "candidate")
                if repetition % 2 == 0
                else ("candidate", "baseline")
            )
            evidence["phase_order"].append(list(phase_order))
            for mode in phase_order:
                phase_name = f"{repetition:02d}-{mode}"
                server = ServerProcess(
                    command=_server_command(
                        args,
                        mode,
                        server_port,
                        event_port,
                        replay_port,
                        extra_server_args,
                    ),
                    environment=environment,
                    log_path=result_dir / f"server-{phase_name}.log",
                )
                try:
                    server.start()
                    _wait_for_health(server, server_url, args.startup_timeout)
                    for scenario in scenarios:
                        filename = f"{phase_name}-{scenario.name}.json"
                        result_path = result_dir / filename
                        benchmark_log = result_dir / (
                            f"benchmark-{phase_name}-{scenario.name}.log"
                        )
                        result = _run_benchmark(
                            _benchmark_command(
                                args,
                                scenario,
                                server_url,
                                result_dir,
                                filename,
                            ),
                            environment,
                            benchmark_log,
                            result_path,
                            scenario.num_prompts,
                        )
                        results[mode][scenario.name].append(result)
                finally:
                    server.stop()
                time.sleep(args.cooldown_seconds)

        _validate_workload_equivalence(results, scenarios)
        passed, comparisons = compare_performance(results, rules)
        evidence["comparisons"] = comparisons
        evidence["success"] = passed
        if not passed:
            failed_metrics = [
                f"{item['scenario']}.{item['field']}"
                for item in comparisons
                if not item["passed"]
            ]
            evidence["error"] = (
                "performance regression threshold exceeded: "
                + ", ".join(failed_metrics)
            )
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["finished_at_unix"] = time.time()
        evidence["duration_seconds"] = (
            evidence["finished_at_unix"] - evidence["started_at_unix"]
        )
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"prefix routing performance evidence: {evidence_path}")

    for comparison in evidence["comparisons"]:
        print(
            f"{comparison['scenario']}.{comparison['field']}: "
            f"baseline={comparison['baseline_median']:.4f} "
            f"candidate={comparison['candidate_median']:.4f} "
            f"regression={comparison['regression_pct']:.2f}% "
            f"passed={comparison['passed']}"
        )
    if not evidence["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
