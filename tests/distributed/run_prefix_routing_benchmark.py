# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure two-node prefix-routing benefit against a seeded-random proxy."""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tests.distributed.run_prefix_routing_performance import (
    ServerProcess,
    _free_ports,
    _git,
    _run_benchmark,
    _vllm_cli,
    _wait_for_health,
)


@dataclass(frozen=True)
class BenefitMetric:
    field: str
    direction: Literal["higher", "lower"]


def compare_benefit(
    results: dict[str, list[dict[str, Any]]],
    metrics: list[BenefitMetric],
) -> list[dict[str, Any]]:
    """Compare median candidate metrics with median baseline metrics."""
    comparisons: list[dict[str, Any]] = []
    for metric in metrics:
        for mode in ("baseline", "candidate"):
            if any(metric.field not in result for result in results[mode]):
                raise RuntimeError(f"{mode} result is missing metric {metric.field!r}")
        baseline = statistics.median(
            float(result[metric.field]) for result in results["baseline"]
        )
        candidate = statistics.median(
            float(result[metric.field]) for result in results["candidate"]
        )
        if baseline <= 0:
            raise ValueError(f"baseline metric {metric.field} must be positive")
        ratio = candidate / baseline
        improvement_pct = (
            (ratio - 1.0) * 100.0
            if metric.direction == "higher"
            else (1.0 - ratio) * 100.0
        )
        comparisons.append(
            {
                **asdict(metric),
                "baseline_median": baseline,
                "candidate_median": candidate,
                "candidate_to_baseline_ratio": ratio,
                "improvement_pct": improvement_pct,
            }
        )
    return comparisons


def _parse_prometheus_counters(text: str) -> dict[str, float]:
    counters = {
        "prefix_cache_queries": 0.0,
        "prefix_cache_hits": 0.0,
    }
    names = {
        "vllm:prefix_cache_queries": "prefix_cache_queries",
        "vllm:prefix_cache_queries_total": "prefix_cache_queries",
        "vllm:prefix_cache_hits": "prefix_cache_hits",
        "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        field = names.get(metric_name)
        if field is None:
            continue
        try:
            counters[field] += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return counters


def _fetch_cache_counters(url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        return _parse_prometheus_counters(response.read().decode("utf-8"))


def _cache_delta(
    before: list[dict[str, float]], after: list[dict[str, float]]
) -> dict[str, float]:
    queries = sum(
        end["prefix_cache_queries"] - start["prefix_cache_queries"]
        for start, end in zip(before, after)
    )
    hits = sum(
        end["prefix_cache_hits"] - start["prefix_cache_hits"]
        for start, end in zip(before, after)
    )
    return {
        "prefix_cache_queries": queries,
        "prefix_cache_hits": hits,
        "prefix_cache_hit_rate": hits / queries if queries > 0 else 0.0,
    }


def _routing_counts(log_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not log_path.exists():
        return counts
    pattern = re.compile(r"Prefix routing chose node=([^ ]+)")
    for match in pattern.finditer(
        log_path.read_text(encoding="utf-8", errors="replace")
    ):
        node_id = match.group(1)
        counts[node_id] = counts.get(node_id, 0) + 1
    return counts


def _server_command(
    args: argparse.Namespace,
    port: int,
    kv_events_config: dict[str, Any] | None,
    routing_config: dict[str, Any] | None,
    extra_server_args: list[str],
) -> list[str]:
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
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-prefix-caching",
    ]
    if kv_events_config is not None:
        command.extend(
            [
                "--kv-events-config",
                json.dumps(kv_events_config, separators=(",", ":")),
            ]
        )
    if routing_config is not None:
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
    ingress_url: str,
    result_dir: Path,
    filename: str,
) -> list[str]:
    return [
        _vllm_cli(),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        ingress_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "prefix_repetition",
        "--num-prompts",
        str(args.num_prompts),
        "--prefix-repetition-prefix-len",
        str(args.prefix_len),
        "--prefix-repetition-suffix-len",
        str(args.suffix_len),
        "--prefix-repetition-num-prefixes",
        str(args.num_prefixes),
        "--prefix-repetition-output-len",
        str(args.output_len),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(args.max_concurrency),
        "--seed",
        str(args.workload_seed),
        "--ignore-eos",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        filename,
    ]


def _proxy_command(
    port: int,
    node_urls: list[str],
    seed: int,
    stats_file: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("prefix_routing_random_proxy.py")),
        "--port",
        str(port),
        "--seed",
        str(seed),
        "--stats-file",
        str(stats_file),
    ]
    for node_url in node_urls:
        command.extend(["--upstream", node_url])
    return command


def _validate_workload_equivalence(
    results: dict[str, list[dict[str, Any]]],
) -> None:
    totals = {
        (
            result["total_input_tokens"],
            result["total_output_tokens"],
            result["completed"],
        )
        for mode in ("baseline", "candidate")
        for result in results[mode]
    }
    if len(totals) != 1:
        raise RuntimeError(f"workload differs between benchmark phases: {totals}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="prefix-routing-benchmark")
    parser.add_argument("--device-env-var", default="ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--prefix-len", type=int, default=1536)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-prefixes", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=400)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--workload-seed", type=int, default=0)
    parser.add_argument("--proxy-seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--sync-settle-seconds", type=float, default=3.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--extra-server-args-json", default="[]")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if len(args.expected_sha) != 40:
        raise SystemExit("--expected-sha must be an exact 40-character Git SHA")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise SystemExit("--devices must contain exactly two distinct device IDs")
    positive = {
        "max-model-len": args.max_model_len,
        "prefix-len": args.prefix_len,
        "suffix-len": args.suffix_len,
        "output-len": args.output_len,
        "num-prefixes": args.num_prefixes,
        "num-prompts": args.num_prompts,
        "request-rate": args.request_rate,
        "max-concurrency": args.max_concurrency,
        "startup-timeout": args.startup_timeout,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"--{name} must be positive")
    if args.num_prompts % args.num_prefixes != 0:
        raise SystemExit("--num-prompts must be divisible by --num-prefixes")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if args.prefix_len + args.suffix_len + args.output_len > args.max_model_len:
        raise SystemExit("prefix, suffix, and output lengths exceed --max-model-len")
    if args.repetitions < 3:
        raise SystemExit("--repetitions must be at least 3")
    if args.sync_settle_seconds < 0 or args.cooldown_seconds < 0:
        raise SystemExit("settle and cooldown durations must be non-negative")
    extra_server_args = json.loads(args.extra_server_args_json)
    if not isinstance(extra_server_args, list) or not all(
        isinstance(value, str) for value in extra_server_args
    ):
        raise SystemExit("--extra-server-args-json must be a JSON string list")
    return devices, extra_server_args


def main() -> None:
    args = _parse_args()
    devices, extra_server_args = _validate_args(args)
    actual_sha = _git("rev-parse", "HEAD")
    if actual_sha != args.expected_sha:
        raise SystemExit(
            f"expected exact Git SHA {args.expected_sha!r}, found {actual_sha}"
        )
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("fixed-SHA benchmark requires a clean worktree")

    result_dir = args.result_dir or Path("/tmp") / (
        f"prefix-routing-benefit-{actual_sha}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = result_dir / "prefix-routing-benefit.json"
    ports = _free_ports(7)
    node_ports = ports[:2]
    event_ports = ports[2:4]
    replay_ports = ports[4:6]
    proxy_port = ports[6]
    node_urls = [f"http://127.0.0.1:{port}" for port in node_ports]
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    routing_token = "prefix-routing-benchmark-token"

    common_environment = os.environ.copy()
    common_environment["PYTHONHASHSEED"] = "0"
    environments = []
    for device in devices:
        environment = common_environment.copy()
        environment[args.device_env_var] = device
        environments.append(environment)

    metrics = [
        BenefitMetric("request_throughput", "higher"),
        BenefitMetric("total_token_throughput", "higher"),
        BenefitMetric("mean_ttft_ms", "lower"),
        BenefitMetric("p50_ttft_ms", "lower"),
        BenefitMetric("p95_ttft_ms", "lower"),
        BenefitMetric("p99_ttft_ms", "lower"),
        BenefitMetric("mean_tpot_ms", "lower"),
        BenefitMetric("p95_e2el_ms", "lower"),
    ]
    results: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "candidate": [],
    }
    phase_evidence: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "schema_version": "prefix-routing-benefit/v1",
        "git_sha": actual_sha,
        "pythonhashseed": "0",
        "model": args.model,
        "served_model_name": args.served_model_name,
        "hardware": {
            "device_env_var": args.device_env_var,
            "devices": devices,
            "chip_count": 2,
            "node_count": 1,
        },
        "server_config": {
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "extra_server_args": extra_server_args,
        },
        "workload": {
            "dataset_name": "prefix_repetition",
            "prefix_len": args.prefix_len,
            "suffix_len": args.suffix_len,
            "output_len": args.output_len,
            "num_prefixes": args.num_prefixes,
            "num_prompts": args.num_prompts,
            "prompts_per_prefix": args.num_prompts // args.num_prefixes,
            "request_rate": args.request_rate,
            "max_concurrency": args.max_concurrency,
            "seed": args.workload_seed,
            "artificial_warmup": False,
        },
        "baseline": {
            "policy": "seeded-random external proxy",
            "global_prefix_scheduler": False,
            "proxy_seed": args.proxy_seed,
        },
        "candidate": {
            "policy": "longest cached prefix, round-robin tie break",
            "global_prefix_scheduler": True,
            "event_transport": "zmq snapshot/replay",
        },
        "repetitions": args.repetitions,
        "phase_order": [],
        "raw_results": results,
        "phases": phase_evidence,
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
                kv_configs = [
                    {
                        "enable_kv_cache_events": True,
                        "publisher": "zmq",
                        "endpoint": f"tcp://*:{event_ports[index]}",
                        "replay_endpoint": f"tcp://*:{replay_ports[index]}",
                    }
                    for index in range(2)
                ]
                routing_config = {
                    "nodes": [
                        {
                            "id": "node0",
                            "url": "local",
                            "local": True,
                            "event_endpoint": (f"tcp://127.0.0.1:{event_ports[0]}"),
                            "replay_endpoint": (f"tcp://127.0.0.1:{replay_ports[0]}"),
                            "data_parallel_rank": 0,
                        },
                        {
                            "id": "node1",
                            "url": node_urls[1],
                            "event_endpoint": (f"tcp://127.0.0.1:{event_ports[1]}"),
                            "replay_endpoint": (f"tcp://127.0.0.1:{replay_ports[1]}"),
                            "data_parallel_rank": 0,
                            "routing_token": routing_token,
                        },
                    ],
                    "routing_token": routing_token,
                    "event_sync_interval": 1.0,
                    "event_replay_timeout": 2.0,
                    "request_timeout": 600.0,
                }
                servers = [
                    ServerProcess(
                        command=_server_command(
                            args,
                            node_ports[index],
                            kv_configs[index] if mode == "candidate" else None,
                            routing_config
                            if mode == "candidate" and index == 0
                            else None,
                            extra_server_args,
                        ),
                        environment=environments[index],
                        log_path=result_dir / f"server-{phase_name}-node{index}.log",
                    )
                    for index in range(2)
                ]
                proxy_stats_path = result_dir / f"proxy-{phase_name}.json"
                proxy = ServerProcess(
                    command=_proxy_command(
                        proxy_port,
                        node_urls,
                        args.proxy_seed,
                        proxy_stats_path,
                    ),
                    environment=common_environment,
                    log_path=result_dir / f"proxy-{phase_name}.log",
                )
                ingress_url = proxy_url if mode == "baseline" else node_urls[0]
                phase: dict[str, Any] = {
                    "name": phase_name,
                    "mode": mode,
                    "ingress_url": ingress_url,
                }
                phase_evidence.append(phase)
                try:
                    servers[1].start()
                    _wait_for_health(servers[1], node_urls[1], args.startup_timeout)
                    servers[0].start()
                    _wait_for_health(servers[0], node_urls[0], args.startup_timeout)
                    if mode == "baseline":
                        proxy.start()
                        _wait_for_health(proxy, proxy_url, 30.0)
                    else:
                        time.sleep(args.sync_settle_seconds)

                    before = [_fetch_cache_counters(url) for url in node_urls]
                    filename = f"benchmark-{phase_name}.json"
                    result = _run_benchmark(
                        _benchmark_command(args, ingress_url, result_dir, filename),
                        common_environment,
                        result_dir / f"benchmark-{phase_name}.log",
                        result_dir / filename,
                        args.num_prompts,
                    )
                    results[mode].append(result)
                    after = [_fetch_cache_counters(url) for url in node_urls]
                    phase["cache"] = _cache_delta(before, after)
                    if mode == "baseline":
                        with urllib.request.urlopen(
                            f"{proxy_url}/_prefix_routing_benchmark/stats",
                            timeout=10,
                        ) as response:
                            phase["routing"] = json.loads(response.read())
                    else:
                        phase["routing"] = {
                            "request_counts": _routing_counts(servers[0].log_path)
                        }
                finally:
                    proxy.stop()
                    for server in reversed(servers):
                        server.stop()
                time.sleep(args.cooldown_seconds)

        _validate_workload_equivalence(results)
        evidence["comparisons"] = compare_benefit(results, metrics)
        ttft = next(
            item for item in evidence["comparisons"] if item["field"] == "p95_ttft_ms"
        )
        throughput = next(
            item
            for item in evidence["comparisons"]
            if item["field"] == "total_token_throughput"
        )
        evidence["optimization_observed"] = (
            ttft["improvement_pct"] > 0.0 and throughput["improvement_pct"] >= -5.0
        )
        evidence["success"] = True
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
        print(f"prefix routing benefit evidence: {evidence_path}")

    for comparison in evidence["comparisons"]:
        print(
            f"{comparison['field']}: "
            f"baseline={comparison['baseline_median']:.4f} "
            f"candidate={comparison['candidate_median']:.4f} "
            f"improvement={comparison['improvement_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
