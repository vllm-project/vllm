# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional measured context-capacity sweep for a selected Recipes TP/DP layout.

Run this after parallel-layout recommendation and before concurrency tuning.
Uses a fixed workload and selects the *largest* context meeting startup and
explicit P99 latency constraints; it is not a throughput optimizer.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_candidates(raw: str) -> list[int]:
    try:
        values = sorted({positive_int(x.strip()) for x in raw.split(",")})
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one length is required")
    return values


def metric(data: dict, name: str) -> float | None:
    value = data.get(name)
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else None
    )


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def wait_for_health(server: subprocess.Popen, port: int, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM server exited with status {server.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM /health did not become ready within {seconds}s")


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    env_path = Path(args.env).resolve()
    output_dir = Path(args.output_dir).resolve()
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(base.get("model"), str):
        raise ValueError("Config must contain a model")
    if not env_path.is_file():
        raise FileNotFoundError(env_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = base["model"]
    served_name = base.get("served-model-name", model)
    if isinstance(served_name, list):
        served_name = served_name[0]
    if not isinstance(served_name, str):
        raise ValueError("Invalid served-model-name")
    minimum = args.input_tokens + args.output_tokens + args.context_margin
    candidates = [c for c in args.candidates if c >= minimum]
    if not candidates:
        raise ValueError(
            f"No max-model-len candidate meets input + output + margin ({minimum})"
        )

    results = []
    best: int | None = None
    for length in candidates:
        path = output_dir / f"context-{length}.yml"
        config = dict(base)
        config["max-model-len"] = length
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result_path = output_dir / f"bench-{length}.json"
        server_log = output_dir / f"server-{length}.log"
        bench_log = output_dir / f"bench-{length}.log"
        serve = ["vllm", "serve", "--config", str(path), "--port", str(args.port)]
        bench = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--model",
            served_name,
            "--tokenizer",
            model,
            "--dataset-name",
            "random",
            "--request-rate",
            "inf",
            "--random-input-len",
            str(args.input_tokens),
            "--random-output-len",
            str(args.output_tokens),
            "--num-prompts",
            str(args.num_prompts),
            "--max-concurrency",
            str(args.concurrency),
            "--port",
            str(args.port),
            "--ignore-eos",
            "--metric-percentiles",
            "99",
            "--save-result",
            "--result-filename",
            str(result_path),
        ]
        record: dict = {"max_model_len": length, "feasible": False}
        results.append(record)
        if args.dry_run:
            print("SERVER:", shlex.join(serve))
            print("BENCH: ", shlex.join(bench))
            record["status"] = "dry_run"
            continue
        server = None
        try:
            # Source env.sh in the subprocess, not in the calling Python process.
            with server_log.open("w", encoding="utf-8") as sf:
                command = (
                    "source "
                    + shlex.quote(str(env_path))
                    + " && exec "
                    + shlex.join(serve)
                )
                server = subprocess.Popen(
                    ["bash", "-c", command],
                    stdout=sf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd="/vllm-workspace" if Path("/vllm-workspace").is_dir() else None,
                )
                wait_for_health(server, args.port, args.server_timeout)
                with bench_log.open("w", encoding="utf-8") as bf:
                    bench_command = (
                        "source "
                        + shlex.quote(str(env_path))
                        + " && "
                        + shlex.join(bench)
                    )
                    subprocess.run(
                        ["bash", "-c", bench_command],
                        stdout=bf,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.bench_timeout,
                    )
            data = json.loads(result_path.read_text(encoding="utf-8"))
            p99_ttft = metric(data, "p99_ttft_ms")
            p99_tpot = metric(data, "p99_tpot_ms")
            if p99_ttft is None or p99_tpot is None:
                raise ValueError("Bench JSON missing p99_ttft_ms or p99_tpot_ms")
            record.update(
                status="measured",
                p99_ttft_ms=p99_ttft,
                p99_tpot_ms=p99_tpot,
                output_throughput=metric(data, "output_throughput"),
            )
            record["feasible"] = (
                args.ttft_sla_ms is None or p99_ttft <= args.ttft_sla_ms
            ) and (args.tpot_sla_ms is None or p99_tpot <= args.tpot_sla_ms)
            if record["feasible"]:
                best = length
        except (
            OSError,
            RuntimeError,
            TimeoutError,
            subprocess.SubprocessError,
            ValueError,
            KeyError,
        ) as exc:
            record.update(status="failed", error=str(exc))
        finally:
            if server is not None:
                stop_server(server)
            (output_dir / "context-length-results.json").write_text(
                json.dumps(
                    {
                        "objective": "largest_feasible_context",
                        "results": results,
                        "selected_max_model_len": best,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            f"max-model-len={length}: {record['status']}, feasible={record['feasible']}"
        )

    if args.dry_run:
        return 0
    if best is None:
        print(
            "No context length passed startup, benchmark, and SLA checks",
            file=sys.stderr,
        )
        return 1
    selected = dict(base)
    selected["max-model-len"] = best
    selected_path = output_dir / "context-length-config.yml"
    selected_path.write_text(
        yaml.safe_dump(selected, sort_keys=False), encoding="utf-8"
    )
    print(f"Selected max-model-len={best}; wrote {selected_path}")
    if args.apply:
        backup = config_path.with_suffix(config_path.suffix + ".before-context-sweep")
        if not backup.exists():
            backup.write_bytes(config_path.read_bytes())
        config_path.write_bytes(selected_path.read_bytes())
        print(f"Updated {config_path}; original saved to {backup}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="parallel-layout-config.yml")
    p.add_argument("--env", default="../env.sh")
    p.add_argument("--output-dir", default="context-length")
    p.add_argument("--candidates", required=True, type=parse_candidates)
    p.add_argument("--input-tokens", type=positive_int, required=True)
    p.add_argument("--output-tokens", type=positive_int, required=True)
    p.add_argument("--concurrency", type=positive_int, required=True)
    p.add_argument("--num-prompts", type=positive_int, default=100)
    p.add_argument("--context-margin", type=int, default=128)
    p.add_argument("--ttft-sla-ms", type=float)
    p.add_argument("--tpot-sla-ms", type=float)
    p.add_argument("--port", type=positive_int, default=8001)
    p.add_argument("--server-timeout", type=positive_int, default=1800)
    p.add_argument("--bench-timeout", type=positive_int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Update selected TP/DP config after successful sweep",
    )
    args = p.parse_args()
    if args.context_margin < 0:
        p.error("--context-margin must be nonnegative")
    if args.ttft_sla_ms is not None and args.ttft_sla_ms <= 0:
        p.error("--ttft-sla-ms must be positive")
    if args.tpot_sla_ms is not None and args.tpot_sla_ms <= 0:
        p.error("--tpot-sla-ms must be positive")
    try:
        return run(args)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        p.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
