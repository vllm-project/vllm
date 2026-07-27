#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the speculative-decoding throughput matrix."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib import request

ROOT = Path("/apdcephfs_sgfd2/share_300532381/ruicen/draft_models")
JSON_ROOT = ROOT / "deepspec_eval_datasets"
DATASETS = {
    "math500": "math500.jsonl",
    "humaneval": "humaneval.jsonl",
    "gsm8k": "gsm8k.jsonl",
    "mtbench": "mt-bench.jsonl",
    "livecodebench": "livecodebench.jsonl",
    "mbpp": "mbpp.jsonl",
}
DATASET_LIMITS = {
    "gsm8k": 500,
    "math500": 500,
    "humaneval": 164,
    "mbpp": 256,
    "livecodebench": 500,
    "mtbench": 80,
}
CONCURRENCIES = [4, 8, 16, 32, 64, 128]
CLIENT = Path(__file__).with_name("load_generator.py")
CONFIGS = Path(__file__).with_name("configs.json")
CHAT_KWARGS = {
    "hy3": {"reasoning_effort": "no_think"},
    "qwen3-8b": {"enable_thinking": False},
}


def cudagraph_mode(config: dict[str, Any], method: str) -> str:
    speculative = config["methods"][method]
    return (
        "PIECEWISE"
        if speculative and speculative.get("dflash_dcut")
        else "FULL_AND_PIECEWISE"
    )


def server_command(
    config: dict[str, Any],
    method: str,
    port: int,
    max_num_seqs: int = 128,
    gpu_memory_utilization: float = 0.97,
) -> list[str]:
    speculative = config["methods"][method]
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        config["target"],
        "--tensor-parallel-size",
        str(config["tp"]),
        "--served-model-name",
        config["served_name"],
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--safetensors-load-strategy",
        "prefetch",
        "--safetensors-prefetch-num-threads",
        "16",
        "--compilation-config",
        json.dumps(
            {"cudagraph_mode": cudagraph_mode(config, method)},
            separators=(",", ":"),
        ),
    ]
    if speculative:
        command += [
            "--speculative-config",
            json.dumps(speculative, separators=(",", ":")),
        ]
    return command


def wait_ready(process: subprocess.Popen[Any], port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with request.urlopen(
                f"http://127.0.0.1:{port}/v1/models",
                timeout=5,
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("server startup timed out")


def stop(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def complete(path: Path, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    dataset = result.get("dataset_name")
    max_prompts = args.max_prompts or DATASET_LIMITS.get(dataset)
    return (
        result.get("status") == "ok"
        and result.get("warmup_seconds") == args.warmup_seconds
        and result.get("measure_seconds") == args.measure_seconds
        and result.get("repeats") == args.repeats
        and result.get("max_tokens") == args.max_tokens
        and result.get("max_prompts") == max_prompts
        and result.get("prompt_seed") == args.prompt_seed
    )


def cell_path(
    output: Path,
    method: str,
    dataset: str,
    concurrency: int,
) -> Path:
    return output / "cells" / method / f"{dataset}_c{concurrency}.json"


def run_cell(
    config: dict[str, Any],
    method: str,
    dataset: str,
    concurrency: int,
    args: argparse.Namespace,
) -> None:
    result = cell_path(args.output_dir, method, dataset, concurrency)
    if args.resume and complete(result, args):
        print(f"skip {method} {dataset} c{concurrency}", flush=True)
        return
    max_prompts = args.max_prompts or DATASET_LIMITS[dataset]
    command = [
        sys.executable,
        str(CLIENT),
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model",
        config["served_name"],
        "--dataset",
        str(args.dataset_root / DATASETS[dataset]),
        "--output",
        str(result),
        "--concurrency",
        str(concurrency),
        "--warmup-seconds",
        str(args.warmup_seconds),
        "--measure-seconds",
        str(args.measure_seconds),
        "--repeats",
        str(args.repeats),
        "--max-tokens",
        str(args.max_tokens),
        "--max-prompts",
        str(max_prompts),
        "--prompt-seed",
        str(args.prompt_seed),
        "--request-prefix",
        f"{args.model_family}-{method}-{dataset}-c{concurrency}",
    ]
    if config["methods"][method]:
        command.append("--require-spec")
    log_path = args.output_dir / "logs" / f"cell_{method}_{dataset}_c{concurrency}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        return_code = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if return_code:
        raise RuntimeError(f"cell failed; see {log_path}")
    data = json.loads(result.read_text(encoding="utf-8"))
    data.update(
        {
            "model_family": args.model_family,
            "method": method,
            "dataset_name": dataset,
            "speculative_config": config["methods"][method],
        }
    )
    result.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rates = [round(item["output_throughput"], 2) for item in data["windows"]]
    print(f"done {method} {dataset} c{concurrency}: {rates}", flush=True)


def summarize(output: Path) -> None:
    summary = []
    for path in sorted((output / "cells").glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        windows = data["windows"]
        summary.append(
            {
                "model": data.get("model_family"),
                "method": data.get("method"),
                "dataset": data.get("dataset_name"),
                "concurrency": data["concurrency"],
                "output_throughput": sum(item["output_throughput"] for item in windows)
                / len(windows),
                "mean_acceptance_length": sum(
                    item["accepted_tokens"] for item in windows
                )
                / sum(item["drafts"] for item in windows)
                + 1
                if sum(item["drafts"] for item in windows)
                else None,
            }
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_method(
    config: dict[str, Any],
    method: str,
    args: argparse.Namespace,
) -> None:
    paths = [
        cell_path(args.output_dir, method, dataset, concurrency)
        for dataset in args.datasets
        for concurrency in args.concurrencies
    ]
    if args.resume and all(complete(path, args) for path in paths):
        print(f"skip completed method: {method}", flush=True)
        return

    command = server_command(
        config,
        method,
        args.port,
        args.max_num_seqs,
        args.gpu_memory_utilization,
    )
    log_path = args.output_dir / "logs" / f"server_{method}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_DSPARK_HPC_CORRECTION": "1",
    }
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    try:
        wait_ready(process, args.port, args.server_timeout)
        print(f"server ready: {method}", flush=True)
        for dataset in args.datasets:
            for concurrency in args.concurrencies:
                run_cell(config, method, dataset, concurrency, args)
                summarize(args.output_dir)
    finally:
        stop(process)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-family",
        required=True,
        choices=("hy3", "qwen3-8b"),
    )
    parser.add_argument("--methods")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--concurrencies",
        default=",".join(map(str, CONCURRENCIES)),
    )
    parser.add_argument("--dataset-root", type=Path, default=JSON_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--warmup-seconds", type=float, default=30)
    parser.add_argument("--measure-seconds", type=float, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--prompt-seed", type=int, default=980406)
    parser.add_argument("--server-timeout", type=int, default=2400)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_configs = json.loads(CONFIGS.read_text(encoding="utf-8"))
    config = all_configs[args.model_family]
    args.methods = parse_csv(args.methods) if args.methods else list(config["methods"])
    args.datasets = parse_csv(args.datasets)
    args.concurrencies = [int(item) for item in parse_csv(args.concurrencies)]
    if set(args.methods) - config["methods"].keys():
        parser.error("unknown method")
    if set(args.datasets) - DATASETS.keys():
        parser.error("unknown dataset")

    if args.dry_run:
        commands = {
            method: server_command(
                config,
                method,
                args.port,
                args.max_num_seqs,
                args.gpu_memory_utilization,
            )
            for method in args.methods
        }
        print(json.dumps(commands, indent=2, ensure_ascii=False))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model_family": args.model_family,
        "methods": {method: config["methods"][method] for method in args.methods},
        "datasets": args.datasets,
        "concurrencies": args.concurrencies,
        "warmup_seconds": args.warmup_seconds,
        "measure_seconds": args.measure_seconds,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "dataset_root": str(args.dataset_root),
        "dataset_prompt_limits": {
            dataset: args.max_prompts or DATASET_LIMITS[dataset]
            for dataset in args.datasets
        },
        "prompt_seed": args.prompt_seed,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "request": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 0,
            "max_completion_tokens": args.max_tokens,
            "chat_template_kwargs": CHAT_KWARGS[args.model_family],
            "stream": True,
            "ignore_eos": False,
        },
        "server_commands": {
            method: server_command(
                config,
                method,
                args.port,
                args.max_num_seqs,
                args.gpu_memory_utilization,
            )
            for method in args.methods
        },
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for method in args.methods:
        run_method(config, method, args)
    summarize(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
