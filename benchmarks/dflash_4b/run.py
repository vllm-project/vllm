# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run single-GPU, concurrency-one Qwen3.5 comparisons (4B by default)."""

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
DRAFT = ("z-lab/Qwen3.5-4B-DFlash", "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf")
TARGET_27B = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
DRAFT_27B = ("z-lab/Qwen3.5-27B-DFlash", "25ee0025ff950496a634e100b75c2db4515e9824")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("vllm", "sglang"))
    parser.add_argument("mode", choices=("baseline", "dflash"))
    parser.add_argument("--model-size", choices=("4B", "27B"), default="4B")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--request-count", type=int, default=200)
    parser.add_argument("--warmup-request-count", type=int, default=20)
    parser.add_argument("--attn-group-size", type=int)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target_model, draft_model = (
        (TARGET_27B, DRAFT_27B) if args.model_size == "27B" else (TARGET, DRAFT)
    )
    if args.attn_group_size is not None and (
        args.engine != "vllm" or args.attn_group_size <= 0
    ):
        parser.error("--attn-group-size requires vllm and a positive integer")
    if args.dry_run:
        print(json.dumps(server_command(args, target_model[0], draft_model[0])))
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Reserve a GPU using run.sh (canhazgpu) first")
    output = args.output.resolve() / f"{args.engine}_{args.mode}"
    output.mkdir(parents=True, exist_ok=False)
    target = download_model(target_model)
    draft = download_model(draft_model) if args.mode == "dflash" else draft_model[0]
    command = server_command(args, target, draft)
    base = f"http://127.0.0.1:{args.port}"
    # Refuse to accidentally benchmark an existing server on this port.
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    config = {
        "engine": args.engine,
        "mode": args.mode,
        "models": {"target": target_model, "draft": draft_model},
        "public_dataset": "spec_al_gsm8k",
        "versions": versions(args.engine),
        "server_command": command,
        "vllm_runner": "V2" if args.engine == "vllm" else None,
        "concurrency": 1,
        "warmup_requests": args.warmup_request_count,
        "measured_requests": args.request_count,
        "requested_output_tokens": 256,
        "attn_group_size": args.attn_group_size,
    }
    env = os.environ.copy()
    if args.engine == "vllm":
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    with (output / "server.log").open("w") as log:
        startup_start = time.monotonic()
        server = subprocess.Popen(
            command, stdout=log, stderr=log, env=env, start_new_session=True
        )
        try:
            wait_ready(server, base)
            startup_seconds = time.monotonic() - startup_start
            snapshot(base, output, "before", args.engine)
            bench = [
                str(Path(sys.executable).parent / "aiperf"),
                "profile",
                "--model",
                "qwen",
                "--tokenizer",
                target,
                "--url",
                base,
                "--request-count",
                str(args.request_count),
                "--warmup-request-count",
                str(args.warmup_request_count),
                "--public-dataset",
                "spec_al_gsm8k",
                "--extra-inputs",
                "max_completion_tokens:256",
                "--concurrency",
                "1",
                "--endpoint-type",
                "chat",
                "--streaming",
                "--output-artifact-dir",
                str(output / "aiperf"),
            ]
            config["benchmark_command"] = bench
            write_json(output / "run_config.json", config)
            print(f"Running {args.engine} {args.mode}; logs: {output}", flush=True)
            with (output / "benchmark.log").open("w") as bench_log:
                benchmark_start = time.monotonic()
                subprocess.run(bench, stdout=bench_log, stderr=bench_log, check=True)
                benchmark_seconds = time.monotonic() - benchmark_start
            snapshot(base, output, "after", args.engine)
            summary = summarize(output, args.engine, args.mode, args.request_count)
            summary["timing_seconds"] = {
                "server_startup": startup_seconds,
                "benchmark_wall_clock": benchmark_seconds,
            }
            write_json(output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
        finally:
            stop(server)


def download_model(model):
    # The Hub CLI works in either environment without importing vLLM into SGLang.
    command = [
        str(Path(sys.executable).parent / "hf"),
        "download",
        model[0],
        "--revision",
        model[1],
        "--quiet",
    ]
    return subprocess.check_output(command, text=True).strip()


def server_command(args, target, draft):
    common = ["--host", "127.0.0.1", "--port", str(args.port)]
    if args.engine == "vllm":
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--disable-uvicorn-access-log",
            "--model",
            target,
            "--served-model-name",
            "qwen",
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--quantization",
            "fp8",
            "--language-model-only",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "128",
            "--enable-prefix-caching",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--trust-remote-code",
            "--cudagraph-capture-sizes",
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",
        ]
        if args.attn_group_size is not None:
            command += ["--attn-group-size", str(args.attn_group_size)]
        if args.mode == "dflash":
            command += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dflash",
                        "model": draft,
                        "num_speculative_tokens": 15,
                    }
                ),
            ]
    else:
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--log-level-http",
            "warning",
            "--model-path",
            target,
            "--served-model-name",
            "qwen",
            "--tp-size",
            "1",
            "--dtype",
            "bfloat16",
            "--quantization",
            "fp8",
            "--context-length",
            "32768",
            "--mem-fraction-static",
            "0.8",
            "--enable-metrics",
            "--trust-remote-code",
        ]
        if args.mode == "dflash":
            command += [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                draft,
                "--speculative-num-draft-tokens",
                "16",
            ]
    return command + common


def summarize(output, engine, mode, request_count=200):
    report = json.loads((output / "aiperf/profile_export_aiperf.json").read_text())
    if report["request_count"]["avg"] != request_count:
        raise RuntimeError(
            f"Expected {request_count} successful measured requests; inspect logs"
        )
    metrics = json.loads((output / "aiperf/server_metrics_export.json").read_text())[
        "metrics"
    ]
    al = None
    al_method = None
    if mode == "dflash":
        if engine == "vllm":
            drafts = metric_stat(metrics, "vllm:spec_decode_num_drafts", "total")
            accepted = metric_stat(
                metrics, "vllm:spec_decode_num_accepted_tokens", "total"
            )
            al = 1 + accepted / drafts
            al_method = "1 + accepted draft tokens / draft iterations (measured phase)"
        else:
            al = metric_stat(metrics, "sglang:spec_accept_length", "avg")
            al_method = "Mean sampled AL gauge (measured phase); different weighting"
    return {
        "engine": engine,
        "mode": mode,
        "itl_p50_ms": report["inter_chunk_latency"]["p50"],
        "tpot_p50_ms": report["inter_token_latency"]["p50"],
        "output_tokens_per_second": report["output_token_throughput"]["avg"],
        "acceptance_length": al,
        "acceptance_length_method": al_method,
    }


def metric_stat(metrics, name, stat):
    return sum(s["stats"][stat] for s in metrics[name]["series"])


def versions(engine):
    result = {}
    for name in (engine, "torch", "flashinfer-python", "aiperf"):
        result[name] = importlib.metadata.version(name)
    return result


def wait_ready(server, base):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("Server exited during startup; inspect server.log")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError("Server startup exceeded 30 minutes")


def snapshot(base, output, phase, engine):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
        (output / f"metrics_{phase}.prom").write_bytes(response.read())
    if engine == "sglang":
        with urllib.request.urlopen(base + "/get_server_info", timeout=30) as response:
            (output / f"server_info_{phase}.json").write_bytes(response.read())


def stop(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
