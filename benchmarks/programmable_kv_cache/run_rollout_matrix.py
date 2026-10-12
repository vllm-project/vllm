# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run isolated, repeated serving comparisons; only use explicitly selected GPUs."""

import argparse
import gzip
import hashlib
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests


def stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=45)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def require_free_port(port):
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=1)
    except OSError:
        return
    connection.close()
    raise RuntimeError(f"Port {port} already belongs to another process")


def reset_cache(base, controlled):
    deadline = time.monotonic() + 120
    while True:
        reset = requests.post(
            base + "/reset_prefix_cache",
            params={"reset_external": controlled},
            timeout=120,
        )
        reset.raise_for_status()
        if reset.json()["success"]:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Cache reset remained busy")
        time.sleep(1)


def source_hashes():
    names = subprocess.check_output(
        ["git", "diff", "--name-only"], text=True
    ).splitlines()
    names.extend(str(p) for p in Path("benchmarks/programmable_kv_cache").glob("*.py"))
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in sorted(set(names))
        if Path(name).is_file()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-gib", type=float, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--arms", default="native,noop,priority,movement,composed,policy"
    )
    parser.add_argument("--gpus", default="0,3")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--store-port", type=int, default=19151)
    parser.add_argument("--prediction-seconds", type=float, default=0.25)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--policy-components", default="retention,movement,quota")
    parser.add_argument("--replay-pause-floor", type=float, default=0)
    parser.add_argument("--live-after-replay", action="store_true")
    parser.add_argument("--verl-root", type=Path, default=Path("../verl"))
    args = parser.parse_args()
    components = set(args.policy_components.split(","))
    if not components <= {"retention", "movement", "quota"}:
        parser.error("Unknown policy component")
    if not 0 <= args.replay_pause_floor <= 60:
        parser.error("replay-pause-floor must be between zero and 60 seconds")
    if args.live_after_replay and (args.oracle or args.replay_pause_floor):
        parser.error("Live runs cannot use replay-only prediction settings")
    require_free_port(args.port)
    require_free_port(args.store_port)
    require_free_port(args.store_port + 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpus,
        OMP_NUM_THREADS="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_SERVER_DEV_MODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=str(args.verl_root.resolve()),
    )
    config_path = args.output_dir.resolve() / "mooncake-config.json"
    config_path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "master_server_address": f"127.0.0.1:{args.store_port}",
                "protocol": "tcp",
                "device_name": "",
                "global_segment_size": "8GB",
                "local_buffer_size": "256MB",
            }
        )
    )
    env["MOONCAKE_CONFIG_PATH"] = str(config_path)
    record = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": source_hashes(),
        "runs": [],
        "live_runs": [],
        "commands": [],
        "passed": False,
    }
    manifest = args.output_dir / "matrix.json"
    arms = args.arms.split(",")
    groups = [
        [a for a in arms if a == "native"],
        [a for a in arms if a not in ("native", "policy")],
        [a for a in arms if a == "policy"],
    ]
    server = store = sampler = None
    try:
        with (args.output_dir / "gpu.csv").open("w") as gpu_log:
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--id={args.gpus}",
                    "--query-gpu=timestamp,index,name,memory.used,memory.total,"
                    "utilization.gpu,utilization.memory,power.draw,clocks.sm",
                    "--format=csv",
                    "--loop=1",
                ],
                stdout=gpu_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for group in groups:
            if not group:
                continue
            controlled = group != ["native"]
            label = (
                "policy"
                if group == ["policy"]
                else "control"
                if controlled
                else "native"
            )
            if controlled:
                command = [
                    str(Path(sys.executable).parent / "mooncake_master"),
                    "--rpc_address=127.0.0.1",
                    f"--rpc_port={args.store_port}",
                    "--metrics_host=127.0.0.1",
                    f"--metrics_port={args.store_port + 2}",
                    "--enable_metric_reporting=false",
                ]
                record["commands"].append(command)
                store_log = (args.output_dir / f"{label}-store.log").open("w")
                store = subprocess.Popen(
                    command,
                    env=env,
                    stdout=store_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(args.model),
                "--served-model-name",
                "qwen-rollout",
                "--tensor-parallel-size",
                "2",
                "--max-model-len",
                str(args.max_context),
                "--max-num-seqs",
                str(args.concurrency),
                "--max-num-batched-tokens",
                "4096",
                "--kv-cache-memory-bytes",
                str(int(args.cache_gib * 2**30)),
                "--mamba-cache-mode",
                "align",
                "--dtype",
                "bfloat16",
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--disable-uvicorn-access-log",
            ]
            if args.enforce_eager:
                command.append("--enforce-eager")
            if args.disable_custom_all_reduce:
                command.append("--disable-custom-all-reduce")
            if controlled:
                policies = []
                if label == "policy":
                    policies = [
                        {"name": "retention", "implementation": "retention"},
                        {
                            "name": "movement",
                            "implementation": "tool_wait",
                            "options": {"min_pressure": 0.6},
                        },
                        {
                            "name": "quota",
                            "implementation": "quota",
                            "options": {
                                "reserve_running_growth": True,
                                "borrow_idle_capacity": True,
                                "reuse_score_weight": 0.5,
                                "max_wait_seconds": 30,
                            },
                        },
                    ]
                    policies = [p for p in policies if p["name"] in components]
                settings = {
                    "engine_id": "rollout",
                    "policies": policies,
                    "policy_decode_lookahead_tokens": 512,
                    "max_policy_wait_seconds": 30,
                    "max_inflight_load_blocks": 32,
                }
                connector = {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "enable_lookup": False,
                        "enable_save": False,
                        "save_decode_cache": False,
                    },
                }
                command.extend(
                    [
                        "--kv-control-config",
                        json.dumps(settings),
                        "--kv-transfer-config",
                        json.dumps(connector),
                    ]
                )
            record["commands"].append(command)
            server_log = (args.output_dir / f"{label}-server.log").open("w")
            server = subprocess.Popen(
                command,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"{label} server exited: {server.returncode}")
                try:
                    if requests.get(base + "/health", timeout=3).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Server startup exceeded 900 seconds")
            with gzip.open(args.trace, "rt") as file:
                trace = json.load(file)
            for row in trace["requests"][:2]:
                warmup = requests.post(
                    base + "/v1/completions",
                    json={
                        "model": "qwen-rollout",
                        "prompt": row["prompt_ids"],
                        "max_tokens": 16,
                        "temperature": 0,
                        "cache_salt": "warmup",
                    },
                    timeout=300,
                )
                warmup.raise_for_status()
            for repeat in range(args.repeats):
                order = list(group)
                random.Random(repeat).shuffle(order)
                for arm in order:
                    reset_cache(base, controlled)
                    output = args.output_dir / f"{arm}-{repeat}.json.gz"
                    command = [
                        sys.executable,
                        "-m",
                        "benchmarks.programmable_kv_cache.verl_rollout",
                        "--base-url",
                        base,
                        "--model",
                        str(args.model),
                        "--dataset",
                        str(args.dataset),
                        "--trace",
                        str(args.trace),
                        "--output",
                        str(output),
                        "--mode",
                        arm,
                        "--concurrency",
                        str(args.concurrency),
                        "--prediction-seconds",
                        str(args.prediction_seconds),
                    ]
                    if args.oracle:
                        command.append("--oracle")
                    if args.replay_pause_floor:
                        command.extend(
                            ["--replay-pause-floor", str(args.replay_pause_floor)]
                        )
                    record["commands"].append(command)
                    with (args.output_dir / f"{arm}-{repeat}.log").open("w") as log:
                        result = subprocess.run(
                            command, env=env, stdout=log, stderr=subprocess.STDOUT
                        )
                    if result.returncode:
                        raise RuntimeError(f"{arm}/{repeat} failed; inspect its report")
                    with gzip.open(output, "rt") as file:
                        report = json.load(file)
                    assert report["passed"], report.get("error")
                    record["runs"].append(
                        {
                            "arm": arm,
                            "repeat": repeat,
                            "seconds": report["seconds"],
                            **report["summary"],
                        }
                    )
                    manifest.write_text(json.dumps(record, indent=2) + "\n")
                    print(json.dumps(record["runs"][-1]), flush=True)
            if args.live_after_replay:
                for arm in group:
                    reset_cache(base, controlled)
                    output = args.output_dir / f"{arm}-live.json.gz"
                    command = [
                        sys.executable,
                        "-m",
                        "benchmarks.programmable_kv_cache.verl_rollout",
                        "--base-url",
                        base,
                        "--model",
                        str(args.model),
                        "--dataset",
                        str(args.dataset),
                        "--output",
                        str(output),
                        "--mode",
                        arm,
                        "--concurrency",
                        str(args.concurrency),
                        "--max-context",
                        str(args.max_context),
                        "--prediction-seconds",
                        str(args.prediction_seconds),
                    ]
                    for key in (
                        "prompts",
                        "samples",
                        "families",
                        "fewshot",
                        "max_tokens",
                        "response_length",
                        "tool_delay",
                        "temperature",
                        "seed",
                    ):
                        command.extend(
                            ["--" + key.replace("_", "-"), str(trace["config"][key])]
                        )
                    record["commands"].append(command)
                    with (args.output_dir / f"{arm}-live.log").open("w") as log:
                        subprocess.run(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    with gzip.open(output, "rt") as file:
                        report = json.load(file)
                    assert report["passed"], report.get("error")
                    record["live_runs"].append(
                        {"arm": arm, "seconds": report["seconds"], **report["summary"]}
                    )
                    manifest.write_text(json.dumps(record, indent=2) + "\n")
                    print(json.dumps({"live": record["live_runs"][-1]}), flush=True)
            stop(server)
            server = None
            server_log.close()
            stop(store)
            store = None
        medians = {
            arm: statistics.median(
                r["rollouts_per_second"] for r in record["runs"] if r["arm"] == arm
            )
            for arm in arms
        }
        if "native" in medians:
            record["throughput_ratios"] = {
                arm: value / medians["native"] for arm, value in medians.items()
            }
            record["target_25_percent_met"] = {
                arm: ratio >= 1.25
                for arm, ratio in record["throughput_ratios"].items()
                if arm not in ("native", "noop")
            }
        record["passed"] = True
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop(server)
        stop(store)
        stop(sampler)
        manifest.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
