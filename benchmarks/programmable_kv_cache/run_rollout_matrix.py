# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run isolated, repeated serving comparisons; only use explicitly selected GPUs."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pstats
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def install_signal_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, interrupted)


def stop(process, *, interrupt=False):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT if interrupt else signal.SIGTERM)
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
    names.extend(
        str(p)
        for p in Path("benchmarks/programmable_kv_cache/profile_bootstrap").glob("*.py")
    )
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in sorted(set(names))
        if Path(name).is_file()
    }


def dataset_identity(directory):
    metadata = (
        directory
        / ".cache/huggingface/download/main/test-00000-of-00001.parquet.metadata"
    )
    return {
        "revision": metadata.read_text().splitlines()[0] if metadata.exists() else None,
        "sha256": {
            str(path.relative_to(directory)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((directory / "main").glob("*.parquet"))
        },
    }


def read_cpu_stat(path):
    stat = path.read_text()
    end = stat.rfind(")")
    values = stat[end + 2 :].split()
    return {
        "name": stat[stat.index("(") + 1 : end],
        "start_ticks": int(values[19]),
        "cpu_seconds": (int(values[11]) + int(values[12])) / os.sysconf("SC_CLK_TCK"),
    }


def process_cpu(root_pid):
    parents = {}
    for line in subprocess.check_output(
        ["ps", "-e", "-o", "pid=,ppid="], text=True
    ).splitlines():
        pid, parent = map(int, line.split())
        parents.setdefault(parent, []).append(pid)
    pending, result = [root_pid], {}
    while pending:
        pid = pending.pop()
        pending.extend(parents.get(pid, []))
        try:
            result[pid] = read_cpu_stat(Path(f"/proc/{pid}/stat"))
        except (FileNotFoundError, ProcessLookupError):
            continue
        threads = result[pid]["threads"] = {}
        for path in Path(f"/proc/{pid}/task").glob("*/stat"):
            try:
                threads[int(path.parent.name)] = read_cpu_stat(path)
            except (FileNotFoundError, ProcessLookupError):
                continue
    return result


def toggle_cpu_profiles(directory, processes, expected, *, strict=True):
    for pid, identity in processes.items():
        try:
            metadata = json.loads((directory / f"{pid}.json").read_text())
            current = Path(f"/proc/{pid}/stat").read_text()
            start = int(current[current.rfind(")") + 2 :].split()[19])
            if (
                start != identity["start_ticks"]
                or start != metadata["start_ticks"]
                or metadata["state"] != expected
            ):
                raise RuntimeError(f"Unexpected CPU profiler state: {metadata}")
            os.kill(pid, signal.SIGUSR2)
        except (OSError, RuntimeError):
            if strict:
                raise


def wait_cpu_profiles(directory, processes, expected):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if all(
            json.loads((directory / f"{pid}.json").read_text())["state"] == expected
            for pid in processes
        ):
            return
        time.sleep(0.05)
    raise TimeoutError(f"CPU profilers did not reach {expected}")


def gpu_process_window(path, start, finish, owned_pids):
    with path.open() as file:
        lines = file.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    devices, unexpected = {}, {}
    for row in csv.DictReader(lines, skipinitialspace=True):
        timestamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
        epoch = timestamp.replace(tzinfo=timezone.utc).timestamp()
        if not start <= epoch <= finish:
            continue
        device, pid = row["gpu_uuid"], int(row["pid"])
        devices.setdefault(device, set()).add(epoch)
        if pid not in owned_pids:
            key = device, pid
            entry = unexpected.setdefault(
                key,
                {
                    "gpu_uuid": device,
                    "pid": pid,
                    "process_name": row["process_name"],
                    "first_epoch": epoch,
                    "last_epoch": epoch,
                },
            )
            entry["last_epoch"] = epoch
    coverage = {}
    for device, timestamps in devices.items():
        times = sorted(timestamps)
        coverage[device] = {
            "samples": len(times),
            "max_gap_seconds": max(
                b - a for a, b in zip([start, *times], [*times, finish])
            ),
        }
    return {"devices": coverage, "unexpected_processes": list(unexpected.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--cache-gib", type=float)
    memory.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--families", type=int, default=32)
    parser.add_argument("--fewshot", type=int, default=48)
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
    parser.add_argument("--store-gib", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-delay-iterations", type=int, default=128)
    parser.add_argument("--profile-iterations", type=int, default=32)
    parser.add_argument("--prediction-seconds", type=float, default=0.25)
    parser.add_argument(
        "--prediction-mode", choices=("fixed", "adaptive"), default="adaptive"
    )
    parser.add_argument("--movement-min-pause-seconds", type=float, default=0.5)
    parser.add_argument("--movement-min-pressure", type=float, default=0.8)
    parser.add_argument("--prefetch-on-admission", action="store_true")
    parser.add_argument("--movement-pressure-only", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--policy-components", default="retention,movement,quota")
    parser.add_argument("--policy-max-wait-seconds", type=float, default=30)
    parser.add_argument("--quota-aging-seconds", type=float, default=1)
    parser.add_argument("--quota-reuse-score-weight", type=float, default=0.5)
    parser.add_argument("--policy-decode-lookahead-tokens", type=int, default=512)
    parser.add_argument("--replay-pause-floor", type=float, default=0)
    parser.add_argument("--live-after-replay", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--live-repeats", type=int, default=1)
    parser.add_argument("--verl-root", type=Path, default=Path("../verl"))
    args = parser.parse_args()
    install_signal_handlers()
    if args.live_repeats < 1:
        parser.error("live-repeats must be positive")
    if args.live_only:
        if args.trace is None:
            parser.error("live-only requires a trace providing workload configuration")
        if args.profile:
            parser.error("live-only profiling is not supported")
        args.live_after_replay = True
    allowed_arms = {
        "native",
        "idle",
        "noop",
        "priority",
        "movement",
        "composed",
        "policy",
    }
    if not set(args.arms.split(",")) <= allowed_arms:
        parser.error("Unknown comparison arm")
    components = set(args.policy_components.split(","))
    if args.trace is None and "native" not in args.arms.split(","):
        parser.error("Trace generation requires the native arm")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.cache_gib is not None and args.cache_gib <= 0:
        parser.error("cache-gib must be positive")
    if not components <= {"retention", "movement", "quota"}:
        parser.error("Unknown policy component")
    if (
        not math.isfinite(args.movement_min_pause_seconds)
        or args.movement_min_pause_seconds < 0
    ):
        parser.error("movement-min-pause-seconds must be nonnegative and finite")
    if (
        not math.isfinite(args.movement_min_pressure)
        or not 0 <= args.movement_min_pressure <= 1
    ):
        parser.error("movement-min-pressure must be finite and between zero and one")
    if (
        not math.isfinite(args.prediction_seconds)
        or not 0 < args.prediction_seconds <= 300
    ):
        parser.error("prediction-seconds must be in (0, 300]")
    for name in ("policy_max_wait_seconds", "quota_aging_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive and finite")
    if (
        not math.isfinite(args.quota_reuse_score_weight)
        or args.quota_reuse_score_weight < 0
        or args.policy_decode_lookahead_tokens < 0
    ):
        parser.error("Quota reuse weight and decode lookahead must be nonnegative")
    policy_mode = (
        "policy"
        if "movement" in components
        else "retention"
        if "retention" in components
        else "quota"
    )
    if not 0 <= args.replay_pause_floor <= 60:
        parser.error("replay-pause-floor must be between zero and 60 seconds")
    if args.profile_delay_iterations < 0 or args.profile_iterations < 1:
        parser.error("Invalid profiling iteration counts")
    if args.profile and importlib.util.find_spec("yappi") is None:
        parser.error("Profiling requires yappi: uv pip install yappi")
    if args.live_after_replay and (args.oracle or args.replay_pause_floor):
        parser.error("Live runs cannot use replay-only prediction settings")
    require_free_port(args.port)
    require_free_port(args.store_port)
    require_free_port(args.store_port + 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "source.patch").open("w") as patch:
        subprocess.run(["git", "diff", "--binary"], stdout=patch, check=True)
    base = f"http://127.0.0.1:{args.port}"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpus,
        OMP_NUM_THREADS="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_SERVER_DEV_MODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join(
            filter(
                None, (str(args.verl_root.resolve()), os.environ.get("PYTHONPATH", ""))
            )
        ),
    )
    config_path = args.output_dir.resolve() / "mooncake-config.json"
    config_path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "master_server_address": f"127.0.0.1:{args.store_port}",
                "protocol": "tcp",
                "device_name": "",
                "global_segment_size": f"{args.store_gib}GB",
                "local_buffer_size": "256MB",
            }
        )
    )
    env["MOONCAKE_CONFIG_PATH"] = str(config_path)
    record = {
        "started_at_epoch": time.time(),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": source_hashes(),
        "dataset_identity": dataset_identity(args.dataset),
        "runs": [],
        "live_runs": [],
        "commands": [],
        "passed": False,
    }
    manifest = args.output_dir / "matrix.json"
    if args.profile:
        record["cpu_profiler"] = {
            "name": "yappi",
            "version": importlib.metadata.version("yappi"),
            "clock": "cpu",
            "scope": "Python threads",
        }
    arms = args.arms.split(",")
    groups = [
        [a for a in arms if a == "native"],
        [a for a in arms if a == "idle"],
        [a for a in arms if a not in ("native", "idle", "policy")],
        [a for a in arms if a == "policy"],
    ]
    server = store = sampler = process_sampler = None
    cpu_directory = cpu_profiled = None
    try:
        with (args.output_dir / "gpu-processes.csv").open("w") as log:
            log.write("timestamp,gpu_uuid,pid,process_name,used_memory\n")
            log.flush()
            process_sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--id={args.gpus}",
                    "--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader",
                    "--loop=5",
                ],
                env=dict(os.environ, TZ="UTC"),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record["gpu_process_log_timezone"] = "UTC"
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
                else "idle"
                if group == ["idle"]
                else "control"
                if controlled
                else "native"
            )
            needs_connector = (
                controlled
                and label != "idle"
                and not (label == "policy" and "movement" not in components)
            )
            if needs_connector:
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
                str(args.max_num_seqs or args.concurrency),
                "--max-num-batched-tokens",
                str(args.batch_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
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
            if args.cache_gib is not None:
                command.extend(
                    ["--kv-cache-memory-bytes", str(int(args.cache_gib * 2**30))]
                )
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
                            "options": {
                                "min_pressure": args.movement_min_pressure,
                                "min_pause_seconds": args.movement_min_pause_seconds,
                                "prefetch_on_admission": args.prefetch_on_admission,
                                "consider_pause_duration": (
                                    not args.movement_pressure_only
                                ),
                            },
                        },
                        {
                            "name": "quota",
                            "implementation": "quota",
                            "options": {
                                "reserve_running_growth": True,
                                "borrow_idle_capacity": True,
                                "reuse_score_weight": args.quota_reuse_score_weight,
                                "aging_seconds": args.quota_aging_seconds,
                                "max_wait_seconds": args.policy_max_wait_seconds,
                            },
                        },
                    ]
                    policies = [p for p in policies if p["name"] in components]
                settings = {
                    "engine_id": "rollout",
                    "policies": policies,
                    "policy_decode_lookahead_tokens": (
                        args.policy_decode_lookahead_tokens
                    ),
                    "max_policy_wait_seconds": args.policy_max_wait_seconds,
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
                command.extend(["--kv-control-config", json.dumps(settings)])
                if needs_connector:
                    command.extend(["--kv-transfer-config", json.dumps(connector)])
            if args.profile:
                command.extend(
                    [
                        "--profiler-config",
                        json.dumps(
                            {
                                "profiler": "torch",
                                "torch_profiler_dir": str(
                                    (args.output_dir / f"{label}-cuda").resolve()
                                ),
                                "torch_profiler_with_stack": False,
                                "torch_profiler_record_shapes": True,
                                "torch_profiler_dump_cuda_time_total": False,
                                "torch_profiler_use_gzip": False,
                                "ignore_frontend": True,
                                "delay_iterations": args.profile_delay_iterations,
                                "max_iterations": args.profile_iterations,
                            }
                        ),
                    ]
                )
            server_env = env
            if args.profile:
                cpu_directory = (args.output_dir / f"{label}-cpu").resolve()
                bootstrap = Path(__file__).parent / "profile_bootstrap"
                server_env = dict(
                    env,
                    KV_BENCH_CPU_PROFILE_DIR=str(cpu_directory),
                    PYTHONPATH=os.pathsep.join(
                        (str(bootstrap.resolve()), env["PYTHONPATH"])
                    ),
                )
            record["commands"].append(command)
            server_log = (args.output_dir / f"{label}-server.log").open("w")
            server = subprocess.Popen(
                command,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 1800
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
                raise TimeoutError("Server startup exceeded 1800 seconds")
            if args.trace is None:
                args.trace = args.output_dir / "trace.json.gz"
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
                    str(args.trace),
                    "--mode",
                    "native",
                    "--concurrency",
                    str(args.concurrency),
                    "--max-context",
                    str(args.max_context),
                    "--prompts",
                    str(args.prompts),
                    "--samples",
                    str(args.samples),
                    "--families",
                    str(args.families),
                    "--fewshot",
                    str(args.fewshot),
                ]
                record["commands"].append(command)
                with (args.output_dir / "trace.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
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
            for repeat in range(0 if args.live_only else args.repeats):
                order = list(group)
                random.Random(repeat).shuffle(order)
                for arm in order:
                    reset_cache(base, controlled)
                    if args.profile and repeat == 0 and arm == order[0]:
                        cpu_profiled = {
                            pid: row
                            for pid, row in process_cpu(server.pid).items()
                            if pid == server.pid or row["name"].startswith("VLLM::")
                        }
                        assert any(
                            "EngineCor" in row["name"] for row in cpu_profiled.values()
                        ), "Missing engine CPU profiler target"
                        toggle_cpu_profiles(cpu_directory, cpu_profiled, "ready")
                        requests.post(
                            base + "/start_profile", timeout=60
                        ).raise_for_status()
                        wait_cpu_profiles(cpu_directory, cpu_profiled, "running")
                        record.setdefault("profiled_runs", []).append(
                            {
                                "arm": arm,
                                "repeat": repeat,
                                "delay_iterations": args.profile_delay_iterations,
                                "max_iterations": args.profile_iterations,
                                "cpu_profiler": "yappi",
                                "cpu_timer": "cpu",
                                "cpu_processes": cpu_profiled,
                            }
                        )
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
                        policy_mode if arm == "policy" else arm,
                        "--concurrency",
                        str(args.concurrency),
                        "--prediction-seconds",
                        str(args.prediction_seconds),
                        "--prediction-mode",
                        args.prediction_mode,
                    ]
                    if args.oracle:
                        command.append("--oracle")
                    if args.prefetch_on_admission:
                        command.extend(
                            [
                                "--control-timeout",
                                str(
                                    args.policy_max_wait_seconds
                                    + max(
                                        args.prediction_seconds, args.replay_pause_floor
                                    )
                                    + 30
                                ),
                            ]
                        )
                    if args.profile:
                        command.append("--trace-control-rpc")
                    if args.replay_pause_floor:
                        command.extend(
                            ["--replay-pause-floor", str(args.replay_pause_floor)]
                        )
                    record["commands"].append(command)
                    cpu_before = process_cpu(server.pid)
                    if store is not None:
                        cpu_before.update(process_cpu(store.pid))
                    cpu_started = time.time()
                    cpu_boot_started = time.clock_gettime(time.CLOCK_BOOTTIME)
                    with (args.output_dir / f"{arm}-{repeat}.log").open("w") as log:
                        result = subprocess.run(
                            command, env=env, stdout=log, stderr=subprocess.STDOUT
                        )
                    if result.returncode:
                        raise RuntimeError(f"{arm}/{repeat} failed; inspect its report")
                    cpu_after = process_cpu(server.pid)
                    if store is not None:
                        cpu_after.update(process_cpu(store.pid))
                    cpu_ended = time.time()
                    with gzip.open(output, "rt") as file:
                        report = json.load(file)
                    assert report["passed"], report.get("error")
                    started = report["clock"]["wall_time"] + (
                        report["workload_started"] - report["clock"]["monotonic"]
                    )
                    stable_pids = {
                        pid
                        for pid, row in cpu_after.items()
                        if pid in cpu_before
                        and row["start_ticks"] == cpu_before[pid]["start_ticks"]
                    }
                    audit = gpu_process_window(
                        args.output_dir / "gpu-processes.csv",
                        started,
                        started + report["seconds"],
                        stable_pids,
                    )
                    record.setdefault("resource_validation", []).append(
                        {"arm": arm, "repeat": repeat, **audit}
                    )
                    if audit["unexpected_processes"]:
                        raise RuntimeError(f"GPU sharing invalidates this run: {audit}")
                    if len(audit["devices"]) != len(args.gpus.split(",")) or any(
                        device["max_gap_seconds"] > 10
                        for device in audit["devices"].values()
                    ):
                        raise RuntimeError(
                            f"GPU process log coverage incomplete: {audit}"
                        )
                    if arm == "policy" and policy_mode == "retention":
                        assert any(
                            (r.get("origin") or {}).get("policy_name") == "retention"
                            and "priority_applied" in r.get("completed_steps", [])
                            for r in report["controls"]
                        ), "Retention comparison applied no policy priorities"
                    if args.profile and repeat == 0 and arm == order[0]:
                        toggle_cpu_profiles(cpu_directory, cpu_profiled, "running")
                        requests.post(
                            base + "/stop_profile", timeout=600
                        ).raise_for_status()
                        wait_cpu_profiles(cpu_directory, cpu_profiled, "saved")
                        engine_pid = next(
                            pid
                            for pid, row in cpu_profiled.items()
                            if "EngineCor" in row["name"]
                        )
                        profiles = {
                            pid: pstats.Stats(str(cpu_directory / f"{pid}.pstats"))
                            for pid in cpu_profiled
                        }
                        assert all(
                            value[2] >= -1e-9 and value[3] >= -1e-9
                            for profile in profiles.values()
                            for value in profile.stats.values()
                        ), "CPU profiler produced negative function times"
                        for pid, profile in profiles.items():
                            metadata = json.loads(
                                (cpu_directory / f"{pid}.json").read_text()
                            )
                            assert (
                                profile.total_tt
                                <= metadata["process_cpu_seconds"] * 1.05 + 0.01
                            ), "Function CPU time exceeds process CPU time"
                        stats = profiles[engine_pid]
                        assert any(
                            key[0].endswith("vllm/v1/core/sched/scheduler.py")
                            and key[2].rsplit(".", 1)[-1] == "schedule"
                            and value[1] > 0
                            for key, value in stats.stats.items()
                        ), "CPU profile contains no engine scheduling work"
                        record["profiled_runs"][-1]["cpu_files"] = [
                            str(cpu_directory / f"{pid}.pstats") for pid in cpu_profiled
                        ]
                        cpu_profiled = None
                    digest = report["summary"]["workload_sha256"]
                    if "workload_sha256" in record:
                        assert digest == record["workload_sha256"], (
                            "Replay work changed"
                        )
                    record["workload_sha256"] = digest
                    record["runs"].append(
                        {
                            "arm": arm,
                            "repeat": repeat,
                            "seconds": report["seconds"],
                            "server_cpu_launch_window": {
                                "server_pid": server.pid,
                                "store_pid": store.pid if store is not None else None,
                                "started_at_epoch": cpu_started,
                                "started_at_boottime_seconds": cpu_boot_started,
                                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                                "ended_at_epoch": cpu_ended,
                                "components": [
                                    {
                                        "pid": pid,
                                        "name": row["name"],
                                        "cpu_seconds": row["cpu_seconds"]
                                        - cpu_before[pid]["cpu_seconds"],
                                        "threads_before": cpu_before[pid]["threads"],
                                        "threads_after": row["threads"],
                                    }
                                    for pid, row in cpu_after.items()
                                    if pid in cpu_before
                                    and row["start_ticks"]
                                    == cpu_before[pid]["start_ticks"]
                                ],
                            },
                            **report["summary"],
                        }
                    )
                    manifest.write_text(json.dumps(record, indent=2) + "\n")
                    print(json.dumps(record["runs"][-1]), flush=True)
            if args.live_after_replay:
                for live_repeat, arm in (
                    (i, a) for i in range(args.live_repeats) for a in group
                ):
                    reset_cache(base, controlled)
                    suffix = "live" if live_repeat == 0 else f"live-{live_repeat}"
                    output = args.output_dir / f"{arm}-{suffix}.json.gz"
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
                        policy_mode if arm == "policy" else arm,
                        "--concurrency",
                        str(args.concurrency),
                        "--max-context",
                        str(args.max_context),
                        "--prediction-seconds",
                        str(args.prediction_seconds),
                        "--prediction-mode",
                        args.prediction_mode,
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
                    if args.prefetch_on_admission:
                        command.extend(
                            [
                                "--control-timeout",
                                str(
                                    args.policy_max_wait_seconds
                                    + args.prediction_seconds
                                    + 30
                                ),
                            ]
                        )
                    record["commands"].append(command)
                    cpu_before = process_cpu(server.pid)
                    if store is not None:
                        cpu_before.update(process_cpu(store.pid))
                    with (args.output_dir / f"{arm}-{suffix}.log").open("w") as log:
                        subprocess.run(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    cpu_after = process_cpu(server.pid)
                    if store is not None:
                        cpu_after.update(process_cpu(store.pid))
                    with gzip.open(output, "rt") as file:
                        report = json.load(file)
                    assert report["passed"], report.get("error")
                    started = report["clock"]["wall_time"] + (
                        report["workload_started"] - report["clock"]["monotonic"]
                    )
                    audit = gpu_process_window(
                        args.output_dir / "gpu-processes.csv",
                        started,
                        started + report["seconds"],
                        {
                            pid
                            for pid, row in cpu_after.items()
                            if pid in cpu_before
                            and row["start_ticks"] == cpu_before[pid]["start_ticks"]
                        },
                    )
                    record.setdefault("live_resource_validation", []).append(
                        {"arm": arm, "repeat": live_repeat, **audit}
                    )
                    if (
                        audit["unexpected_processes"]
                        or len(audit["devices"]) != len(args.gpus.split(","))
                        or any(
                            device["max_gap_seconds"] > 10
                            for device in audit["devices"].values()
                        )
                    ):
                        raise RuntimeError(f"Live GPU process audit failed: {audit}")
                    record["live_runs"].append(
                        {
                            "arm": arm,
                            "repeat": live_repeat,
                            "seconds": report["seconds"],
                            **report["summary"],
                        }
                    )
                    manifest.write_text(json.dumps(record, indent=2) + "\n")
                    print(json.dumps({"live": record["live_runs"][-1]}), flush=True)
            stop(server, interrupt=args.profile)
            server = None
            server_log.close()
            stop(store)
            store = None
        medians = {
            arm: statistics.median(
                r["rollouts_per_second"] for r in record["runs"] if r["arm"] == arm
            )
            for arm in arms
            if any(r["arm"] == arm for r in record["runs"])
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
        if cpu_profiled:
            toggle_cpu_profiles(cpu_directory, cpu_profiled, "running", strict=False)
        stop(server, interrupt=args.profile)
        stop(store)
        stop(sampler)
        stop(process_sampler)
        manifest.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
