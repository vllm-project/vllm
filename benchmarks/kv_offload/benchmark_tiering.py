# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the KV-offload tiering subsystem with guidellm.

Currently only the FS secondary tier is supported.

Starts a vLLM server with a TieringOffloadingSpec config,
drives it with `guidellm`, and records per-run metrics.

Quick sweep (~1 hour):
    python benchmarks/kv_offload/benchmark_tiering.py bench <fs-dir> --quick

Full sweep:
    python benchmarks/kv_offload/benchmark_tiering.py bench <fs-dir>

Single run:
    python benchmarks/kv_offload/benchmark_tiering.py run <fs-dir> \
        <concurrency> <small|large> [<prefill|decode|prefill-decode>] \
        [--files small|large] [--prefix-tokens small|medium|large]

Compare two session dirs:
    python benchmarks/kv_offload/benchmark_tiering.py compare main_dir pr_dir
"""

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import regex as re

MODEL = "openai/gpt-oss-120b"
TP_SIZE = 2
PORT = 8000

BENCH_MAX_SECONDS = 420  # 7 minutes per guidellm run
BENCH_RANDOM_SEED = 889
BENCH_TURNS = 5
BENCH_PROMPT_TOKENS = 4096

# FS-file granularity: KV block_size 16 -> small files, 1024 -> large files.
FILE_SIZES = {"block16": 16, "block1024": 1024}
# Shared prefix length (tokens) driven into the FS tier.
PROMPT_SIZES = {"small": 5_000, "medium": 20_000, "large": 100_000}
# CPU tier budget in bytes.
CPU_SIZES = {"24g": 24 * 1024**3, "200g": 200 * 1024**3}
CONCURRENCY_SWEEP = (16, 32, 64)

# Phase tag -> guidellm output_tokens per turn.
PHASES = {"prefill": 1, "decode": 512, "prefill-decode": 256}

METRICS = [
    "request_throughput",
    "request_latency_mean",
    "request_latency_p99",
    "output_token_throughput",
    "total_token_throughput",
    "ttft_mean",
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
]


@dataclass
class RunSpec:
    phase: str  # PHASES key
    files: str  # FILE_SIZES key
    prompt: str  # PROMPT_SIZES key
    concurrency: int
    cpu_size: str  # CPU_SIZES key

    @property
    def name(self) -> str:
        return (
            f"{self.phase}_files-{self.files}_prompt-{self.prompt}"
            f"_c{self.concurrency}_cpu-{self.cpu_size}"
        )


def build_sweeps(mode: str) -> list[RunSpec]:
    if mode == "quick":
        # ~1 hour: 8 runs x ~7.5 min (bench + startup + teardown).
        return [
            RunSpec(p, f, "medium", 64, cpu)
            for cpu in CPU_SIZES
            for p in ("prefill", "decode")
            for f in FILE_SIZES
        ]
    # --full: everything except prefill-decode (run that phase on request).
    return [
        RunSpec(p, f, s, c, cpu)
        for p in ("prefill", "decode")
        for f in FILE_SIZES
        for s in PROMPT_SIZES
        for c in CONCURRENCY_SWEEP
        for cpu in CPU_SIZES
    ]


def kill_vllm(proc: subprocess.Popen) -> None:
    """SIGINT x5 (10s waits), then SIGKILL; exit loudly if unkillable."""
    for attempt in range(1, 6):
        if proc.poll() is not None:
            print(f"      vLLM process ({proc.pid}) has exited gracefully.")
            return
        print(f"      [Attempt {attempt}/5] Sending SIGINT to PID {proc.pid}...")
        proc.send_signal(signal.SIGINT)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
    if proc.poll() is None:
        print("      vLLM did not exit after 5 attempts. Force killing...")
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
    if proc.poll() is None:
        raise RuntimeError(
            f"vLLM process ({proc.pid}) could NOT be killed; GPU VRAM is "
            "likely still locked. Aborting to prevent cascading failures."
        )
    print("      vLLM successfully terminated.")


def cleanup(fs_dir: Path) -> None:
    for entry in fs_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def write_kv_config(path: Path, spec: RunSpec, fs_dir: Path) -> None:
    config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": CPU_SIZES[spec.cpu_size],
            "eviction_policy": "lru",
            "secondary_tiers": [
                {
                    "type": "fs",
                    "root_dir": str(fs_dir) + "/",
                }
            ],
            "enable_cross_layers_blocks": "True",
            "block_size": FILE_SIZES[spec.files],
        },
    }
    path.write_text(json.dumps(config, indent=2))


def start_vllm(config_path: Path, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHON_HASH_SEED="0")
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--kv-transfer-config",
        config_path.read_text(),
        "--enable-prefix-caching",
        "--no-disable-hybrid-kv-cache-manager",
        "--port",
        str(PORT),
    ]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc


def wait_for_vllm(proc: subprocess.Popen, log_path: Path, timeout: int = 1800) -> None:
    print(f"      Waiting for vLLM on port {PORT}...")
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{PORT}/v1/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(_crash_report(proc.pid, log_path))
        try:
            urlopen(url, timeout=5)
            print("      vLLM is ready!")
            return
        except OSError:
            time.sleep(5)
    raise RuntimeError(
        _crash_report(proc.pid, log_path, "timed out waiting to accept connections")
    )


def _crash_report(pid: int, log_path: Path, why: str = "crashed during startup") -> str:
    tail = "\n".join(log_path.read_text().splitlines()[-20:])
    return f"vLLM process (PID {pid}) {why}.\nLast 20 lines of {log_path}:\n{tail}"


def run_guidellm(spec: RunSpec, log_path: Path) -> dict:
    prefix_count = max(4 * spec.concurrency, 256)
    data = {
        "kind": "synthetic_text",
        "prompt_tokens": BENCH_PROMPT_TOKENS,
        "output_tokens": PHASES[spec.phase],
        "turns": BENCH_TURNS,
        "prefix_buckets": [
            {
                "bucket_weight": 100,
                "prefix_count": prefix_count,
                "prefix_tokens": PROMPT_SIZES[spec.prompt],
            }
        ],
    }
    cmd = [
        "guidellm",
        "run",
        "--backend",
        "kind=openai_http,"
        f"target=http://127.0.0.1:{PORT},"
        "request_format=/v1/completions",
        "--profile",
        f"kind=concurrent,streams={spec.concurrency}",
        "--constraint",
        f"kind=max_duration,seconds={BENCH_MAX_SECONDS}",
        "--seed",
        f"kind=static,value={BENCH_RANDOM_SEED}",
        "--data",
        json.dumps(data),
    ]
    with log_path.open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return parse_guidellm_report(log_path.read_text())


def parse_guidellm_report(text: str) -> dict:
    """Extract `Metric: value` pairs from the guidellm stdout report."""
    metrics = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9 ()/]+?):\s+([0-9.]+)\s", line + " ")
        if m:
            key = (
                m.group(1)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            metrics[key] = m.group(2)
    return {k: v for k, v in metrics.items() if k in METRICS}


def run_one(spec: RunSpec, fs_dir: Path, session_dir: Path) -> dict:
    run_dir = session_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = run_dir / "vllm.log"
    guidellm_log = run_dir / "guidellm.log"

    print("=" * 70)
    print(f"Run: {spec.name}")
    print(f"Logs: {run_dir}/")
    print("=" * 70)

    print("  [1/5] Cleanup...")
    cleanup(fs_dir)
    print("  [2/5] Writing KV transfer config...")
    config_path = run_dir / "kv_transfer_config.json"
    write_kv_config(config_path, spec, fs_dir)
    print("  [3/5] Starting vLLM server...")
    proc = start_vllm(config_path, vllm_log)
    try:
        wait_for_vllm(proc, vllm_log)
        print("  [4/5] Running guidellm benchmark...")
        result = {
            "spec": asdict(spec),
            "status": "ok",
            **run_guidellm(spec, guidellm_log),
        }
    finally:
        print("  [5/5] Teardown: stopping vLLM...")
        kill_vllm(proc)
    time.sleep(5)
    return result


def compare(session_a: Path, session_b: Path) -> None:
    def load(d: Path) -> dict:
        results = json.loads((d / "results.json").read_text())
        return {r["spec"]["name"] if "spec" in r else r["name"]: r for r in results}

    a, b = load(session_a), load(session_b)
    print(f"{'run':<60} {'metric':<30} {session_a.name:>12} {session_b.name:>12}")
    for name in sorted(set(a) & set(b)):
        for metric in METRICS:
            va, vb = a[name].get(metric), b[name].get(metric)
            if va is not None and vb is not None:
                delta = (float(vb) - float(va)) / float(va) * 100
                print(f"{name:<60} {metric:<30} {va:>12} {vb:>12}  ({delta:+.1f}%)")


def preflight(fs_dir: Path) -> None:
    if shutil.which("guidellm") is None:
        answer = (
            input(
                "guidellm not found on PATH. Install it now "
                "(pip install guidellm)? [y/N] "
            )
            .strip()
            .lower()
        )
        if answer != "y":
            raise RuntimeError("Install guidellm and re-run: pip install guidellm")
        subprocess.run([sys.executable, "-m", "pip", "install", "guidellm"], check=True)
        if shutil.which("guidellm") is None:
            raise RuntimeError(
                "guidellm was installed but is not on PATH; "
                "check that it landed in this environment's bin dir."
            )
    if shutil.which("vllm") is None:
        raise RuntimeError("vllm not found on PATH")
    fs_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    try:
        _main()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(f"ERROR: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser(
        "bench", help="run the sweep matrix against a vLLM server"
    )
    bench.add_argument(
        "fs_dir",
        type=Path,
        help="folder backing the tier store (wiped between runs)",
    )
    bench.add_argument(
        "--quick",
        action="store_true",
        help="pre-selected sweep, completes in about an hour",
    )

    run = subparsers.add_parser("run", help="run a single benchmark configuration")
    run.add_argument(
        "fs_dir",
        type=Path,
        help="folder backing the tier store (wiped between runs)",
    )
    run.add_argument("concurrency", type=int)
    run.add_argument("cpu_size", choices=sorted(CPU_SIZES))
    run.add_argument(
        "config_tag",
        nargs="?",
        choices=sorted(PHASES),
        default="prefill",
        help="phase tag: prefill, decode, or prefill-decode (default: prefill)",
    )
    run.add_argument(
        "--files",
        choices=sorted(FILE_SIZES),
        default="block1024",
        help="file granularity (block_size 16 vs 1024)",
    )
    run.add_argument(
        "--prefix-tokens",
        choices=sorted(PROMPT_SIZES),
        default="medium",
        help="turn-0 prompt length (shared prefix driven into the FS tier)",
    )

    compare_parser = subparsers.add_parser("compare", help="compare two session dirs")
    compare_parser.add_argument("session_a", type=Path, metavar="MAIN_DIR")
    compare_parser.add_argument("session_b", type=Path, metavar="PR_DIR")
    return parser


def _main() -> None:
    args = build_parser().parse_args()

    if args.command == "compare":
        compare(args.session_a, args.session_b)
        return
    if args.command == "run":
        specs = [
            RunSpec(
                args.config_tag,
                args.files,
                args.prefix_tokens,
                args.concurrency,
                args.cpu_size,
            )
        ]
    else:
        specs = build_sweeps("quick" if args.quick else "full")

    try:
        preflight(args.fs_dir)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(f"Preflight failed: {e}")

    session_dir = Path(f"benchmark_session_{datetime.now():%Y%m%d_%H%M%S}")
    session_dir.mkdir()
    print(f"Session dir: {session_dir}/")
    print(
        f"{len(specs)} runs "
        f"(~{len(specs) * (BENCH_MAX_SECONDS + 120) // 60} min total):"
    )
    for spec in specs:
        print(f"  {spec.name}")

    results = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}]")
        r = run_one(spec, args.fs_dir, session_dir)
        results.append(r)
        (session_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"{spec.name}: {r['status']}")
        for m in METRICS:
            print(f"  {m}: {r.get(m, 'n/a')}")
    print(f"\nAll runs complete. Results in {session_dir}/")


if __name__ == "__main__":
    main()
