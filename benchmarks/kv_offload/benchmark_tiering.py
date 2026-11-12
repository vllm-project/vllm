# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the KV-offload tiering subsystem with guidellm.

Currently only the FS secondary tier is supported.

Starts a vLLM server with a TieringOffloadingSpec config,
drives it with `guidellm`, and records per-run metrics.

Run all scenarios:
    python benchmarks/kv_offload/benchmark_tiering.py bench-scenario <fs-dir>

Run selected scenarios:
    python benchmarks/kv_offload/benchmark_tiering.py bench-scenario <fs-dir> \
        --scenario fit-c16 overflow-c64

List scenarios with their estimated KV footprint:
    python benchmarks/kv_offload/benchmark_tiering.py list

Compare two session dirs:
    python benchmarks/kv_offload/benchmark_tiering.py compare main_dir pr_dir
"""

import argparse
import contextlib
import csv
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

MODEL = "openai/gpt-oss-120b"
TP_SIZE = 2
PORT = 8000

# gpu block size tokens
BLOCK_SIZE = 64
# size of a single GPU block; If not set via the env-var, this will be
# computed at runtime using a vllm dummy run.
# for gptoss120b, on H100, BLOCK_BYTES is 4.5MB.
BLOCK_BYTES = os.environ.get("BLOCK_BYTES")

BENCH_MAX_SECONDS = 420  # 7 minutes per guidellm run
BENCH_RANDOM_SEED = 889
BENCH_TURNS = 5
BENCH_PROMPT_TOKENS = 4096

# Phase tag -> guidellm output_tokens per turn.
PHASES = {"prefill": 1, "mixed": 256}

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


def gib(n: float) -> int:
    return int(n * 1024**3)


def gpu_block_bytes_from_engine(block_size: int = BLOCK_SIZE) -> int:
    """GPU KV bytes for one BLOCK_SIZE-token block, whole model.

    Ground truth: builds a TP=1 engine with dummy weights (uniproc executor,
    worker in-process) and reads the allocated cache tensors from the model
    runner directly. Needs a GPU. vLLM logging is disabled.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        load_format="dummy",
        max_model_len=8192,
        disable_log_stats=True,
        numa_bind=False,
        block_size=block_size,
    )
    runner = (
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    )
    total = sum(t.numel() * t.element_size() for t in runner.kv_caches)
    return total // runner.kv_cache_config.num_blocks


def _cell(v: object) -> str:
    """CSV cell: blank for missing, floats rounded to 2 decimals."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


@dataclass
class Capacity:
    """Tier capacity given as requests or bytes; materialize fills both."""

    requests: int | None = None
    bytes: int | None = None

    def __post_init__(self) -> None:
        assert (self.requests is None) != (self.bytes is None), (
            "exactly one of requests or bytes must be set"
        )

    def get_num_requests(self) -> int:
        assert self.requests is not None, "not materialized"
        return self.requests

    def get_num_bytes(self) -> int:
        assert self.bytes is not None, "not materialized"
        return self.bytes

    def materialize(self, per_request_bytes: float) -> None:
        if self.bytes is None:
            assert self.requests is not None
            self.bytes = int(self.requests * per_request_bytes)
        if self.requests is None:
            assert self.bytes is not None
            self.requests = int(self.bytes // per_request_bytes)

    def is_materialized(self) -> bool:
        return all([x is not None for x in [self.bytes, self.requests]])


@dataclass
class Scenario:
    """One benchmark configuration; add entries to SCENARIOS to extend."""

    name: str
    desc: str
    concurrency: int
    offload_block_size: int  # KV block_size (tokens) written to the FS tier
    prefix_tokens: int  # shared prefix length (tokens) driven into the FS tier
    gpu: Capacity  # total GPU KV across all TP ranks
    cpu: Capacity  # whole-model CPU tier budget
    isl_tokens: int = BENCH_PROMPT_TOKENS  # per-turn prompt tokens
    phase: str = "prefill"  # PHASES key
    _block_bytes: float | None = None

    def per_request_bytes(self, block_bytes: float) -> float:
        """Last-turn KV footprint of a single request."""
        osl = PHASES[self.phase]
        turns = BENCH_TURNS - 1
        tokens = self.prefix_tokens + turns * self.isl_tokens + turns * osl
        return tokens / BLOCK_SIZE * block_bytes

    def materialize(self, block_bytes: float) -> None:
        """Resolve each tier's bytes/requests; all four values set after."""
        self._block_bytes = block_bytes
        per_request = self.per_request_bytes(block_bytes)
        self.gpu.materialize(per_request)
        self.cpu.materialize(per_request)

    def kv_size_roofline(self) -> int:
        """Worst-case KV bytes: last-turn footprint per request x concurrency."""
        assert self._block_bytes is not None
        return int(self.concurrency * self.per_request_bytes(self._block_bytes))

    def describe(self) -> None:
        """Print what the scenario holds; unmaterialized fields show as '?'."""
        print(self.describe_str(), end="")

    def describe_str(self) -> str:
        """describe() as a string; unmaterialized fields show as '?'."""

        def tier_str(t: Capacity) -> str:
            bytes_s = f"{t.bytes / 1024**3:.1f}G" if t.bytes is not None else "?"
            reqs_s = t.requests if t.requests is not None else "?"
            return f"{bytes_s} ({reqs_s} requests)"

        s = f"{self.name}: {self.desc}\n"
        s += (
            f"  concurrency={self.concurrency} phase={self.phase} "
            f"offload_block_size={self.offload_block_size}\n"
        )
        s += f"  prefix_tokens={self.prefix_tokens} isl_tokens={self.isl_tokens}\n"
        s += f"  gpu={tier_str(self.gpu)}\n"
        s += f"  cpu={tier_str(self.cpu)}\n"
        if self.gpu.requests is not None:
            assert self.cpu.requests is not None
            fits = self.gpu.requests + self.cpu.requests
            over = "OVER" if self.concurrency > fits else "under"
            s += (
                f" requests: need {self.concurrency} vs fits {fits} "
                f" (gpu {self.gpu.requests}, cpu {self.cpu.requests})"
                f"  ({over})\n"
            )
        if self._block_bytes is not None:
            assert self.gpu.bytes is not None
            assert self.cpu.bytes is not None
            roofline = self.kv_size_roofline()
            budget = self.gpu.bytes + self.cpu.bytes
            over = "OVER" if roofline > budget else "under"
            s += (
                f"  kv roofline={roofline / 1024**3:.1f}G"
                f" vs budget={budget / 1024**3:.1f}G ({over})"
            )
        return s


QUICK = [
    Scenario(
        "fit-c16",
        "cpu kv cache barely used: fits in gpu",
        concurrency=16,
        offload_block_size=64,
        prefix_tokens=20_000,
        gpu=Capacity(requests=17),
        cpu=Capacity(requests=36),
    ),
    Scenario(
        "tight-c32",
        "cpu kv cache heavily used, never overflows",
        concurrency=32,
        offload_block_size=64,
        prefix_tokens=20_000,
        gpu=Capacity(requests=17),
        cpu=Capacity(requests=36),
    ),
    Scenario(
        "overflow-c64",
        "cpu kv cache overflows, triggering promotion allocation errors",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=20_000,
        gpu=Capacity(requests=17),
        cpu=Capacity(requests=36),
    ),
    Scenario(
        "overflow-c64-large-files",
        "effect of promoting/storing large (1024-token) vs small blocks",
        concurrency=64,
        offload_block_size=1024,
        prefix_tokens=20_000,
        gpu=Capacity(requests=17),
        cpu=Capacity(requests=36),
    ),
    Scenario(
        "overflow-c64-mixed",
        "cpu kv cache overflows, triggering promotion allocation errors;"
        " prefill + decode",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=20_000,
        gpu=Capacity(requests=17),
        cpu=Capacity(requests=36),
        phase="mixed",
    ),
    Scenario(
        "large-ctx-c16",
        "Large context prompts with barely enough space in GPU for a few"
        "requests in GPU; CPU can hold a bit more.",
        concurrency=16,
        offload_block_size=64,
        prefix_tokens=100_000,
        gpu=Capacity(requests=4),
        cpu=Capacity(requests=8),
    ),
]

SCENARIOS = QUICK + [
    Scenario(
        "staging-cpu",
        "cpu cache is just a staging area (small cpu, big gpu)",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=20_000,
        gpu=Capacity(requests=32),
        cpu=Capacity(requests=16),
    ),
    Scenario(
        "buffered-cpu",
        "cpu cache has slack beyond stores and promotions",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=20_000,
        gpu=Capacity(requests=32),
        cpu=Capacity(requests=100),
    ),
    Scenario(
        "large-ctx-large-cpu",
        "100k context with a large cpu tier. But not large enough to hold all requests",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=100_000,
        gpu=Capacity(requests=4),
        cpu=Capacity(requests=50),
    ),
    Scenario(
        "large-ctx-small-cpu",
        "100k context with a small cpu tier; (Triggers CPU overflow)",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=100_000,
        gpu=Capacity(requests=4),
        cpu=Capacity(requests=10),
    ),
    Scenario(
        "small-ctx-large-cpu",
        "5k context, 1k subsequent ISL, big cpu tier (No CPU overflow)",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=5_000,
        gpu=Capacity(requests=16),
        cpu=Capacity(requests=96),
        isl_tokens=1024,
    ),
    Scenario(
        "small-ctx-small-cpu",
        "5k context, 1k subsequent ISL, small cpu tier; (Triggers CPU overflow)",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=5_000,
        gpu=Capacity(requests=16),
        cpu=Capacity(requests=32),
        isl_tokens=1024,
    ),
]


class VLLMServer:
    """Lifecycle of a single `vllm serve` process."""

    def __init__(
        self, config_path: Path, log_path: Path, port: int, kv_cache_bytes: int
    ) -> None:
        self.config_path = config_path
        self.log_path = log_path
        self.port = port
        self.kv_cache_bytes = kv_cache_bytes
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = dict(os.environ, PYTHONHASHSEED="0")
        cmd = [
            "vllm",
            "serve",
            MODEL,
            "--tensor-parallel-size",
            str(TP_SIZE),
            "--kv-cache-memory-bytes",
            str(self.kv_cache_bytes),
            "--kv-transfer-config",
            self.config_path.read_text(),
            "--enable-prefix-caching",
            "--no-disable-hybrid-kv-cache-manager",
            "--port",
            str(self.port),
        ]
        with self.log_path.open("w") as log:
            self.proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env
            )

    def wait_ready(self, timeout: int = 1800) -> None:
        assert self.proc is not None
        print(f"      Waiting for vLLM on port {self.port}...")
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(self._crash_report("crashed during startup"))
            try:
                urlopen(url, timeout=5)
                print("      vLLM is ready!")
                return
            except OSError:
                time.sleep(5)
        raise RuntimeError(
            self._crash_report("timed out waiting to accept connections")
        )

    def stop(self) -> None:
        """SIGINT x5 (10s waits), then SIGKILL; exit loudly if unkillable."""
        assert self.proc is not None
        proc = self.proc
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

    def _crash_report(self, why: str) -> str:
        assert self.proc is not None
        tail = "\n".join(self.log_path.read_text().splitlines()[-20:])
        return (
            f"vLLM process (PID {self.proc.pid}) {why}.\n"
            f"Last 20 lines of {self.log_path}:\n{tail}"
        )

    def __enter__(self) -> "VLLMServer":
        self.start()
        try:
            self.wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class GuideLLM:
    """Drives the guidellm load against a running server and parses metrics."""

    def __init__(self, port: int, max_seconds: int) -> None:
        self.port = port
        self.max_seconds = max_seconds

    def run(self, spec: Scenario, log_path: Path) -> dict:
        prefix_count = max(4 * spec.concurrency, 256)
        data = {
            "kind": "synthetic_text",
            "prompt_tokens": spec.isl_tokens,
            "output_tokens": PHASES[spec.phase],
            "turns": BENCH_TURNS,
            "prefix_buckets": [
                {
                    "bucket_weight": 100,
                    "prefix_count": prefix_count,
                    "prefix_tokens": spec.prefix_tokens,
                }
            ],
        }
        cmd = [
            "guidellm",
            "run",
            "--backend",
            (
                "kind=openai_http,"
                f"target=http://127.0.0.1:{self.port},"
                "request_format=/v1/completions"
            ),
            "--profile",
            f"kind=concurrent,streams={spec.concurrency}",
            "--constraint",
            f"kind=max_duration,seconds={self.max_seconds}",
            "--seed",
            f"kind=static,value={BENCH_RANDOM_SEED}",
            "--data",
            json.dumps(data),
            "--output",
            f"kind=json,path={log_path.parent / 'guidellm.json'}",
        ]
        with log_path.open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        return self._parse_report(log_path.with_name("guidellm.json"))

    def _parse_report(self, json_path: Path) -> dict:
        """Extract aggregate metrics from the guidellm JSON report.

        guidellm also records per-request stats under `requests`; those are
        dropped after extraction to keep the report file small.
        """
        report = json.loads(json_path.read_text())
        benchmark = report["benchmarks"][0]
        metrics = benchmark["metrics"]

        def successful(field: str) -> dict:
            return metrics.get(field, {}).get("successful") or {}

        def percentile(field: str, p: str) -> float | None:
            return successful(field).get("percentiles", {}).get(p)

        result = {
            "request_throughput": successful("requests_per_second").get("mean"),
            "request_latency_mean": successful("request_latency").get("mean"),
            "request_latency_p99": percentile("request_latency", "p99"),
            "output_token_throughput": successful("output_tokens_per_second").get(
                "mean"
            ),
            "total_token_throughput": successful("tokens_per_second").get("mean"),
            "ttft_mean": successful("time_to_first_token_ms").get("mean"),
            "ttft_p50": percentile("time_to_first_token_ms", "p50"),
            "ttft_p95": percentile("time_to_first_token_ms", "p95"),
            "ttft_p99": percentile("time_to_first_token_ms", "p99"),
        }
        benchmark.pop("requests", None)
        json_path.write_text(json.dumps(report, indent=2))
        return {k: v for k, v in result.items() if v is not None}


class BenchmarkRun:
    """One benchmark run: cleanup, config, server, load, teardown."""

    def __init__(
        self,
        spec: Scenario,
        fs_dir: Path,
        session_dir: Path,
        port: int,
        max_seconds: int,
    ) -> None:
        self.spec = spec
        self.fs_dir = fs_dir
        self.port = port
        self.max_seconds = max_seconds
        self.run_dir = session_dir / spec.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.guidellm = GuideLLM(port, max_seconds)

    def execute(self) -> dict:
        print("=" * 70)
        print(f"Run: {self.spec.name}")
        print(f"Logs: {self.run_dir}/")
        print("=" * 70)

        print("  [1/5] Cleanup...")
        self._cleanup()
        print("  [2/5] Writing KV transfer config...")
        config_path = self.run_dir / "kv_transfer_config.json"
        self._write_config(config_path)
        print("  [3/5] Starting vLLM server...")
        with VLLMServer(
            config_path,
            self.run_dir / "vllm.log",
            self.port,
            self.spec.gpu.get_num_bytes() // TP_SIZE,
        ):
            print("  [4/5] Running guidellm benchmark...")
            result = {
                "name": self.spec.name,
                "spec": asdict(self.spec),
                "status": "ok",
                **self.guidellm.run(self.spec, self.run_dir / "guidellm.log"),
            }
        print("  [5/5] Teardown: vLLM stopped.")
        time.sleep(5)
        return result

    def _cleanup(self) -> None:
        for shm in Path("/dev/shm").glob("vllm_offload*"):
            shm.unlink()
        for entry in self.fs_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def _write_config(self, path: Path) -> None:
        config = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": self.spec.cpu.get_num_bytes(),
                "eviction_policy": "lru",
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(self.fs_dir) + "/",
                    }
                ],
                "block_size": self.spec.offload_block_size,
            },
        }
        path.write_text(json.dumps(config, indent=2))


class BenchmarkSession:
    """Runs a list of specs and checkpoints results into a session dir."""

    def __init__(
        self, fs_dir: Path, specs: list[Scenario], port: int, max_seconds: int
    ) -> None:
        self.fs_dir = fs_dir
        self.specs = specs
        self.port = port
        self.max_seconds = max_seconds
        self.session_dir = Path(f"benchmark_session_{datetime.now():%Y%m%d_%H%M%S}")
        self.session_dir.mkdir()
        self.results: list[dict] = []

    def run(self) -> list[dict]:
        print(f"Session dir: {self.session_dir}/")
        print(
            f"{len(self.specs)} runs "
            f"(~{len(self.specs) * (self.max_seconds + 120) // 60} min total):"
        )
        for spec in self.specs:
            spec.describe()

        for i, spec in enumerate(self.specs, 1):
            print(f"\n[{i}/{len(self.specs)}]")
            result = BenchmarkRun(
                spec, self.fs_dir, self.session_dir, self.port, self.max_seconds
            ).execute()
            self.results.append(result)
            self._checkpoint()
            print(f"{spec.name}: {result['status']}")
            for m in METRICS:
                print(f"  {m}: {result.get(m, 'n/a')}")
        self._write_table()
        print(f"\nAll runs complete. Results in {self.session_dir}/")
        return self.results

    def _write_table(self) -> None:
        """Save scenarios x metrics as CSV."""
        csv_path = self.session_dir / "results.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario"] + METRICS)
            for r in self.results:
                writer.writerow([r["name"]] + [_cell(r.get(m)) for m in METRICS])
        print(f"Results table saved to {csv_path}")

    def _checkpoint(self) -> None:
        (self.session_dir / "results.json").write_text(
            json.dumps(self.results, indent=2)
        )


def compare(session_a: Path, session_b: Path) -> None:
    def load(d: Path) -> dict:
        results = json.loads((d / "results.json").read_text())
        return {r.get("name") or Scenario(**r["spec"]).name: r for r in results}

    a, b = load(session_a), load(session_b)
    print(f"{'run':<60} {'metric':<30} {session_a.name:>12} {session_b.name:>12}")
    for name in sorted(set(a) & set(b)):
        for metric in METRICS:
            va, vb = a[name].get(metric), b[name].get(metric)
            if va is not None and vb is not None:
                delta = (float(vb) - float(va)) / float(va) * 100
                print(f"{name:<60} {metric:<30} {va:>12} {vb:>12}  ({delta:+.1f}%)")


def preflight(fs_dir: Path, port: int) -> None:
    if shutil.which("guidellm") is None:
        raise RuntimeError("guidellm not found on PATH (pip install guidellm)")

    if shutil.which("vllm") is None:
        raise RuntimeError("vllm not found on PATH")
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        raise RuntimeError(
            f"Port {port} is already in use; stop the other server or pass --port."
        ) from None
    finally:
        probe.close()
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
    parser.add_argument(
        "--bench-seconds",
        type=int,
        default=BENCH_MAX_SECONDS,
        help="max duration per guidellm run in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help="port for the vLLM server (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser(
        "bench", help="run all scenarios, or just the QUICK set with --quick"
    )
    bench.add_argument(
        "fs_dir",
        type=Path,
        help="folder backing the tier store (wiped between runs)",
    )
    bench.add_argument(
        "--quick",
        action="store_true",
        help="run only the QUICK scenarios",
    )

    bench_scenario = subparsers.add_parser(
        "bench-scenario", help="run named scenarios against a vLLM server"
    )
    bench_scenario.add_argument(
        "fs_dir",
        type=Path,
        help="folder backing the tier store (wiped between runs)",
    )
    bench_scenario.add_argument(
        "--scenario",
        nargs="+",
        required=True,
        metavar="NAME",
        help="scenarios to run (see `list`)",
        choices=[sc.name for sc in SCENARIOS],
    )

    run = subparsers.add_parser("run", help="run one scenario built from explicit args")
    run.add_argument(
        "fs_dir",
        type=Path,
        help="folder backing the tier store (wiped between runs)",
    )
    run.add_argument(
        "--concurrency", type=int, required=True, help="concurrent requests"
    )
    run.add_argument(
        "--offload-block-size",
        type=int,
        required=True,
        help="KV block_size (tokens) written to the FS tier",
    )
    run.add_argument(
        "--prefix-tokens", type=int, required=True, help="shared prefix length"
    )
    run.add_argument("--cpu-gib", type=float, help="CPU tier budget in GiB")
    run.add_argument("--gpu-gib", type=float, help="GPU KV cache size in GiB")
    run.add_argument("--cpu-requests", type=int, help="CPU tier capacity in requests")
    run.add_argument("--gpu-requests", type=int, help="GPU tier capacity in requests")
    run.add_argument("--isl-tokens", type=int, help="per-turn prompt tokens")
    run.add_argument(
        "--phase", choices=sorted(PHASES), help="prefill or mixed (default: prefill)"
    )

    subparsers.add_parser("list", help="list scenarios and estimated footprints")

    compare_parser = subparsers.add_parser("compare", help="compare two session dirs")
    compare_parser.add_argument("session_a", type=Path, metavar="MAIN_DIR")
    compare_parser.add_argument("session_b", type=Path, metavar="PR_DIR")
    return parser


def _resolve_block_bytes() -> float:
    """BLOCK_BYTES from the environment, else calibrate with a dummy engine."""
    if BLOCK_BYTES is not None:
        return float(BLOCK_BYTES)
    print("BLOCK_BYTES not set; calibrating with a dummy engine...")
    return gpu_block_bytes_from_engine()


def _main() -> None:
    args = build_parser().parse_args()

    if args.command == "compare":
        compare(args.session_a, args.session_b)
        return

    by_name = {sc.name: sc for sc in SCENARIOS}
    if args.command == "list":
        for sc in SCENARIOS:
            sc.describe()
        return

    if args.command == "run":
        specs = [
            Scenario(
                "adhoc",
                "ad-hoc scenario",
                concurrency=args.concurrency,
                offload_block_size=args.offload_block_size,
                prefix_tokens=args.prefix_tokens,
                cpu=Capacity(args.cpu_requests, args.cpu_gib),
                gpu=Capacity(args.gpu_requests, args.gpu_gib),
                isl_tokens=args.isl_tokens or BENCH_PROMPT_TOKENS,
                phase=args.phase or "prefill",
            )
        ]
    elif args.command == "bench":
        specs = QUICK if args.quick else SCENARIOS
    else:
        assert args.command == "bench-scenario"
        specs = [by_name[n] for n in args.scenario]

    block_bytes = _resolve_block_bytes()
    print(f" ===== Using GPU block bytes: {block_bytes} =====")
    for spec in specs:
        spec.materialize(block_bytes)
        spec.describe()

    try:
        preflight(args.fs_dir, args.port)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(f"Preflight failed: {e}")

    BenchmarkSession(args.fs_dir, specs, args.port, args.bench_seconds).run()


if __name__ == "__main__":
    main()
