# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the KV tiering subsystem with guidellm.

Starts a vLLM server, drives it with `guidellm`, and records per-run
metrics. The tier configuration is set with `--mode`:

- `gpu`: GPU KV only; no kv-transfer-config is passed to vLLM.
- `cpu`: GPU + CPU tiers; kv-transfer-config without secondary tiers.
- `fs`: GPU + CPU + FS tiers (requires `--fs-dir`).

Run the quick scenarios against the FS tier:
    python benchmarks/kv_offload/benchmark_tiering.py --mode fs \
        --fs-dir /mnt/nvme --fs-read-threads 16 --fs-write-threads 16 bench

Run all scenarios against the FS tier:
    python benchmarks/kv_offload/benchmark_tiering.py --mode fs \
        --fs-dir /mnt/nvme bench --full

Run selected scenarios CPU-only:
    python benchmarks/kv_offload/benchmark_tiering.py --mode cpu \
        bench-scenario --scenario fit-c16 overflow-c64

Run one ad-hoc scenario:
    python benchmarks/kv_offload/benchmark_tiering.py --mode fs \
        --fs-dir /mnt/nvme run --concurrency 16 --offload-block-size 1024 \
        --prefix-tokens 20000

List scenarios:
    python benchmarks/kv_offload/benchmark_tiering.py list

Compare two session dirs:
    python benchmarks/kv_offload/benchmark_tiering.py compare main_dir pr_dir
"""

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from utils import GuideLLM, KVCacheCalibration, VLLMServer

"""
Parts of the benchmark are built for the gpt-oss-120b model.
 1. gpt-oss-120b has 2 attention groups FA and SWA.
 2. These attention groups have the same gpu block size (2.3M on H100).
This fact is exploited in `KVCacheCalibration.from_engine` and
`Scenario::max_per_request_block_size` functions that are used,
 - to determine the maximum KV cache memory that a benchmark run would
   require and subsequently,
 - to design GPU/CPU cache overflow scenarios.

 Updating to a different model, requires updating these functions.
"""
MODEL = "openai/gpt-oss-120b"
TP_SIZE = 2
PORT = 8000

# gpu block size tokens
BLOCK_SIZE = 64
# size of a single attention group in bytes.
# gptoss-120b has a FA group and a SWA group. Both groups have the same
# size. This lets us use a single PER_GROUP_BLOCK_BYTES value for both
# groups in the benchmark.
# If unset via an environment variable, this is computed once at the
# start of the benchmark via `KVCacheCalibration.from_engine`.
# Notes:
# - This value is 2.3M on H100.
# - With FS offloading, this value should match the offloaded file size.
PER_GROUP_BLOCK_BYTES = os.environ.get("PER_GROUP_BLOCK_BYTES")

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

    def materialize(self, max_per_request_bytes: float) -> None:
        if self.bytes is None:
            assert self.requests is not None
            self.bytes = int(self.requests * max_per_request_bytes)
        if self.requests is None:
            assert self.bytes is not None
            self.requests = int(self.bytes // max_per_request_bytes)

    def is_materialized(self) -> bool:
        return all([x is not None for x in [self.bytes, self.requests]])


@dataclass(frozen=True)
class FsOptions:
    root_dir: Path
    n_read_threads: int | None
    n_write_threads: int | None

    def tier_config(self) -> dict[str, str | int]:
        config: dict[str, str | int] = {
            "type": "fs",
            "root_dir": str(self.root_dir) + "/",
        }
        if self.n_read_threads is not None:
            config["n_read_threads"] = self.n_read_threads
        if self.n_write_threads is not None:
            config["n_write_threads"] = self.n_write_threads
        return config


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
    _per_group_block_bytes: float | None = None

    def max_per_request_bytes(self, per_group_block_bytes: float) -> float:
        """Worst-case KV bytes resident for one request at its last turn.

        For gpt-oss-120b,
        FA group: full token history at BLOCK_SIZE tokens per block.
        SWA group: bounded by the 128-token sliding window (2 blocks).
        """
        osl = PHASES[self.phase]
        fa_tokens = (
            self.prefix_tokens + BENCH_TURNS * self.isl_tokens + (BENCH_TURNS - 1) * osl
        )
        fa_blocks = fa_tokens / BLOCK_SIZE
        swa_blocks = 128 / BLOCK_SIZE  # gptoss has a sliding window of 128 tokens.
        return (fa_blocks + swa_blocks) * per_group_block_bytes

    def materialize(self, per_group_block_bytes: float) -> None:
        """Resolve each tier's bytes/requests"""
        self._per_group_block_bytes = per_group_block_bytes
        max_per_request = self.max_per_request_bytes(per_group_block_bytes)
        self.gpu.materialize(max_per_request)
        self.cpu.materialize(max_per_request)

    def kv_size_roofline(self) -> int:
        """Worst-case KV bytes: last-turn footprint per request x concurrency."""
        assert self._per_group_block_bytes is not None
        return int(
            self.concurrency * self.max_per_request_bytes(self._per_group_block_bytes)
        )

    def describe(self) -> None:
        def tier_str(t: Capacity) -> str:
            bytes_s = f"{t.bytes / 1024**3:.1f}G" if t.bytes is not None else "?"
            reqs_s = t.requests if t.requests is not None else "?"
            return f"{bytes_s} ({reqs_s} requests)"

        s = f"==== {self.name} == \n"
        s += f" description: {self.desc}\n"
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
            over = "OVER" if self.concurrency > fits else "Under"
            s += (
                f"  requests: need {self.concurrency} vs fits {fits} "
                f" (gpu {self.gpu.requests}, cpu {self.cpu.requests})"
                f"  ({over})\n"
            )
        if self._per_group_block_bytes is not None:
            assert self.gpu.bytes is not None
            assert self.cpu.bytes is not None
            roofline = self.kv_size_roofline()
            budget = self.gpu.bytes + self.cpu.bytes
            over = "OVER" if roofline > budget else "Under"
            s += (
                f"  kv roofline={roofline / 1024**3:.1f}G"
                f" vs budget={budget / 1024**3:.1f}G ({over})\n"
            )

        print(s)


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
    Scenario(
        "staging-cpu",
        "cpu cache is just a staging area (small cpu, big gpu)",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=20_000,
        gpu=Capacity(requests=32),
        cpu=Capacity(requests=16),
    ),
]

SCENARIOS = QUICK + [
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
        gpu=Capacity(requests=24),
        cpu=Capacity(requests=96),
        isl_tokens=1024,
    ),
    Scenario(
        "small-ctx-small-cpu",
        "5k context, 1k subsequent ISL, small cpu tier; (Triggers CPU overflow)",
        concurrency=64,
        offload_block_size=64,
        prefix_tokens=5_000,
        gpu=Capacity(requests=24),
        cpu=Capacity(requests=32),
        isl_tokens=1024,
    ),
]


class BenchmarkRun:
    """One benchmark run: cleanup, config, server, load, teardown."""

    def __init__(
        self,
        spec: Scenario,
        mode: str,
        session_dir: Path,
        port: int,
        max_seconds: int,
        fs_options: FsOptions | None = None,
    ) -> None:
        self.fs_options = fs_options
        self.spec = spec
        self.port = port
        self.max_seconds = max_seconds
        self.mode = mode
        self.shm_engine_id = f"benchmark_{os.getpid()}"
        self.run_dir = session_dir / spec.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.guidellm = GuideLLM(
            port, max_seconds, PHASES, BENCH_TURNS, BENCH_RANDOM_SEED
        )

    def execute(self) -> dict:
        print("=" * 70)
        print(f"Run: {self.spec.name}")
        print(f"Logs: {self.run_dir}/")
        print("=" * 70)

        print("  [1/5] Cleanup...")
        self._cleanup()
        print("  [2/5] Writing KV transfer config...")
        config_path = self._write_config(self.run_dir / "kv_transfer_config.json")
        print("  [3/5] Starting vLLM server...")
        with VLLMServer(
            MODEL,
            TP_SIZE,
            self.spec.gpu.get_num_bytes() // TP_SIZE,
            config_path,
            self.port,
            self.run_dir / "vllm.log",
            shm_engine_id=self.shm_engine_id if config_path is not None else None,
        ) as server:
            print("  [4/5] Running guidellm benchmark...")
            guidellm_result = self.guidellm.run(
                self.spec, self.run_dir / "guidellm.log"
            )
            server.assert_running()
            result = {
                "name": self.spec.name,
                "spec": asdict(self.spec),
                "status": "ok",
                **guidellm_result,
            }
        return result

    def _cleanup(self) -> None:
        if self.mode == "fs":
            assert self.fs_options is not None
            root_dir = self.fs_options.root_dir
            root_dir.mkdir(parents=True, exist_ok=True)
            for entry in root_dir.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

    def _write_config(self, path: Path) -> Path | None:
        """Write the KV transfer config; returns None in gpu mode."""
        if self.mode == "gpu":
            print("  GPU mode: skipping KV transfer config.")
            return None
        extra = {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": self.spec.cpu.get_num_bytes(),
            "eviction_policy": "lru",
            "block_size": self.spec.offload_block_size,
        }
        if self.mode == "fs":
            assert self.fs_options is not None
            extra["secondary_tiers"] = [self.fs_options.tier_config()]
        config = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "engine_id": self.shm_engine_id,
            "kv_connector_extra_config": extra,
        }
        path.write_text(json.dumps(config, indent=2))
        return path


class BenchmarkSession:
    """Runs a list of specs and checkpoints results into a session dir."""

    def __init__(
        self,
        mode: str,
        specs: list[Scenario],
        port: int,
        max_seconds: int,
        fs_options: FsOptions | None = None,
    ) -> None:
        self.fs_options = fs_options
        self.specs = specs
        self.port = port
        self.max_seconds = max_seconds
        self.mode = mode
        self.session_dir = Path(f"benchmark_session_{datetime.now():%Y%m%d_%H%M%S}")
        self.session_dir.mkdir()
        self.results: list[dict] = []

    def run(self) -> list[dict]:
        print(f"Session dir: {self.session_dir}/")
        max_gpu = max(spec.gpu.get_num_bytes() for spec in self.specs)
        max_cpu = max(spec.cpu.get_num_bytes() for spec in self.specs)
        print(
            f"{len(self.specs)} runs : "
            f"~{len(self.specs) * (self.max_seconds + 120) // 60} min total ; "
            f" max gpu memory {max_gpu / 1024**3:.1f} G ; "
            f" max cpu memory {max_cpu / 1024**3:.1f} G ; "
        )

        for i, spec in enumerate(self.specs, 1):
            spec.describe()
            print(f"\n[{i}/{len(self.specs)}]")
            result = BenchmarkRun(
                spec,
                self.mode,
                self.session_dir,
                self.port,
                self.max_seconds,
                self.fs_options,
            ).execute()
            self.results.append(result)
            self._checkpoint()
            print(f"{spec.name}: {result['status']}")
            for m in METRICS:
                print(f"  {m}: {result.get(m, 'n/a')}")
        print(f"\nAll runs complete. Results in {self.session_dir}/")
        return self.results

    def _write_table(self) -> None:
        """Atomically save scenarios x metrics as CSV."""
        csv_path = self.session_dir / "results.csv"
        tmp_path = csv_path.with_name(csv_path.name + ".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario"] + METRICS)
            for r in self.results:
                writer.writerow([r["name"]] + [_cell(r.get(m)) for m in METRICS])
        tmp_path.replace(csv_path)

    def _checkpoint(self) -> None:
        (self.session_dir / "results.json").write_text(
            json.dumps({"mode": self.mode, "results": self.results}, indent=2)
        )
        self._write_table()


def compare(session_a: Path, session_b: Path) -> None:
    def load(d: Path) -> dict:
        data = json.loads((d / "results.json").read_text())
        results = data["results"] if isinstance(data, dict) else data
        return {r.get("name") or Scenario(**r["spec"]).name: r for r in results}

    a, b = load(session_a), load(session_b)
    print(f"{'run':<60} {'metric':<30} {session_a.name:>12} {session_b.name:>12}")
    for name in sorted(set(a) & set(b)):
        for metric in METRICS:
            va, vb = a[name].get(metric), b[name].get(metric)
            if va is not None and vb is not None:
                delta = (float(vb) - float(va)) / float(va) * 100
                print(f"{name:<60} {metric:<30} {va:>12} {vb:>12}  ({delta:+.1f}%)")


def preflight(port: int) -> None:
    if shutil.which("guidellm") is None:
        raise RuntimeError("guidellm not found on PATH (pip install guidellm)")

    if shutil.which("vllm") is None:
        raise RuntimeError("vllm not found on PATH")
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        raise RuntimeError(
            f"Port {port} is already in use; stop the other server or pass --port."
        ) from None
    finally:
        probe.close()


def main() -> None:
    try:
        _main()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(f"ERROR: {e}")


def add_fs_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Filesystem tier options")
    group.add_argument("--fs-dir", type=Path, help="folder backing the FS tier")
    group.add_argument(
        "--fs-read-threads",
        type=int,
        default=16,
        help="number of read-priority I/O threads",
    )
    group.add_argument(
        "--fs-write-threads",
        type=int,
        default=16,
        help="number of write-priority I/O threads",
    )


def parse_fs_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> FsOptions | None:
    if args.mode != "fs":
        return None
    if args.fs_dir is None:
        parser.error("--mode fs requires --fs-dir")
    return FsOptions(args.fs_dir, args.fs_read_threads, args.fs_write_threads)


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
    parser.add_argument(
        "--mode",
        choices=["gpu", "cpu", "fs"],
        default="fs",
        help="tier configuration (default: %(default)s)",
    )
    add_fs_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser(
        "bench", help="run the QUICK scenarios, or all scenarios with --full"
    )
    bench.add_argument(
        "--full",
        action="store_true",
        help="run all scenarios instead of the QUICK scenarios",
    )

    bench_scenario = subparsers.add_parser(
        "bench-scenario", help="run named scenarios against a vLLM server"
    )
    bench_scenario.add_argument(
        "--scenario",
        nargs="+",
        required=True,
        metavar="NAME",
        help="scenarios to run (see `list`)",
        choices=[sc.name for sc in SCENARIOS],
    )

    subparsers.add_parser(
        "calibrate",
        help="print the calibrated per-attention-group block size",
    )

    run = subparsers.add_parser("run", help="run one scenario built from explicit args")
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

    subparsers.add_parser("list", help="list scenarios")

    compare_parser = subparsers.add_parser("compare", help="compare two session dirs")
    compare_parser.add_argument("session_a", type=Path, metavar="MAIN_DIR")
    compare_parser.add_argument("session_b", type=Path, metavar="PR_DIR")
    return parser


def _main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    by_name = {sc.name: sc for sc in SCENARIOS}
    if args.command == "list":
        for sc in SCENARIOS:
            sc.describe()
        return

    if args.command == "compare":
        compare(args.session_a, args.session_b)
        return

    if args.command == "calibrate":
        per_group_block_bytes = KVCacheCalibration(
            MODEL, TP_SIZE, BLOCK_SIZE, PER_GROUP_BLOCK_BYTES
        ).from_engine()
        print(f"PER_GROUP_BLOCK_BYTES={per_group_block_bytes}")
        return

    fs_options = parse_fs_options(parser, args)

    if args.command == "run":
        specs = [
            Scenario(
                "adhoc",
                "ad-hoc scenario",
                concurrency=args.concurrency,
                offload_block_size=args.offload_block_size,
                prefix_tokens=args.prefix_tokens,
                cpu=Capacity(
                    args.cpu_requests,
                    gib(args.cpu_gib) if args.cpu_gib is not None else None,
                ),
                gpu=Capacity(
                    args.gpu_requests,
                    gib(args.gpu_gib) if args.gpu_gib is not None else None,
                ),
                isl_tokens=args.isl_tokens or BENCH_PROMPT_TOKENS,
                phase=args.phase or "prefill",
            )
        ]
    elif args.command == "bench":
        specs = SCENARIOS if args.full else QUICK
    else:
        assert args.command == "bench-scenario"
        specs = [by_name[n] for n in args.scenario]

    per_group_block_bytes = KVCacheCalibration(
        MODEL, TP_SIZE, BLOCK_SIZE, PER_GROUP_BLOCK_BYTES
    ).resolve()
    print(f" ===== Using GPU per-group block bytes: {per_group_block_bytes} =====")
    for spec in specs:
        spec.materialize(per_group_block_bytes)
        spec.describe()

    try:
        preflight(args.port)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(f"Preflight failed: {e}")

    BenchmarkSession(args.mode, specs, args.port, args.bench_seconds, fs_options).run()


if __name__ == "__main__":
    main()
