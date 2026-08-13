#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for the CPU EC connector over NIXL.

Spins up two `vllm serve` instances on a single pod (producer + consumer)
and drives them via the disaggregated ``/inference/v1/generate`` endpoint
to assert that an image's encoder cache flows producer → consumer via
NIXL without re-encoding on the consumer.

Run from the repo root::

    python scripts/cpu_ec_connector/test_cpu_ec_e2e.py

Requires two CUDA devices visible to the host (defaults to GPUs 0/1). Each
instance can be widened with tensor parallelism by listing more devices::

    python scripts/cpu_ec_connector/test_cpu_ec_e2e.py \
        --producer-gpus 0,1,2,3 --consumer-gpus 4,5,6,7

The device count per instance is its --tensor-parallel-size, so that example
needs eight GPUs. At TP>1, ``--load-format fastsafetensors`` splits the
checkpoint read across ranks rather than having each rank read all of it.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shared import (
    ALL_TESTS,
    DEFAULT_IMAGE,
    DEFAULT_MODEL,
    HEALTH_TIMEOUT_S,
    TESTS_REQUIRING_CUSTOM_HARNESS,
    TESTS_REQUIRING_DEFAULT_HARNESS,
    LocalEventLog,
    ServerSpec,
    test_baseline,
    test_cache_reuse,
    test_concurrent_ec,
    test_multi_image,
    test_pool_exhaustion,
    test_producer_restart,
    wait_for_health,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHES_DIR = Path(__file__).resolve().parent  # contains sitecustomize.py

DEFAULT_PRODUCER_GPUS = "0"
DEFAULT_CONSUMER_GPUS = "1"
DEFAULT_PRODUCER_PORT = 8001
DEFAULT_CONSUMER_PORT = 8002
DEFAULT_PRODUCER_SIDE_PORT = 5601
DEFAULT_CONSUMER_SIDE_PORT = 5602

REQUEST_TIMEOUT_S = 120

# Set to True by --debug; read by spawn_server so every spawned process
# (including restart_producer) picks it up without signature changes.
_DEBUG_LOGGING = False


# -----------------------------------------------------------------------------
# Server lifecycle
# -----------------------------------------------------------------------------


def build_vllm_argv(spec: ServerSpec, model: str) -> list[str]:
    """Argv for a single `vllm serve` invocation."""
    ec_role = "ec_producer" if spec.role == "producer" else "ec_consumer"
    ec_cfg = {
        "ec_connector": "ECCPUConnector",
        "ec_role": ec_role,
        "engine_id": spec.engine_id,
        "ec_enable_nixl": True,
        "ec_connector_extra_config": {
            "ec_enable_nixl": "True",
            "ec_cpu_bytes": spec.ec_cpu_bytes,
        },
    }
    argv = [
        "vllm",
        "serve",
        model,
        "--port",
        str(spec.http_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--enforce-eager",
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(spec.tp_size),
        "--ec-transfer-config",
        json.dumps(ec_cfg),
    ]
    if spec.load_format is not None:
        argv += ["--load-format", spec.load_format]
    if spec.role == "producer":
        argv.append("--no-enable-prefix-caching")
    return argv


def spawn_server(spec: ServerSpec, model: str) -> subprocess.Popen:
    """Spawn one vllm serve instance with sitecustomize patches pre-loaded.

    The patches must apply to every Python interpreter in the server's
    process tree (the vision encoder forward runs in vLLM's EngineCore
    subprocess, not in the API-server parent). Delivering them via a
    `sitecustomize.py` on PYTHONPATH gets every interpreter — parent,
    EngineCore, and any helper subprocess.
    """
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    event_path = spec.log_path.with_suffix(".events.jsonl")
    event_path.unlink(missing_ok=True)
    env = {
        **os.environ,
        "EC_TEST_EVENT_FILE": str(event_path),
        "CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in spec.device_list),
        "VLLM_EC_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_EC_SIDE_CHANNEL_PORT": str(spec.side_channel_port),
        "EC_TEST_ROLE": spec.role,
        "PYTHONPATH": str(PATCHES_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        # Gate for the /reset_mm_cache and /reset_prefix_cache routes.
        "VLLM_SERVER_DEV_MODE": "1",
    }
    if _DEBUG_LOGGING:
        env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    argv = build_vllm_argv(spec, model)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", *argv[1:]]
    log_fh = spec.log_path.open("wb", buffering=0)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        # Place each server in its own process group so the driver can SIGTERM
        # the whole tree (worker subprocesses included) cleanly on teardown.
        start_new_session=True,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def shutdown_server(proc: subprocess.Popen, name: str, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        print(
            f"[teardown] {name} did not exit on SIGTERM within "
            f"{grace_s:.0f}s; sending SIGKILL",
            file=sys.stderr,
        )
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    fh = getattr(proc, "_log_fh", None)
    if fh is not None:
        fh.close()


# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------


def _parse_devices(raw: str, flag: str) -> tuple[int, ...]:
    """Parse a comma-separated device list; its length is the TP size."""
    try:
        devices = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        # `from None`: the int() ValueError adds nothing over the message here,
        # and a traceback would bury a plain CLI usage error.
        raise SystemExit(
            f"{flag}: expected comma-separated integers, got {raw!r}"
        ) from None
    if not devices:
        raise SystemExit(f"{flag}: at least one device index is required")
    if len(set(devices)) != len(devices):
        raise SystemExit(f"{flag}: duplicate device indices in {raw!r}")
    return devices


def cleanup_dev_shm() -> None:
    """Wipe any stale `/dev/shm/vllm_ec_*.mmap` files."""
    stale = list(Path("/dev/shm").glob("vllm_ec_*.mmap"))
    for f in stale:
        try:
            f.unlink()
        except OSError as e:
            print(f"[setup] warning: could not remove {f}: {e}", file=sys.stderr)
    if stale:
        print(f"[setup] cleared {len(stale)} stale mmap file(s) from /dev/shm")


def make_specs(
    args,
    log_dir: Path,
    *,
    producer_ec_cpu_bytes: int = 4 * 1024**3,
    consumer_ec_cpu_bytes: int = 4 * 1024**3,
) -> tuple[ServerSpec, ServerSpec]:
    """Default ServerSpec pair: GPU 0 producer, GPU 1 consumer, distinct ports."""
    producer = ServerSpec(
        role="producer",
        gpu=args.producer_devices[0],
        devices=args.producer_devices,
        http_port=args.producer_port,
        side_channel_port=DEFAULT_PRODUCER_SIDE_PORT,
        engine_id="ec-producer-0",
        gpu_memory_utilization=0.01,
        log_path=log_dir / "producer.log",
        ec_cpu_bytes=producer_ec_cpu_bytes,
        load_format=args.load_format,
    )
    consumer = ServerSpec(
        role="consumer",
        gpu=args.consumer_devices[0],
        devices=args.consumer_devices,
        http_port=args.consumer_port,
        side_channel_port=DEFAULT_CONSUMER_SIDE_PORT,
        engine_id="ec-consumer-0",
        gpu_memory_utilization=0.5,
        log_path=log_dir / "consumer.log",
        ec_cpu_bytes=consumer_ec_cpu_bytes,
        load_format=args.load_format,
    )
    return producer, consumer


# -----------------------------------------------------------------------------
# LocalHarness
# -----------------------------------------------------------------------------


class LocalHarness:
    def __init__(
        self,
        producer: ServerSpec,
        consumer: ServerSpec,
        model: str,
        *,
        keep_on_exit: bool = False,
    ):
        self.producer = producer
        self.consumer = consumer
        self.model = model
        self.keep_on_exit = keep_on_exit
        self.producer_proc: subprocess.Popen | None = None
        self.consumer_proc: subprocess.Popen | None = None
        for spec in (producer, consumer):
            spec.base_url = f"http://127.0.0.1:{spec.http_port}"
            spec.events = LocalEventLog(spec.log_path.with_suffix(".events.jsonl"))

    def __enter__(self) -> LocalHarness:
        print(
            f"[setup] spawning producer (gpus={self.producer.device_list}, "
            f"tp={self.producer.tp_size}, port={self.producer.http_port}, "
            f"ec_cpu_bytes={self.producer.ec_cpu_bytes})"
        )
        self.producer_proc = spawn_server(self.producer, self.model)
        print(
            f"[setup] spawning consumer (gpus={self.consumer.device_list}, "
            f"tp={self.consumer.tp_size}, port={self.consumer.http_port}, "
            f"ec_cpu_bytes={self.consumer.ec_cpu_bytes})"
        )
        self.consumer_proc = spawn_server(self.consumer, self.model)
        self._wait_both_healthy()
        return self

    def __exit__(self, *_exc) -> None:
        if self.keep_on_exit:
            print("\n[teardown] --keep-servers set; leaving both servers running.")
            for proc, port in [
                (self.producer_proc, self.producer.http_port),
                (self.consumer_proc, self.consumer.http_port),
            ]:
                if proc is not None:
                    print(f"  pid={proc.pid}  port={port}")
            return
        print("\n[teardown] stopping servers")
        if self.consumer_proc is not None:
            shutdown_server(self.consumer_proc, "consumer")
        if self.producer_proc is not None:
            shutdown_server(self.producer_proc, "producer")

    def _wait_both_healthy(self) -> None:
        print(f"[setup] waiting on /health for both (up to {HEALTH_TIMEOUT_S}s)…")
        pairs = [
            (self.producer, self.producer_proc),
            (self.consumer, self.consumer_proc),
        ]
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {
                ex.submit(wait_for_health, s.base_url, p, HEALTH_TIMEOUT_S): (
                    s.role,
                    s.base_url,
                )
                for s, p in pairs
                if p is not None
            }
            for fut in as_completed(futs):
                fut.result()
                name, url = futs[fut]
                print(f"  ✓ {name} healthy at {url}")

    def restart_producer(self) -> None:
        assert self.producer_proc is not None
        print("\n[restart] SIGTERM producer")
        shutdown_server(self.producer_proc, "producer")
        # No manual mmap removal: the region is named by vllm_config.instance_id
        # (random per process), so the restarted producer opens a fresh file with
        # no collision, and ECSharedRegion.cleanup() unlinks its own mmap on the
        # graceful SIGTERM above.
        print("[restart] spawning new producer")
        self.producer_proc = spawn_server(self.producer, self.model)
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(
                wait_for_health,
                self.producer.base_url,
                self.producer_proc,
                HEALTH_TIMEOUT_S,
            ).result()
        print(f"  ✓ producer healthy on {self.producer.base_url}")


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument(
        "--producer-gpus",
        default=DEFAULT_PRODUCER_GPUS,
        help="comma-separated device indices for the producer; the count is "
        "its tensor-parallel size (e.g. '0,1,2,3' for TP=4)",
    )
    parser.add_argument(
        "--consumer-gpus",
        default=DEFAULT_CONSUMER_GPUS,
        help="comma-separated device indices for the consumer; the count is "
        "its tensor-parallel size (e.g. '4,5,6,7' for TP=4)",
    )
    parser.add_argument("--producer-port", type=int, default=DEFAULT_PRODUCER_PORT)
    parser.add_argument("--consumer-port", type=int, default=DEFAULT_CONSUMER_PORT)
    parser.add_argument(
        "--load-format",
        default=None,
        help="passed through to both instances as --load-format. Use "
        "'fastsafetensors' to have each TP rank read only its own subset of "
        "checkpoint files and redistribute over NCCL instead of every rank "
        "re-reading the whole checkpoint. Default leaves vLLM on 'auto'",
    )
    parser.add_argument(
        "--keep-servers",
        action="store_true",
        help="leave the shared-harness servers running on success",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="set VLLM_LOGGING_LEVEL=DEBUG on all spawned servers",
    )
    parser.add_argument(
        "--test",
        action="append",
        choices=("all", *ALL_TESTS),
        default=None,
        help="test(s) to run; pass multiple times or 'all'. default: all",
    )
    args = parser.parse_args()

    global _DEBUG_LOGGING
    _DEBUG_LOGGING = args.debug

    args.producer_devices = _parse_devices(args.producer_gpus, "--producer-gpus")
    args.consumer_devices = _parse_devices(args.consumer_gpus, "--consumer-gpus")
    if args.load_format == "fastsafetensors" and (
        importlib.util.find_spec("fastsafetensors") is None
    ):
        # Checked here rather than left to the servers: otherwise the failure
        # surfaces as a health-check timeout with the real cause buried in a log.
        print(
            "--load-format fastsafetensors needs the extra installed: "
            "pip install 'vllm[fastsafetensors]'",
            file=sys.stderr,
        )
        return 2

    shared_devices = set(args.producer_devices) & set(args.consumer_devices)
    if shared_devices:
        # Both instances reserve gpu_memory_utilization of every device they
        # see, so an overlap silently starves one of them rather than failing.
        print(
            f"producer and consumer must not share devices; both list "
            f"{sorted(shared_devices)}",
            file=sys.stderr,
        )
        return 2

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    selected = set(args.test or ["all"])
    if "all" in selected:
        selected = set(ALL_TESTS)
    normal_tests = selected & set(TESTS_REQUIRING_DEFAULT_HARNESS)
    custom_tests = selected & set(TESTS_REQUIRING_CUSTOM_HARNESS)

    log_dir = REPO_ROOT / "logs" / "cpu_ec_e2e" / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] log dir: {log_dir}")
    print(f"[setup] tests selected: {sorted(selected)}")

    failures: list[tuple[str, BaseException]] = []

    if normal_tests:
        cleanup_dev_shm()
        producer, consumer = make_specs(args, log_dir)
        with LocalHarness(
            producer, consumer, args.model, keep_on_exit=args.keep_servers
        ) as h:
            for test_name in TESTS_REQUIRING_DEFAULT_HARNESS:
                if test_name not in normal_tests:
                    continue
                try:
                    if test_name == "baseline":
                        test_baseline(h, args.image, args.prompt)
                    elif test_name == "cache-reuse":
                        test_cache_reuse(h, args.prompt)
                    elif test_name == "multi-image":
                        test_multi_image(h, args.prompt)
                    elif test_name == "concurrent":
                        test_concurrent_ec(h, args.prompt)
                except AssertionError as e:
                    failures.append((test_name, e))
                    print(f"  ✗ {test_name} FAILED: {e}", file=sys.stderr)

    for test_name in TESTS_REQUIRING_CUSTOM_HARNESS:
        if test_name not in custom_tests:
            continue
        try:
            if test_name == "pool-exhaustion":
                test_pool_exhaustion(
                    log_dir,
                    args.model,
                    args.prompt,
                    make_specs_fn=lambda ld, **kw: make_specs(args, ld, **kw),
                    make_harness=LocalHarness,
                    pre_harness=cleanup_dev_shm,
                )
            elif test_name == "producer-restart":
                test_producer_restart(
                    log_dir,
                    args.model,
                    args.prompt,
                    make_specs_fn=lambda ld, **kw: make_specs(args, ld, **kw),
                    make_harness=LocalHarness,
                    pre_harness=cleanup_dev_shm,
                )
        except AssertionError as e:
            failures.append((test_name, e))
            print(f"  ✗ {test_name} FAILED: {e}", file=sys.stderr)

    print(f"\n[teardown] logs preserved at {log_dir}")
    if failures:
        print(f"\n[done] {len(failures)} of {len(selected)} tests FAILED:")
        for test_name, e in failures:
            print(f"  - {test_name}: {e}")
        return 1
    print(f"\n[done] all {len(selected)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
