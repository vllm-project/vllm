# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for running KV tiering benchmark scenarios."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

if TYPE_CHECKING:
    from benchmark_tiering import Scenario


class KVCacheCalibration:
    """Calibrate the benchmark's per-attention-group block size."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        block_size: int,
        per_group_block_bytes: str | None,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.block_size = block_size
        self.per_group_block_bytes = per_group_block_bytes

    @staticmethod
    @contextlib.contextmanager
    def _redirect_process_output(log_file):
        """Redirect stdout/stderr descriptors for this process and children."""
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        try:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    @staticmethod
    @contextlib.contextmanager
    def _engine_core_in_process():
        """Keep the V1 engine core in-process for callable worker RPCs."""
        setting = "VLLM_ENABLE_V1_MULTIPROCESSING"
        previous_value = os.environ.get(setting)
        os.environ[setting] = "0"
        try:
            yield
        finally:
            if previous_value is None:
                os.environ.pop(setting, None)
            else:
                os.environ[setting] = previous_value

    @staticmethod
    def _worker_kv_group_info(worker) -> tuple[tuple[str, int], ...]:
        """Return each KV group's kind and rank-local page size."""
        from vllm.v1.core.single_type_kv_cache_manager import get_kv_cache_kind
        from vllm.v1.kv_cache_interface import AttentionSpec

        config = worker.model_runner.kv_cache_config
        group_info = []
        for group in config.kv_cache_groups:
            assert all(
                isinstance(spec, AttentionSpec)
                for spec in group.kv_cache_specs.values()
            ), f"Unexpected spec type in group: {group.manager_config}"
            kind = get_kv_cache_kind(
                group.manager_config, group.kv_cache_specs.values()
            )
            group_info.append((kind.value, group.page_size_bytes))
        return tuple(group_info)

    def from_engine(self) -> int:
        """Infer the whole-model per-group block size from a dummy engine."""
        log = Path("block_bytes_calibration.log")
        print(f"Calibration log: {log}")
        with (
            log.open("w") as f,
            self._redirect_process_output(f),
            self._engine_core_in_process(),
        ):
            from vllm import LLM

            llm = LLM(
                model=self.model,
                tensor_parallel_size=self.tensor_parallel_size,
                load_format="dummy",
                max_model_len=8192,
                disable_log_stats=True,
                numa_bind=False,
                enforce_eager=True,
                block_size=self.block_size,
            )
            rank_group_info = llm.collective_rpc(
                self._worker_kv_group_info, timeout=120
            )
            assert len(rank_group_info) == self.tensor_parallel_size
            first_rank_group_info = rank_group_info[0]
            assert all(
                group_info == first_rank_group_info for group_info in rank_group_info
            ), f"TP ranks disagree on KV groups: {rank_group_info}"
            assert len(first_rank_group_info) == 2, "gptoss has FA and SWA"
            assert {kind for kind, _ in first_rank_group_info} == {
                "full_attention",
                "sliding_window",
            }, f"expected only full attention and SWA groups: {first_rank_group_info}"
            rank_local_page_sizes = {
                page_size for _, page_size in first_rank_group_info
            }
            assert len(rank_local_page_sizes) == 1, (
                f"attention groups disagree on block size: {first_rank_group_info}"
            )
            del llm
        return sum(group_info[0][1] for group_info in rank_group_info)

    def resolve(self) -> float:
        """Return the environment override, or calibrate a new value."""
        if self.per_group_block_bytes is not None:
            return float(self.per_group_block_bytes)
        print("PER_GROUP_BLOCK_BYTES not set; calibrating with a dummy engine...")
        return self.from_engine()


class VLLMServer:
    """Lifecycle of a single `vllm serve` process."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        kv_cache_bytes: int,
        config_path: Path | None,
        port: int,
        log_path: Path,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.kv_cache_bytes = kv_cache_bytes
        self.config_path = config_path
        self.port = port
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._previous_signal_handlers: dict[int, object] = {}

    def _handle_termination(self, signum: int, _frame: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signum)
        raise SystemExit(128 + signum)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_signal_handlers.items():
            signal.signal(signum, handler)
        self._previous_signal_handlers = {}

    def start(self) -> None:
        env = dict(os.environ, PYTHONHASHSEED="0")
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--kv-cache-memory-bytes",
            str(self.kv_cache_bytes),
        ]
        if self.config_path is not None:
            cmd += ["--kv-transfer-config", self.config_path.read_text()]
        cmd += [
            "--enable-prefix-caching",
            "--no-disable-hybrid-kv-cache-manager",
            "--port",
            str(self.port),
        ]
        try:
            self._previous_signal_handlers = {
                signal.SIGINT: signal.getsignal(signal.SIGINT),
                signal.SIGTERM: signal.getsignal(signal.SIGTERM),
            }
            signal.signal(signal.SIGINT, self._handle_termination)
            signal.signal(signal.SIGTERM, self._handle_termination)
            with self.log_path.open("w") as log:
                self.proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env
                )
        except BaseException:
            self._restore_signal_handlers()
            raise

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
        self._restore_signal_handlers()
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

    def assert_running(self) -> None:
        """Raise if the server exited before expected."""
        assert self.proc is not None
        return_code = self.proc.poll()
        if return_code is not None:
            raise RuntimeError(
                self._crash_report(f"Exited before expected: return code {return_code}")
            )

    def _crash_report(self, why: str) -> str:
        assert self.proc is not None
        tail = "\n".join(self.log_path.read_text().splitlines()[-20:])
        return (
            f"vLLM process (PID {self.proc.pid}) {why}.\n"
            f"Last 20 lines of {self.log_path}:\n{tail}"
        )

    def __enter__(self) -> VLLMServer:
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

    def __init__(
        self,
        port: int,
        max_seconds: int,
        phases: dict[str, int],
        turns: int,
        random_seed: int,
    ) -> None:
        self.port = port
        self.max_seconds = max_seconds
        self.phases = phases
        self.turns = turns
        self.random_seed = random_seed

    def run(self, spec: Scenario, log_path: Path) -> dict:
        prefix_count = max(4 * spec.concurrency, 256)
        data = {
            "kind": "synthetic_text",
            "prompt_tokens": spec.isl_tokens,
            "output_tokens": self.phases[spec.phase],
            "turns": self.turns,
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
            f"kind=static,value={self.random_seed}",
            "--data",
            json.dumps(data),
            "--output",
            f"kind=json,path={log_path.parent / 'guidellm.json'}",
        ]
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                return_code = proc.wait()
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
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
