# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the IPC weight cache loader.

A cold start (no weight cache daemon, so the loader falls back to disk) and
warm restarts (weights mapped from the daemon via CUDA IPC) must both serve
identical outputs.
"""

import shutil
import subprocess
import sys
import tempfile
import threading
import time

from vllm import SamplingParams

MODEL = "Qwen/Qwen3.5-0.8B"
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]
DAEMON_TIMEOUT_S = 600


class WeightCacheDaemon:
    """Context manager running the real weight cache daemon as a subprocess."""

    _READY_MARKER = "Weight cache daemon READY"

    def __init__(self, model: str, tp_size: int):
        # Short base path: Unix socket paths are limited to ~107 characters.
        self.socket_dir = tempfile.mkdtemp(prefix="vllm_ipc_")
        self._cmd = [
            sys.executable,
            "-m",
            "vllm.model_executor.model_loader.weight_cache.daemon",
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp_size),
            "--weight-cache-socket-dir",
            self.socket_dir,
            "--enforce-eager",
        ]
        self._proc: subprocess.Popen | None = None
        self._lines: list[str] = []

    def __enter__(self) -> "WeightCacheDaemon":
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        try:
            self._wait_ready(DAEMON_TIMEOUT_S)
        except Exception:
            self._stop()
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop()

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._lines.append(line)

    def _logs(self) -> str:
        return "".join(self._lines)

    def _wait_ready(self, timeout_s: float) -> None:
        assert self._proc is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Weight cache daemon exited with {self._proc.returncode}:\n"
                    f"{self._logs()}"
                )
            if self._READY_MARKER in self._logs():
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"Weight cache daemon not ready after {timeout_s}s:\n{self._logs()}"
        )

    def _stop(self) -> None:
        assert self._proc is not None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        shutil.rmtree(self.socket_dir, ignore_errors=True)


def generate(vllm_runner, socket_dir: str | None, fallback: bool, **kwargs):
    extra_config = (
        {} if socket_dir is None else {"socket_dir": socket_dir, "fallback": fallback}
    )
    with vllm_runner(
        MODEL,
        load_format="auto" if socket_dir is None else "ipc_cache",
        model_loader_extra_config=extra_config,
        # Greedy outputs are compared across engine restarts; cap the batch
        # size so scheduling cannot change the results.
        max_num_seqs=1,
        **kwargs,
    ) as llm:
        sampling_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        return llm.generate(PROMPTS, sampling_params)


def test_ipc_cache_cold_start_and_warm_restart(vllm_runner):
    """Cold start falls back to disk; warm restarts load weights via CUDA IPC.

    All runs must produce outputs identical to a default-loader baseline. The
    warm runs disable the disk fallback, so they only pass if the weights
    really came from the daemon.
    """
    llm_kwargs = dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        # Qwen3.5 is a hybrid mamba model; its mamba cache mode requires
        # chunked prefill.
        enable_chunked_prefill=True,
    )

    # Baseline: plain disk loading with the default loader.
    baseline_outputs = generate(vllm_runner, None, fallback=True, **llm_kwargs)
    assert all(text for _, texts in baseline_outputs for text in texts)

    # Cold start: no daemon is serving, so the loader falls back to disk.
    with tempfile.TemporaryDirectory(prefix="vllm_ipc_empty_") as empty_socket_dir:
        cold_outputs = generate(
            vllm_runner, empty_socket_dir, fallback=True, **llm_kwargs
        )

    with WeightCacheDaemon(MODEL, tp_size=1) as daemon:
        warm_outputs = generate(
            vllm_runner, daemon.socket_dir, fallback=False, **llm_kwargs
        )
        # Warm restart: a second engine lifetime against the same daemon.
        restart_outputs = generate(
            vllm_runner, daemon.socket_dir, fallback=False, **llm_kwargs
        )

    assert cold_outputs == baseline_outputs
    assert warm_outputs == baseline_outputs
    assert restart_outputs == baseline_outputs
