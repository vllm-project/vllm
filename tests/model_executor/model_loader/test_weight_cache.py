# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the IPC weight cache loader.

A cold start (no weight cache daemon, so the loader falls back to disk) and
warm restarts (weights mapped from the daemon via CUDA IPC) must both serve
identical outputs.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.model_executor.model_loader.weight_cache.protocol import get_socket_path
from vllm.platforms import current_platform

DAEMON_TIMEOUT_S = 600


class WeightCacheDaemon:
    """Context manager running the real weight cache daemon as a subprocess."""

    def __init__(self, model: str, tp_size: int, extra_args: list[str] | None = None):
        # Short base path: Unix socket paths are limited to ~107 characters.
        self.socket_dir = tempfile.mkdtemp(prefix="vllm_ipc_")
        self.tp_size = tp_size
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
            *(extra_args or []),
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
        # Each rank binds its socket only once the model is fully cached.
        # Poll for the socket files rather than a log line: model loading can
        # pull in JIT compilers that swap the process's stderr and swallow
        # everything logged afterwards, making log-based readiness flaky.
        expected = [
            get_socket_path(gpu_id, self.socket_dir) for gpu_id in range(self.tp_size)
        ]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Weight cache daemon exited with {self._proc.returncode}:\n"
                    f"{self._logs()}"
                )
            if all(os.path.exists(path) for path in expected):
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


@dataclass
class ModelCase:
    model: str
    prompts: list[str]
    images: list | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    daemon_args: list[str] = field(default_factory=list)


def generate(
    vllm_runner,
    case: ModelCase,
    socket_dir: str | None,
    fallback: bool,
):
    extra_config = (
        {} if socket_dir is None else {"socket_dir": socket_dir, "fallback": fallback}
    )
    with vllm_runner(
        case.model,
        load_format="auto" if socket_dir is None else "ipc_cache",
        model_loader_extra_config=extra_config,
        # Greedy outputs are compared across engine restarts; cap the batch
        # size so scheduling cannot change the results.
        max_num_seqs=1,
        **case.llm_kwargs,
    ) as llm:
        sampling_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        return llm.generate(case.prompts, sampling_params, images=case.images)


# Both hybrid-recurrent models need chunked prefill for their mamba-style
# cache mode.
QWEN_CASE = ModelCase(
    model="Qwen/Qwen3.5-0.8B",
    prompts=[
        "Hello, my name is",
        "The capital of France is",
    ],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_chunked_prefill=True,
    ),
)

# Kimi K3 adds the mxfp4-pack MoE path (post-load kernel setup runs in
# pre-processed mode) and a vision tower, exercised with an image request.
K3_CASE = ModelCase(
    model="tiny-random/kimi-k3",
    prompts=[
        "The capital of France is",
        "def quicksort(arr):",
        "<|kimi_image_placeholder|>What is shown in the image?",
    ],
    images=[None, None, ImageAsset("stop_sign").pil_image],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        # The image placeholder expands to >1k tokens for the stop_sign asset.
        max_model_len=4096,
    ),
    daemon_args=["--trust-remote-code"],
)


@pytest.mark.parametrize("case", [QWEN_CASE, K3_CASE], ids=["qwen3.5", "kimi-k3"])
def test_ipc_cache_cold_start_and_warm_restart(vllm_runner, case: ModelCase):
    """Cold start falls back to disk; warm restarts load weights via CUDA IPC.

    All runs must produce outputs identical to a default-loader baseline. The
    warm runs disable the disk fallback, so they only pass if the weights
    really came from the daemon.
    """
    if not current_platform.is_cuda_alike():
        pytest.skip("Weight cache IPC sharing requires CUDA or ROCm")
    if case is K3_CASE and not current_platform.is_device_capability_family(100):
        pytest.skip("Kimi K3 IPC weight cache requires an SM100 MXFP4 backend")

    # Baseline: plain disk loading with the default loader.
    baseline_outputs = generate(
        vllm_runner,
        case,
        None,
        fallback=True,
    )
    assert all(text for _, texts in baseline_outputs for text in texts)

    # Cold start: no daemon is serving, so the loader falls back to disk.
    with tempfile.TemporaryDirectory(prefix="vllm_ipc_empty_") as empty_socket_dir:
        cold_outputs = generate(vllm_runner, case, empty_socket_dir, fallback=True)

    with WeightCacheDaemon(case.model, tp_size=1, extra_args=case.daemon_args) as d:
        warm_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)
        # Warm restart: a second engine lifetime against the same daemon.
        restart_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)

    assert cold_outputs == baseline_outputs
    assert warm_outputs == baseline_outputs
    assert restart_outputs == baseline_outputs
