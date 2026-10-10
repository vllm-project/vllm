# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import requests

from tests.utils import RemoteOpenAIServer

from .fault_injection import FAULT_DIR_ENV

REPO_ROOT = Path(__file__).resolve().parents[4]

STALL_TIMEOUT_S = 5.0
PROBE_TIMEOUT_S = 5.0
DETECT_DEADLINE_S = STALL_TIMEOUT_S + PROBE_TIMEOUT_S + 15.0


class ReadyProbeServer:
    """A vLLM server with fault injection hooks and readiness helpers."""

    def __init__(self, server: RemoteOpenAIServer, model: str, fault_dir: Path):
        self.server = server
        self.model = model
        self.fault_dir = fault_dir

    def status(self, path: str = "/ready") -> int | None:
        """Return the HTTP status, or None if the server is unreachable."""
        try:
            return requests.get(
                self.server.url_for(path.lstrip("/")), timeout=30
            ).status_code
        except requests.RequestException:
            return None

    def ready_body(self) -> dict:
        resp = requests.get(self.server.url_for("ready"), timeout=30)
        return resp.json() if resp.status_code != 200 else {}

    def wait_until_unready(self) -> float | None:
        """Return seconds until ``/ready`` stops returning 200, or None."""
        start = time.monotonic()
        while time.monotonic() - start < DETECT_DEADLINE_S:
            if self.status() != 200:
                return time.monotonic() - start
            time.sleep(0.5)
        return None

    def wait_until_ready(self) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < DETECT_DEADLINE_S:
            if self.status() == 200:
                return True
            time.sleep(0.5)
        return False

    def stays_ready(self, duration_s: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < duration_s:
            if self.status() != 200:
                return False
            time.sleep(0.5)
        return True

    def generate(self, max_tokens: int = 8, dp_rank: int | None = None) -> int:
        headers = {} if dp_rank is None else {"X-data-parallel-rank": str(dp_rank)}
        resp = requests.post(
            self.server.url_for("v1/completions"),
            json={
                "model": self.model,
                "prompt": "Hello",
                "max_tokens": max_tokens,
                "ignore_eos": True,
            },
            headers=headers,
            timeout=600,
        )
        return resp.status_code

    def generate_in_background(self, max_tokens: int, dp_rank: int | None = None):
        threading.Thread(
            target=self._generate_quietly, args=(max_tokens, dp_rank), daemon=True
        ).start()

    @contextmanager
    def busy(self, dp_ranks: tuple[int | None, ...] = (None,)):
        """Keep the given DP ranks continuously busy with requests."""
        stop = threading.Event()

        def loop(dp_rank):
            while not stop.is_set():
                self._generate_quietly(64, dp_rank)

        threads = [
            threading.Thread(target=loop, args=(rank,), daemon=True)
            for rank in dp_ranks
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        try:
            yield
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=60)

    def _generate_quietly(self, max_tokens: int, dp_rank: int | None) -> None:
        with suppress(requests.RequestException):
            self.generate(max_tokens, dp_rank)

    def rpc(self, method: str, *args: str, background: bool = False) -> None:
        """Call a worker method through the dev ``/collective_rpc`` endpoint."""

        def call():
            with suppress(requests.RequestException):
                requests.post(
                    self.server.url_for("collective_rpc"),
                    json={"method": method, "args": list(args)},
                    timeout=600,
                )

        if background:
            threading.Thread(target=call, daemon=True).start()
        else:
            call()

    def set_fault(self, name: str) -> None:
        (self.fault_dir / name).touch()

    def clear_fault(self, name: str) -> None:
        (self.fault_dir / name).unlink(missing_ok=True)


@contextmanager
def ready_probe_server(model: str, extra_args: list[str], fault_dir: Path):
    fault_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_READY_STALL_TIMEOUT_S": str(STALL_TIMEOUT_S),
        "VLLM_HEALTH_CHECK_GPU_TIMEOUT": str(PROBE_TIMEOUT_S),
        "VLLM_READY_IDLE_PROBE_CACHE_TTL_S": "0",
        FAULT_DIR_ENV: str(fault_dir),
        # The engine processes import the fault injection classes by name.
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p
        ),
    }
    args = [
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "512",
        "--kernel-config",
        '{"enable_flashinfer_autotune": false}',
        # A module-scoped server and a per-test server can share a GPU.
        "--gpu-memory-utilization",
        "0.2",
        "--scheduler-cls",
        "tests.v1.engine.readiness.fault_injection.FaultInjectionScheduler",
        "--worker-extension-cls",
        "tests.v1.engine.readiness.fault_injection.FaultInjectionWorkerExtension",
        *extra_args,
    ]
    with RemoteOpenAIServer(model, args, env_dict=env) as server:
        yield ReadyProbeServer(server, model, fault_dir)
