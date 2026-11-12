# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import requests

from tests.utils import RemoteOpenAIServer

REPO_ROOT = Path(__file__).resolve().parents[4]

PROBE_TIMEOUT_S = 5.0
DETECT_DEADLINE_S = PROBE_TIMEOUT_S + 15.0


class ReadyProbeServer:
    """A vLLM server with fault injection hooks and readiness helpers."""

    def __init__(self, server: RemoteOpenAIServer, model: str):
        self.server = server
        self.model = model

    def _health_generate(self) -> requests.Response:
        return requests.get(
            self.server.url_for("health_generate"),
            params={"timeout_s": PROBE_TIMEOUT_S},
            timeout=30,
        )

    def status(self, path: str = "/health_generate") -> int | None:
        """Return the HTTP status, or None if the server is unreachable."""
        try:
            if path == "/health_generate":
                return self._health_generate().status_code
            return requests.get(
                self.server.url_for(path.lstrip("/")), timeout=30
            ).status_code
        except requests.RequestException:
            return None

    def ready_body(self) -> dict:
        resp = self._health_generate()
        return resp.json() if resp.status_code != 200 else {}

    def wait_until_unready(self) -> dict | None:
        """Return the first not-ready `/health_generate` body, or None.

        A passing idle probe is cached for a few seconds, so a fault may take
        that long to show.
        """
        start = time.monotonic()
        while time.monotonic() - start < DETECT_DEADLINE_S:
            with suppress(requests.RequestException):
                resp = self._health_generate()
                if resp.status_code != 200:
                    return resp.json()
            time.sleep(0.5)
        return None

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
        """Call a worker method through the dev `/collective_rpc` endpoint."""

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


@contextmanager
def ready_probe_server(model: str, extra_args: list[str]):
    env = {
        "VLLM_SERVER_DEV_MODE": "1",
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
        "--worker-extension-cls",
        "tests.v1.engine.readiness.fault_injection.FaultInjectionWorkerExtension",
        *extra_args,
    ]
    with RemoteOpenAIServer(model, args, env_dict=env) as server:
        yield ReadyProbeServer(server, model)
