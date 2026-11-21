# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared in-process FakeLauncher (PlatformAdapter) for CPU unit/contract tests."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.executor.interfaces import (
    CacheResetUnsupportedError,
    DirectCommandResult,
    PlatformAdapter,
    ServerHandle,
    ServerTransportError,
)
from tools.byog.schemas.result import ProtocolVariant

_DIRECT_ENV_ALLOWLIST = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONPATH")


class FakeLauncher(PlatformAdapter):
    """In-process PlatformAdapter backed by `FakeVLLMServer`."""

    def __init__(
        self,
        *,
        allow_reset_cache: bool = True,
        fail_reset_transport: bool = False,
        fail_after_n: int | None = None,
        fail_on_calls: set[int] | None = None,
        wrong_answer_mode: bool = False,
        strict_model: bool = False,
        cleanup_ok: bool = True,
        vram_reclaimed: bool | None = True,
        fail_start_timeout: bool = False,
        hardware: dict[str, Any] | None = None,
    ):
        self.allow_reset_cache = allow_reset_cache
        self.fail_reset_transport = fail_reset_transport
        self.fail_after_n = fail_after_n
        self.fail_on_calls = set(fail_on_calls or set())
        self.wrong_answer_mode = wrong_answer_mode
        self.strict_model = strict_model
        self.cleanup_ok = cleanup_ok
        self.vram_reclaimed = vram_reclaimed
        self.fail_start_timeout = fail_start_timeout
        self._hardware = hardware or {
            "platform": "nvidia",
            "gpu_id": "h200",
            "gpu_name": "NVIDIA H200 (fake)",
            "gpu_count": 1,
            "driver_version": "570.00-fake",
            "gpu_memory_mib": [143771],
            "device_memory_used_mib": [0],
        }
        self.starts: int = 0
        self.stops: int = 0
        self.resets: int = 0
        self.direct_calls: int = 0
        self.last_enable_dev_mode: bool | None = None
        self.server_states: list[FakeServerState] = []
        self._active: dict[str, FakeVLLMServer] = {}

    @property
    def total_requests(self) -> int:
        return sum(s.total_inference_calls for s in self.server_states)

    def detect_hardware(self) -> dict[str, Any]:
        return dict(self._hardware)

    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
        *,
        enable_dev_mode: bool = False,
    ) -> ServerHandle:
        self.starts += 1
        self.last_enable_dev_mode = enable_dev_mode
        if self.fail_start_timeout:
            raise TimeoutError(f"Fake server did not become ready within {timeout_s}s.")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"fake_server_{self.starts}.log"
        log_path.write_text(
            f"FakeVLLMServer start #{self.starts} dev_mode={enable_dev_mode}\n",
            encoding="utf-8",
        )
        model_name = str(
            serve_config.get("served-model-name")
            or serve_config.get("served_model_name")
            or serve_config.get("model", "fake-model")
        )
        state = FakeServerState(
            model_name=model_name,
            strict_model=self.strict_model,
            allow_reset_cache=self.allow_reset_cache and enable_dev_mode,
            fail_after_n=self.fail_after_n,
            fail_on_calls=set(self.fail_on_calls),
            wrong_answer_mode=self.wrong_answer_mode,
        )
        self.server_states.append(state)
        srv = FakeVLLMServer(state=state)
        base_url = srv.start()
        self._active[base_url] = srv
        return ServerHandle(
            base_url=base_url,
            command=["fake-vllm-serve", "--model", model_name],
            log_path=log_path,
            pid=None,
            metadata={"fake": True, "enable_dev_mode": enable_dev_mode},
        )

    def reset_cache(self, handle: ServerHandle) -> ProtocolVariant:
        self.resets += 1
        if self.fail_reset_transport:
            raise ServerTransportError("Simulated connection reset during cache reset.")
        srv = self._active.get(handle.base_url)
        if srv is None or not srv.state.allow_reset_cache:
            raise CacheResetUnsupportedError(
                "Fake server rejected /reset_prefix_cache with HTTP 404."
            )
        srv.state.reset_calls += 1
        return ProtocolVariant.RESET_CACHE

    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> dict[str, Any]:
        self.stops += 1
        srv = self._active.pop(handle.base_url, None)
        if srv is not None:
            srv.stop()
        return {
            "terminated_pgid": self.cleanup_ok,
            "vram_reclaimed": self.vram_reclaimed,
            "baseline_memory_mib": baseline_memory_mib,
        }

    def run_direct(
        self,
        cmd: list[str],
        serve_env: dict[str, str],
        timeout_s: int,
    ) -> DirectCommandResult:
        self.direct_calls += 1
        env = {k: v for k, v in os.environ.items() if k in _DIRECT_ENV_ALLOWLIST}
        env.update(serve_env)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=max(1, timeout_s),
                check=False,
            )
            return DirectCommandResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_s=time.monotonic() - t0,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return DirectCommandResult(
                returncode=-9,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else "",
                stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else "",
                elapsed_s=time.monotonic() - t0,
                timed_out=True,
            )
