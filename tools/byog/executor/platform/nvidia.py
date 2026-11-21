# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4: NVIDIA subprocess platform adapter with PGID cleanup and GPU telemetry."""

from __future__ import annotations

import atexit
import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tools.byog.executor.interfaces import (
    CacheResetUnsupportedError,
    DirectCommandResult,
    PlatformAdapter,
    ServerHandle,
    ServerTransportError,
)
from tools.byog.schemas.result import ProtocolVariant

_HOST_ENV_ALLOWLIST_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "HF_HUB_CACHE",
        "HF_HOME",
        "BYOG_ASSET_CACHE",
    }
)
_HOST_ENV_ALLOWLIST_PREFIXES = ("CUDA_", "NCCL_", "TORCH_", "VLLM_")
_SECRET_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")
_VRAM_TOLERANCE_MIB = 256

_GPU_NAME_TO_ID: tuple[tuple[str, str], ...] = (
    ("gb300", "gb300"),
    ("gb200", "gb200"),
    ("b300", "b300"),
    ("b200", "b200"),
    ("h200", "h200"),
    ("h100", "h100"),
    ("a100", "a100"),
    ("l40s", "l40s"),
    ("l4", "l4"),
    ("rtx pro 6000", "rtx_pro_6000"),
    ("rtx 5090", "rtx_5090"),
)


def _is_secret_var(name: str) -> bool:
    upper = name.upper()
    if any(tok in upper for tok in _SECRET_TOKENS):
        return True
    parts = {p for p in upper.split("_") if p}
    return bool(parts & {"KEY", "PAT"})


def _build_subprocess_env(
    serve_env: dict[str, str],
    *,
    enable_dev_mode: bool = False,
    allow_hf_token: bool = False,
) -> dict[str, str]:
    """Construct a clean allowlisted environment for child processes."""
    env: dict[str, str] = {}
    for key, val in os.environ.items():
        if allow_hf_token and key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env[key] = val
            continue
        if _is_secret_var(key):
            continue
        if key in _HOST_ENV_ALLOWLIST_EXACT or key.startswith(
            _HOST_ENV_ALLOWLIST_PREFIXES
        ):
            env[key] = val

    env.update(serve_env)
    if enable_dev_mode:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    else:
        env.pop("VLLM_SERVER_DEV_MODE", None)
    return env


def _allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _canonical_gpu_id(gpu_name: str) -> str:
    lower = gpu_name.lower()
    for needle, gpu_id in _GPU_NAME_TO_ID:
        if needle in lower:
            return gpu_id
    return lower.replace(" ", "_")


def _parse_visible_gpu_indices() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    indices: list[int] = []
    for item in raw.split(","):
        tok = item.strip()
        if not tok:
            continue
        if tok.isdigit():
            indices.append(int(tok))
        else:
            # UUID or MIG device string: cannot filter by numeric index.
            return None
    return indices or None


def build_vllm_serve_command(
    serve_config: dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> list[str]:
    """Translate `serve.config` dict into a `vllm serve` argv list."""
    model = str(serve_config["model"])
    cmd = ["vllm", "serve", model, "--host", host, "--port", str(port)]
    for key, val in sorted(serve_config.items()):
        if key in ("model", "host", "port") or val is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif isinstance(val, list):
            for item in val:
                cmd.extend([flag, str(item)])
        elif isinstance(val, dict):
            cmd.extend([flag, json.dumps(val, separators=(",", ":"))])
        else:
            cmd.extend([flag, str(val)])
    return cmd


def _query_nvidia_smi() -> list[dict[str, Any]] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None

    visible = _parse_visible_gpu_indices()
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if visible is not None and idx not in visible:
            continue
        total_mib = int(parts[3]) if parts[3].isdigit() else None
        used_mib = int(parts[4]) if parts[4].isdigit() else None
        rows.append(
            {
                "index": idx,
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": total_mib,
                "memory_used_mib": used_mib,
            }
        )
    return rows or None


def _query_used_memory_mib() -> list[int] | None:
    rows = _query_nvidia_smi()
    if not rows:
        return None
    used: list[int] = []
    for row in rows:
        val = row.get("memory_used_mib")
        if not isinstance(val, int):
            # MIG or unreadable memory counter -> honestly unknown.
            return None
        used.append(val)
    return used


def _kill_process_group(pgid: int, proc: subprocess.Popen[Any] | None) -> bool:
    """Send SIGTERM then unconditional SIGKILL to `pgid`; never raises."""
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGTERM)

    if proc is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)

    # Always SIGKILL the process group to catch orphaned EngineCore children
    # even if the group leader already exited.
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGKILL)

    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return False
    return True


class NvidiaServerLauncher(PlatformAdapter):
    """NVIDIA subprocess launcher with process-group cleanup and VRAM verification."""

    def __init__(self, *, ready_poll_s: float = 1.0):
        self._ready_poll_s = ready_poll_s
        self._procs: dict[str, tuple[subprocess.Popen[Any], int, Any]] = {}
        atexit.register(self._cleanup_all)

    def _cleanup_all(self) -> None:
        for key in list(self._procs):
            proc, pgid, log_fh = self._procs.pop(key)
            _kill_process_group(pgid, proc)
            with contextlib.suppress(OSError):
                log_fh.close()

    def detect_hardware(self) -> dict[str, Any]:
        rows = _query_nvidia_smi()
        if not rows:
            return {
                "platform": "nvidia",
                "gpu_id": "unknown",
                "gpu_name": "unknown",
                "gpu_count": 0,
                "driver_version": None,
                "gpu_memory_mib": None,
                "device_memory_used_mib": None,
            }
        first = rows[0]
        mem_used = (
            [int(r["memory_used_mib"]) for r in rows]
            if all(isinstance(r["memory_used_mib"], int) for r in rows)
            else None
        )
        mem_total = (
            [int(r["memory_total_mib"]) for r in rows]
            if all(isinstance(r["memory_total_mib"], int) for r in rows)
            else None
        )
        return {
            "platform": "nvidia",
            "gpu_id": _canonical_gpu_id(str(first["name"])),
            "gpu_name": str(first["name"]),
            "gpu_count": len(rows),
            "driver_version": str(first["driver_version"]),
            "gpu_memory_mib": mem_total,
            "device_memory_used_mib": mem_used,
        }

    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
        *,
        enable_dev_mode: bool = False,
    ) -> ServerHandle:
        log_dir.mkdir(parents=True, exist_ok=True)
        port = _allocate_free_port()
        cmd = build_vllm_serve_command(serve_config, host="127.0.0.1", port=port)
        env = _build_subprocess_env(
            serve_env, enable_dev_mode=enable_dev_mode, allow_hf_token=True
        )
        log_path = log_dir / f"vllm_server_{port}.log"
        log_fh = log_path.open("w", encoding="utf-8")

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log_fh.close()
            raise

        pgid = os.getpgid(proc.pid)
        base_url = f"http://127.0.0.1:{port}"
        self._procs[base_url] = (proc, pgid, log_fh)
        handle = ServerHandle(
            base_url=base_url,
            command=cmd,
            log_path=log_path,
            pid=proc.pid,
            metadata={"pgid": pgid, "enable_dev_mode": enable_dev_mode},
        )

        deadline = time.monotonic() + max(1, timeout_s)
        try:
            while time.monotonic() < deadline:
                rc = proc.poll()
                if rc is not None:
                    tail = self._tail_log(log_path)
                    raise RuntimeError(
                        f"vLLM server exited early with code {rc}.\nLog tail:\n{tail}"
                    )
                if self._is_server_ready(base_url):
                    return handle
                time.sleep(
                    min(self._ready_poll_s, max(0.05, deadline - time.monotonic()))
                )
            tail = self._tail_log(log_path)
            raise TimeoutError(
                f"vLLM server did not become ready within {timeout_s}s.\n"
                f"Log tail:\n{tail}"
            )
        except BaseException:
            self.stop(handle)
            raise

    @staticmethod
    def _is_server_ready(base_url: str) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(f"{base_url}{path}", timeout=2) as resp:
                    if resp.status != 200:
                        return False
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
            ):
                return False
        return True

    @staticmethod
    def _tail_log(log_path: Path, max_lines: int = 40) -> str:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-max_lines:])
        except OSError:
            return "<unable to read server log>"

    def reset_cache(self, handle: ServerHandle) -> ProtocolVariant:
        req = urllib.request.Request(
            f"{handle.base_url}/reset_prefix_cache",
            data=b"",
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 200 <= resp.status < 300:
                    return ProtocolVariant.RESET_CACHE
                raise ServerTransportError(
                    f"Unexpected HTTP {resp.status} from /reset_prefix_cache."
                )
        except urllib.error.HTTPError as err:
            if err.code in (404, 405, 501):
                raise CacheResetUnsupportedError(
                    f"/reset_prefix_cache unsupported (HTTP {err.code})."
                ) from err
            raise ServerTransportError(
                f"/reset_prefix_cache failed with HTTP {err.code}: {err.reason}"
            ) from err
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
        ) as err:
            raise ServerTransportError(
                f"/reset_prefix_cache transport error: {err}"
            ) from err

    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> dict[str, Any]:
        entry = self._procs.pop(handle.base_url, None)
        terminated_pgid = True
        if entry is not None:
            proc, pgid, log_fh = entry
            terminated_pgid = _kill_process_group(pgid, proc)
            with contextlib.suppress(OSError):
                log_fh.close()
        elif "pgid" in handle.metadata:
            terminated_pgid = _kill_process_group(int(handle.metadata["pgid"]), None)

        vram_reclaimed: bool | None = None
        if baseline_memory_mib is not None:
            reclaim_deadline = time.monotonic() + 5.0
            while True:
                current = _query_used_memory_mib()
                if current is None or len(current) != len(baseline_memory_mib):
                    vram_reclaimed = None
                    break
                if all(
                    cur <= base + _VRAM_TOLERANCE_MIB
                    for cur, base in zip(current, baseline_memory_mib)
                ):
                    vram_reclaimed = True
                    break
                if time.monotonic() >= reclaim_deadline:
                    vram_reclaimed = False
                    break
                time.sleep(0.5)

        return {
            "terminated_pgid": terminated_pgid,
            "vram_reclaimed": vram_reclaimed,
            "baseline_memory_mib": baseline_memory_mib,
        }

    def run_direct(
        self,
        cmd: list[str],
        serve_env: dict[str, str],
        timeout_s: int,
    ) -> DirectCommandResult:
        env = _build_subprocess_env(
            serve_env, enable_dev_mode=False, allow_hf_token=False
        )
        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=max(1, timeout_s))
            return DirectCommandResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                elapsed_s=time.monotonic() - t0,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            _kill_process_group(pgid, proc)
            stdout, stderr = proc.communicate()
            return DirectCommandResult(
                returncode=-9,
                stdout=stdout or "",
                stderr=stderr or "",
                elapsed_s=time.monotonic() - t0,
                timed_out=True,
            )
