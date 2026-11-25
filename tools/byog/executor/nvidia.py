# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA platform adapter (C4): subprocess launcher with PGID and VRAM cleanup."""

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

from tools.byog.interfaces import (
    CacheResetUnsupportedError,
    CleanupResult,
    DetectedHardware,
    DirectCommandResult,
    PlatformAdapter,
    ServerHandle,
    ServerTransportError,
)

_HOST_ALLOWLIST = frozenset(
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
    }
)
_HOST_PREFIXES = ("CUDA_", "NCCL_", "TORCH_", "VLLM_")
_SECRET_SUBSTRS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "APIKEY",
)
# Checked against whitespace/punctuation-delimited tokens so GH200 != H200, L40 != L4.
_GPU_TOKENS = (
    "gb300",
    "gb200",
    "gh200",
    "b300",
    "b200",
    "h200",
    "h100",
    "a100",
    "l40s",
    "l40",
    "l4",
)


def _normalize_gpu_id(name: str) -> str:
    cleaned = name.lower().replace("-", " ").replace("/", " ").replace("_", " ")
    tokens = set(cleaned.split())
    for gid in _GPU_TOKENS:
        if gid in tokens:
            return gid
    return name.lower()


def _build_env(
    serve_env: dict[str, str], *, enable_dev_mode: bool = False
) -> dict[str, str]:
    """Build sanitized env for candidate subprocesses (`HF_TOKEN` always stripped)."""
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        up = k.upper()
        if any(s in up for s in _SECRET_SUBSTRS) or bool(
            set(up.split("_")) & {"KEY", "PAT"}
        ):
            continue
        if k in _HOST_ALLOWLIST or k.startswith(_HOST_PREFIXES):
            env[k] = v
    env.update(serve_env)
    if enable_dev_mode:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    else:
        env.pop("VLLM_SERVER_DEV_MODE", None)
    return env


def build_vllm_serve_command(
    serve_config: dict[str, Any], *, host: str = "127.0.0.1", port: int = 8000
) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        str(serve_config["model"]),
        "--host",
        host,
        "--port",
        str(port),
    ]
    for k, v in sorted(serve_config.items()):
        if k in ("model", "host", "port") or v is None:
            continue
        opt = k.replace("_", "-")
        flag = f"--{opt}"
        if isinstance(v, bool):
            cmd.append(flag if v else f"--no-{opt}")
        elif isinstance(v, list):
            for item in v:
                cmd.extend([flag, str(item)])
        elif isinstance(v, dict):
            cmd.extend([flag, json.dumps(v, separators=(",", ":"))])
        else:
            cmd.extend([flag, str(v)])
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

    raw_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    vis = (
        [int(x) for x in raw_vis.split(",") if x.strip().isdigit()]
        if raw_vis and all(x.strip().isdigit() for x in raw_vis.split(",") if x.strip())
        else None
    )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        if vis is not None and idx not in vis:
            continue
        rows.append(
            {
                "index": idx,
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": int(parts[3]) if parts[3].isdigit() else None,
                "memory_used_mib": int(parts[4]) if parts[4].isdigit() else None,
            }
        )
    return rows or None


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _kill_pgid(pgid: int, proc: subprocess.Popen[Any] | None) -> bool:
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGTERM)
    if proc is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
    if _pgid_alive(pgid):
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
        if proc is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        deadline = time.monotonic() + 5.0
        while _pgid_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
    return not _pgid_alive(pgid)


class NvidiaServerLauncher(PlatformAdapter):
    """Launches `vllm serve` on NVIDIA GPUs with PGID and VRAM cleanup checks."""

    def __init__(self, *, ready_poll_s: float = 1.0):
        self._ready_poll_s = ready_poll_s
        self._procs: dict[str, tuple[subprocess.Popen[Any], int, Any]] = {}
        atexit.register(self._cleanup_all)

    def _cleanup_all(self) -> None:
        for key in list(self._procs):
            proc, pgid, fh = self._procs.pop(key)
            _kill_pgid(pgid, proc)
            with contextlib.suppress(OSError):
                fh.close()

    def detect_hardware(self) -> DetectedHardware:
        rows = _query_nvidia_smi()
        if not rows:
            return DetectedHardware(
                platform="nvidia",
                gpu_id="unknown",
                gpu_count=0,
            )
        name = str(rows[0]["name"])
        used = (
            [int(r["memory_used_mib"]) for r in rows]
            if all(isinstance(r["memory_used_mib"], int) for r in rows)
            else None
        )
        return DetectedHardware(
            platform="nvidia",
            gpu_id=_normalize_gpu_id(name),
            gpu_count=len(rows),
            gpu_name=name,
            driver_version=str(rows[0]["driver_version"]),
            device_memory_used_mib=used,
        )

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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])

        cmd = build_vllm_serve_command(serve_config, port=port)
        env = _build_env(serve_env, enable_dev_mode=enable_dev_mode)
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
            metadata={"pgid": pgid},
        )

        deadline = time.monotonic() + max(1, timeout_s)
        try:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    tail = "\n".join(
                        log_path.read_text(errors="replace").splitlines()[-30:]
                    )
                    raise RuntimeError(
                        f"vLLM exited early (code {proc.returncode}):\n{tail}"
                    )
                if self._ready(base_url):
                    return handle
                time.sleep(
                    min(self._ready_poll_s, max(0.05, deadline - time.monotonic()))
                )
            raise TimeoutError(f"vLLM server not ready within {timeout_s}s.")
        except BaseException:
            self.stop(handle)
            raise

    @staticmethod
    def _ready(base_url: str) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(f"{base_url}{path}", timeout=2) as r:
                    if r.status != 200:
                        return False
            except (urllib.error.URLError, http.client.HTTPException, OSError):
                return False
        return True

    def reset_cache(self, handle: ServerHandle) -> None:
        req = urllib.request.Request(
            f"{handle.base_url}/reset_prefix_cache", data=b"", method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                if 200 <= r.status < 300:
                    return
                raise ServerTransportError(f"Unexpected HTTP {r.status}.")
        except urllib.error.HTTPError as err:
            if err.code in (404, 405, 501):
                raise CacheResetUnsupportedError(f"HTTP {err.code}") from err
            raise ServerTransportError(f"HTTP {err.code}: {err.reason}") from err
        except (urllib.error.URLError, http.client.HTTPException, OSError) as err:
            raise ServerTransportError(str(err)) from err

    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> CleanupResult:
        entry = self._procs.pop(handle.base_url, None)
        terminated = False
        if entry is not None:
            proc, pgid, fh = entry
            terminated = _kill_pgid(pgid, proc)
            with contextlib.suppress(OSError):
                fh.close()
        elif "pgid" in handle.metadata:
            terminated = _kill_pgid(int(handle.metadata["pgid"]), None)

        vram_reclaimed: bool | None = None
        after_mem: list[int] | None = None
        if baseline_memory_mib is not None:
            deadline = time.monotonic() + 5.0
            while True:
                rows = _query_nvidia_smi()
                if (
                    not rows
                    or len(rows) != len(baseline_memory_mib)
                    or not all(isinstance(r["memory_used_mib"], int) for r in rows)
                ):
                    vram_reclaimed = False
                    break
                after_mem = [int(r["memory_used_mib"]) for r in rows]
                if all(c <= b + 256 for c, b in zip(after_mem, baseline_memory_mib)):
                    vram_reclaimed = True
                    break
                if time.monotonic() >= deadline:
                    vram_reclaimed = False
                    break
                time.sleep(0.5)
        return CleanupResult(
            terminated_pgid=terminated,
            vram_reclaimed=vram_reclaimed,
            baseline_memory_mib=baseline_memory_mib,
            after_memory_mib=after_mem,
        )

    def run_direct(
        self, cmd: list[str], serve_env: dict[str, str], timeout_s: int
    ) -> DirectCommandResult:
        env = _build_env(serve_env, enable_dev_mode=False)
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
            out, err = proc.communicate(timeout=max(1, timeout_s))
            return DirectCommandResult(proc.returncode, out, err, time.monotonic() - t0)
        except subprocess.TimeoutExpired:
            _kill_pgid(pgid, proc)
            out, err = proc.communicate()
            return DirectCommandResult(
                -9, out or "", err or "", time.monotonic() - t0, timed_out=True
            )
        except BaseException:
            _kill_pgid(pgid, proc)
            raise
