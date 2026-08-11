# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re


class SnapshotProvider(ABC):
    @abstractmethod
    def capture(self, root_pid: int, snapshot_dir: Path) -> dict[str, Any]: ...

    @abstractmethod
    def restore(self, snapshot_dir: Path) -> dict[str, Any]: ...

    def discard_restored(self, root_pid: int) -> None:
        self._kill_tree(root_pid)

    def rollback_capture(self, snapshot_dir: Path) -> dict[str, Any] | None:
        return None

    @property
    def restore_is_retriable(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, Any]:
        return {"provider": type(self).__name__}

    @staticmethod
    def _process_tree(root_pid: int) -> list[int]:
        children: dict[int, list[int]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                fields = stat[stat.rfind(")") + 2 :].split()
                children.setdefault(int(fields[1]), []).append(int(entry.name))
            except (FileNotFoundError, IndexError, PermissionError, ValueError):
                continue
        tree = []
        pending = [root_pid]
        while pending:
            pid = pending.pop()
            tree.append(pid)
            pending.extend(children.get(pid, ()))
        return sorted(tree)

    @classmethod
    def _kill_tree(cls, root_pid: int) -> None:
        for pid in reversed(cls._process_tree(root_pid)):
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@dataclass
class FakeSnapshotProvider(SnapshotProvider):
    delay: float = 0.0

    def capture(self, root_pid: int, snapshot_dir: Path) -> dict[str, Any]:
        started = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        (snapshot_dir / "fake.img").write_bytes(b"vllm-engine-snapshot\n")
        os.kill(root_pid, signal.SIGSTOP)
        return {
            "root_pid": root_pid,
            "source_exited": False,
            "capture_seconds": time.monotonic() - started,
            "provider": "fake",
        }

    def restore(self, snapshot_dir: Path) -> dict[str, Any]:
        started = time.monotonic()
        root_pid = int((snapshot_dir / "root.pid").read_text())
        os.kill(root_pid, signal.SIGCONT)
        return {
            "root_pid": root_pid,
            "source_exited": False,
            "restore_seconds": time.monotonic() - started,
            "provider": "fake",
        }

    def runtime_identity(self) -> dict[str, Any]:
        return {"provider": "fake"}


@dataclass
class CriuCudaSnapshotProvider(SnapshotProvider):
    criu_path: str
    cuda_checkpoint_path: str
    lock_timeout_ms: int = 30000
    cuda_operation_timeout_seconds: float = 300
    criu_operation_timeout_seconds: float = 1800

    def capture(self, root_pid: int, snapshot_dir: Path) -> dict[str, Any]:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        images_dir = snapshot_dir / "images"
        work_dir = snapshot_dir / "work"
        images_dir.mkdir()
        work_dir.mkdir()
        cuda_pid = self._cuda_pid(root_pid)

        started = time.monotonic()
        locked = False
        try:
            self._run_cuda("lock", cuda_pid, "--timeout", str(self.lock_timeout_ms))
            locked = True
            self._run_cuda("checkpoint", cuda_pid)
            cuda_end = time.monotonic()
            self._run(
                [
                    self.criu_path,
                    "dump",
                    "--tree",
                    str(root_pid),
                    "--images-dir",
                    str(images_dir),
                    "--work-dir",
                    str(work_dir),
                    "--shell-job",
                    "--file-locks",
                    "--ext-unix-sk",
                    "--link-remap",
                    "--tcp-established",
                    "--skip-in-flight",
                    "--network-lock",
                    "nftables",
                    "-o",
                    "dump.log",
                    "-v4",
                ],
                env=self._criu_env(),
                timeout=self.criu_operation_timeout_seconds,
            )
            criu_end = time.monotonic()
        except Exception as exc:
            cleanup_errors = []
            if locked:
                try:
                    self._run_cuda("restore", cuda_pid)
                except Exception as cleanup_error:
                    cleanup_errors.append(f"restore: {cleanup_error}")
                try:
                    self._run_cuda("unlock", cuda_pid)
                except Exception as cleanup_error:
                    cleanup_errors.append(f"unlock: {cleanup_error}")
            error = self._with_criu_log(exc, work_dir / "dump.log")
            if cleanup_errors:
                error = RuntimeError(
                    f"{error}\nCUDA rollback cleanup failed: "
                    + "; ".join(cleanup_errors)
                )
            raise error from exc
        return {
            "root_pid": root_pid,
            "source_exited": True,
            "cuda_pids": [cuda_pid],
            "cuda_checkpoint_seconds": cuda_end - started,
            "criu_dump_seconds": criu_end - cuda_end,
            "capture_seconds": criu_end - started,
            "provider": "criu_cuda",
        }

    def restore(self, snapshot_dir: Path) -> dict[str, Any]:
        images_dir = snapshot_dir / "images"
        work_dir = snapshot_dir / f"restore-work-{uuid.uuid4().hex}"
        work_dir.mkdir(exist_ok=False)
        pidfile = work_dir / "root.pid"
        started = time.monotonic()
        root_pid: int | None = None
        try:
            self._run(
                [
                    self.criu_path,
                    "restore",
                    "--images-dir",
                    str(images_dir),
                    "--work-dir",
                    str(work_dir),
                    "--shell-job",
                    "--file-locks",
                    "--ext-unix-sk",
                    "--tcp-established",
                    "--restore-detached",
                    "--pidfile",
                    str(pidfile),
                    "-o",
                    "restore.log",
                    "-v4",
                ],
                env=self._criu_env(),
                timeout=self.criu_operation_timeout_seconds,
            )
            criu_end = time.monotonic()
            root_pid = int(pidfile.read_text())
            pidfile_read_end = time.monotonic()
            cuda_pid = self._cuda_pid(root_pid, restored=True)
            cuda_pid_discovery_end = time.monotonic()
            self._run_cuda("restore", cuda_pid)
            cuda_restore_action_end = time.monotonic()
            self._run_cuda("unlock", cuda_pid)
            ended = time.monotonic()
            shutil.rmtree(work_dir)
        except Exception as exc:
            if root_pid is not None:
                self._kill_tree(root_pid)
            error = self._with_criu_log(exc, work_dir / "restore.log")
            try:
                shutil.rmtree(work_dir)
            except OSError as cleanup_error:
                error = RuntimeError(
                    f"{error}\nrestore work cleanup failed: {cleanup_error}"
                )
            raise error from exc
        return {
            "root_pid": root_pid,
            "source_exited": False,
            "cuda_pids": [cuda_pid],
            "criu_restore_seconds": criu_end - started,
            "root_pid_read_seconds": pidfile_read_end - criu_end,
            "cuda_pid_discovery_seconds": (cuda_pid_discovery_end - pidfile_read_end),
            "cuda_restore_action_seconds": (
                cuda_restore_action_end - cuda_pid_discovery_end
            ),
            "cuda_unlock_seconds": ended - cuda_restore_action_end,
            "cuda_restore_seconds": ended - criu_end,
            "restore_seconds": ended - started,
            "provider": "criu_cuda",
        }

    def rollback_capture(self, snapshot_dir: Path) -> dict[str, Any]:
        return self.restore(snapshot_dir)

    @property
    def restore_is_retriable(self) -> bool:
        return False

    def runtime_identity(self) -> dict[str, Any]:
        criu_version = self._run(
            [self.criu_path, "--version"],
            check=False,
            env=self._criu_env(),
        )
        return {
            "provider": "criu_cuda",
            "criu": self._binary_identity(self.criu_path, criu_version.stdout.strip()),
            "cuda_checkpoint": self._binary_identity(
                self.cuda_checkpoint_path, self._cuda_checkpoint_version()
            ),
        }

    def _cuda_checkpoint_version(self) -> str:
        result = self._run([self.cuda_checkpoint_path, "--help"], check=False)
        output = f"{result.stdout}\n{result.stderr}"
        version = re.search(r"(?:version|Version)\s*:?\s*([^\s]+)", output)
        return version.group(1) if version else "unknown"

    @staticmethod
    def _binary_identity(path: str, version: str) -> dict[str, Any]:
        binary = Path(path)
        digest = hashlib.sha256()
        with binary.open("rb") as binary_file:
            for chunk in iter(lambda: binary_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(binary.resolve()),
            "size": binary.stat().st_size,
            "sha256": digest.hexdigest(),
            "version": version,
        }

    def _cuda_pids(self, root_pid: int, restored: bool = False) -> list[int]:
        result = []
        for pid in self._process_tree(root_pid):
            command = [self.cuda_checkpoint_path]
            command.append("--get-restore-tid" if restored else "--get-state")
            command.extend(["--pid", str(pid)])
            if subprocess.run(command, capture_output=True, timeout=10).returncode == 0:
                result.append(pid)
        return result

    def _cuda_pid(self, root_pid: int, restored: bool = False) -> int:
        cuda_pids = self._cuda_pids(root_pid, restored=restored)
        if len(cuda_pids) != 1:
            process_type = "restored CUDA process" if restored else "CUDA process"
            raise RuntimeError(
                f"expected exactly one {process_type} in EngineCore tree, "
                f"found {len(cuda_pids)}"
            )
        return cuda_pids[0]

    def _run_cuda(self, action: str, pid: int, *extra: str) -> None:
        self._run(
            self._cuda_command(action, pid, *extra),
            timeout=self.cuda_operation_timeout_seconds,
        )

    def _cuda_command(self, action: str, pid: int, *extra: str) -> list[str]:
        return [
            self.cuda_checkpoint_path,
            "--action",
            action,
            "--pid",
            str(pid),
            *extra,
        ]

    @staticmethod
    def _criu_env() -> dict[str, str]:
        env = os.environ.copy()
        tunables = env.get("GLIBC_TUNABLES")
        if tunables is None:
            return env
        filtered = ":".join(
            value
            for value in tunables.split(":")
            if not value.startswith("glibc.pthread.rseq=")
        )
        if filtered:
            env["GLIBC_TUNABLES"] = filtered
        else:
            env.pop("GLIBC_TUNABLES")
        return env

    @staticmethod
    def _with_criu_log(exc: BaseException, path: Path) -> RuntimeError:
        try:
            log_tail = path.read_text(errors="replace")[-12000:].strip()
        except FileNotFoundError:
            log_tail = "CRIU log file is missing"
        return RuntimeError(f"{exc}\nCRIU log tail:\n{log_tail}")

    @staticmethod
    def _run(
        command: list[str],
        check: bool = True,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise CriuCudaSnapshotProvider._command_error(
                command,
                result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    @staticmethod
    def _command_error(
        command: list[str],
        returncode: int,
        *,
        stdout: str | None,
        stderr: str | None,
    ) -> RuntimeError:
        output = (stderr or stdout or "").strip()
        return RuntimeError(
            f"command failed ({returncode}): {' '.join(command)}: {output[-4000:]}"
        )


def resolve_snapshot_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"engine snapshots require '{name}' to be available on PATH"
        )
    return str(Path(path).resolve())


def make_snapshot_provider(name: str) -> SnapshotProvider:
    if name == "fake":
        return FakeSnapshotProvider()
    if name == "criu_cuda":
        return CriuCudaSnapshotProvider(
            criu_path=resolve_snapshot_executable("criu"),
            cuda_checkpoint_path=resolve_snapshot_executable("cuda-checkpoint"),
        )
    raise ValueError(f"unknown engine snapshot provider: {name}")
