# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host runtime operations for local snapshots."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import regex as re

from vllm.snapshot.manifest import (
    SnapshotManifest,
    SnapshotRuntimeIdentity,
    _fsync_directory,
    _write_json_atomic,
    validate_artifact_root,
    write_manifest_atomic,
)
from vllm.snapshot.types import Oracle


class SnapshotCreateError(RuntimeError):
    """Snapshot creation did not produce a complete artifact."""


class SnapshotRestoreError(RuntimeError):
    """Snapshot restore failed."""


_COMMON_CRIU_OPTIONS = (
    "--shell-job",
    "--ext-unix-sk",
    "--tcp-established",
    "--link-remap",
    "--file-locks",
)


def _error_detail(error: BaseException) -> str:
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _close_pidfds(handles: Iterable[tuple[int, int]]) -> None:
    for _pid, pidfd in handles:
        with suppress(OSError):
            os.close(pidfd)


@dataclass(frozen=True)
class ProcessInventory:
    root_pid: int
    process_tree: tuple[int, ...]
    cuda_holders: tuple[int, ...]
    gpu_uuid: str


@dataclass(frozen=True)
class _TcpSocketRecord:
    family: str
    local_raw: str
    remote_raw: str
    inode: int


def _validate_tcp_connections(
    records: tuple[_TcpSocketRecord, ...], owned_inodes: set[int]
) -> None:
    owned = tuple(record for record in records if record.inode in owned_inodes)
    endpoints = {
        (record.family, record.local_raw, record.remote_raw) for record in owned
    }
    if any(
        (record.family, record.remote_raw, record.local_raw) not in endpoints
        for record in owned
    ):
        raise SnapshotCreateError(
            "snapshot tree has an external established TCP connection"
        )


class LocalSnapshotTools:
    """Host-owned CRIU and cuda-checkpoint implementation."""

    def __init__(self) -> None:
        self.criu = shutil.which("criu") or "criu"
        self.cuda_checkpoint = shutil.which("cuda-checkpoint") or "cuda-checkpoint"
        self.nvidia_smi = shutil.which("nvidia-smi") or "nvidia-smi"
        plugin_dir = os.environ.get("CRIU_CUDA_PLUGIN_DIR", "")
        self.plugin_dir = Path(plugin_dir) if plugin_dir else None
        self.shm_dir = Path("/dev/shm")
        self.timeout_s = float(os.environ.get("VLLM_SNAPSHOT_TIMEOUT_S", "900"))
        self._children: dict[int, subprocess.Popen[bytes]] = {}
        self._restored_processes: dict[int, tuple[tuple[int, int], ...]] = {}

    def _privileged(self) -> list[str]:
        return [] if os.geteuid() == 0 else ["sudo", "-n"]

    def _run(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout_s,
        )

    def preflight(self, action: str, artifact: Path) -> None:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("snapshot requires Linux x86_64")
        if action == "restore" and (
            not callable(getattr(os, "pidfd_open", None))
            or not callable(getattr(signal, "pidfd_send_signal", None))
        ):
            raise RuntimeError("snapshot requires Linux pidfd support")
        for command in (self.criu, self.cuda_checkpoint, self.nvidia_smi):
            if shutil.which(command) is None:
                raise RuntimeError(f"snapshot dependency not found: {command}")
        if self.plugin_dir is None:
            raise RuntimeError("CRIU_CUDA_PLUGIN_DIR is required")
        if not (self.plugin_dir / "cuda_plugin.so").is_file():
            raise RuntimeError(f"CRIU CUDA plugin not found in {self.plugin_dir}")
        if os.geteuid() != 0:
            self._run(["sudo", "-n", "true"], timeout=5)
        if action == "restore":
            validate_artifact_root(artifact, creating=False)
        else:
            validate_artifact_root(artifact, creating=True)

    def launch_child(
        self,
        workdir: Path,
        engine_argv: tuple[str, ...],
    ) -> int:
        log_file = (workdir / "child.log").open("wb")
        command = [
            sys.executable,
            "-m",
            "vllm.snapshot.server",
            "--ready-file",
            str(workdir / "ready.json"),
            "--release-file",
            str(workdir / "release.json"),
            "--release-timeout-s",
            str(self.timeout_s),
            "--",
            *engine_argv,
        ]
        child_environment = os.environ.copy()
        child_environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        process = subprocess.Popen(
            command,
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        self._children[process.pid] = process
        return process.pid

    def wait_ready(self, workdir: Path, root_pid: int) -> Oracle:
        ready_file = workdir / "ready.json"
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if ready_file.is_file() and ready_file.stat().st_size:
                payload = json.loads(ready_file.read_text())
                return Oracle(
                    token_ids=tuple(payload["token_ids"]),
                    text=payload["text"],
                    sampled_token_logprob=payload["sampled_token_logprob"],
                )
            process = self._children.get(root_pid)
            if process is not None and process.poll() is not None:
                log = (workdir / "child.log").read_text(errors="replace")
                raise SnapshotCreateError(f"snapshot child exited before ready:\n{log}")
            try:
                os.kill(root_pid, 0)
            except ProcessLookupError as error:
                log = (workdir / "child.log").read_text(errors="replace")
                raise SnapshotCreateError(
                    f"snapshot child exited before ready:\n{log}"
                ) from error
            time.sleep(0.05)
        raise SnapshotCreateError("snapshot child did not become ready before timeout")

    def _tree_pids(self, root_pid: int) -> tuple[int, ...]:
        rows = self._run(["ps", "-eo", "pid=,ppid="]).stdout.splitlines()
        children: dict[int, list[int]] = {}
        for row in rows:
            pid, parent_pid = map(int, row.split())
            children.setdefault(parent_pid, []).append(pid)
        stack = [root_pid]
        found: list[int] = []
        while stack:
            pid = stack.pop()
            if pid in found:
                continue
            found.append(pid)
            stack.extend(children.get(pid, []))
        return tuple(found)

    def _cuda_process_rows(self) -> tuple[str, ...]:
        output = self._run(
            [
                self.nvidia_smi,
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ]
        ).stdout
        return tuple(line for line in output.splitlines() if line.strip())

    def _cuda_pids(self, rows: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(sorted({int(line.split(",", 1)[0].strip()) for line in rows}))

    def _descriptor_targets(self, pid: int) -> tuple[str, ...]:
        descriptor_dir = Path("/proc") / str(pid) / "fd"
        try:
            descriptors = tuple(descriptor_dir.iterdir())
        except FileNotFoundError as error:
            raise SnapshotCreateError(
                f"snapshot process exited during descriptor inventory: {pid}"
            ) from error
        targets: list[str] = []
        for descriptor in descriptors:
            try:
                targets.append(os.readlink(descriptor))
            except FileNotFoundError:
                continue
        return tuple(targets)

    def _descriptor_inventory(
        self, process_tree: tuple[int, ...]
    ) -> tuple[tuple[int, ...], set[int]]:
        io_uring_pids: list[int] = []
        socket_inodes: set[int] = set()
        for pid in process_tree:
            targets = self._descriptor_targets(pid)
            if "anon_inode:[io_uring]" in targets:
                io_uring_pids.append(pid)
            for target in targets:
                if target.startswith("socket:[") and target.endswith("]"):
                    socket_inodes.add(int(target[len("socket:[") : -1]))
        return tuple(io_uring_pids), socket_inodes

    def _tcp_records(self) -> tuple[_TcpSocketRecord, ...]:
        records: list[_TcpSocketRecord] = []
        for family, table in (
            ("AF_INET", Path("/proc/net/tcp")),
            ("AF_INET6", Path("/proc/net/tcp6")),
        ):
            for row in table.read_text().splitlines()[1:]:
                fields = row.split()
                if fields[3] != "01":
                    continue
                records.append(
                    _TcpSocketRecord(
                        family=family,
                        local_raw=fields[1],
                        remote_raw=fields[2],
                        inode=int(fields[9]),
                    )
                )
        return tuple(records)

    def inventory(self, root_pid: int) -> ProcessInventory:
        process_tree = self._tree_pids(root_pid)
        cuda_rows = self._cuda_process_rows()
        cuda_holders = tuple(
            pid for pid in self._cuda_pids(cuda_rows) if pid in process_tree
        )
        if not cuda_holders:
            raise SnapshotCreateError("snapshot tree has no CUDA-holding process")
        io_uring_pids, socket_inodes = self._descriptor_inventory(process_tree)
        if io_uring_pids:
            raise SnapshotCreateError(
                "snapshot tree owns io_uring state that CRIU cannot dump; set "
                "kernel.io_uring_disabled=1 before starting an unprivileged "
                "vLLM process, or use =2 to disable it host-wide "
                f"(pids: {', '.join(map(str, io_uring_pids))})"
            )
        _validate_tcp_connections(self._tcp_records(), socket_inodes)
        gpu_uuid = self._gpu_uuid_for_pids(cuda_holders, cuda_rows)
        return ProcessInventory(
            root_pid=root_pid,
            process_tree=process_tree,
            cuda_holders=cuda_holders,
            gpu_uuid=gpu_uuid,
        )

    def _gpu_uuid_for_pids(self, pids: tuple[int, ...], rows: tuple[str, ...]) -> str:
        wanted = set(pids)
        matches: set[str] = set()
        try:
            for line in rows:
                pid, gpu_uuid = (item.strip() for item in line.split(",", 1))
                if int(pid) in wanted:
                    matches.add(gpu_uuid)
        except (TypeError, ValueError) as error:
            raise SnapshotCreateError(
                "could not identify the GPU used by the snapshot process tree"
            ) from error
        if len(matches) != 1:
            raise SnapshotCreateError(
                "snapshot process tree must use exactly one identifiable GPU"
            )
        return matches.pop()

    def _criu(
        self,
        action: str,
        artifact: Path,
        arguments: list[str],
    ) -> None:
        if self.plugin_dir is None:
            raise RuntimeError("CRIU_CUDA_PLUGIN_DIR is required")
        env = os.environ.copy()
        env["PATH"] = f"{Path(self.cuda_checkpoint).parent}:{env['PATH']}"
        env["VLLM_SNAPSHOT_IGNORED_FD_MAP"] = str(artifact / "ignored-nvidia-fds.tsv")
        env["RIG_CRIU_IGNORED_FD_MAP"] = env["VLLM_SNAPSHOT_IGNORED_FD_MAP"]
        command = [
            *self._privileged(),
            "env",
            f"PATH={env['PATH']}",
            f"VLLM_SNAPSHOT_IGNORED_FD_MAP={env['VLLM_SNAPSHOT_IGNORED_FD_MAP']}",
            f"RIG_CRIU_IGNORED_FD_MAP={env['RIG_CRIU_IGNORED_FD_MAP']}",
            self.criu,
            action,
            "-L",
            str(self.plugin_dir),
            *arguments,
        ]
        self._run(command)

    def _record_child_log_size(self, artifact: Path) -> None:
        child_log = artifact / "child.log"
        size_path = artifact / "child.log.snapshot-size"
        payload = f"{child_log.stat().st_size}\n".encode()
        descriptor = os.open(
            size_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

    def _link_remap_names(self) -> set[str]:
        return {
            path.name
            for path in self.shm_dir.glob("link_remap.*")
            if path.name.removeprefix("link_remap.").isdigit()
        }

    def _capture_link_remaps(self, artifact: Path, names: set[str]) -> None:
        if not names:
            return
        destination_dir = artifact / "link-remaps"
        destination_dir.mkdir(mode=0o700)
        for name in sorted(names):
            source = self.shm_dir / name
            source_fd = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                metadata = os.fstat(source_fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                    raise SnapshotCreateError(f"invalid CRIU link remap: {source}")
                destination_fd = os.open(
                    destination_dir / name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                with os.fdopen(destination_fd, "wb") as destination:
                    with os.fdopen(os.dup(source_fd), "rb") as contents:
                        shutil.copyfileobj(contents, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
            finally:
                os.close(source_fd)
            source.unlink()

    def _rollback_link_remaps(self, created: dict[str, tuple[int, int]]) -> None:
        for name, identity in created.items():
            target = self.shm_dir / name
            try:
                metadata = target.lstat()
            except FileNotFoundError:
                continue
            if (metadata.st_dev, metadata.st_ino) == identity:
                target.unlink()

    def _restore_link_remaps(self, artifact: Path) -> dict[str, tuple[int, int]]:
        source_dir = artifact / "link-remaps"
        if not source_dir.exists():
            return {}
        created: dict[str, tuple[int, int]] = {}
        try:
            for source in sorted(source_dir.iterdir()):
                if (
                    not source.name.removeprefix("link_remap.").isdigit()
                    or source.is_symlink()
                    or not source.is_file()
                ):
                    raise SnapshotRestoreError(f"invalid saved link remap: {source}")
                payload = source.read_bytes()
                target = self.shm_dir / source.name
                try:
                    descriptor = os.open(
                        target,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                except FileExistsError as error:
                    try:
                        descriptor = os.open(
                            target,
                            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        )
                    except OSError:
                        raise SnapshotRestoreError(
                            f"conflicting CRIU link remap exists: {target}"
                        ) from error
                    with os.fdopen(descriptor, "rb") as existing:
                        metadata = os.fstat(existing.fileno())
                        if (
                            not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_uid != os.getuid()
                            or existing.read() != payload
                        ):
                            raise SnapshotRestoreError(
                                f"conflicting CRIU link remap exists: {target}"
                            ) from error
                    continue
                metadata = os.fstat(descriptor)
                created[source.name] = (metadata.st_dev, metadata.st_ino)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
        except BaseException:
            self._rollback_link_remaps(created)
            raise
        return created

    def _reset_child_log(self, artifact: Path) -> None:
        size_path = artifact / "child.log.snapshot-size"
        descriptor = os.open(
            size_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "r") as source:
            size = int(source.read().strip())
        if size < 0:
            raise SnapshotRestoreError("captured child log size is invalid")

        child_log = artifact / "child.log"
        descriptor = os.open(
            child_log,
            os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < size:
                raise SnapshotRestoreError("captured child log is invalid")
            os.ftruncate(descriptor, size)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def dump(self, workdir: Path, inventory: ProcessInventory) -> None:
        images = workdir / "images"
        images.mkdir(mode=0o700)
        fd_map = workdir / "ignored-nvidia-fds.tsv"
        fd_map.touch(mode=0o600)
        remaps_before = self._link_remap_names()
        try:
            self._criu(
                "dump",
                workdir,
                [
                    "--tree",
                    str(inventory.root_pid),
                    "--images-dir",
                    str(images),
                    "--log-file",
                    "dump.log",
                    *_COMMON_CRIU_OPTIONS,
                ],
            )
            self._capture_link_remaps(
                workdir,
                self._link_remap_names() - remaps_before,
            )
        except BaseException:
            for name in self._link_remap_names() - remaps_before:
                (self.shm_dir / name).unlink(missing_ok=True)
            raise
        self._record_child_log_size(workdir)

    def verify_dead(self, inventory: ProcessInventory) -> None:
        process = self._children.get(inventory.root_pid)
        if process is not None:
            process.wait(timeout=10)
        for pid in inventory.process_tree:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise SnapshotCreateError(f"process survived CRIU dump: {pid}")
        self._children.pop(inventory.root_pid, None)

    def _version(self, command: list[str]) -> str:
        try:
            result = self._run(command, timeout=10)
        except subprocess.CalledProcessError as error:
            return (error.stdout or error.stderr).splitlines()[0]
        return result.stdout.splitlines()[0] if result.stdout else "unknown"

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _plugin_identity(self) -> tuple[tuple[str, str], ...]:
        if self.plugin_dir is None:
            return ()
        return tuple(
            (f"snapshot_plugin_sha256:{path.name}", self._sha256(path))
            for path in sorted(self.plugin_dir.glob("*.so"))
        )

    def _torch_identity(self) -> tuple[str, str]:
        torch_version = importlib.metadata.version("torch")
        torch_spec = importlib.util.find_spec("torch")
        locations = torch_spec and torch_spec.submodule_search_locations
        if not locations:
            raise RuntimeError("installed torch package could not be located")
        version_path = Path(next(iter(locations))) / "version.py"
        syntax = ast.parse(version_path.read_text(), filename=str(version_path))
        cuda_runtime = ""
        for statement in syntax.body:
            target = None
            value = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                value = statement.value
            elif isinstance(statement, ast.AnnAssign):
                target = statement.target
                value = statement.value
            if isinstance(target, ast.Name) and target.id == "cuda" and value:
                parsed = ast.literal_eval(value)
                cuda_runtime = str(parsed or "")
                break
        return torch_version, cuda_runtime

    def _gpu_identity(self, gpu_uuid: str) -> tuple[str, str, str]:
        output = self._run(
            [
                self.nvidia_smi,
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
                f"--id={gpu_uuid}",
            ]
        ).stdout.strip()
        name, uuid, driver = (item.strip() for item in output.split(",", 2))
        return name, uuid, driver

    def _artifact_bytes(self, workdir: Path) -> int:
        return sum(path.stat().st_size for path in workdir.rglob("*") if path.is_file())

    def _environment_identity(self) -> tuple[tuple[str, str], ...]:
        prefixes = ("VLLM_", "CUDA_", "NCCL_", "TORCH_", "TRITON_")
        selected = tuple(
            sorted(
                (key, value)
                for key, value in os.environ.items()
                if key.startswith(prefixes) and key not in {"VLLM_SNAPSHOT_TIMEOUT_S"}
            )
        )
        return selected + self._plugin_identity()

    def make_manifest(
        self,
        args: argparse.Namespace,
        engine_argv: tuple[str, ...],
        inventory: ProcessInventory,
        oracle: Oracle,
        workdir: Path,
    ) -> SnapshotManifest:
        identity = self.current_identity(inventory.gpu_uuid)
        revision = str(getattr(args, "revision", None) or "")
        tokenizer_revision = str(getattr(args, "tokenizer_revision", None) or revision)
        source_model = str(getattr(args, "model_tag", None) or args.model)
        from vllm.config.model import get_served_model_name

        served_model_name = get_served_model_name(
            source_model, getattr(args, "served_model_name", None)
        )
        return SnapshotManifest(
            schema_version=1,
            boundary="post-engine-init-reloadable-state-released",
            created_at=datetime.now(timezone.utc).isoformat(),
            artifact_bytes=self._artifact_bytes(workdir),
            **identity.model_dump(),
            model=source_model,
            served_model_name=str(served_model_name),
            model_revision=revision,
            tokenizer_revision=tokenizer_revision,
            engine_argv=engine_argv,
            process_tree=inventory.process_tree,
            cuda_holders=inventory.cuda_holders,
            oracle_token_ids=oracle.token_ids,
            oracle_text=oracle.text,
            oracle_sampled_token_logprob=oracle.sampled_token_logprob,
        )

    def publish(self, workdir: Path, manifest: SnapshotManifest) -> None:
        write_manifest_atomic(workdir, manifest)
        _fsync_directory(workdir.parent)

    def current_identity(self, gpu_uuid: str) -> SnapshotRuntimeIdentity:
        gpu_name, gpu_uuid, driver_version = self._gpu_identity(gpu_uuid)
        torch_version, cuda_runtime = self._torch_identity()
        host_id = Path("/etc/machine-id").read_text().strip()
        return SnapshotRuntimeIdentity(
            vllm_version=importlib.metadata.version("vllm"),
            python_version=platform.python_version(),
            torch_version=torch_version,
            cuda_runtime=cuda_runtime,
            driver_version=driver_version,
            criu_version=self._version([self.criu, "--version"]),
            cuda_checkpoint_sha256=self._sha256(Path(self.cuda_checkpoint)),
            kernel_release=platform.release(),
            host_id=host_id,
            gpu_name=gpu_name,
            gpu_uuid=gpu_uuid,
            environment=self._environment_identity(),
        )

    def _read_restored_pid(self, pidfile: Path) -> int:
        try:
            payload = self._run([*self._privileged(), "cat", str(pidfile)]).stdout
        except Exception as error:
            raise SnapshotRestoreError("restored PID file is missing") from error
        match = re.fullmatch(r"([1-9][0-9]{0,9})\n?", payload)
        if match is None:
            raise SnapshotRestoreError("restored PID file is invalid")
        restored_pid = int(match.group(1))
        if restored_pid > 2**31 - 1:
            raise SnapshotRestoreError("restored PID file is invalid")
        return restored_pid

    def _process_state(self, pid: int) -> tuple[int, int, int, int]:
        payload = (Path("/proc") / str(pid) / "stat").read_text()
        command_end = payload.rfind(")")
        fields = payload[command_end + 2 :].split()
        if command_end < 0 or len(fields) < 20:
            raise ValueError(f"invalid process state for PID {pid}")
        return (int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19]))

    def _process_states(self) -> dict[int, tuple[int, int, int, int]]:
        states: dict[int, tuple[int, int, int, int]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                states[pid] = self._process_state(pid)
            except (OSError, ValueError):
                continue
        return states

    def _process_command(self, pid: int) -> tuple[str, ...]:
        payload = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        return tuple(
            os.fsdecode(argument) for argument in payload.split(b"\0") if argument
        )

    def _pidfd_open(self, pid: int) -> int:
        pidfd_open = getattr(os, "pidfd_open", None)
        if not callable(pidfd_open):
            raise SnapshotRestoreError("snapshot requires Linux pidfd support")
        return pidfd_open(pid)

    def _pidfd_exited(self, pidfd: int) -> bool:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        return any(self._pidfd_event_is_exit(event) for _fd, event in poller.poll(0))

    def _pidfd_event_is_exit(self, event: int) -> bool:
        if event & (select.POLLERR | select.POLLNVAL):
            raise SnapshotRestoreError("restored process pidfd poll failed")
        return bool(event & (select.POLLIN | select.POLLHUP))

    def _pin_restored_tree(
        self,
        artifact: Path,
        process_tree: tuple[int, ...],
        cuda_holders: tuple[int, ...],
    ) -> None:
        root_pid = process_tree[0]
        expected_pids = set(process_tree)
        expected_cuda_holders = set(cuda_holders)
        if not expected_cuda_holders or not expected_cuda_holders.issubset(
            expected_pids
        ):
            raise SnapshotRestoreError(
                "captured CUDA holders are outside the captured process tree"
            )
        if root_pid in self._restored_processes:
            raise SnapshotRestoreError("restored PID already has a pinned transaction")
        handles: list[tuple[int, int]] = []
        try:
            for pid in process_tree:
                handles.append((pid, self._pidfd_open(pid)))
            handles_by_pid = dict(handles)
            if any(self._pidfd_exited(pidfd) for _pid, pidfd in handles):
                raise SnapshotRestoreError(
                    "restored process exited during verification"
                )

            root_state = self._process_state(root_pid)
            if root_state[1:3] != (root_pid, root_pid):
                raise SnapshotRestoreError(
                    "restored root is not the session and process-group leader"
                )
            command = self._process_command(root_pid)
            release_file = str(artifact / "release.json")
            if not any(
                command[index : index + 2] == ("-m", "vllm.snapshot.server")
                for index in range(len(command) - 1)
            ) or not any(
                command[index : index + 2] == ("--release-file", release_file)
                for index in range(len(command) - 1)
            ):
                raise SnapshotRestoreError(
                    "restored root command does not match the snapshot barrier"
                )
            if self._process_state(root_pid) != root_state:
                raise SnapshotRestoreError("restored PID changed during verification")
            if self._pidfd_exited(handles[0][1]):
                raise SnapshotRestoreError("restored root exited during verification")

            first_states = self._process_states()
            live_session = {
                pid for pid, state in first_states.items() if state[2] == root_pid
            }
            if not expected_cuda_holders.issubset(live_session):
                raise SnapshotRestoreError(
                    "restored session is missing captured CUDA holders"
                )
            if live_session != expected_pids:
                raise SnapshotRestoreError(
                    "restored session does not match the captured process tree"
                )
            if first_states[root_pid] != root_state:
                raise SnapshotRestoreError("restored PID changed during verification")
            for pid in process_tree[1:]:
                state = first_states[pid]
                pidfd = handles_by_pid[pid]
                if self._process_state(pid) != state or self._pidfd_exited(pidfd):
                    raise SnapshotRestoreError(
                        "restored process tree changed during verification"
                    )

            second_states = self._process_states()
            second_session = {
                pid for pid, state in second_states.items() if state[2] == root_pid
            }
            if second_session != expected_pids or any(
                second_states[pid] != first_states[pid] for pid in process_tree
            ):
                raise SnapshotRestoreError(
                    "restored process tree changed during verification"
                )
            if any(self._pidfd_exited(pidfd) for _pid, pidfd in handles):
                raise SnapshotRestoreError(
                    "restored process exited during verification"
                )
        except BaseException as error:
            _close_pidfds(handles)
            if isinstance(error, SnapshotRestoreError):
                raise
            if not isinstance(error, Exception):
                raise
            raise SnapshotRestoreError(
                "restored process tree could not be pinned"
            ) from error
        self._restored_processes[root_pid] = tuple(handles)

    def _abort_failed_restore(
        self,
        artifact: Path,
        staged_remaps: dict[str, tuple[int, int]],
        primary: BaseException,
    ) -> NoReturn:
        secondary: list[str] = []
        try:
            _write_json_atomic(
                artifact / "release.json", {"release": False}, overwrite=True
            )
        except BaseException as error:
            secondary.append(f"abort marker failed: {_error_detail(error)}")
        try:
            self._rollback_link_remaps(staged_remaps)
        except BaseException as error:
            secondary.append(f"remap rollback failed: {_error_detail(error)}")
        if secondary:
            raise SnapshotRestoreError(
                f"snapshot restore failed: {_error_detail(primary)}; "
                + "; ".join(secondary)
            ) from primary
        raise primary

    def restore(self, artifact: Path, manifest: SnapshotManifest) -> int:
        # This is necessarily a best-effort TOCTOU check. CRIU restore and the
        # rollback below remain authoritative if a PID is claimed after it.
        for pid in manifest.process_tree:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                pass
            except Exception as error:
                raise SnapshotRestoreError(
                    f"captured PID availability probe failed: {pid}"
                ) from error
            raise SnapshotRestoreError(f"captured PID is already occupied: {pid}")
        expected_root_pid = manifest.process_tree[0]
        (artifact / "release.json").unlink(missing_ok=True)
        pidfile = artifact / "restored.pid"
        pidfile.unlink(missing_ok=True)
        self._reset_child_log(artifact)
        staged_remaps = self._restore_link_remaps(artifact)
        try:
            self._criu(
                "restore",
                artifact,
                [
                    "--images-dir",
                    str(artifact / "images"),
                    "--log-file",
                    "restore.log",
                    *_COMMON_CRIU_OPTIONS,
                    "--restore-detached",
                    "--pidfile",
                    str(pidfile),
                ],
            )
            restored_pid = self._read_restored_pid(pidfile)
            if restored_pid != expected_root_pid:
                raise SnapshotRestoreError(
                    "restored PID does not match the captured process tree"
                )
            self._pin_restored_tree(
                artifact, manifest.process_tree, manifest.cuda_holders
            )
        except BaseException as error:
            self._abort_failed_restore(artifact, staged_remaps, error)
        return restored_pid

    def release(self, artifact: Path, host: str | None, port: int) -> None:
        if not 1 <= port <= 65535:
            raise SnapshotRestoreError(f"snapshot restore port is invalid: {port}")
        family = socket.AF_INET6 if host and ":" in host else socket.AF_INET
        with socket.socket(family=family, type=socket.SOCK_STREAM) as listener_probe:
            listener_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener_probe.bind((host or "", port))
            except OSError as error:
                address = f"{host or '0.0.0.0'}:{port}"
                raise SnapshotRestoreError(
                    f"snapshot restore address is already in use: {address}"
                ) from error

        _write_json_atomic(
            artifact / "release.json",
            {"release": True, "host": host, "port": port},
            overwrite=True,
        )

    def _connect_host(self, host: str | None) -> str:
        if not host or host == "0.0.0.0":
            return "127.0.0.1"
        if host == "::":
            return "::1"
        return host

    def wait_listener(self, root_pid: int, host: str | None, port: int) -> None:
        deadline = time.monotonic() + self.timeout_s
        connect_host = self._connect_host(host)
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((connect_host, port), timeout=1):
                    return
            except OSError as error:
                handles = self._restored_processes.get(root_pid)
                if handles is None or self._pidfd_exited(handles[0][1]):
                    raise SnapshotRestoreError(
                        "restored process exited before binding HTTP"
                    ) from error
                time.sleep(0.1)
        raise SnapshotRestoreError("restored HTTP listener did not become ready")

    def request_oracle(
        self, host: str | None, port: int, manifest: SnapshotManifest
    ) -> Oracle:
        connect_host = self._connect_host(host)
        url_host = f"[{connect_host}]" if ":" in connect_host else connect_host
        payload = json.dumps(
            {
                "model": manifest.served_model_name,
                "prompt": "The capital of France is",
                "min_tokens": 1,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 0,
                "logprobs": 0,
                "return_token_ids": True,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://{url_host}:{port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
        choice = body["choices"][0]
        try:
            return Oracle(
                token_ids=tuple(choice["token_ids"]),
                text=choice["text"],
                sampled_token_logprob=choice["logprobs"]["token_logprobs"][0],
            )
        except ValueError as error:
            raise SnapshotRestoreError(
                "snapshot HTTP canary must return exactly one finite token logprob"
            ) from error
        except (IndexError, KeyError, TypeError) as error:
            raise SnapshotRestoreError(
                "snapshot HTTP canary response is missing sampled token logprob"
            ) from error

    def cleanup(self, root_pid: int) -> None:
        handles = self._restored_processes.get(root_pid)
        if handles is None:
            raise SnapshotRestoreError("no pinned restore transaction for cleanup")
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if not callable(pidfd_send_signal):
            raise SnapshotRestoreError("restored process cleanup is incomplete")
        try:
            for _pid, pidfd in handles:
                with suppress(ProcessLookupError):
                    pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
            time.sleep(min(0.5, self.timeout_s))
            for _pid, pidfd in handles:
                with suppress(ProcessLookupError):
                    pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
        except BaseException as error:
            raise SnapshotRestoreError(
                f"restored process cleanup is incomplete: {_error_detail(error)}"
            ) from error

        poller = select.poll()
        for _pid, pidfd in handles:
            poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        pending = {pidfd for _pid, pidfd in handles}
        deadline = time.monotonic() + self.timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SnapshotRestoreError("restored process cleanup is incomplete")
            for pidfd, event in poller.poll(max(1, int(remaining * 1000))):
                if not self._pidfd_event_is_exit(event):
                    continue
                pending.discard(pidfd)
                with suppress(KeyError, OSError):
                    poller.unregister(pidfd)
        self._restored_processes.pop(root_pid)
        _close_pidfds(handles)

    def complete_restore(self, root_pid: int) -> None:
        handles = self._restored_processes.pop(root_pid, None)
        if handles is None:
            raise SnapshotRestoreError("no pinned restore transaction to complete")
        _close_pidfds(handles)

    def abort_create(self, root_pid: int) -> None:
        process = self._children.get(root_pid)
        if process is None:
            return
        with suppress(ProcessLookupError):
            os.killpg(root_pid, signal.SIGKILL)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return
        self._children.pop(root_pid, None)
