# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import ast
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from vllm_cli.snapshot import snapshot_environment
from vllm_cli.snapshot.manifest import (
    SnapshotCompatibilityError,
    SnapshotManifest,
    SocketIdentity,
    validate_artifact_root,
    validate_identity,
    write_manifest_atomic,
)
from vllm_cli.snapshot.types import Oracle


class SnapshotCreateError(RuntimeError):
    """Snapshot creation did not produce a complete artifact."""


class SnapshotRestoreError(RuntimeError):
    """Snapshot restore failed and the restored tree was cleaned up."""


@dataclass(frozen=True)
class ProcessInventory:
    root_pid: int
    process_tree: tuple[int, ...]
    cuda_holders: tuple[int, ...]
    sockets: tuple[SocketIdentity, ...]


@dataclass(frozen=True)
class _TcpSocketRecord:
    family: str
    local_raw: str
    remote_raw: str
    inode: int


def _format_tcp_endpoint(family: str, raw: str) -> str:
    address_hex, port_hex = raw.rsplit(":", 1)
    packed = bytes.fromhex(address_hex)
    if family == "AF_INET":
        address = socket.inet_ntop(socket.AF_INET, packed[::-1])
        return f"{address}:{int(port_hex, 16)}"
    if family == "AF_INET6":
        packed = b"".join(
            packed[offset : offset + 4][::-1] for offset in range(0, 16, 4)
        )
        address = socket.inet_ntop(socket.AF_INET6, packed)
        return f"[{address}]:{int(port_hex, 16)}"
    raise SnapshotCreateError(f"unsupported TCP address family: {family}")


def _validate_tcp_connections(
    records: tuple[_TcpSocketRecord, ...], owned_inodes: set[int]
) -> tuple[SocketIdentity, ...]:
    owned = tuple(record for record in records if record.inode in owned_inodes)
    endpoints = {
        (record.family, record.local_raw, record.remote_raw) for record in owned
    }
    external = tuple(
        record
        for record in owned
        if (record.family, record.remote_raw, record.local_raw) not in endpoints
    )
    if external:
        details = ", ".join(
            f"{_format_tcp_endpoint(record.family, record.local_raw)} to "
            f"{_format_tcp_endpoint(record.family, record.remote_raw)}"
            for record in external
        )
        raise SnapshotCreateError(
            f"snapshot tree has an external established TCP connection: {details}"
        )
    return tuple(
        sorted(
            (
                SocketIdentity(
                    family=record.family,
                    socket_type="SOCK_STREAM",
                    local_address=_format_tcp_endpoint(record.family, record.local_raw),
                    remote_address=_format_tcp_endpoint(
                        record.family, record.remote_raw
                    ),
                    state="ESTABLISHED",
                )
                for record in owned
            ),
            key=lambda identity: (
                identity.family,
                identity.local_address,
                identity.remote_address or "",
            ),
        )
    )


class SnapshotTools(Protocol):
    def preflight(self, action: str, artifact: Path) -> None: ...

    def launch_child(self, workdir: Path, engine_argv: tuple[str, ...]) -> int: ...

    def wait_ready(self, workdir: Path, root_pid: int) -> Oracle: ...

    def inventory(self, root_pid: int) -> ProcessInventory: ...

    def dump(self, workdir: Path, inventory: ProcessInventory) -> None: ...

    def verify_dead(self, inventory: ProcessInventory) -> None: ...

    def make_manifest(
        self,
        args: argparse.Namespace,
        engine_argv: tuple[str, ...],
        inventory: ProcessInventory,
        oracle: Oracle,
        workdir: Path,
    ) -> SnapshotManifest: ...

    def publish(
        self, workdir: Path, target: Path, manifest: SnapshotManifest
    ) -> None: ...

    def current_identity(self, manifest: SnapshotManifest) -> SnapshotManifest: ...

    def restore(self, artifact: Path) -> int: ...

    def release(self, artifact: Path, host: str | None, port: int) -> None: ...

    def wait_listener(self, root_pid: int, host: str | None, port: int) -> None: ...

    def request_oracle(
        self, host: str | None, port: int, manifest: SnapshotManifest
    ) -> Oracle: ...

    def cleanup(self, root_pid: int, manifest: SnapshotManifest) -> None: ...


def _remove_snapshot_option(argv: tuple[str, ...]) -> tuple[str, ...]:
    remaining: list[str] = []
    iterator = iter(argv)
    for item in iterator:
        if item == "--snapshot-dir":
            next(iterator, None)
        elif item.startswith("--snapshot-dir="):
            continue
        else:
            remaining.append(item)
    return tuple(remaining)


def _current_engine_argv(args: argparse.Namespace) -> tuple[str, ...]:
    process_argv = tuple(sys.argv[1:])
    if len(process_argv) >= 2 and process_argv[:2] == ("snapshot", "create"):
        return _remove_snapshot_option(process_argv[2:])
    model = getattr(args, "model_tag", None) or getattr(args, "model", None)
    if not model:
        raise SnapshotCreateError("snapshot create requires a model argument")
    return (str(model),)


def create_snapshot(
    args: argparse.Namespace,
    *,
    engine_argv: tuple[str, ...] | None = None,
    tools: SnapshotTools | None = None,
) -> None:
    target = Path(args.snapshot_dir).absolute()
    toolset = tools or LocalSnapshotTools()
    toolset.preflight("create", target)
    if target.exists() or target.is_symlink():
        raise SnapshotCreateError(f"snapshot target already exists: {target}")
    validate_artifact_root(target, creating=True)
    target.parent.mkdir(parents=False, exist_ok=True)
    target.mkdir(mode=0o700)
    published = False
    root_pid: int | None = None
    inventory: ProcessInventory | None = None
    try:
        child_argv = engine_argv or _current_engine_argv(args)
        root_pid = toolset.launch_child(target, child_argv)
        oracle = toolset.wait_ready(target, root_pid)
        inventory = toolset.inventory(root_pid)
        toolset.dump(target, inventory)
        toolset.verify_dead(inventory)
        manifest = toolset.make_manifest(args, child_argv, inventory, oracle, target)
        if not manifest.complete:
            raise SnapshotCreateError("snapshot manifest is incomplete")
        toolset.publish(target, target, manifest)
        published = True
    except BaseException:
        if root_pid is not None and hasattr(toolset, "abort_create"):
            toolset.abort_create(root_pid, inventory)  # type: ignore[attr-defined]
        raise
    finally:
        if not published and target.exists():
            shutil.rmtree(target)


def restore_snapshot(
    args: argparse.Namespace, *, tools: SnapshotTools | None = None
) -> None:
    artifact = Path(args.snapshot_dir).absolute()
    toolset = tools or LocalSnapshotTools()
    toolset.preflight("restore", artifact)
    from vllm_cli.snapshot.manifest import read_manifest

    manifest = read_manifest(artifact)
    if not manifest.complete:
        raise SnapshotCompatibilityError("snapshot manifest is incomplete")
    current = toolset.current_identity(manifest)
    validate_identity(manifest, current)

    root_pid: int | None = None
    try:
        root_pid = toolset.restore(artifact)
        toolset.release(artifact, args.host, args.port)
        toolset.wait_listener(root_pid, args.host, args.port)
        actual = toolset.request_oracle(args.host, args.port, manifest)
        expected = Oracle(
            token_ids=manifest.oracle_token_ids,
            text=manifest.oracle_text,
        )
        if actual != expected:
            raise SnapshotRestoreError(
                f"snapshot oracle mismatch: expected={expected!r}, actual={actual!r}"
            )
    except BaseException as error:
        if root_pid is not None:
            toolset.cleanup(root_pid, manifest)
        if isinstance(error, SnapshotRestoreError):
            raise
        raise SnapshotRestoreError(str(error)) from error


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

    def _privileged(self) -> list[str]:
        return [] if os.geteuid() == 0 else ["sudo", "-n"]

    def _run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout or self.timeout_s,
        )

    def preflight(self, action: str, artifact: Path) -> None:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("snapshot requires Linux x86_64")
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

    def launch_child(self, workdir: Path, engine_argv: tuple[str, ...]) -> int:
        log_file = (workdir / "child.log").open("wb")
        command = [
            sys.executable,
            "-m",
            "vllm.snapshot.server",
            "--ready-file",
            "ready.json",
            "--release-file",
            "release.json",
            "--release-timeout-s",
            str(self.timeout_s),
            "--",
            *engine_argv,
        ]
        process = subprocess.Popen(
            command,
            cwd=workdir,
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
                )
            process = self._children.get(root_pid)
            if process is not None and process.poll() is not None:
                self._children.pop(root_pid)
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

    def _cuda_pids(self) -> tuple[int, ...]:
        output = self._run(
            [
                self.nvidia_smi,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        ).stdout
        return tuple(
            sorted({int(line.strip()) for line in output.splitlines() if line.strip()})
        )

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

    def _socket_inodes(self, process_tree: tuple[int, ...]) -> set[int]:
        inodes: set[int] = set()
        for pid in process_tree:
            for target in self._descriptor_targets(pid):
                if target.startswith("socket:[") and target.endswith("]"):
                    inodes.add(int(target[len("socket:[") : -1]))
        return inodes

    def _io_uring_pids(self, process_tree: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            pid
            for pid in process_tree
            if "anon_inode:[io_uring]" in self._descriptor_targets(pid)
        )

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
        cuda_holders = tuple(pid for pid in self._cuda_pids() if pid in process_tree)
        if not cuda_holders:
            raise SnapshotCreateError("snapshot tree has no CUDA-holding process")
        io_uring_pids = self._io_uring_pids(process_tree)
        if io_uring_pids:
            raise SnapshotCreateError(
                "snapshot tree owns io_uring state that CRIU cannot dump; set "
                "kernel.io_uring_disabled=1 before starting an unprivileged "
                "vLLM process, or use =2 to disable it host-wide "
                f"(pids: {', '.join(map(str, io_uring_pids))})"
            )
        sockets = _validate_tcp_connections(
            self._tcp_records(), self._socket_inodes(process_tree)
        )
        return ProcessInventory(
            root_pid=root_pid,
            process_tree=process_tree,
            cuda_holders=cuda_holders,
            sockets=sockets,
        )

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

    def _restore_link_remaps(self, artifact: Path) -> None:
        source_dir = artifact / "link-remaps"
        if not source_dir.exists():
            return
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
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except FileExistsError as error:
                if target.is_symlink() or target.read_bytes() != payload:
                    raise SnapshotRestoreError(
                        f"conflicting CRIU link remap exists: {target}"
                    ) from error
                continue
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())

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
                    "--shell-job",
                    "--ext-unix-sk",
                    "--tcp-established",
                    "--link-remap",
                ],
            )
        except BaseException:
            for name in self._link_remap_names() - remaps_before:
                (self.shm_dir / name).unlink(missing_ok=True)
            raise
        self._capture_link_remaps(
            workdir,
            self._link_remap_names() - remaps_before,
        )
        self._record_child_log_size(workdir)

    def verify_dead(self, inventory: ProcessInventory) -> None:
        process = self._children.pop(inventory.root_pid, None)
        if process is not None:
            process.wait(timeout=10)
        for pid in inventory.process_tree:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise SnapshotCreateError(f"process survived CRIU dump: {pid}")

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

    def _source_revision(self) -> str:
        vllm_spec = importlib.util.find_spec("vllm")
        if vllm_spec is None or vllm_spec.origin is None:
            return self._binary_revision()
        source_root = Path(vllm_spec.origin).resolve().parents[1]
        try:
            dirty = self._run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                timeout=10,
            ).stdout.strip()
            if dirty:
                raise SnapshotCompatibilityError(
                    "snapshot source is an editable dirty checkout"
                )
            return self._run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return self._binary_revision()

    def _binary_revision(self) -> str:
        return importlib.metadata.version("vllm")

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

    def _gpu_identity(self) -> tuple[str, str, str]:
        output = self._run(
            [
                self.nvidia_smi,
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
                "--id=0",
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
                for key, value in snapshot_environment().items()
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
        gpu_name, gpu_uuid, driver_version = self._gpu_identity()
        torch_version, cuda_runtime = self._torch_identity()
        host_id_path = Path("/etc/machine-id")
        host_id = host_id_path.read_text().strip()
        revision = str(getattr(args, "revision", None) or "")
        tokenizer_revision = str(getattr(args, "tokenizer_revision", None) or revision)
        source_revision = self._source_revision()
        return SnapshotManifest(
            schema_version=1,
            boundary="post-engine-init-pre-http-bind",
            complete=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            artifact_bytes=self._artifact_bytes(workdir),
            source_revision=source_revision,
            binary_revision=self._binary_revision(),
            python_version=platform.python_version(),
            torch_version=torch_version,
            cuda_runtime=cuda_runtime,
            driver_version=driver_version,
            criu_version=self._version([self.criu, "--version"]),
            cuda_checkpoint_version=self._sha256(Path(self.cuda_checkpoint)),
            kernel_release=platform.release(),
            host_id=host_id,
            gpu_name=gpu_name,
            gpu_uuid=gpu_uuid,
            model=str(getattr(args, "model_tag", None) or args.model),
            model_revision=revision,
            tokenizer_revision=tokenizer_revision,
            engine_args=(("argv", json.dumps(engine_argv)),),
            environment=self._environment_identity(),
            process_tree=inventory.process_tree,
            cuda_holders=inventory.cuda_holders,
            socket_inventory=inventory.sockets,
            oracle_token_ids=oracle.token_ids,
            oracle_text=oracle.text,
        )

    def publish(self, workdir: Path, target: Path, manifest: SnapshotManifest) -> None:
        if workdir != target:
            raise SnapshotCreateError("snapshot work directory changed before publish")
        write_manifest_atomic(workdir, manifest)
        parent_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def current_identity(self, manifest: SnapshotManifest) -> SnapshotManifest:
        gpu_name, gpu_uuid, driver_version = self._gpu_identity()
        torch_version, cuda_runtime = self._torch_identity()
        host_id = Path("/etc/machine-id").read_text().strip()
        source_revision = self._source_revision()
        return dataclasses.replace(
            manifest,
            source_revision=source_revision,
            binary_revision=self._binary_revision(),
            python_version=platform.python_version(),
            torch_version=torch_version,
            cuda_runtime=cuda_runtime,
            driver_version=driver_version,
            criu_version=self._version([self.criu, "--version"]),
            cuda_checkpoint_version=self._sha256(Path(self.cuda_checkpoint)),
            kernel_release=platform.release(),
            host_id=host_id,
            gpu_name=gpu_name,
            gpu_uuid=gpu_uuid,
            environment=self._environment_identity(),
        )

    def restore(self, artifact: Path) -> int:
        release = artifact / "release.json"
        release.unlink(missing_ok=True)
        pidfile = artifact / "restored.pid"
        pidfile.unlink(missing_ok=True)
        self._reset_child_log(artifact)
        self._restore_link_remaps(artifact)
        self._criu(
            "restore",
            artifact,
            [
                "--images-dir",
                str(artifact / "images"),
                "--log-file",
                "restore.log",
                "--shell-job",
                "--ext-unix-sk",
                "--tcp-established",
                "--link-remap",
                "--restore-detached",
                "--pidfile",
                str(pidfile),
            ],
        )
        return int(self._run([*self._privileged(), "cat", str(pidfile)]).stdout)

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

        path = artifact / "release.json"
        temporary = artifact / "release.json.tmp"
        payload = json.dumps(
            {"release": True, "host": host, "port": port},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

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
            except OSError:
                try:
                    os.kill(root_pid, 0)
                except ProcessLookupError as error:
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
                "model": manifest.model,
                "prompt": "The capital of France is",
                "max_tokens": 1,
                "temperature": 0,
                "seed": 0,
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
        return Oracle(token_ids=tuple(choice["token_ids"]), text=choice["text"])

    def cleanup(self, root_pid: int, manifest: SnapshotManifest) -> None:
        with suppress(ProcessLookupError):
            os.killpg(root_pid, signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(root_pid, signal.SIGKILL)
        remaining = set(self._cuda_pids()).intersection(manifest.process_tree)
        for pid in remaining:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        still_present = set(self._cuda_pids()).intersection(manifest.process_tree)
        if still_present:
            raise SnapshotRestoreError(
                f"CUDA processes survived cleanup: {sorted(still_present)}"
            )

    def abort_create(self, root_pid: int, _inventory: ProcessInventory | None) -> None:
        with suppress(ProcessLookupError):
            os.killpg(root_pid, signal.SIGKILL)
        process = self._children.pop(root_pid, None)
        if process is not None:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
