# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from vllm.snapshot.protocol import SnapshotControlError, _read_message
from vllm.snapshot.providers import (
    CriuCudaSnapshotProvider,
    SnapshotProvider,
    make_snapshot_provider,
)
from vllm.snapshot.resources import parse_snapshot_resource_policy


class SnapshotState(str, Enum):
    READY = "READY"
    DRAINING = "DRAINING"
    PREPARING = "PREPARING"
    SNAPSHOTTING = "SNAPSHOTTING"
    HIBERNATED = "HIBERNATED"
    RESTORING = "RESTORING"
    ATTACHING = "ATTACHING"
    VERIFYING = "VERIFYING"
    FAILED = "FAILED"


@dataclass
class SnapshotStatus:
    state: SnapshotState = SnapshotState.READY
    generation: int = 0
    snapshot_id: str | None = None
    root_pid: int | None = None
    last_error: str | None = None
    operation_started_at: float | None = None
    operation_finished_at: float | None = None


@dataclass(frozen=True)
class SnapshotTicket:
    snapshot_id: str
    nonce: str
    generation: int
    config_hash: str
    root_pid: int
    marker_path: str


class SnapshotProcessManager(Protocol):
    def prepare_snapshot(self) -> int: ...

    def cancel_snapshot(self) -> None: ...

    def wait_for_snapshot_exit(self, timeout: float = 30) -> None: ...

    def adopt_restored_process(self, pid: int) -> None: ...

    def discard_restored_process(self) -> None: ...


_MAX_CONTROL_CONNECTIONS = 8
_CONTROL_REQUEST_TIMEOUT_SECONDS = 5.0


class EngineSnapshotManager:
    def __init__(
        self,
        socket_path: str,
        snapshot_root: str,
        root_pid: int,
        *,
        provider: str | SnapshotProvider = "fake",
        config_hash: str = "",
        snapshot_config: dict[str, Any] | None = None,
        resource_policy: str = "full",
        persistence: str = "durable",
        integrity: str = "optimistic",
        process_manager: SnapshotProcessManager | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.snapshot_root = Path(snapshot_root)
        self.status = SnapshotStatus(root_pid=root_pid)
        self.config_hash = config_hash
        self.provider = (
            provider
            if isinstance(provider, SnapshotProvider)
            else make_snapshot_provider(provider)
        )
        self.snapshot_config = _json_value(snapshot_config or {})
        self.resource_policy = parse_snapshot_resource_policy(resource_policy)
        self.persistence = self._parse_persistence(persistence)
        self.integrity = self._parse_integrity(integrity)
        self.compatibility = self._runtime_compatibility()
        self.configured_artifacts = self._configured_artifacts()
        self.process_manager = process_manager
        self._state_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._stop = threading.Event()
        self._request_slots = threading.BoundedSemaphore(_MAX_CONTROL_CONNECTIONS)
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._ticket: SnapshotTicket | None = None
        self._previous_snapshot_id: str | None = None

    def start(self) -> None:
        self.snapshot_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        entries = list(self.snapshot_root.iterdir())
        if entries:
            names = ", ".join(sorted(entry.name for entry in entries))
            raise RuntimeError(
                "engine snapshot directory is not empty; inspect or explicitly "
                f"remove its contents before cold start: {names}"
            )
        self.snapshot_root.chmod(0o700)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._server.listen(_MAX_CONTROL_CONNECTIONS)
        self._server.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._serve, name="engine-snapshot-control", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.socket_path.unlink(missing_ok=True)

    def snapshot_status(self) -> dict[str, Any]:
        with self._state_lock:
            return self._status_dict()

    def prepare_capture(self) -> dict[str, Any]:
        with self._state_lock:
            if self.status.state is not SnapshotState.READY:
                raise SnapshotControlError(
                    f"capture requires READY, got {self.status.state}",
                    code="invalid_state",
                )
            root_pid = self._require_live_root()
            self.status.state = SnapshotState.DRAINING
            self.status.operation_started_at = time.time()
            self.status.operation_finished_at = None
            self.status.last_error = None
            snapshot_id = uuid.uuid4().hex
            marker_path = self.snapshot_root / f".{snapshot_id}.detached.json"
            marker_path.unlink(missing_ok=True)
            self._ticket = SnapshotTicket(
                snapshot_id=snapshot_id,
                nonce=uuid.uuid4().hex,
                generation=self.status.generation + 1,
                config_hash=self.config_hash,
                root_pid=root_pid,
                marker_path=str(marker_path),
            )
            self._previous_snapshot_id = self.status.snapshot_id
            return self._status_dict() | {"ticket": asdict(self._ticket)}

    def abort_capture(self) -> dict[str, Any]:
        with self._state_lock:
            if self.status.state is not SnapshotState.DRAINING:
                raise SnapshotControlError(
                    f"abort requires DRAINING, got {self.status.state}",
                    code="invalid_state",
                )
            self.status.state = SnapshotState.READY
            self.status.operation_finished_at = time.time()
            self._clear_ticket()
            return self._status_dict()

    def rollback_capture(self) -> dict[str, Any]:
        with self._state_lock:
            if self.status.state is not SnapshotState.DRAINING:
                raise SnapshotControlError(
                    f"rollback requires DRAINING, got {self.status.state}",
                    code="invalid_state",
                )
            ticket = self._require_ticket()
            self.status.state = SnapshotState.ATTACHING
            self.status.last_error = "capture was canceled after EngineCore I/O detach"
            self.status.operation_finished_at = time.time()
            if not self._signal_snapshot_resume(ticket.root_pid):
                self.status.state = SnapshotState.FAILED
                self.status.root_pid = None
                raise RuntimeError("EngineCore root exited before snapshot I/O resume")
            return self._status_dict() | {
                "ticket": asdict(ticket),
            }

    def capture(self) -> dict[str, Any]:
        if not self._operation_lock.acquire(blocking=False):
            raise SnapshotControlError(
                "another snapshot operation is running", code="busy"
            )
        manager_started = time.monotonic()
        manager_timings: dict[str, float] = {}
        ticket: SnapshotTicket | None = None
        root_pid: int | None = None
        staging: Path | None = None
        previous_snapshot_id: str | None = None
        try:
            with self._state_lock:
                if self.status.state is not SnapshotState.DRAINING:
                    raise SnapshotControlError(
                        f"capture requires DRAINING, got {self.status.state}",
                        code="invalid_state",
                    )
                ticket = self._require_ticket()
                self.status.state = SnapshotState.PREPARING
            phase_started = time.monotonic()
            self._validate_detach_marker(ticket)
            manager_timings["detach_marker_seconds"] = time.monotonic() - phase_started
            phase_started = time.monotonic()
            root_pid = (
                self.process_manager.prepare_snapshot()
                if self.process_manager is not None
                else self._require_live_root()
            )
            manager_timings["process_prepare_seconds"] = (
                time.monotonic() - phase_started
            )
            snapshot_id = ticket.snapshot_id
            previous_snapshot_id = self._previous_snapshot_id
            with self._state_lock:
                self.status.snapshot_id = snapshot_id
                self.status.state = SnapshotState.SNAPSHOTTING
            staging = self.snapshot_root / f".{snapshot_id}.staging"
            committed = self.snapshot_root / snapshot_id
            result: dict[str, Any] | None = None
            try:
                phase_started = time.monotonic()
                result = self.provider.capture(root_pid, staging)
                manager_timings["provider_capture_seconds"] = (
                    time.monotonic() - phase_started
                )
                phase_started = time.monotonic()
                if (
                    result.get("source_exited", False)
                    and self.process_manager is not None
                ):
                    self.process_manager.wait_for_snapshot_exit()
                manager_timings["source_exit_wait_seconds"] = (
                    time.monotonic() - phase_started
                )
                (staging / "root.pid").write_text(f"{root_pid}\n")
                phase_started = time.monotonic()
                artifacts = self._artifact_inventory(staging, self.integrity)
                inventory_seconds = time.monotonic() - phase_started
                manager_timings["artifact_inventory_seconds"] = inventory_seconds
                if self.integrity == "strict":
                    manager_timings["artifact_sha256_seconds"] = inventory_seconds
                phase_started = time.monotonic()
                manifest = self._manifest(result, artifacts)
                manifest_path = staging / "manifest.json"
                manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
                manifest_path.write_bytes(manifest_bytes)
                (staging / "manifest.sha256").write_text(
                    hashlib.sha256(manifest_bytes).hexdigest() + "\n"
                )
                manager_timings["manifest_write_seconds"] = (
                    time.monotonic() - phase_started
                )
                phase_started = time.monotonic()
                self._sync_tree(staging)
                manager_timings["fsync_seconds"] = time.monotonic() - phase_started
                phase_started = time.monotonic()
                staging.rename(committed)
                manager_timings["rename_seconds"] = time.monotonic() - phase_started
                phase_started = time.monotonic()
                self._sync_directory(self.snapshot_root)
                manager_timings["parent_fsync_seconds"] = (
                    time.monotonic() - phase_started
                )
                phase_started = time.monotonic()
                if previous_snapshot_id is not None:
                    shutil.rmtree(self.snapshot_root / previous_snapshot_id)
                manager_timings["previous_snapshot_cleanup_seconds"] = (
                    time.monotonic() - phase_started
                )
                with self._state_lock:
                    self.status.state = SnapshotState.HIBERNATED
                    if result.get("source_exited", False):
                        self.status.root_pid = None
                    self.status.operation_finished_at = time.time()
                    self._remove_marker(ticket)
                    manager_timings["total_seconds"] = (
                        time.monotonic() - manager_started
                    )
                    return self._status_dict() | {
                        "manifest": manifest,
                        "manager_timings": manager_timings,
                    }
            except BaseException as exc:
                rollback_result: dict[str, Any] | None = None
                rollback_error: BaseException | None = None
                if result is not None and result.get("source_exited", False):
                    rollback_dir = committed if committed.exists() else staging
                    try:
                        rollback_result = self.provider.rollback_capture(rollback_dir)
                        if rollback_result is None:
                            raise RuntimeError(
                                "snapshot provider cannot roll back an exited source"
                            )
                        if self.process_manager is not None:
                            self.process_manager.adopt_restored_process(
                                int(rollback_result["root_pid"])
                            )
                    except BaseException as rollback_exc:
                        rollback_error = rollback_exc
                elif self.process_manager is not None:
                    self.process_manager.cancel_snapshot()
                resume_succeeded = False
                with self._state_lock:
                    error = str(exc)
                    if rollback_error is not None:
                        error += f"; capture rollback failed: {rollback_error}"
                    self.status.last_error = error
                    self.status.snapshot_id = previous_snapshot_id
                    self.status.operation_finished_at = time.time()
                    self._remove_marker(ticket)
                    if rollback_result is not None:
                        resume_pid = int(rollback_result["root_pid"])
                    else:
                        resume_pid = root_pid
                    if resume_pid is not None and self._signal_snapshot_resume(
                        resume_pid
                    ):
                        resume_succeeded = True
                        self.status.root_pid = resume_pid
                        self.status.state = SnapshotState.ATTACHING
                    else:
                        self.status.root_pid = None
                        self.status.state = SnapshotState.FAILED
                cleanup_error = None
                if rollback_result is not None or (
                    result is None and resume_succeeded
                ):
                    cleanup_error = self._remove_snapshot_trees(staging, committed)
                if cleanup_error is not None:
                    with self._state_lock:
                        self.status.last_error = (
                            f"{self.status.last_error}; "
                            f"snapshot cleanup failed: {cleanup_error}"
                        )
                    raise RuntimeError(
                        f"{exc}; snapshot cleanup failed: {cleanup_error}"
                    ) from exc
                raise
        except BaseException as exc:
            if ticket is not None and root_pid is None:
                with self._state_lock:
                    self.status.last_error = str(exc)
                    self.status.snapshot_id = self._previous_snapshot_id
                    self.status.operation_finished_at = time.time()
                    if self._signal_snapshot_resume(ticket.root_pid):
                        self.status.state = SnapshotState.ATTACHING
                    else:
                        self.status.root_pid = None
                        self.status.state = SnapshotState.FAILED
            raise
        finally:
            self._operation_lock.release()

    def restore(self) -> dict[str, Any]:
        if not self._operation_lock.acquire(blocking=False):
            raise SnapshotControlError(
                "another snapshot operation is running", code="busy"
            )
        manager_started = time.monotonic()
        manager_timings: dict[str, float] = {}
        try:
            with self._state_lock:
                if self.status.state is not SnapshotState.HIBERNATED:
                    raise SnapshotControlError(
                        f"restore requires HIBERNATED, got {self.status.state}",
                        code="invalid_state",
                    )
                self.status.state = SnapshotState.RESTORING
                self.status.operation_started_at = time.time()
                self.status.operation_finished_at = None
                snapshot_dir = self._current_snapshot_dir()
            result: dict[str, Any] | None = None
            adopted = False
            restore_attempted = False
            try:
                manager_timings.update(self._validate_manifest(snapshot_dir))
                ticket = self._require_ticket()
                phase_started = time.monotonic()
                self._mark_restore_attempt(snapshot_dir)
                manager_timings["restore_attempt_marker_seconds"] = (
                    time.monotonic() - phase_started
                )
                restore_attempted = True
                phase_started = time.monotonic()
                result = self.provider.restore(snapshot_dir)
                manager_timings["provider_restore_seconds"] = (
                    time.monotonic() - phase_started
                )
                phase_started = time.monotonic()
                if self.process_manager is not None:
                    self.process_manager.adopt_restored_process(int(result["root_pid"]))
                    adopted = True
                manager_timings["pidfd_adopt_seconds"] = (
                    time.monotonic() - phase_started
                )
                with self._state_lock:
                    self.status.state = SnapshotState.ATTACHING
                    self.status.root_pid = int(result["root_pid"])
                    phase_started = time.monotonic()
                    if not self._signal_snapshot_resume(self.status.root_pid):
                        raise RuntimeError("restored EngineCore root is not alive")
                    manager_timings["resume_signal_seconds"] = (
                        time.monotonic() - phase_started
                    )
                    manager_timings["total_seconds"] = (
                        time.monotonic() - manager_started
                    )
                    return self._status_dict() | {
                        "provider_result": result,
                        "ticket": asdict(ticket) | {"root_pid": self.status.root_pid},
                        "manager_timings": manager_timings,
                    }
            except BaseException as exc:
                cleanup_error = None
                if result is not None:
                    root_pid = int(result["root_pid"])
                    try:
                        if adopted and self.process_manager is not None:
                            self.process_manager.discard_restored_process()
                        elif self._pid_alive(root_pid):
                            self._discard_provider_process(root_pid)
                    except BaseException as cleanup_exc:
                        cleanup_error = cleanup_exc
                with self._state_lock:
                    self.status.last_error = str(exc)
                    if cleanup_error is not None:
                        self.status.last_error += (
                            f"; restored Engine cleanup failed: {cleanup_error}"
                        )
                    self.status.operation_finished_at = time.time()
                    self.status.state = (
                        SnapshotState.HIBERNATED
                        if cleanup_error is None
                        and (
                            self.provider.restore_is_retriable or not restore_attempted
                        )
                        else SnapshotState.FAILED
                    )
                    self.status.root_pid = (
                        int(result["root_pid"])
                        if cleanup_error is not None
                        and result is not None
                        and self._pid_alive(int(result["root_pid"]))
                        else None
                    )
                raise
        finally:
            self._operation_lock.release()

    def complete_restore(self) -> dict[str, Any]:
        with self._state_lock:
            if self.status.state is not SnapshotState.VERIFYING:
                raise SnapshotControlError(
                    f"complete restore requires VERIFYING, got {self.status.state}",
                    code="invalid_state",
                )
            self.status.state = SnapshotState.READY
            self.status.generation = self._require_ticket().generation
            self.status.operation_finished_at = time.time()
            self._clear_ticket()
            return self._status_dict()

    def confirm_attach(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            if self.status.state is not SnapshotState.ATTACHING:
                raise SnapshotControlError(
                    f"attach confirmation requires ATTACHING, got {self.status.state}",
                    code="invalid_state",
                )
            ticket = self._require_ticket()
            expected = asdict(ticket) | {"root_pid": self.status.root_pid}
            for field in (
                "snapshot_id",
                "nonce",
                "generation",
                "config_hash",
                "root_pid",
            ):
                if payload.get(field) != expected[field]:
                    raise SnapshotControlError(
                        f"attach confirmation {field} mismatch",
                        code="identity_mismatch",
                    )
            self.status.state = SnapshotState.VERIFYING
            return self._status_dict()

    def complete_capture_rollback(self) -> dict[str, Any]:
        with self._state_lock:
            if (
                self.status.state is not SnapshotState.VERIFYING
                or self.status.snapshot_id != self._previous_snapshot_id
            ):
                raise SnapshotControlError(
                    "capture rollback requires the previous snapshot in VERIFYING",
                    code="invalid_state",
                )
            self.status.state = SnapshotState.READY
            self.status.generation = self._require_ticket().generation
            self.status.operation_finished_at = time.time()
            self._clear_ticket()
            return self._status_dict()

    def fail_restore(self, error: str) -> dict[str, Any]:
        with self._state_lock:
            if self.status.state not in (
                SnapshotState.ATTACHING,
                SnapshotState.VERIFYING,
            ):
                raise SnapshotControlError(
                    f"fail restore requires VERIFYING, got {self.status.state}",
                    code="invalid_state",
                )
            cleanup_error = None
            root_pid = self.status.root_pid
            try:
                if self.process_manager is not None:
                    self.process_manager.discard_restored_process()
                elif root_pid is not None and self._pid_alive(root_pid):
                    self._discard_provider_process(root_pid)
            except BaseException as exc:
                cleanup_error = exc
            self.status.state = (
                SnapshotState.HIBERNATED
                if self.provider.restore_is_retriable and cleanup_error is None
                else SnapshotState.FAILED
            )
            self.status.root_pid = (
                root_pid
                if cleanup_error is not None
                and root_pid is not None
                and self._pid_alive(root_pid)
                else None
            )
            self.status.last_error = error
            if cleanup_error is not None:
                self.status.last_error += (
                    f"; restored Engine cleanup failed: {cleanup_error}"
                )
            self.status.operation_finished_at = time.time()
            return self._status_dict()

    def fail_capture(self, error: str) -> dict[str, Any]:
        with self._state_lock:
            if self.status.state not in (
                SnapshotState.DRAINING,
                SnapshotState.ATTACHING,
                SnapshotState.VERIFYING,
            ):
                raise SnapshotControlError(
                    "fail capture requires DRAINING, ATTACHING, or VERIFYING, "
                    f"got {self.status.state}",
                    code="invalid_state",
                )
            cleanup_error = None
            root_pid = self.status.root_pid
            try:
                if self.process_manager is not None:
                    self.process_manager.discard_restored_process()
                elif root_pid is not None and self._pid_alive(root_pid):
                    self._discard_provider_process(root_pid)
            except BaseException as exc:
                cleanup_error = exc
            self.status.state = SnapshotState.FAILED
            self.status.root_pid = (
                root_pid
                if cleanup_error is not None
                and root_pid is not None
                and self._pid_alive(root_pid)
                else None
            )
            self.status.last_error = error
            if cleanup_error is not None:
                self.status.last_error += f"; Engine cleanup failed: {cleanup_error}"
            self.status.operation_finished_at = time.time()
            self._clear_ticket()
            return self._status_dict()

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except (TimeoutError, OSError):
                continue
            if not self._request_slots.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(
                target=self._handle_connection_with_slot,
                args=(conn,),
                daemon=True,
                name="engine-snapshot-request",
            ).start()

    def _handle_connection_with_slot(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(_CONTROL_REQUEST_TIMEOUT_SECONDS)
            self._handle_connection(conn)
        finally:
            conn.close()
            self._request_slots.release()

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                request = _read_message(conn)
                command = request.get("command")
                if command == "status":
                    result = self.snapshot_status()
                elif command == "prepare_capture":
                    result = self.prepare_capture()
                elif command == "abort_capture":
                    result = self.abort_capture()
                elif command == "rollback_capture":
                    result = self.rollback_capture()
                elif command == "capture":
                    result = self.capture()
                elif command == "restore":
                    result = self.restore()
                elif command == "complete_restore":
                    result = self.complete_restore()
                elif command == "confirm_attach":
                    result = self.confirm_attach(dict(request.get("payload", {})))
                elif command == "complete_capture_rollback":
                    result = self.complete_capture_rollback()
                elif command == "fail_restore":
                    result = self.fail_restore(
                        str(request.get("payload", {}).get("error", "verify failed"))
                    )
                elif command == "fail_capture":
                    result = self.fail_capture(
                        str(request.get("payload", {}).get("error", "prepare failed"))
                    )
                else:
                    raise SnapshotControlError(
                        f"unknown snapshot command: {command}",
                        code="unknown_command",
                    )
                response = {"ok": True, "result": result}
            except SnapshotControlError as exc:
                response = {"ok": False, "code": exc.code, "error": str(exc)}
            except Exception as exc:
                response = {
                    "ok": False,
                    "code": "snapshot_error",
                    "error": str(exc),
                }
            conn.sendall(json.dumps(response).encode() + b"\n")

    def _status_dict(self) -> dict[str, Any]:
        return asdict(self.status) | {
            "state": self.status.state.value,
            "resource_policy": self.resource_policy.to_wire(),
            "persistence": self.persistence,
            "integrity": self.integrity,
        }

    def _manifest(
        self,
        provider_result: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        ticket = self._require_ticket()
        return {
            "format_version": 2,
            "snapshot_id": self.status.snapshot_id,
            "generation": ticket.generation,
            "nonce": ticket.nonce,
            "config_hash": self.config_hash,
            "snapshot_config": self.snapshot_config,
            "root_pid": self.status.root_pid,
            "created_at": time.time(),
            "resource_policy": self.resource_policy.to_wire(),
            "persistence": self.persistence,
            "integrity": self.integrity,
            "compatibility": self.compatibility,
            "configured_artifacts": self.configured_artifacts,
            "provider": provider_result,
            "artifacts": artifacts,
        }

    def _validate_manifest(self, snapshot_dir: Path) -> dict[str, float]:
        timings = {}
        phase_started = time.monotonic()
        manifest_path = snapshot_dir / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = (snapshot_dir / "manifest.sha256").read_text().strip()
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_digest:
            raise RuntimeError("snapshot manifest digest mismatch")
        manifest = json.loads(manifest_bytes)
        timings["manifest_digest_seconds"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        ticket = self._require_ticket()
        if manifest.get("format_version") != 2:
            raise RuntimeError("unsupported snapshot format")
        if manifest.get("snapshot_id") != self.status.snapshot_id:
            raise RuntimeError("snapshot id mismatch")
        if manifest.get("generation") != ticket.generation:
            raise RuntimeError("snapshot generation mismatch")
        if manifest.get("nonce") != ticket.nonce:
            raise RuntimeError("snapshot nonce mismatch")
        if manifest.get("root_pid") != ticket.root_pid:
            raise RuntimeError("snapshot root PID mismatch")
        if manifest.get("config_hash") != self.config_hash:
            raise RuntimeError("snapshot config hash mismatch")
        if manifest.get("snapshot_config") != self.snapshot_config:
            raise RuntimeError("snapshot config mismatch")
        if manifest.get("resource_policy") != self.resource_policy.to_wire():
            raise RuntimeError("snapshot resource policy mismatch")
        if manifest.get("persistence") != self.persistence:
            raise RuntimeError("snapshot persistence mode mismatch")
        if manifest.get("integrity") != self.integrity:
            raise RuntimeError("snapshot integrity mode mismatch")
        if manifest.get("compatibility") != self.compatibility:
            raise RuntimeError("snapshot runtime compatibility mismatch")
        if manifest.get("configured_artifacts") != self.configured_artifacts:
            raise RuntimeError("snapshot model or tokenizer artifact mismatch")
        timings["compatibility_validation_seconds"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        if manifest.get("artifacts") != self._artifact_inventory(
            snapshot_dir, self.integrity
        ):
            raise RuntimeError("snapshot artifact inventory mismatch")
        validation_seconds = time.monotonic() - phase_started
        timings["artifact_validation_seconds"] = validation_seconds
        if self.integrity == "strict":
            timings["artifact_sha256_seconds"] = validation_seconds
        return timings

    def _mark_restore_attempt(self, snapshot_dir: Path) -> None:
        if self.provider.restore_is_retriable:
            return
        marker_path = snapshot_dir / "restore-attempt.json"
        marker = {
            "attempted_at": time.time(),
            "snapshot_id": self.status.snapshot_id,
            "generation": self._require_ticket().generation,
        }
        try:
            descriptor = os.open(
                marker_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "snapshot restore was already attempted and is not retriable"
            ) from exc
        with os.fdopen(descriptor, "w") as marker_file:
            json.dump(marker, marker_file)
            marker_file.write("\n")
            marker_file.flush()
            if self.persistence == "durable":
                os.fsync(marker_file.fileno())
        self._sync_directory(snapshot_dir)

    @staticmethod
    def _parse_persistence(name: str) -> str:
        if name not in ("durable", "page_cache"):
            raise ValueError(f"unknown snapshot persistence mode: {name}")
        return name

    @staticmethod
    def _parse_integrity(name: str) -> str:
        if name not in ("optimistic", "strict"):
            raise ValueError(f"unknown snapshot integrity mode: {name}")
        return name

    def _runtime_compatibility(self) -> dict[str, Any]:
        import torch

        from vllm.platforms import current_platform
        from vllm.version import __version__ as vllm_version

        driver_path = Path("/proc/driver/nvidia/version")
        capability = current_platform.get_device_capability()
        gpu_uuid = (
            current_platform.get_device_uuid()
            if isinstance(self.provider, CriuCudaSnapshotProvider)
            else None
        )
        compute_capability = (
            {"major": capability.major, "minor": capability.minor}
            if capability is not None
            else None
        )
        return {
            "python": sys.version,
            "platform": {
                "machine": platform.machine(),
                "kernel": platform.release(),
                "libc": list(platform.libc_ver()),
            },
            "vllm_version": vllm_version,
            "torch_version": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "gpu_uuid": gpu_uuid,
            "compute_capability": compute_capability,
            "nvidia_driver": (
                driver_path.read_text(errors="replace").strip()
                if driver_path.exists()
                else None
            ),
            "provider": self.provider.runtime_identity(),
        }

    def _configured_artifacts(self) -> dict[str, Any]:
        identities: dict[str, Any] = {}
        path_identities: dict[str, dict[str, Any]] = {}
        for name in ("model", "tokenizer"):
            value = self.snapshot_config.get(name)
            if value is None:
                identities[name] = None
                continue
            path = Path(str(value))
            if not path.exists():
                identities[name] = {"locator": str(value)}
                continue
            resolved = str(path.resolve())
            if resolved not in path_identities:
                path_identities[resolved] = self._path_identity(path, self.integrity)
            identities[name] = path_identities[resolved]
        return identities

    @classmethod
    def _path_identity(cls, path: Path, integrity: str) -> dict[str, Any]:
        if path.is_file():
            files = [path]
            root = path.parent
        elif path.is_dir():
            files = sorted(
                entry
                for entry in path.rglob("*")
                if entry.is_file() and ".git" not in entry.relative_to(path).parts
            )
            root = path
        else:
            raise RuntimeError(
                f"configured artifact is not a file or directory: {path}"
            )
        return {"root": str(path.resolve())} | cls._file_inventory(
            root, files, integrity
        )

    @classmethod
    def _artifact_inventory(cls, root: Path, integrity: str) -> dict[str, Any]:
        excluded = {"manifest.json", "manifest.sha256", "restore-attempt.json"}
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and str(path.relative_to(root)) not in excluded
            and "restore-work-" not in str(path.relative_to(root))
        )
        return cls._file_inventory(root, files, integrity)

    @classmethod
    def _file_inventory(
        cls, root: Path, files: list[Path], integrity: str
    ) -> dict[str, Any]:
        strict = cls._parse_integrity(integrity) == "strict"
        entries: list[dict[str, Any]] = []
        for path in files:
            entry = {
                "path": str(path.relative_to(root)),
                "type": "symlink" if path.is_symlink() else "file",
                "size": path.stat().st_size,
            }
            if strict:
                entry["sha256"] = cls._file_digest(path)
            entries.append(entry)
        inventory = {
            "files": entries,
            "total_bytes": sum(entry["size"] for entry in entries),
        }
        if strict:
            inventory["sha256"] = config_digest(entries)
        return inventory

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _current_snapshot_dir(self) -> Path:
        if self.status.snapshot_id is None:
            raise RuntimeError("snapshot id is missing")
        return self.snapshot_root / self.status.snapshot_id

    def _require_ticket(self) -> SnapshotTicket:
        if self._ticket is None:
            raise RuntimeError("snapshot ticket is missing")
        return self._ticket

    def _validate_detach_marker(self, ticket: SnapshotTicket) -> None:
        deadline = time.monotonic() + 30
        marker_path = Path(ticket.marker_path)
        while True:
            try:
                marker = json.loads(marker_path.read_text())
            except FileNotFoundError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for EngineCore I/O detach"
                    ) from exc
                time.sleep(0.05)
                continue
            expected = {
                "pid": ticket.root_pid,
                "nonce": ticket.nonce,
                "generation": ticket.generation,
                "snapshot_id": ticket.snapshot_id,
                "config_hash": ticket.config_hash,
                "state": "detached",
            }
            if marker != expected:
                raise RuntimeError("EngineCore I/O detach marker mismatch")
            return

    @staticmethod
    def _remove_marker(ticket: SnapshotTicket) -> None:
        Path(ticket.marker_path).unlink(missing_ok=True)

    def _clear_ticket(self) -> None:
        if self._ticket is not None:
            self._remove_marker(self._ticket)
        self._ticket = None
        self._previous_snapshot_id = None

    def _require_live_root(self) -> int:
        root_pid = self.status.root_pid
        if root_pid is None or not self._pid_alive(root_pid):
            raise RuntimeError("EngineCore root is not alive")
        return root_pid

    def _signal_snapshot_resume(self, root_pid: int) -> bool:
        if not self._pid_alive(root_pid):
            return False
        try:
            os.kill(root_pid, signal.SIGUSR2)
        except ProcessLookupError:
            return False
        return True

    @staticmethod
    def _remove_snapshot_trees(*paths: Path) -> OSError | None:
        cleanup_errors = []
        for path in paths:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                cleanup_errors.append(f"{path}: {exc}")
        if cleanup_errors:
            return OSError("; ".join(cleanup_errors))
        return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            stat = (Path("/proc") / str(pid) / "stat").read_text().split()
        except ProcessLookupError:
            return False
        except FileNotFoundError:
            return False
        return len(stat) > 2 and stat[2] != "Z"

    def _discard_provider_process(self, root_pid: int) -> None:
        self.provider.discard_restored(root_pid)
        deadline = time.monotonic() + 5
        while self._pid_alive(root_pid):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"EngineCore process tree rooted at PID {root_pid} did not exit"
                )
            time.sleep(0.05)

    def _sync_tree(self, path: Path) -> None:
        if self.persistence != "durable":
            return
        for root, _, files in os.walk(path):
            root_path = Path(root)
            for name in files:
                entry = root_path / name
                if entry.is_file():
                    with entry.open("rb") as file:
                        os.fsync(file.fileno())
        for root, directories, _ in os.walk(path, topdown=False):
            root_path = Path(root)
            for name in directories:
                directory = root_path / name
                if not directory.is_symlink():
                    self._sync_directory(directory)
        self._sync_directory(path)

    def _sync_directory(self, path: Path) -> None:
        if self.persistence != "durable":
            return
        dir_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def config_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
