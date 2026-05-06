# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone writer thread pool for the filesystem capture consumer.

Accepts raw byte payloads from the capture path and persists them to a
POSIX filesystem using a columnar layout.  Driven by
:class:`vllm.v1.capture.consumers.filesystem.consumer.FilesystemConsumer`.

Design highlights:

- One ``queue.Queue`` per worker thread, partitioned by
  ``hash(request_id) % num_threads``. This preserves append ordering
  for multi-step captures sharing a ``(request_id, layer, hook)`` key
  without any cross-thread locks.
- Per-thread LRU FD cache (``OrderedDict``) so the same ``.bin.tmp``
  file descriptor is reused across multi-step ``WriteTask``s without
  reopening. Evicting a key ``fsync``s and ``close``s the fd.
- Atomic publish: writes land on ``<path>.tmp`` with an explicit
  ``fsync`` before ``os.replace`` promotes them to their final name.
- Structured ``WriteError``s with errno, path, and ``(request_id,
  layer, hook)`` so partial failures can be surfaced back to callers.
- Graceful shutdown with a bounded drain: in-flight tasks finish,
  remaining tasks are marked ``error`` in the result map so the
  engine can propagate ``partial_error`` to the owning requests.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import pathlib
import queue
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

CaptureKey = tuple[str, int, str]
CollisionPolicy = Literal["overwrite", "error", "suffix"]


class WriteError(Exception):
    """Typed error surfaced by the writer.

    Carries enough structured context that the caller can attribute
    the failure back to the owning ``(request_id, layer, hook)``
    capture and log a human-readable reason.
    """

    def __init__(
        self,
        message: str,
        *,
        path: pathlib.Path | None = None,
        key: CaptureKey | None = None,
        errno_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.key = key
        self.errno_code = errno_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "path": str(self.path) if self.path is not None else None,
            "key": list(self.key) if self.key is not None else None,
            "errno": self.errno_code,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"WriteError(message={self.message!r}, path={self.path!r}, "
            f"key={self.key!r}, errno={self.errno_code!r})"
        )


@dataclass
class WriteTask:
    """Append raw bytes to a capture's ``.bin.tmp`` file.

    The writer opens ``path`` lazily and caches the fd in the
    per-thread LRU cache keyed by ``key``. Multi-step captures emit
    several ``WriteTask``s with ``append=True`` for the same key in
    submission order.
    """

    path: pathlib.Path
    payload: bytes
    append: bool
    key: CaptureKey


@dataclass
class FinalizeTask:
    """Finalize a capture: fsync + rename the bin, then emit the sidecar.

    ``sidecar_payload`` is a plain dict; the writer serializes it with
    ``json.dumps``. If the caller passes a non-serializable value the
    writer raises ``WriteError`` synchronously inside ``submit`` so
    the bad input never reaches a worker thread.
    """

    bin_path: pathlib.Path
    sidecar_path: pathlib.Path
    sidecar_payload: dict[str, Any]
    key: CaptureKey


@dataclass
class WriteResult:
    """Outcome for a single ``(request_id, layer, hook)`` capture."""

    key: CaptureKey
    status: Literal["pending", "ok", "error"] = "pending"
    error: WriteError | None = None
    # Final on-disk paths once the capture has been successfully
    # finalized. Populated by ``FinalizeTask`` handling. When the
    # ``suffix`` collision policy rewrites the path, the actual
    # resolved paths appear here.
    bin_path: pathlib.Path | None = None
    sidecar_path: pathlib.Path | None = None


# Sentinel used to signal worker threads to drain and exit.
_SHUTDOWN = object()


@dataclass
class _PartitionState:
    """Per-thread mutable state held inside the worker loop.

    Kept outside ``ActivationWriter`` so that each thread's FD cache
    is lexically scoped to that thread and never touched by anyone
    else. No locking required.
    """

    fd_cache: OrderedDict[CaptureKey, int] = field(default_factory=OrderedDict)
    cache_capacity: int = 256


class ActivationWriter:
    """Thread pool that persists captured activations to disk.

    Example usage (phase 4 will wire this into ``gpu_model_runner``)::

        writer = ActivationWriter(pathlib.Path("/mnt/nas/activations"))
        writer.submit(WriteTask(path=..., payload=..., append=False, key=key))
        writer.submit(FinalizeTask(bin_path=..., sidecar_path=..., ...))
        writer.shutdown()
    """

    def __init__(
        self,
        root: pathlib.Path,
        *,
        num_threads: int = 4,
        queue_size: int = 1024,
        timeout_seconds: float = 180.0,
        on_collision: CollisionPolicy = "overwrite",
        fd_cache_size: int = 256,
    ) -> None:
        if num_threads <= 0:
            raise ValueError(f"num_threads must be >= 1, got {num_threads}")
        if queue_size <= 0:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        if fd_cache_size <= 0:
            raise ValueError(f"fd_cache_size must be >= 1, got {fd_cache_size}")
        if on_collision not in ("overwrite", "error", "suffix"):
            raise ValueError(
                f"on_collision must be one of 'overwrite' / 'error' / "
                f"'suffix', got {on_collision!r}"
            )

        self.root = pathlib.Path(root)
        self.num_threads = num_threads
        self.queue_size = queue_size
        self.timeout_seconds = timeout_seconds
        self.on_collision: CollisionPolicy = on_collision
        self.fd_cache_size = fd_cache_size

        self._queues: list[queue.Queue[Any]] = [
            queue.Queue(maxsize=queue_size) for _ in range(num_threads)
        ]
        self._threads: list[threading.Thread] = []
        self._results: dict[CaptureKey, WriteResult] = {}
        self._results_lock = threading.Lock()
        self._shutdown_flag = threading.Event()
        self._shutdown_lock = threading.Lock()
        # Callbacks fired on every transition to terminal status.
        # Tests use this to observe events without polling.
        self._status_callbacks: list[Callable[[WriteResult], None]] = []

        for idx in range(num_threads):
            partition = _PartitionState(cache_capacity=fd_cache_size)
            thread = threading.Thread(
                target=self._worker_loop,
                args=(idx, self._queues[idx], partition),
                name=f"ActivationWriter-{idx}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

        self._finalizer = weakref.finalize(self, _finalize_writer, self)

    # ------------------------------------------------------------------
    # Public API

    def submit(self, task: WriteTask | FinalizeTask) -> None:
        """Enqueue ``task`` on the partition owning ``task.key``.

        Raises:
            WriteError: if the writer is already shutting down, if the
                target queue is full longer than ``timeout_seconds``,
                or (for ``FinalizeTask``) if ``sidecar_payload`` is
                not JSON-serializable.
        """
        if self._shutdown_flag.is_set():
            raise WriteError(
                "writer is shutting down; cannot accept new tasks",
                key=task.key,
            )

        if isinstance(task, FinalizeTask):
            # Fail fast on bad sidecar inputs rather than burying the
            # error on a worker thread.
            try:
                json.dumps(task.sidecar_payload)
            except (TypeError, ValueError) as exc:
                raise WriteError(
                    f"sidecar payload is not JSON-serializable: {exc}",
                    path=task.sidecar_path,
                    key=task.key,
                ) from exc

        # Make sure a pending result row exists before the task can be
        # observed by the worker. This way tests (and phase 4) can
        # query the result map without races.
        with self._results_lock:
            if task.key not in self._results:
                self._results[task.key] = WriteResult(key=task.key)

        partition_idx = self._partition_for(task.key)
        try:
            self._queues[partition_idx].put(task, timeout=self.timeout_seconds)
        except queue.Full as exc:
            err = WriteError(
                "writer queue full; timed out waiting for a slot",
                key=task.key,
            )
            self._record_error(task.key, err)
            raise err from exc

    def shutdown(self, timeout: float = 30.0) -> None:
        """Drain the queues and join the worker threads.

        After ``timeout`` seconds, any tasks still pending are marked
        with an ``error`` status so the engine can surface a
        ``partial_error`` capture for the owning requests.
        """
        with self._shutdown_lock:
            if self._shutdown_flag.is_set():
                return
            self._shutdown_flag.set()

        # Signal every worker exactly once.
        for q in self._queues:
            try:
                q.put_nowait(_SHUTDOWN)
            except queue.Full:
                # If the queue is full the worker will still observe
                # _SHUTDOWN after it drains one task. Best-effort.
                logger.warning(
                    "activation writer queue full at shutdown; "
                    "worker will exit after next drain iteration"
                )

        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)

        # Anything still alive is beyond the grace period. Mark every
        # remaining queued task as failed so owning requests get a
        # terminal result.
        for idx, q in enumerate(self._queues):
            while True:
                try:
                    task = q.get_nowait()
                except queue.Empty:
                    break
                if task is _SHUTDOWN:
                    continue
                self._record_error(
                    task.key,
                    WriteError(
                        "writer shutdown before task completed",
                        key=task.key,
                    ),
                )

        # Mark any keys still in "pending" state as errored. This
        # covers the case where a task is mid-execution when the
        # worker is abandoned past the grace period.
        with self._results_lock:
            for key, result in self._results.items():
                if result.status == "pending":
                    result.status = "error"
                    result.error = WriteError(
                        "writer shutdown before task completed",
                        key=key,
                    )

        for thread in self._threads:
            if thread.is_alive():
                logger.warning(
                    "activation writer thread %s did not exit within grace period",
                    thread.name,
                )

    # ------------------------------------------------------------------
    # Observability hooks

    def add_status_callback(self, callback: Callable[[WriteResult], None]) -> None:
        """Register a callback fired whenever a result goes terminal.

        Invoked inside the lock that updates the result row; keep the
        callback cheap. Used by tests to avoid polling.
        """
        with self._results_lock:
            self._status_callbacks.append(callback)

    def get_result(self, key: CaptureKey) -> WriteResult | None:
        """Return the current result row for ``key``, if any."""
        with self._results_lock:
            result = self._results.get(key)
            if result is None:
                return None
            # Shallow copy so callers can't mutate our state.
            return WriteResult(
                key=result.key,
                status=result.status,
                error=result.error,
                bin_path=result.bin_path,
                sidecar_path=result.sidecar_path,
            )

    def results_snapshot(self) -> dict[CaptureKey, WriteResult]:
        """Return a snapshot of all result rows for inspection."""
        with self._results_lock:
            return {
                k: WriteResult(
                    key=v.key,
                    status=v.status,
                    error=v.error,
                    bin_path=v.bin_path,
                    sidecar_path=v.sidecar_path,
                )
                for k, v in self._results.items()
            }

    # ------------------------------------------------------------------
    # Partitioning

    def _partition_for(self, key: CaptureKey) -> int:
        # Partition by request_id only. Multi-step tasks for the same
        # (request_id, layer, hook) must land on the same worker, and
        # all captures for one request share the same request_id.
        request_id = key[0]
        return hash(request_id) % self.num_threads

    # ------------------------------------------------------------------
    # Worker loop

    def _worker_loop(
        self,
        thread_idx: int,
        q: queue.Queue[Any],
        partition: _PartitionState,
    ) -> None:
        try:
            while True:
                task = q.get()
                if task is _SHUTDOWN:
                    break
                try:
                    if isinstance(task, WriteTask):
                        self._handle_write(partition, task)
                    elif isinstance(task, FinalizeTask):
                        self._handle_finalize(partition, task)
                    else:  # pragma: no cover - defensive
                        logger.error(
                            "activation writer thread %d got unknown task type %s",
                            thread_idx,
                            type(task).__name__,
                        )
                except WriteError as exc:
                    self._record_error(task.key, exc)
                    self._drop_fd(partition, task.key, fsync=False)
                except Exception as exc:
                    self._record_error(
                        task.key,
                        WriteError(
                            f"unexpected writer failure: {exc}",
                            key=task.key,
                        ),
                    )
                    self._drop_fd(partition, task.key, fsync=False)
        finally:
            # Flush and close every cached fd so the thread never
            # leaks an open descriptor on shutdown.
            self._close_all_fds(partition)

    # ------------------------------------------------------------------
    # WriteTask handling

    def _handle_write(self, partition: _PartitionState, task: WriteTask) -> None:
        tmp_path = _tmp_path(task.path)
        fd = self._acquire_fd(partition, task.key, tmp_path, task.append)
        try:
            _full_write(fd, task.payload)
        except OSError as exc:
            # Evict the fd on I/O error so the next attempt reopens.
            self._drop_fd(partition, task.key, fsync=False)
            raise WriteError(
                f"write to {tmp_path} failed: {exc}",
                path=tmp_path,
                key=task.key,
                errno_code=exc.errno,
            ) from exc

    def _acquire_fd(
        self,
        partition: _PartitionState,
        key: CaptureKey,
        tmp_path: pathlib.Path,
        append: bool,
    ) -> int:
        cache = partition.fd_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WriteError(
                f"mkdir {tmp_path.parent} failed: {exc}",
                path=tmp_path,
                key=key,
                errno_code=exc.errno,
            ) from exc

        flags = os.O_WRONLY | os.O_CREAT
        if append:
            # O_APPEND honors the "append in submission order"
            # invariant even if the same fd is somehow shared.
            flags |= os.O_APPEND
        else:
            flags |= os.O_TRUNC
        try:
            fd = os.open(tmp_path, flags, 0o644)
        except OSError as exc:
            raise WriteError(
                f"open {tmp_path} failed: {exc}",
                path=tmp_path,
                key=key,
                errno_code=exc.errno,
            ) from exc

        cache[key] = fd
        cache.move_to_end(key)
        self._enforce_cache_limit(partition)
        return fd

    def _enforce_cache_limit(self, partition: _PartitionState) -> None:
        cache = partition.fd_cache
        while len(cache) > partition.cache_capacity:
            evict_key, evict_fd = cache.popitem(last=False)
            try:
                os.fsync(evict_fd)
            except OSError as exc:
                logger.warning(
                    "fsync on evicted fd for key=%s failed: %s",
                    evict_key,
                    exc,
                )
            try:
                os.close(evict_fd)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning(
                    "close on evicted fd for key=%s failed: %s",
                    evict_key,
                    exc,
                )

    def _drop_fd(
        self,
        partition: _PartitionState,
        key: CaptureKey,
        *,
        fsync: bool,
    ) -> int | None:
        fd = partition.fd_cache.pop(key, None)
        if fd is None:
            return None
        if fsync:
            try:
                os.fsync(fd)
            except OSError as exc:
                logger.warning("fsync on fd for key=%s failed: %s", key, exc)
        try:
            os.close(fd)
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning("close on fd for key=%s failed: %s", key, exc)
        return fd

    def _close_all_fds(self, partition: _PartitionState) -> None:
        while partition.fd_cache:
            key, fd = partition.fd_cache.popitem(last=False)
            try:
                os.fsync(fd)
            except OSError as exc:
                logger.warning("fsync on shutdown fd for key=%s failed: %s", key, exc)
            try:
                os.close(fd)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning("close on shutdown fd for key=%s failed: %s", key, exc)

    # ------------------------------------------------------------------
    # FinalizeTask handling

    def _handle_finalize(self, partition: _PartitionState, task: FinalizeTask) -> None:
        # Short-circuit if an earlier task in this capture's sequence
        # already errored. Running the rename would publish a
        # half-written .bin file under the final name, which is a
        # partial-data leak. Leave the .tmp in place for operators
        # to inspect.
        with self._results_lock:
            existing = self._results.get(task.key)
            if existing is not None and existing.status == "error":
                self._drop_fd(partition, task.key, fsync=False)
                return

        bin_tmp = _tmp_path(task.bin_path)
        sidecar_tmp = _tmp_path(task.sidecar_path)

        # Ensure the .bin.tmp exists. If no WriteTask was ever sent
        # for this key (zero-byte capture), touch the tmp file now so
        # the promote path is uniform.
        if task.key not in partition.fd_cache and not bin_tmp.exists():
            try:
                bin_tmp.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(
                    bin_tmp,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o644,
                )
                partition.fd_cache[task.key] = fd
                partition.fd_cache.move_to_end(task.key)
            except OSError as exc:
                raise WriteError(
                    f"open {bin_tmp} during finalize failed: {exc}",
                    path=bin_tmp,
                    key=task.key,
                    errno_code=exc.errno,
                ) from exc

        # Flush and close the .bin.tmp fd.
        fd = partition.fd_cache.pop(task.key, None)
        if fd is not None:
            try:
                os.fsync(fd)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise WriteError(
                    f"fsync {bin_tmp} failed: {exc}",
                    path=bin_tmp,
                    key=task.key,
                    errno_code=exc.errno,
                ) from exc
            try:
                os.close(fd)
            except OSError as exc:
                raise WriteError(
                    f"close {bin_tmp} failed: {exc}",
                    path=bin_tmp,
                    key=task.key,
                    errno_code=exc.errno,
                ) from exc

        # Promote .bin.tmp -> task.bin_path under the collision policy.
        final_bin = self._promote(
            tmp_path=bin_tmp,
            target=task.bin_path,
            key=task.key,
        )

        # Write the sidecar: json -> .json.tmp -> fsync -> promote.
        try:
            sidecar_tmp.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WriteError(
                f"mkdir {sidecar_tmp.parent} failed: {exc}",
                path=sidecar_tmp,
                key=task.key,
                errno_code=exc.errno,
            ) from exc

        try:
            payload = json.dumps(task.sidecar_payload).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise WriteError(
                f"sidecar payload is not JSON-serializable: {exc}",
                path=task.sidecar_path,
                key=task.key,
            ) from exc

        try:
            sidecar_fd = os.open(
                sidecar_tmp,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o644,
            )
        except OSError as exc:
            raise WriteError(
                f"open {sidecar_tmp} failed: {exc}",
                path=sidecar_tmp,
                key=task.key,
                errno_code=exc.errno,
            ) from exc
        try:
            _full_write(sidecar_fd, payload)
            try:
                os.fsync(sidecar_fd)
            except OSError as exc:
                raise WriteError(
                    f"fsync {sidecar_tmp} failed: {exc}",
                    path=sidecar_tmp,
                    key=task.key,
                    errno_code=exc.errno,
                ) from exc
        except OSError as exc:
            raise WriteError(
                f"write {sidecar_tmp} failed: {exc}",
                path=sidecar_tmp,
                key=task.key,
                errno_code=exc.errno,
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                os.close(sidecar_fd)

        final_sidecar = self._promote(
            tmp_path=sidecar_tmp,
            target=task.sidecar_path,
            key=task.key,
        )

        self._record_ok(task.key, bin_path=final_bin, sidecar_path=final_sidecar)

    def _promote(
        self,
        *,
        tmp_path: pathlib.Path,
        target: pathlib.Path,
        key: CaptureKey,
    ) -> pathlib.Path:
        """Promote ``tmp_path`` to its final location under the policy.

        Returns the actual path that now holds the data; this differs
        from ``target`` only for the ``suffix`` policy.
        """
        policy = self.on_collision
        if policy == "overwrite":
            destination = target
        elif policy == "error":
            if target.exists():
                # Leave the .tmp alone so operators can inspect it.
                raise WriteError(
                    f"refusing to overwrite existing capture: {target}",
                    path=target,
                    key=key,
                    errno_code=errno.EEXIST,
                )
            destination = target
        elif policy == "suffix":
            if target.exists():
                stamp_ms = int(time.time() * 1000)
                destination = target.with_name(
                    f"{target.stem}.{stamp_ms}{target.suffix}"
                )
            else:
                destination = target
        else:  # pragma: no cover - validated in __init__
            raise WriteError(
                f"unknown collision policy: {policy}",
                path=target,
                key=key,
            )

        try:
            os.replace(tmp_path, destination)
        except OSError as exc:
            raise WriteError(
                f"rename {tmp_path} -> {destination} failed: {exc}",
                path=destination,
                key=key,
                errno_code=exc.errno,
            ) from exc
        return destination

    # ------------------------------------------------------------------
    # Result recording

    def _record_error(self, key: CaptureKey, err: WriteError) -> None:
        fired: list[tuple[Callable[[WriteResult], None], WriteResult]] = []
        with self._results_lock:
            result = self._results.setdefault(key, WriteResult(key=key))
            if result.status != "ok":
                result.status = "error"
                result.error = err
                snapshot = WriteResult(
                    key=result.key,
                    status=result.status,
                    error=result.error,
                    bin_path=result.bin_path,
                    sidecar_path=result.sidecar_path,
                )
                for cb in self._status_callbacks:
                    fired.append((cb, snapshot))
        for cb, snap in fired:
            try:
                cb(snap)
            except Exception:  # pragma: no cover - callback safety
                logger.exception("activation writer status callback raised")

    def _record_ok(
        self,
        key: CaptureKey,
        *,
        bin_path: pathlib.Path,
        sidecar_path: pathlib.Path,
    ) -> None:
        fired: list[tuple[Callable[[WriteResult], None], WriteResult]] = []
        with self._results_lock:
            result = self._results.setdefault(key, WriteResult(key=key))
            # "error" is sticky: if any task in the capture's
            # sequence already failed, the final state must stay
            # errored even if the finalize rename itself happened to
            # succeed. The underlying bytes are incomplete.
            if result.status == "error":
                return
            result.status = "ok"
            result.error = None
            result.bin_path = bin_path
            result.sidecar_path = sidecar_path
            snapshot = WriteResult(
                key=result.key,
                status=result.status,
                error=result.error,
                bin_path=result.bin_path,
                sidecar_path=result.sidecar_path,
            )
            for cb in self._status_callbacks:
                fired.append((cb, snapshot))
        for cb, snap in fired:
            try:
                cb(snap)
            except Exception:  # pragma: no cover - callback safety
                logger.exception("activation writer status callback raised")


def _finalize_writer(writer: ActivationWriter) -> None:
    """Module-level finalizer so ``weakref.finalize`` can collect us."""
    try:
        writer.shutdown(timeout=5.0)
    except Exception:  # pragma: no cover - best effort at GC time
        logger.exception("ActivationWriter finalizer raised")


def _tmp_path(path: pathlib.Path) -> pathlib.Path:
    """Return the canonical ``.tmp`` sibling for ``path``."""
    return path.with_name(path.name + ".tmp")


def _full_write(fd: int, payload: bytes) -> None:
    """``os.write`` that retries short writes until ``payload`` is drained.

    ``os.write`` is permitted to accept fewer bytes than requested,
    especially on NFS and with large buffers. We loop until the
    caller's whole payload is on the wire so append ordering stays
    consistent with submission order.
    """
    view = memoryview(payload)
    offset = 0
    length = len(view)
    while offset < length:
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "os.write returned 0 bytes")
        offset += written
