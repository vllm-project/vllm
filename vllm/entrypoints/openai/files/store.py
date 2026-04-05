# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-backed upload store with TTL + LRU eviction and audit logging.

Backs the `/v1/files` endpoint. Holds metadata in memory (rebuilt on
server restart) and stores bytes on disk with random sha256 filenames.
Per the tech-spec's capability-handle model, each file has a 128-bit
unguessable ID and is scoped by an optional request-header value.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import tempfile
import time
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.files.mime import SNIFF_HEAD_BYTES, sniff_mime
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import FileUploadConfig
    from vllm.entrypoints.openai.files.protocol import FilePurpose

logger = init_logger(__name__)

# Opportunistic sweep interval: at most once per 5 minutes per store,
# triggered from create_file (see `_maybe_sweep`). No background task.
_SWEEP_INTERVAL_SECONDS = 300

# Streaming write chunk size — large enough to be efficient, small enough
# that we check quota/size often.
_CHUNK_BYTES = 64 * 1024

_FILENAME_MAX_LEN = 255

# Type alias for list-of-Path used in eviction/unlink signatures. Defined
# at module scope so mypy resolves `list` to the builtin, not the
# `FileUploadStore.list` method which shadows it inside the class body.
_PathList = list[Path]


class FileStoreError(Exception):
    """Base class for file-store errors surfaced to the API layer."""


class FileTooLarge(FileStoreError):
    """Single upload exceeded max_size_mb."""


class QuotaExceeded(FileStoreError):
    """Upload would exceed max_total_gb even after LRU eviction."""


class InvalidMime(FileStoreError):
    """Sniffed MIME type is not in the allowlist."""


class ConcurrencyLimitExceeded(FileStoreError):
    """More than `max_concurrent` uploads are already in flight."""


@dataclass
class FileRecord:
    """Internal metadata for one stored file.

    Attributes:
        id: 128-bit capability handle (`file-<32 hex>`).
        filename: Sanitized client filename (metadata only; never used
            as a filesystem path).
        purpose: Upload purpose (`vision` or `user_data`).
        mime_type: Sniffed MIME type, narrowed to the media allowlist.
        bytes: Size on disk.
        created_at: Upload wall-clock time (unix seconds).
        last_accessed: atime-style timestamp, touched on every read.
        scope: Scope header value at upload time, or None.
        on_disk: Path to the bytes under the upload directory.
        client_host: Remote host of the uploading client, for audit.
    """

    id: str
    filename: str  # sanitized client filename (metadata only)
    purpose: FilePurpose
    mime_type: str
    bytes: int
    created_at: int
    last_accessed: float  # monotonic-enough; uses time.time()
    scope: str | None
    on_disk: Path
    # Fields below are NOT logged; they exist only for future retrieval.
    client_host: str | None = field(default=None, repr=False)


def sanitize_filename(raw: str) -> str:
    """Strip path components, remove control chars, cap length.

    The returned string is metadata only and never becomes part of any
    filesystem path — but it is echoed back in API responses, so we keep
    it clean enough to not confuse terminals or leak directory structure.

    Returns:
        A safe-to-echo filename, at most 255 UTF-8 bytes. Returns
        `"upload.bin"` when the input collapses to empty after stripping.
    """
    # Drop any directory components the client may have prefixed.
    # os.path.basename handles both separators on either host OS.
    name = os.path.basename(raw.replace("\\", "/"))
    # Strip control chars (cat Cc) and C1 range. Preserve ordinary printable
    # characters including unicode letters and emoji.
    name = "".join(
        ch for ch in name if unicodedata.category(ch)[0] != "C" and ch not in "\r\n\t"
    )
    # Collapse empty strings — metadata without a filename is fine, but
    # callers expect a non-empty string. Use a placeholder.
    if not name:
        name = "upload.bin"
    # Byte-length cap (255 is the POSIX NAME_MAX; even though we don't use
    # this on disk, we cap to keep responses reasonable).
    encoded = name.encode("utf-8")
    if len(encoded) > _FILENAME_MAX_LEN:
        # Truncate on a codepoint boundary.
        encoded = encoded[:_FILENAME_MAX_LEN]
        name = encoded.decode("utf-8", errors="ignore")
    return name


def new_file_id() -> str:
    """Generate a fresh 128-bit random capability handle.

    Returns:
        An id of the form `file-<32 hex chars>` (37 chars total).
    """
    return f"file-{secrets.token_hex(16)}"


# Per-process registry so that components outside the FastAPI request cycle
# (notably MediaConnector, which resolves vllm-file:// URLs during chat
# completion) can reach the store without threading it through half of
# chat_utils.py. Set once at startup via register_store().
_GLOBAL_STORE: FileUploadStore | None = None


def register_store(store: FileUploadStore | None) -> None:
    """Install `store` as the process-wide upload store. Call with None to
    clear (useful in tests)."""
    global _GLOBAL_STORE
    _GLOBAL_STORE = store


def get_store() -> FileUploadStore | None:
    """Return the process-wide upload store.

    Returns:
        The registered `FileUploadStore`, or None if file uploads are
        not enabled on this server.
    """
    return _GLOBAL_STORE


def _now_wall() -> int:
    return int(time.time())


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FileUploadStore:
    """File-backed upload store. One instance per server.

    Metadata lives in memory and is rebuilt on server restart (the
    upload directory is cleared at construction time). Bytes live on
    disk under `<dir>/<sha256(file_id)>`, so filesystem names leak
    neither the capability handle nor the client filename.

    All filesystem writes and unlinks are dispatched to a thread-pool
    executor so the asyncio event loop stays responsive. Eviction
    paths (quota LRU, TTL sweep, explicit delete) use a two-phase
    pattern: metadata is removed under the store lock, then the
    on-disk bytes are unlinked outside the lock.

    All instance state is private (leading-underscore); the public
    surface is the methods in the "public API" section below.
    """

    def __init__(self, config: FileUploadConfig) -> None:
        self._config = config

        # Resolve upload directory. Two paths:
        #   1. No --file-upload-dir: use tempfile.mkdtemp() which enforces
        #      0o700 permissions, uses a random suffix (no PID-predictable
        #      path for pre-creation attacks on shared hosts), and creates
        #      the directory atomically. No rmtree or marker needed.
        #   2. Operator-supplied dir: acquire a PID lockfile, then apply
        #      the marker-file safety check before clearing.
        using_default_dir = not config.dir
        self._lockfile: Path | None = None
        if using_default_dir:
            self._dir = Path(tempfile.mkdtemp(prefix="vllm-uploads-"))
        else:
            self._dir = Path(config.dir)
            # Acquire a PID lockfile adjacent to the dir BEFORE touching
            # its contents, so two vLLM processes configured with the
            # same --file-upload-dir can't race on rmtree + mkdir and
            # wipe each other's in-memory records. The lockfile lives
            # next to the dir (not inside) since the dir is cleared.
            self._lockfile = self._dir.with_name(self._dir.name + ".lock")
            self._lockfile.parent.mkdir(parents=True, exist_ok=True)
            self._acquire_dir_lock()

            # Clear on startup — non-persistent across restarts is a
            # security invariant. Refuse to rmtree a user-supplied
            # directory unless we previously marked it ourselves, so a
            # typo like --file-upload-dir / never destroys unrelated
            # data. Surface rmtree failures instead of swallowing them.
            marker = self._dir / ".vllm-upload-store"
            if self._dir.exists():
                if marker.exists():
                    try:
                        shutil.rmtree(self._dir)
                    except OSError as e:
                        self._release_dir_lock()
                        raise ValueError(
                            f"Failed to clear file upload directory "
                            f"{self._dir!r}: {e}. Remove stale files "
                            f"manually or point --file-upload-dir at a "
                            f"writable path."
                        ) from e
                else:
                    self._release_dir_lock()
                    raise ValueError(
                        f"Refusing to clear file upload directory "
                        f"{self._dir!r}: it is not marked as managed "
                        f"by vLLM. Point --file-upload-dir at an empty "
                        f"or dedicated path, or at a directory "
                        f"previously initialised by the upload store."
                    )
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Drop the marker so subsequent restarts can safely re-init.
            marker.touch(mode=0o600, exist_ok=True)
            logger.info("File upload store using directory %s", self._dir)

        self._records: dict[str, FileRecord] = {}
        self._lock = asyncio.Lock()
        self._upload_semaphore = asyncio.Semaphore(config.max_concurrent)
        self._total_bytes = 0
        self._last_sweep = time.time()

        self._max_size_bytes = config.max_size_mb * 1024 * 1024
        self._max_total_bytes = config.max_total_gb * 1024 * 1024 * 1024
        self._ttl_enabled = config.ttl_seconds >= 0
        self._ttl_seconds = config.ttl_seconds

    # ------------------------------------------------------------------
    # directory lock (operator-supplied --file-upload-dir only)
    # ------------------------------------------------------------------

    def _acquire_dir_lock(self) -> None:
        """Take exclusive ownership of the upload directory via a PID file.

        On conflict, probe the holder with os.kill(pid, 0). Stale locks
        (dead PIDs) are reclaimed; live holders raise ValueError.
        """
        if self._lockfile is None:
            return
        my_pid = os.getpid()
        while True:
            try:
                fd = os.open(
                    str(self._lockfile),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                try:
                    holder = int(self._lockfile.read_text().strip())
                except (OSError, ValueError):
                    # Lockfile is corrupt — treat as stale and reclaim.
                    self._lockfile.unlink(missing_ok=True)
                    continue
                if holder == my_pid:
                    return
                try:
                    os.kill(holder, 0)
                except ProcessLookupError:
                    # Holder is dead — steal the lock.
                    self._lockfile.unlink(missing_ok=True)
                    continue
                except PermissionError:
                    # PID exists and is not ours — refuse.
                    pass
                raise ValueError(
                    f"File upload directory {self._dir!r} is locked by "
                    f"PID {holder}. Another vLLM server is using it. "
                    f"Stop the other server or use a different "
                    f"--file-upload-dir."
                ) from None
            try:
                os.write(fd, str(my_pid).encode())
            finally:
                os.close(fd)
            return

    def _release_dir_lock(self) -> None:
        """Remove the PID lockfile if we own it."""
        if self._lockfile is None:
            return
        try:
            holder = int(self._lockfile.read_text().strip())
        except (OSError, ValueError):
            return
        if holder == os.getpid():
            self._lockfile.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def create_file(
        self,
        stream: AsyncIterator[bytes],
        filename: str,
        purpose: FilePurpose,
        scope: str | None,
        client_host: str | None = None,
        request_id: str | None = None,
    ) -> FileRecord:
        """Stream an upload to disk, validate MIME, register metadata.

        Writes are dispatched to a thread-pool executor per chunk so
        the asyncio event loop stays responsive under concurrent
        uploads. LRU eviction is performed in two phases — metadata
        removal under the store lock, on-disk unlinks outside the lock.

        Returns:
            The registered `FileRecord` on success.

        Raises:
            ConcurrencyLimitExceeded: `max_concurrent` uploads are already
                in flight (fail-fast rather than queue indefinitely).
            FileTooLarge: Upload exceeds `max_size_mb`.
            InvalidMime: Sniffed MIME type is not in the media allowlist.
            QuotaExceeded: Upload alone exceeds `max_total_gb`.

        On any failure the partial temp file is unlinked before the
        exception propagates, so no on-disk leak occurs.
        """
        clean_name = sanitize_filename(filename)
        await self._maybe_sweep()
        # Non-blocking acquire — if we're at max_concurrent, reject with
        # 503 rather than queueing indefinitely (documented contract).
        if self._upload_semaphore.locked():
            raise ConcurrencyLimitExceeded(
                f"too many in-flight uploads "
                f"(max_concurrent={self._config.max_concurrent})"
            )
        async with self._upload_semaphore:
            file_id = new_file_id()
            on_disk = self._on_disk_path(file_id)
            total = 0
            head = b""
            loop = asyncio.get_running_loop()
            try:
                # Create with 0o600 so only the vLLM process user can
                # read uploaded bytes, even if the enclosing directory
                # permissions regress. O_EXCL guards against the
                # theoretical (but sha256-random) path collision.
                # `out.write` is run in an executor to keep the event
                # loop responsive under concurrent uploads; the
                # containing `open`/`close` is one-shot, not looped.
                fd = os.open(
                    on_disk,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                with os.fdopen(fd, "wb") as out:
                    async for chunk in stream:
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self._max_size_bytes:
                            raise FileTooLarge(
                                f"upload exceeds max_size_mb={self._config.max_size_mb}"
                            )
                        if len(head) < SNIFF_HEAD_BYTES:
                            head += chunk[: SNIFF_HEAD_BYTES - len(head)]
                        await loop.run_in_executor(None, out.write, chunk)
            except BaseException:
                await self._unlink_paths([on_disk])
                raise

            mime_type = sniff_mime(head)
            if mime_type is None or not self._is_allowed_mime(mime_type):
                await self._unlink_paths([on_disk])
                self._audit(
                    "file.reject",
                    file_id=file_id,
                    bytes=total,
                    scope=scope,
                    client_host=client_host,
                    request_id=request_id,
                    reason="mime_mismatch",
                )
                raise InvalidMime(f"unsupported media type (sniffed={mime_type!r})")

            # Two-phase: snapshot any LRU-eviction paths under the lock,
            # then unlink them outside the lock (POSIX keeps open file
            # handles valid across unlink, so concurrent readers are
            # unaffected).
            async with self._lock:
                if total > self._max_total_bytes:
                    quota_reject = True
                    evicted_paths: _PathList = []
                    record = None
                else:
                    quota_reject = False
                    evicted_paths = self._evict_for_quota(total)
                    record = FileRecord(
                        id=file_id,
                        filename=clean_name,
                        purpose=purpose,
                        mime_type=mime_type,
                        bytes=total,
                        created_at=_now_wall(),
                        last_accessed=time.time(),
                        scope=scope,
                        on_disk=on_disk,
                        client_host=client_host,
                    )
                    self._records[file_id] = record
                    self._total_bytes += total

            # Unlink outside the lock.
            await self._unlink_paths(evicted_paths)
            if quota_reject:
                await self._unlink_paths([on_disk])
                self._audit(
                    "file.reject",
                    file_id=file_id,
                    bytes=total,
                    scope=scope,
                    client_host=client_host,
                    request_id=request_id,
                    reason="quota_full",
                )
                raise QuotaExceeded("upload alone exceeds max_total_gb")

            assert record is not None  # quota_reject=False guarantees this
            self._audit(
                "file.upload",
                file_id=file_id,
                bytes=total,
                scope=scope,
                client_host=client_host,
                request_id=request_id,
            )
            return record

    def get(self, file_id: str, scope: str | None) -> FileRecord | None:
        """Return the record for `file_id` if accessible under `scope`.

        Touches `last_accessed` so active conversations keep their files
        alive.

        Returns:
            The matching `FileRecord`, or None when the file doesn't
            exist OR scope mismatches (capability non-disclosure — we
            intentionally do not distinguish the two).
        """
        record = self._records.get(file_id)
        if record is None or record.scope != scope:
            return None
        record.last_accessed = time.time()
        return record

    def list(self, scope: str | None) -> list[FileRecord]:
        """All records visible under `scope`, sorted newest-first.

        Returns:
            A list of `FileRecord` entries whose scope matches `scope`.
        """
        visible = [r for r in self._records.values() if r.scope == scope]
        visible.sort(key=lambda r: r.created_at, reverse=True)
        return visible

    async def delete(
        self,
        file_id: str,
        scope: str | None,
        client_host: str | None = None,
        request_id: str | None = None,
    ) -> bool:
        """Remove a file and its metadata.

        Metadata is removed synchronously under the store lock; the
        on-disk bytes are unlinked outside the lock via a thread-pool
        executor so the event loop is never blocked on filesystem I/O.

        Returns:
            True if the record existed under `scope` and was removed;
            False otherwise (unknown id or scope mismatch).
        """
        async with self._lock:
            record = self._records.get(file_id)
            if record is None or record.scope != scope:
                return False
            path = self._drop_record_metadata(record)
        await self._unlink_paths([path])
        self._audit(
            "file.delete",
            file_id=file_id,
            bytes=record.bytes,
            scope=scope,
            client_host=client_host,
            request_id=request_id,
        )
        return True

    async def stream_content(
        self,
        file_id: str,
        scope: str | None,
        client_host: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Async iterator yielding the file's bytes in 64 KiB chunks.

        The file handle is opened eagerly (before returning the iterator)
        so that a concurrent eviction between `get()` and the first
        iteration surfaces as `FileNotFoundError` from this call — the
        serving layer maps that to 404, avoiding a broken 200 response
        stream. POSIX keeps the open inode alive across subsequent
        unlinks, so reads after open are safe.

        Returns:
            An async iterator over byte chunks from the on-disk file.

        Raises:
            FileNotFoundError: If `file_id` does not resolve under
                `scope`, OR the on-disk bytes were evicted between the
                metadata lookup and the `open()` call.
        """
        record = self.get(file_id, scope)
        if record is None:
            raise FileNotFoundError(file_id)
        loop = asyncio.get_running_loop()
        # Open eagerly so FileNotFoundError fires BEFORE we return an
        # iterator (i.e. before any 200 response has been streamed).
        try:
            fh = await loop.run_in_executor(None, open, record.on_disk, "rb")
        except FileNotFoundError as e:
            raise FileNotFoundError(file_id) from e
        self._audit(
            "file.retrieve",
            file_id=file_id,
            bytes=record.bytes,
            scope=scope,
            client_host=client_host,
            request_id=request_id,
        )

        async def _stream() -> AsyncIterator[bytes]:
            try:
                while True:
                    chunk = await loop.run_in_executor(None, fh.read, _CHUNK_BYTES)
                    if not chunk:
                        return
                    yield chunk
            finally:
                # Dispatch close to the executor too — close() can block
                # on a buffer flush / metadata update on slow filesystems.
                await loop.run_in_executor(None, fh.close)

        return _stream()

    def read_bytes(self, file_id: str, scope: str | None) -> bytes | None:
        """Synchronous whole-file read, scope-enforced.

        Returns:
            The file bytes, or None on unknown id / scope mismatch OR
            when the bytes were evicted between the metadata lookup and
            the disk read (the record's file was unlinked concurrently).
        """
        record = self.get(file_id, scope)
        if record is None:
            return None
        try:
            return record.on_disk.read_bytes()
        except FileNotFoundError:
            return None

    async def read_bytes_by_id_async(self, file_id: str) -> bytes | None:
        """Scope-bypassing async read for `MediaConnector`.

        Split into two phases to avoid cross-thread data races: the
        metadata lookup (dict read + `last_accessed` mutation) happens
        on the event-loop thread, and the blocking disk read is
        dispatched to a thread-pool executor. This keeps `self._records`
        a single-writer / single-reader structure (event loop only)
        even when called from a multimodal resolution path that ran in
        a worker thread prior to this change.

        The 128-bit file ID is the capability handle here — possession
        implies authority to access. Scope enforcement only applies to
        the /v1/files CRUD endpoints (where the file_id may be leaked
        via LIST). The chat-completion layer does not have access to
        request headers in the current architecture, so we treat the
        ID as the access control. Touches atime so files referenced
        in a conversation stay alive.

        Returns:
            The file bytes, or None if the file does not exist OR was
            evicted between the metadata lookup and the disk read.
        """
        record = self._records.get(file_id)
        if record is None:
            return None
        record.last_accessed = time.time()
        self._audit(
            "file.resolve",
            file_id=file_id,
            bytes=record.bytes,
            scope=record.scope,
            client_host=None,
            request_id=None,
        )
        loop = asyncio.get_running_loop()

        def _read() -> bytes | None:
            try:
                return record.on_disk.read_bytes()
            except FileNotFoundError:
                return None

        return await loop.run_in_executor(None, _read)

    def read_bytes_by_id(self, file_id: str) -> bytes | None:
        """Scope-bypassing sync read for `MediaConnector` (sync path only).

        Kept for the synchronous `load_from_url` code path in
        `MediaConnector` (non-async resolution). Prefer
        `read_bytes_by_id_async` from async contexts to avoid blocking
        the event loop on the disk read.

        Returns:
            The file bytes, or None if the file does not exist OR was
            evicted between the metadata lookup and the disk read.
        """
        record = self._records.get(file_id)
        if record is None:
            return None
        record.last_accessed = time.time()
        self._audit(
            "file.resolve",
            file_id=file_id,
            bytes=record.bytes,
            scope=record.scope,
            client_host=None,
            request_id=None,
        )
        try:
            return record.on_disk.read_bytes()
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _on_disk_path(self, file_id: str) -> Path:
        # The ID is already random — we just need a stable filesystem name
        # derived from it. Hashing keeps the on-disk name opaque (doesn't
        # reveal the file_id) and safe for any filesystem.
        import hashlib

        digest = hashlib.sha256(file_id.encode()).hexdigest()
        return self._dir / digest

    def _is_allowed_mime(self, mime: str) -> bool:
        # Re-implemented here (instead of importing) to keep a single
        # source of truth with the store's invariants; the mime module's
        # helper is tested separately.
        return mime.startswith(("video/", "image/", "audio/"))

    def _evict_for_quota(self, incoming_bytes: int) -> _PathList:
        """Evict oldest-accessed records until `incoming_bytes` will fit.

        Caller must hold `self._lock` and must pass the returned paths
        to `_unlink_paths` (outside the lock) to remove the bytes.

        Returns:
            List of on-disk paths for evicted records. Empty list when
            no eviction was required.
        """
        if self._total_bytes + incoming_bytes <= self._max_total_bytes:
            return []
        paths: _PathList = []
        candidates = sorted(self._records.values(), key=lambda r: r.last_accessed)
        for rec in candidates:
            if self._total_bytes + incoming_bytes <= self._max_total_bytes:
                break
            paths.append(self._drop_record_metadata(rec))
            self._audit(
                "file.delete",
                file_id=rec.id,
                bytes=rec.bytes,
                scope=rec.scope,
                client_host=None,
                request_id=None,
                reason="quota_eviction",
            )
        return paths

    def _drop_record_metadata(self, record: FileRecord) -> Path:
        """Drop in-memory state for `record` and return its on-disk path.

        Caller must hold `self._lock` when calling this and must
        subsequently pass the returned path to `_unlink_paths` (outside
        the lock) to actually remove the bytes from disk.

        Returns:
            The on-disk path that needs to be unlinked.
        """
        self._records.pop(record.id, None)
        self._total_bytes = max(0, self._total_bytes - record.bytes)
        return record.on_disk

    async def _unlink_paths(self, paths: _PathList) -> None:
        """Unlink a batch of on-disk files in a thread pool.

        Never holds any store lock — safe to call with `self._lock`
        released. POSIX semantics ensure concurrent open file handles
        remain valid across the unlink.

        Any OSError beyond FileNotFoundError (e.g., permission denied
        on a ro-mount, EBUSY on Windows) is logged and swallowed —
        unlinks are best-effort cleanup and an unlink failure should
        never cascade into a request-handling failure.
        """
        if not paths:
            return
        loop = asyncio.get_running_loop()

        def _bulk_unlink() -> None:
            for p in paths:
                try:
                    p.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("file upload: failed to unlink %s: %s", p, e)

        await loop.run_in_executor(None, _bulk_unlink)

    async def _maybe_sweep(self) -> None:
        """Opportunistic sweep from create_file, so the store works without
        a running lifespan-managed sweeper task. Cheap no-op when TTL is
        disabled or when the last sweep was recent."""
        if not self._ttl_enabled:
            return
        now = time.time()
        if now - self._last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        await self._sweep_expired()

    async def _sweep_expired(self) -> None:
        """Drop records whose last-access timestamp is older than the TTL.

        Two-phase: metadata removal happens under the lock, the unlink
        batch runs outside the lock via a thread pool executor.
        """
        if not self._ttl_enabled:
            return
        cutoff = time.time() - self._ttl_seconds
        paths: _PathList = []
        async with self._lock:
            expired = [r for r in self._records.values() if r.last_accessed < cutoff]
            for rec in expired:
                paths.append(self._drop_record_metadata(rec))
                self._audit(
                    "file.delete",
                    file_id=rec.id,
                    bytes=rec.bytes,
                    scope=rec.scope,
                    client_host=None,
                    request_id=None,
                    reason="ttl_expired",
                )
        await self._unlink_paths(paths)

    def _audit(
        self,
        op: str,
        *,
        file_id: str,
        bytes: int | None,
        scope: str | None,
        client_host: str | None,
        request_id: str | None,
        reason: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "op": op,
            "file_id": file_id,
            "bytes": bytes,
            "scope": scope,
            "client_host": client_host,
            "request_id": request_id,
            "ts": _iso_now(),
        }
        if reason is not None:
            payload["reason"] = reason
        logger.info(json.dumps(payload))
