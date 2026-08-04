# SPDX-License-Identifier: Apache-2.0
"""HTTP and shared-memory client for the HyperPod ai-toolkit cache daemon."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import AsyncIterator
import asyncio
import mmap
import os
import threading
import time
import urllib.parse

# Third Party
import aiohttp

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_HTTP_OK = 200
_HTTP_CONFLICT = 409
_DEFAULT_STREAM_CHUNK_BYTES = 64 * 1024
#: Client-side margin over the daemon's lease wait, so a miss 504 arrives
#: before the client aborts the connection.
_LEASE_WAIT_TIMEOUT_MARGIN_MS = 500

#: Filesystem root of named POSIX shared-memory segments on Linux.
_SHM_DIR = "/dev/shm"


def _shm_segment_path(name: str) -> str:
    """Return the filesystem path of a named POSIX shared-memory segment."""
    return os.path.join(_SHM_DIR, name.lstrip("/"))


class _ReadOnlySharedMemoryMapping:
    """A read-only, untracked mapping of an existing shared-memory segment.

    The ai-toolkit daemon owns the segment; this class only maps it. It
    deliberately avoids :class:`multiprocessing.shared_memory.SharedMemory`
    on Linux for two reasons:

    - **Untracked.** CPython's resource tracker registers even attach-only
      ``SharedMemory`` handles and unlinks them *from the host* at
      interpreter exit. When this process shares the host's ``/dev/shm``
      view (e.g. ``hostIPC`` pods), that cleanup deletes the daemon's
      node-local cache for every client on the node. A plain ``mmap``
      never touches the tracker. (``SharedMemory(track=False)`` on
      Python 3.13+ would also avoid the tracker, but maps writable and
      is unavailable on 3.12.)
    - **Read-only.** The client only reads cache bytes from the segment
      (writes go through the daemon's HTTP API), so pages are mapped
      ``PROT_READ`` and cannot be corrupted from this process.

    On platforms without ``/dev/shm`` (non-Linux development machines) it
    falls back to plain ``SharedMemory`` — tracked and writable, which is
    acceptable because the daemon and its host-owned segment only exist
    on Linux.

    Raises:
        FileNotFoundError: If the segment does not exist, or exists but is
            still empty (a transient state while the daemon recreates it).
    """

    def __init__(self, name: str) -> None:
        self._mmap: mmap.mmap | None = None
        self._shm: shared_memory.SharedMemory | None = None
        if os.path.isdir(_SHM_DIR):
            try:
                fd = os.open(_shm_segment_path(name), os.O_RDONLY)
                try:
                    # length=0 maps the whole file as of this call, avoiding a
                    # stat-then-map race with daemon recreation.
                    self._mmap = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                finally:
                    os.close(fd)
            except (OSError, ValueError) as exc:
                raise FileNotFoundError(
                    f"shared-memory segment {name!r} cannot be opened or mapped"
                ) from exc
            self._size = len(self._mmap)
            self._buf = memoryview(self._mmap)
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._size = self._shm.size
            self._buf = memoryview(self._shm.buf)

    @property
    def buf(self) -> memoryview:
        """The mapped segment bytes (read-only on Linux)."""
        return self._buf

    @property
    def size(self) -> int:
        """The mapped size in bytes."""
        return self._size

    def close(self) -> None:
        """Release the memoryview and unmap the segment. Idempotent.

        Never deletes the segment itself: its lifecycle belongs to the
        daemon.
        """
        self._buf.release()
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None


@dataclass(frozen=True)
class SageMakerHyperPodLease:
    """A daemon lease protecting data stored in shared memory.

    Args:
        lease_id: Identifier returned by the ai-toolkit daemon.
        offsets: Shared-memory ``(offset, length)`` fragments for the object.
        expires_monotonic: ``time.monotonic()`` deadline after which the
            daemon may reclaim the fragments.
    """

    lease_id: str
    offsets: tuple[tuple[int, int], ...]
    expires_monotonic: float = float("inf")

    @property
    def size(self) -> int:
        """Return the total number of bytes covered by the lease."""
        return sum(length for _, length in self.offsets)

    def is_expired(self) -> bool:
        """Return whether the lease is past its server-side TTL and must no
        longer be trusted for reads."""
        return time.monotonic() >= self.expires_monotonic


class SageMakerHyperPodClient:
    """Client for ai-toolkit's HTTP control and shared-memory data planes.

    A circuit breaker opens after ``failure_threshold`` consecutive
    connection failures; requests then fail fast, with one probe per
    ``circuit_cooldown_s`` to recover. HTTP statuses (e.g. the daemon's
    504 miss reply) never trip it.
    """

    #: Consecutive connection-class failures before the circuit opens.
    failure_threshold: int = 3
    #: Seconds between probe requests while the circuit is open.
    circuit_cooldown_s: float = 5.0

    def __init__(
        self,
        url: str,
        bucket: str = "lmcache",
        shared_memory_name: str = "shared_memory",
        max_concurrent_requests: int = 100,
        max_connections: int = 256,
        max_connections_per_host: int = 128,
        timeout_ms: int = 5000,
        lease_wait_timeout_ms: int = 1000,
        lease_ttl_ms: int = 30000,
        put_stream_chunk_bytes: int = _DEFAULT_STREAM_CHUNK_BYTES,
        max_lease_size_mb: float | None = None,
        use_https: bool = False,
    ) -> None:
        """Initialize the ai-toolkit client.

        Args:
            url: Daemon URL (``sagemaker-hyperpod://host:port``).
            bucket: ai-toolkit cache namespace.
            shared_memory_name: POSIX shared-memory segment containing cache data.
            max_concurrent_requests: Maximum concurrent HTTP requests.
            max_connections: Maximum total pooled HTTP connections.
            max_connections_per_host: Maximum pooled connections per host.
            timeout_ms: HTTP transport timeout in milliseconds (PUT and
                lease release).
            lease_wait_timeout_ms: Budget the daemon may spend holding a
                lease request before answering; bounds worst-case lookup
                latency.
            lease_ttl_ms: Server-side lease lifetime in milliseconds.
            put_stream_chunk_bytes: PUT streaming chunk size.
            max_lease_size_mb: Optional upper bound for accepted leases.
            use_https: Use HTTPS instead of HTTP for daemon requests.

        Raises:
            RuntimeError: If the shared-memory segment cannot be opened.
        """
        self._base_url = self.normalize_url(url, use_https=use_https)
        self._bucket = bucket
        self._max_connections = max_connections
        self._max_connections_per_host = max_connections_per_host
        self._timeout_s = timeout_ms / 1000.0
        self._timeout_ms = timeout_ms
        self._lease_wait_timeout_ms = lease_wait_timeout_ms
        self._lease_wait_timeout_s = lease_wait_timeout_ms / 1000.0
        self._lease_ttl_s = lease_ttl_ms / 1000.0
        self._put_stream_chunk_bytes = put_stream_chunk_bytes
        self._max_lease_size_bytes = (
            int(max_lease_size_mb * 1024 * 1024)
            if max_lease_size_mb is not None
            else None
        )
        self._request_gate = asyncio.Semaphore(max_concurrent_requests)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._status_lock = threading.Lock()
        self._consecutive_failures = 0
        self._last_error: str | None = None
        self._last_success_monotonic: float | None = None
        self._next_probe_monotonic = 0.0

        self._shared_memory_name = shared_memory_name
        self._shared_memory_lock = threading.Lock()
        try:
            self._shared_memory = _ReadOnlySharedMemoryMapping(shared_memory_name)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ai-toolkit shared-memory segment {shared_memory_name!r} not found"
            ) from exc
        self._shared_memory_view = self._shared_memory.buf
        self._shared_memory_identity = self._segment_identity()
        self._closed = False
        logger.info(
            "Attached ai-toolkit shared-memory segment %s (%.1f MiB)",
            shared_memory_name,
            self._shared_memory.size / (1 << 20),
        )

    @staticmethod
    def normalize_url(url: str, use_https: bool = False) -> str:
        """Normalize a ``sagemaker-hyperpod://`` daemon URL to its
        transport form.

        Args:
            url: Daemon URL using the ``sagemaker-hyperpod`` scheme.
            use_https: Use HTTPS instead of HTTP for the transport.

        Returns:
            The HTTP(S) transport URL without a trailing slash.

        Raises:
            ValueError: If the URL is empty, uses another scheme, or lacks
                a host.
        """
        value = url.strip().rstrip("/")
        if not value.startswith("sagemaker-hyperpod://"):
            raise ValueError("url must use the sagemaker-hyperpod:// scheme")
        scheme = "https://" if use_https else "http://"
        value = scheme + value.removeprefix("sagemaker-hyperpod://")
        parsed = urllib.parse.urlsplit(value)
        if not parsed.netloc:
            raise ValueError("url must include a host")
        return value

    async def put(self, key: str, data: memoryview) -> bool:
        """Store one object in ai-toolkit.

        Args:
            key: Backend object key before URL escaping.
            data: Byte-oriented object data.

        Returns:
            ``True`` for a successful write or an existing object conflict.
        """
        view = data.cast("B") if data.format != "B" else data

        async def stream() -> AsyncIterator[memoryview]:
            for offset in range(0, len(view), self._put_stream_chunk_bytes):
                yield view[offset : offset + self._put_stream_chunk_bytes]

        response = await self._request(
            "PUT",
            self._object_url(key),
            data=stream(),
            headers={"Content-Length": str(len(view))},
        )
        return response is not None and response[0] in (_HTTP_OK, _HTTP_CONFLICT)

    async def acquire_lease(self, key: str) -> SageMakerHyperPodLease | None:
        """Acquire a daemon lease that pins one object in shared memory.

        The daemon may hold the request for up to
        ``lease_wait_timeout_ms`` before answering; the HTTP timeout is
        that budget plus a margin so the daemon's answer always arrives
        before the client aborts.

        Args:
            key: Backend object key before URL escaping.

        Returns:
            The acquired lease, or ``None`` when the object is missing, the
            lease response is malformed, or the lease exceeds
            ``max_lease_size_mb``. Invalid or oversized leases are released
            best-effort before returning ``None``.
        """
        # Stamp the expiry conservatively, from before the request is sent.
        acquire_start = time.monotonic()
        response = await self._request(
            "POST",
            f"{self._object_url(key)}/leases",
            params={
                "timeout_ms": self._lease_wait_timeout_ms,
                "ttl_s": self._lease_ttl_s,
            },
            timeout=aiohttp.ClientTimeout(
                total=self._lease_wait_timeout_s
                + _LEASE_WAIT_TIMEOUT_MARGIN_MS / 1000.0
            ),
        )
        if response is None or response[0] != _HTTP_OK or response[1] is None:
            return None

        body = response[1]
        if not isinstance(body, dict):
            logger.warning("Malformed ai-toolkit lease response for key %s", key)
            return None
        try:
            offsets = tuple(
                (int(item["offset"]), int(item["len"]))
                for item in body.get("offsets", [])
            )
            lease = SageMakerHyperPodLease(
                str(body["id"]),
                offsets,
                expires_monotonic=acquire_start + self._lease_ttl_s,
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Malformed ai-toolkit lease response for key %s", key)
            return None
        if not offsets or any(offset < 0 or length <= 0 for offset, length in offsets):
            await self.release_lease(lease)
            return None
        if (
            self._max_lease_size_bytes is not None
            and lease.size > self._max_lease_size_bytes
        ):
            await self.release_lease(lease)
            return None
        return lease

    def copy_from_lease(
        self,
        lease: SageMakerHyperPodLease,
        destination: memoryview,
    ) -> bool:
        """Copy a leased object from shared memory into ``destination``.

        The lease and destination sizes must match exactly, all fragments
        are bounds-checked before the first destination byte is written,
        and the whole copy must finish before the lease expiry (an expired
        lease no longer pins its fragments). The mapping is reopened if
        ai-toolkit recreated the shared-memory segment.

        Args:
            lease: Lease whose fragments describe the object to copy.
            destination: Writable byte-oriented buffer for the full object.

        Returns:
            ``True`` when every fragment was validated and copied, ``False``
            when the lease expired (before or during the copy), sizes
            mismatch, any fragment is out of bounds, or the client is
            closed. On ``False`` the destination contents must be treated
            as invalid.
        """
        if lease.is_expired():
            logger.warning("Refusing to copy through expired lease %s", lease.lease_id)
            return False
        view = destination.cast("B") if destination.format != "B" else destination
        if lease.size != len(view):
            return False

        with self._shared_memory_lock:
            if self._closed or not self._refresh_shared_memory_locked():
                return False

            shm_size = self._shared_memory.size
            if any(
                offset < 0 or length <= 0 or offset + length > shm_size
                for offset, length in lease.offsets
            ):
                return False

            copied = 0
            for offset, length in lease.offsets:
                view[copied : copied + length] = self._shared_memory_view[
                    offset : offset + length
                ]
                copied += length
            if lease.is_expired():
                # Expired mid-copy: the destination cannot be trusted.
                logger.warning(
                    "Discarding copy: lease %s expired mid-copy",
                    lease.lease_id,
                )
                return False
            return copied == len(view)

    async def release_lease(self, lease: SageMakerHyperPodLease) -> bool:
        """Release a previously acquired lease.

        Args:
            lease: Lease returned by :meth:`acquire_lease`.

        Returns:
            ``True`` when the daemon confirmed the release, ``False`` on any
            HTTP failure. An unreleased lease expires server-side after its TTL.
        """
        response = await self._request(
            "POST",
            f"{self._base_url}/v1/leases/"
            f"{urllib.parse.quote(lease.lease_id, safe='')}/release",
        )
        return response is not None and response[0] == _HTTP_OK

    async def close(self) -> None:
        """Close HTTP and shared-memory resources. This method is idempotent."""
        with self._shared_memory_lock:
            if self._closed:
                return
            self._closed = True
            self._shared_memory_view.release()
            self._shared_memory.close()
        if self._session is not None:
            await self._session.close()
            self._session = None

    def report_status(self) -> dict[str, object]:
        """Return shared-memory and HTTP transport health.

        Returns:
            A dictionary with ``is_healthy`` plus circuit-breaker and
            shared-memory diagnostics.
        """
        with self._status_lock:
            failures = self._consecutive_failures
            last_error = self._last_error
            last_success = self._last_success_monotonic
        circuit_open = failures >= self.failure_threshold
        with self._shared_memory_lock:
            current_identity = self._segment_identity()
            shared_memory_current = not self._closed and (
                current_identity == self._shared_memory_identity
                if os.path.isdir(_SHM_DIR)
                else True
            )
        return {
            "is_healthy": (
                not self._closed and shared_memory_current and not circuit_open
            ),
            "shared_memory_current": shared_memory_current,
            "circuit_open": circuit_open,
            "consecutive_http_failures": failures,
            "last_http_error": last_error,
            "last_http_success_monotonic": last_success,
        }

    def _segment_identity(self) -> tuple[int, int] | None:
        """Return the current Linux POSIX shared-memory device/inode identity."""
        path = _shm_segment_path(self._shared_memory_name)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return stat.st_dev, stat.st_ino

    def _refresh_shared_memory_locked(self) -> bool:
        """Reattach when ai-toolkit replaced its POSIX shared-memory segment.

        The caller must hold ``_shared_memory_lock``.
        """
        current_identity = self._segment_identity()
        if current_identity is None:
            # No /dev/shm (non-Linux): keep the existing mapping.
            return True
        if current_identity == self._shared_memory_identity:
            return True

        try:
            replacement = _ReadOnlySharedMemoryMapping(self._shared_memory_name)
        except FileNotFoundError:
            return False

        old_view = self._shared_memory_view
        old_shared_memory = self._shared_memory
        self._shared_memory = replacement
        self._shared_memory_view = replacement.buf
        self._shared_memory_identity = current_identity
        old_view.release()
        old_shared_memory.close()
        logger.info(
            "Reattached ai-toolkit shared-memory segment %s (%.1f MiB)",
            self._shared_memory_name,
            replacement.size / (1 << 20),
        )
        return True

    def _record_http_success(self) -> None:
        with self._status_lock:
            self._consecutive_failures = 0
            self._last_error = None
            self._last_success_monotonic = time.monotonic()

    def _record_http_failure(self, error: object) -> None:
        with self._status_lock:
            self._consecutive_failures += 1
            self._last_error = str(error)
            if self._consecutive_failures == self.failure_threshold:
                logger.error(
                    "ai-toolkit circuit opened after %d consecutive failures; "
                    "probing every %.1fs (last error: %s)",
                    self._consecutive_failures,
                    self.circuit_cooldown_s,
                    self._last_error,
                )
            if self._consecutive_failures >= self.failure_threshold:
                self._next_probe_monotonic = time.monotonic() + self.circuit_cooldown_s

    def _fail_fast(self) -> bool:
        """Return ``True`` when the circuit is open and no probe is due.

        When the cooldown has elapsed, the probe window is re-armed and the
        current request is let through as the single probe for this window.
        """
        with self._status_lock:
            if self._consecutive_failures < self.failure_threshold:
                return False
            now = time.monotonic()
            if now < self._next_probe_monotonic:
                return True
            self._next_probe_monotonic = now + self.circuit_cooldown_s
            return False

    def _object_url(self, key: str) -> str:
        encoded_key = urllib.parse.quote(key, safe="")
        encoded_bucket = urllib.parse.quote(self._bucket, safe="")
        return f"{self._base_url}/v1/kv/{encoded_bucket}/{encoded_key}"

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> tuple[int, dict | None] | None:
        if self._fail_fast():
            return None
        try:
            async with self._request_gate:
                session = await self._get_session()
                async with session.request(method, url, **kwargs) as response:
                    body = None
                    if response.content_type == "application/json":
                        try:
                            body = await response.json()
                        except (aiohttp.ContentTypeError, ValueError):
                            logger.warning("Invalid JSON response from %s", url)
                    # Any response proves the transport is healthy; only
                    # connection failures feed the circuit breaker.
                    self._record_http_success()
                    return response.status, body
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._record_http_failure(exc)
            logger.warning("ai-toolkit %s %s failed: %s", method, url, exc)
            return None
        except Exception as exc:
            self._record_http_failure(exc)
            logger.exception("Unexpected ai-toolkit %s %s failure", method, url)
            return None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self._max_connections,
                    limit_per_host=self._max_connections_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True,
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=self._timeout_s),
                    headers={"User-Agent": "LMCache-SageMaker-HyperPod/1.0"},
                )
        return self._session
