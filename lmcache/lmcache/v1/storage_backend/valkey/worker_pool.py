# SPDX-License-Identifier: Apache-2.0
"""
Shared Valkey worker pool built on the ``valkey-glide`` sync client.

A pool of ``num_workers`` threads, each holding its own glide client via
``threading.local()``. The sync glide client is used (rather than async)
because it is the only glide API that supports **copy-reduced** SET
(``bytearray``/``memoryview`` args) and buffer GET — async glide adds an
unavoidable intermediate ``bytes`` allocation on top of that, due to its
Protobuf-based IPC layer. A sync call blocks its calling thread, so
concurrency comes from running N worker threads each with an independent
client.

Both the non-MP ``ValkeyConnector`` (``RemoteConnector``) and the MP-mode
``ValkeyL2Adapter`` (``L2AdapterInterface``) build on this single class,
so the glide client lifecycle, capability probing, buffer-format
handling, and reliable shutdown live in exactly one place.

Requires the ``valkey-glide-sync`` package (``>= 2.3.0``), which provides
the ``glide_sync`` module with copy-reduced SET and buffer GET. Note the
plain ``valkey-glide`` package ships only the *async* client and is not
sufficient. The dependency is imported lazily so installations that don't
use Valkey never need it.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional
import inspect
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

#: Default per-request timeout (seconds).
DEFAULT_REQUEST_TIMEOUT_SECS: float = 5.0
#: Default initial-connection timeout (seconds).
DEFAULT_CONNECTION_TIMEOUT_SECS: float = 10.0

#: Sentinel returned by ``_do_get_into`` when the key is absent.
GET_MISS: int = -1

#: Upper bound on how long ``close()`` waits for all workers to gather at
#: the shutdown barrier. Workers idle at close time gather in microseconds;
#: this only caps the wait when a worker is stuck in a slow glide call, so
#: close falls through to its best-effort path promptly
CLOSE_BARRIER_TIMEOUT_SECS: float = 1.0

#: valkey-glide release that introduced the ``client_info_tag`` client-config
#: option (appends ``(lmcache:<version>)`` to the GLIDE library name reported
#: via ``CLIENT SETINFO LIB-NAME``). Builds older than this do not accept the
#: kwarg, so it is applied only when feature-detected — see
#: ``_glide_config_supports_client_info_tag``. Ref: valkey-io/valkey-glide#6389.
CLIENT_INFO_TAG_MIN_GLIDE_VERSION: str = "2.5.0"


def _lmcache_client_info_tag() -> str:
    """Return the LMCache attribution tag for GLIDE's ``client_info_tag``.

    Produces ``lmcache:<version>`` (e.g. ``lmcache:0.3.9``) so LMCache
    connections are attributable in ``CLIENT INFO`` / ``CLIENT LIST`` as
    ``GlidePySync(lmcache:<version>)``, preserving GLIDE's own library
    identity / adoption metrics while breaking usage down per framework.

    ``client_info_tag`` forbids whitespace (GLIDE raises ``ValueError``
    otherwise), so if the resolved version string is unexpectedly odd we
    fall back to a bare ``lmcache`` tag rather than risk failing every
    connection.
    """
    # First Party
    from lmcache import __version__

    tag = f"lmcache:{__version__}"
    if any(ch.isspace() for ch in tag):
        return "lmcache"
    return tag


def _glide_config_supports_client_info_tag(config_cls: type) -> bool:
    """Whether this glide build's client config accepts ``client_info_tag``.

    Feature-detected from the constructor signature rather than by comparing
    version strings: ``client_info_tag`` was added in valkey-glide
    ``2.5.0`` (see ``CLIENT_INFO_TAG_MIN_GLIDE_VERSION``), and older clients
    — including custom sync builds that predate it — would raise
    ``TypeError`` on the unexpected kwarg. Detecting the parameter directly
    is both the gate the feature needs (only apply the tag once the client
    supports it) and robust to non-standard/absent version strings.
    """
    try:
        return "client_info_tag" in inspect.signature(config_cls).parameters
    except (TypeError, ValueError):
        return False


def _to_str(value: Any) -> str:
    """Decode ``bytes`` to ``str`` (utf-8, replacing bad bytes); pass
    through anything already string-like."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _parse_memory_info(text: str) -> dict[str, int]:
    """Extract ``used_memory`` / ``maxmemory`` from an ``INFO memory`` body.

    Args:
        text: The raw ``INFO memory`` section text from one node.

    Returns:
        ``{"used_memory": int, "maxmemory": int}``. Missing or
        unparsable fields default to ``0``.
    """
    used = 0
    maxm = 0
    for line in text.splitlines():
        if line.startswith("used_memory:"):
            try:
                used = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("maxmemory:"):
            try:
                maxm = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return {"used_memory": used, "maxmemory": maxm}


class ValkeyWorkerPool:
    """A thread pool of per-thread glide sync clients.

    Each worker thread lazily constructs and caches its own
    ``GlideClient`` (standalone) or ``GlideClusterClient`` (cluster) via
    ``threading.local()``. Operations are submitted as ``Future`` objects
    and run on whichever worker thread the executor picks; because each
    thread has its own client, this yields true parallel I/O while the
    GIL is released across the glide FFI boundary.

    The pool speaks only ``str`` keys and byte buffers — it has no
    knowledge of ``ObjectKey`` / ``CacheEngineKey`` or any LMCache type,
    so it can be shared by both the connector and the L2 adapter.

    Args:
        addresses: One or more ``(host, port)`` seed nodes. In standalone
            mode only the first is used; in cluster mode glide
            auto-discovers the full topology from any reachable seed.
        num_workers: Number of worker threads (and clients); must be > 0.
        username: Optional auth username.
        password: Optional auth password.
        request_timeout: Per-request timeout in seconds.
        connection_timeout: Initial connection timeout in seconds.
        tls_enable: Whether to use TLS.
        cluster_mode: ``True`` to build ``GlideClusterClient``.
        database_id: Standalone-only logical DB id; ignored in cluster
            mode.
        ttl_seconds: If set, every key is written with this expiry (in
            seconds) so Valkey/Redis ``volatile-*`` eviction policies can
            reclaim cache keys under memory pressure. ``None`` (default)
            disables TTL — keys are persisted without expiry.

    Raises:
        ValueError: If ``addresses`` is empty or ``num_workers <= 0``.
        RuntimeError: If ``valkey-glide`` is not installed (raised during
            warmup).
    """

    def __init__(
        self,
        addresses: list[tuple[str, int]],
        num_workers: int,
        username: str = "",
        password: str = "",
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECS,
        tls_enable: bool = False,
        cluster_mode: bool = False,
        database_id: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not addresses:
            raise ValueError("addresses must contain at least one (host, port)")
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        self.num_workers: int = num_workers
        self._addresses: list[tuple[str, int]] = list(addresses)
        self._username: str = username
        self._password: str = password
        self._request_timeout: float = request_timeout
        self._request_timeout_ms: int = int(request_timeout * 1000)
        self._connection_timeout: float = connection_timeout
        self._connection_timeout_ms: int = int(connection_timeout * 1000)
        self._tls_enable: bool = tls_enable
        self._cluster_mode: bool = cluster_mode
        self._database_id: Optional[int] = database_id
        self._ttl_seconds: Optional[int] = ttl_seconds
        # Lazily built ExpirySet (see _build_expiry); glide_sync is only
        # imported on worker threads, so this stays None until the first SET.
        self._expiry_obj: Any = None

        # Per-thread glide client holder. The object itself is shared by
        # the pool, but ``self._local.client`` resolves to a separate
        # value per worker thread — so each thread builds and reuses its
        # own client (glide clients are not shared across threads).
        self._local: threading.local = threading.local()
        self._has_buffer_get: Optional[bool] = None
        self._capability_lock: threading.Lock = threading.Lock()
        self._closed: bool = False

        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="valkey",
        )
        # Warm up: force each worker to build its client up-front so any
        # connection / auth / missing-dependency error fails fast here
        # rather than on the first real operation.
        warmup = [self._executor.submit(self._get_client) for _ in range(num_workers)]
        for f in warmup:
            f.result(timeout=connection_timeout)

        logger.info(
            "ValkeyWorkerPool: %d threads, mode=%s, nodes=%s, tls=%s, "
            "buffer_get=%s, ttl_seconds=%s",
            num_workers,
            "cluster" if cluster_mode else "standalone",
            self._addresses,
            tls_enable,
            self._has_buffer_get,
            ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the calling thread's glide client, creating it lazily.

        ``glide_sync`` is imported here (not at module load) so that
        LMCache installations that don't use Valkey never need the
        dependency.

        Returns:
            A ``GlideClient`` or ``GlideClusterClient`` instance.

        Raises:
            RuntimeError: If ``valkey-glide`` is not installed.
        """
        client = getattr(self._local, "client", None)
        if client is not None:
            return client

        try:
            # Third Party
            import glide_sync  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "Valkey support requires the glide_sync module. "
                "Install: pip install 'valkey-glide-sync>=2.3.0' "
                "(note: the plain 'valkey-glide' package is async-only)"
            ) from e

        credentials = None
        if self._username or self._password:
            credentials = glide_sync.ServerCredentials(self._username, self._password)

        addresses = [
            glide_sync.NodeAddress(host, port) for host, port in self._addresses
        ]

        if self._cluster_mode:
            advanced = glide_sync.AdvancedGlideClusterClientConfiguration(
                connection_timeout=self._connection_timeout_ms,
            )
            cfg_kwargs: dict = {
                "addresses": addresses,
                "request_timeout": self._request_timeout_ms,
                "use_tls": self._tls_enable,
                "advanced_config": advanced,
            }
            if credentials is not None:
                cfg_kwargs["credentials"] = credentials
            # Attribute LMCache in CLIENT INFO/LIST as
            # ``GlidePySync(lmcache:<version>)`` — only when the installed
            # glide client supports it (valkey-glide >= 2.5.0); older builds
            # would reject the unknown kwarg.
            if _glide_config_supports_client_info_tag(
                glide_sync.GlideClusterClientConfiguration
            ):
                cfg_kwargs["client_info_tag"] = _lmcache_client_info_tag()
            config = glide_sync.GlideClusterClientConfiguration(**cfg_kwargs)
            client = glide_sync.GlideClusterClient.create(config)
        else:
            # Standalone mode — only the first seed is meaningful; it also
            # supports database_id selection.
            advanced = glide_sync.AdvancedGlideClientConfiguration(
                connection_timeout=self._connection_timeout_ms,
            )
            cfg_kwargs = {
                "addresses": addresses[:1],
                "request_timeout": self._request_timeout_ms,
                "use_tls": self._tls_enable,
                "advanced_config": advanced,
            }
            if credentials is not None:
                cfg_kwargs["credentials"] = credentials
            if self._database_id is not None:
                cfg_kwargs["database_id"] = self._database_id
            # Attribute LMCache in CLIENT INFO/LIST as
            # ``GlidePySync(lmcache:<version>)`` — only when the installed
            # glide client supports it (valkey-glide >= 2.5.0); older builds
            # would reject the unknown kwarg.
            if _glide_config_supports_client_info_tag(
                glide_sync.GlideClientConfiguration
            ):
                cfg_kwargs["client_info_tag"] = _lmcache_client_info_tag()
            config = glide_sync.GlideClientConfiguration(**cfg_kwargs)
            client = glide_sync.GlideClient.create(config)

        self._local.client = client

        # Probe buffer-GET capability once. Concurrent warmup threads race
        # here, so guard the one-time write with a lock.
        with self._capability_lock:
            if self._has_buffer_get is None:
                self._has_buffer_get = (
                    "buffer" in inspect.signature(client.get).parameters
                )

        return client

    @property
    def has_buffer_get(self) -> bool:
        """Whether the glide client supports copy-reduced buffer GET.

        Returns:
            ``True`` if ``client.get`` accepts a ``buffer=`` argument
            (glide >= 2.3.0). Forces creation of a client on a worker
            thread if the capability has not been probed yet.
        """
        if self._has_buffer_get is None:
            self._executor.submit(self._get_client).result(
                timeout=self._connection_timeout
            )
        return bool(self._has_buffer_get)

    # ------------------------------------------------------------------
    # Worker-side primitives (run on a worker thread)
    # ------------------------------------------------------------------

    def _build_expiry(self) -> Any:
        """Lazily build (and cache) the glide ``ExpirySet`` for SET calls.

        Built lazily so ``glide_sync`` is only imported on worker threads
        (matching :meth:`_get_client`). The resulting ``ExpirySet`` is an
        immutable value object, so it is safe to share across threads; a
        benign race during lazy init just rebuilds an equivalent object.

        Returns:
            A ``glide_sync.ExpirySet`` configured with ``ttl_seconds``.

        Raises:
            ValueError: If ``ttl_seconds`` was not configured.
        """
        exp = self._expiry_obj
        if exp is None:
            if self._ttl_seconds is None:
                raise ValueError("ttl_seconds must be set to build an ExpirySet")
            # Third Party
            import glide_sync  # type: ignore[import-untyped]

            exp = glide_sync.ExpirySet(glide_sync.ExpiryType.SEC, self._ttl_seconds)
            self._expiry_obj = exp
        return exp

    def _do_set(self, key_str: str, data: Any) -> None:
        """SET a key. ``data`` is a memoryview / bytes / bytearray.

        With glide >= 2.3.0 and a memoryview/bytearray, this is a
        copy-reduced write. When ``ttl_seconds`` is configured, the key is
        written with an expiry so Valkey/Redis ``volatile-*`` eviction
        policies can reclaim it under memory pressure; otherwise the key
        is persisted indefinitely.
        """
        if self._ttl_seconds is not None:
            self._get_client().set(key_str.encode(), data, expiry=self._build_expiry())
        else:
            self._get_client().set(key_str.encode(), data)

    def _get_scratch(self, size: int) -> bytearray:
        """Per-thread reusable staging buffer for buffer-GET (issue #6215).

        glide writes into this thread-private bytearray (not slab memory),
        then we copy into the destination under the GIL. Eliminates the
        dangling-slot race without pinning or touching ref_count/pin_count.
        The buffer is grown on demand and reused across GETs on the same
        worker thread, so there is no per-call allocation.
        """
        buf = getattr(self._local, "scratch", None)
        if buf is None or len(buf) < size:
            buf = bytearray(size)
            self._local.scratch = buf
        return buf

    def _do_get_into(self, key_str: str, buf: Any) -> int:
        """GET a key directly into ``buf``.

        Fixed-size KV chunks must round-trip exactly, so the read size is
        validated against the destination buffer length here — the single
        place shared by both the connector and the L2 adapter. A short,
        truncated, or oversized value is rejected as a miss rather than
        leaving a partially-filled buffer with stale tail bytes.

        ``buf`` is cast to an unsigned-byte (``"B"``) memoryview if it
        isn't already, because glide's buffer protocol rejects
        non-byte-format views (e.g. a float-typed tensor view).

        Safety (issue #6215): glide's native buffer-GET writes with the
        GIL released, so it must never write directly into ``dst`` when
        ``dst`` aliases slab-allocator memory that another thread could
        recycle mid-write. We stage into a thread-private scratch buffer
        and copy into ``dst`` under the GIL.

        Returns:
            The number of payload bytes copied, which always equals the
            buffer length on a hit, or :data:`GET_MISS` (``-1``) if the
            key is absent or its stored size does not match the buffer.
        """
        client = self._get_client()
        dst = buf if isinstance(buf, memoryview) else memoryview(buf)
        if dst.format != "B":
            dst = dst.cast("B")
        size = len(dst)
        if self.has_buffer_get:
            scratch_view = memoryview(self._get_scratch(size))[:size]
            result = client.get(key_str.encode(), buffer=scratch_view)
            if result is None:
                return GET_MISS
            # glide buffer GET returns the number of bytes written. Real
            # glide sync (>= 2.3.0) reports this as an ASCII byte string
            # (e.g. ``b'4096'``); the fake used in unit tests may report
            # a plain ``int``. Anything else means we cannot trust the
            # byte count, so treat it as a miss rather than assume a
            # full-size read and defeat the size check below.
            if isinstance(result, (bytes, bytearray, memoryview)):
                try:
                    n = int(bytes(result))
                except ValueError:
                    logger.warning(
                        "Valkey buffer GET for key %s returned "
                        "non-numeric byte string %r; treating as miss.",
                        key_str,
                        bytes(result)[:32],
                    )
                    return GET_MISS
            elif isinstance(result, int) and not isinstance(result, bool):
                n = result
            else:
                logger.warning(
                    "Valkey buffer GET for key %s returned unexpected type %s; "
                    "treating as miss.",
                    key_str,
                    type(result).__name__,
                )
                return GET_MISS
            source: Any = scratch_view
        else:
            data = client.get(key_str.encode())
            if data is None:
                return GET_MISS
            n = len(data)
            source = data

        if n != size:
            logger.warning(
                "Valkey GET size mismatch for key %s: read %d bytes, "
                "expected %d; treating as miss.",
                key_str,
                n,
                size,
            )
            return GET_MISS
        # Sizes match: copy the full payload into the destination.
        dst[:] = source[:size]
        return n

    def _do_exists(self, key_str: str) -> bool:
        """Return whether ``key_str`` exists."""
        return bool(self._get_client().exists([key_str.encode()]))

    def _do_delete(self, key_str: str) -> bool:
        """Delete ``key_str``; return ``True`` iff a value was removed."""
        return int(self._get_client().delete([key_str.encode()])) > 0

    # ------------------------------------------------------------------
    # Submit API (returns Futures)
    # ------------------------------------------------------------------

    def submit_set(self, key_str: str, data: Any) -> Future:
        """Submit a SET; the future resolves to ``None`` on success."""
        return self._executor.submit(self._do_set, key_str, data)

    def submit_get_into(self, key_str: str, buf: Any) -> Future:
        """Submit a GET-into-buffer; future resolves to bytes-or-``-1``.

        See :meth:`_do_get_into` for the return-value contract.
        """
        return self._executor.submit(self._do_get_into, key_str, buf)

    def submit_exists(self, key_str: str) -> Future:
        """Submit an EXISTS; the future resolves to ``bool``."""
        return self._executor.submit(self._do_exists, key_str)

    def submit_delete(self, key_str: str) -> Future:
        """Submit a DEL; the future resolves to ``bool`` (removed?)."""
        return self._executor.submit(self._do_delete, key_str)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def node_memory_usage(self) -> dict[str, dict[str, int]]:
        """Return physical memory usage per Valkey node.

        Runs ``INFO memory`` routed to all cluster nodes and parses each
        node's ``used_memory`` / ``maxmemory``. This is the **server's
        real per-node memory**, distinct from LMCache's aggregate logical
        byte accounting — useful for spotting a hot shard in cluster mode.

        Returns:
            A dict mapping ``"host:port"`` to
            ``{"used_memory": int, "maxmemory": int}``. ``maxmemory`` is
            ``0`` when the node has no cap configured. Returns an empty
            dict in standalone mode, or if the query fails (the call is
            best-effort so a slow / unreachable node never breaks status
            reporting).
        """
        if not self._cluster_mode or self._closed:
            return {}
        try:
            return self._executor.submit(self._do_node_memory).result(
                timeout=self._request_timeout
            )
        except Exception as e:
            logger.debug("node_memory_usage query failed: %s", e)
            return {}

    def _do_node_memory(self) -> dict[str, dict[str, int]]:
        """Worker-side ``INFO memory`` fan-out to all nodes."""
        # Third Party
        from glide_shared.commands.core_options import (  # type: ignore[import-untyped]
            InfoSection,
        )
        from glide_shared.routes import AllNodes  # type: ignore[import-untyped]

        client = self._get_client()
        raw = client.info([InfoSection.MEMORY], AllNodes())

        out: dict[str, dict[str, int]] = {}
        if isinstance(raw, dict):
            # Multi-node route: address -> INFO bytes.
            for node, text in raw.items():
                out[_to_str(node)] = _parse_memory_info(_to_str(text))
        else:
            # Single-response fallback (not expected with AllNodes).
            out["<aggregate>"] = _parse_memory_info(_to_str(raw))
        return out

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _close_local_client(self, barrier: threading.Barrier) -> None:
        """Close the calling thread's client after all workers gather.

        The barrier guarantees every worker thread is simultaneously
        executing one close task before any proceeds, so each of the N
        tasks lands on a distinct thread and closes that thread's own
        client exactly once. Without it, a fast thread could grab
        multiple close tasks (no-ops after the first) while another
        thread's client is never closed.

        The wait is capped at :data:`CLOSE_BARRIER_TIMEOUT_SECS`: if a
        worker is stuck in a slow glide call the barrier can never gather,
        and a longer wait would only delay the best-effort close below.
        """
        try:
            barrier.wait(timeout=CLOSE_BARRIER_TIMEOUT_SECS)
        except Exception as exc:
            # BrokenBarrierError, TimeoutError (3.12+), etc. — fall through
            # to a best-effort close of whatever this thread owns so a
            # broken barrier never skips client cleanup.
            logger.debug(
                "ValkeyWorkerPool close barrier did not gather (%s); best-effort close",
                exc,
            )
        client = getattr(self._local, "client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception as exc:
            logger.debug("Error closing per-thread glide client: %s", exc)
        self._local.client = None

    def close(self) -> None:
        """Close every per-thread glide client and stop the worker pool.

        Each client is closed on its owning thread (matching glide's
        per-thread client model) with 1:1 coverage guaranteed by a
        barrier. Idempotent — repeated calls are no-ops.
        """
        if self._closed:
            return
        self._closed = True
        barrier = threading.Barrier(self.num_workers)
        close_futs = [
            self._executor.submit(self._close_local_client, barrier)
            for _ in range(self.num_workers)
        ]
        for f in close_futs:
            try:
                f.result(timeout=self._connection_timeout)
            except Exception as exc:
                logger.debug("Error during ValkeyWorkerPool close: %s", exc)
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("ValkeyWorkerPool closed")
