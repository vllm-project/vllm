# SPDX-License-Identifier: Apache-2.0
"""
Valkey L2 adapter backed by the official ``valkey-glide`` sync client.

Supports both standalone (``GlideClient``) and Valkey Cluster
(``GlideClusterClient``) topologies via the ``cluster_mode`` field. The
sync glide client is used (rather than async) because it is the only
glide API that supports copy-reduced SET (``bytearray``/``memoryview``
args) and buffer GET — async glide adds an unavoidable intermediate
``bytes`` allocation on top of that, due to its Protobuf-based IPC layer.

Concurrency model:
    The glide clients and worker threads live in the shared
    ``ValkeyWorkerPool`` (also used by the non-MP ``ValkeyConnector``). A
    batch ``submit_*_task`` fans out one per-key future per key; the
    per-key callbacks atomically decrement the batch's ``remaining``
    counter and the last finisher publishes the result and signals the
    matching event fd.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
import functools
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    _object_key_to_string,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.storage_backend.valkey.worker_pool import ValkeyWorkerPool

logger = init_logger(__name__)

#: Default per-request timeout (seconds).
DEFAULT_REQUEST_TIMEOUT_SECS: float = 5.0
#: Default initial-connection timeout (seconds).
DEFAULT_CONNECTION_TIMEOUT_SECS: float = 10.0

# Separator used to join ``key_prefix`` and the standard wire key. Mirrors
# the ``@`` separator inside ``_object_key_to_string`` so an empty prefix
# is bit-identical to the unprefixed wire format.
_KEY_SEP = "@"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_startup_nodes(raw: object) -> list[tuple[str, int]]:
    """Parse a comma-separated ``"host:port"`` startup-nodes string.

    Example: ``"host1:6379,host2:6379"``. In standalone mode only the
    first entry is used; in cluster mode glide discovers the rest from
    any reachable seed.

    Args:
        raw: A non-empty ``"host:port[,host:port...]"`` string.

    Returns:
        A list of ``(host, port)`` tuples; never empty.

    Raises:
        ValueError: If ``raw`` is not a string, is empty, or any entry
            is malformed (missing ``:`` or non-integer port).
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "startup_nodes is required and must be a non-empty "
            "'host:port[,host:port...]' string"
        )

    nodes: list[tuple[str, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        host, sep, port_str = chunk.rpartition(":")
        if not sep or not host:
            raise ValueError(f"startup_nodes entry must be 'host:port', got {chunk!r}")
        try:
            port = int(port_str)
        except ValueError as e:
            raise ValueError(
                f"startup_nodes entry has non-integer port: {chunk!r}"
            ) from e
        if port <= 0:
            raise ValueError(f"startup_nodes port must be > 0, got {chunk!r}")
        nodes.append((host, port))

    if not nodes:
        raise ValueError("startup_nodes parsed to an empty list")
    return nodes


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ValkeyL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the Valkey L2 adapter.

    Fields:
      * ``startup_nodes``: List of ``(host, port)`` tuples (the JSON
        config supplies this as a ``"host:port,host:port"`` string). In
        standalone mode only the first is used (a warning is logged for
        the rest). In cluster mode, glide auto-discovers the full
        topology from any reachable seed node.
      * ``cluster_mode``: When ``True``, build a ``GlideClusterClient``;
        otherwise a standalone ``GlideClient``.
      * ``username`` / ``password``: Optional auth credentials.
      * ``key_prefix``: Deployment-level key namespace; joined with the
        standard wire key via ``@``. Must not contain ``@``.
      * ``num_workers``: Size of the internal worker thread pool. Each
        worker holds an independent glide client.
      * ``tls_enable``: Enable TLS (required for managed Valkey services
        like ElastiCache Serverless).
      * ``database_id``: Standalone-only logical database id; ignored in
        cluster mode.
      * ``request_timeout`` / ``connection_timeout``: Glide timeouts in
        seconds.
      * ``max_capacity_gb``: Declared capacity for LMCache-driven
        eviction. ``0`` (default) disables it and defers to server-side
        eviction (recommended); per ``cache_salt`` quota policies still
        operate. ``> 0`` (plus an ``eviction`` block) makes LMCache own
        eviction — trustworthy only for a single writer that privately
        owns the cluster. Do not combine ``> 0`` with any server-side
        eviction or TTL, which drifts LMCache's byte accounting (see the
        design doc's eviction section).
      * ``ttl_seconds``: Opt-in per-key TTL in seconds; ``None`` (default)
        writes keys with no expiry. Enables server-side time-based expiry
        (required by ``volatile-*`` policies). Do not combine with
        ``max_capacity_gb > 0``.
    """

    def __init__(
        self,
        startup_nodes: list[tuple[str, int]],
        cluster_mode: bool = False,
        username: str = "",
        password: str = "",
        key_prefix: str = "",
        num_workers: int = 8,
        tls_enable: bool = False,
        database_id: Optional[int] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECS,
        max_capacity_gb: float = 0.0,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Initialize a ValkeyL2AdapterConfig with validated fields.

        Args:
            startup_nodes: Non-empty list of ``(host, port)`` pairs.
            cluster_mode: ``True`` to use ``GlideClusterClient``.
            username: Optional auth username.
            password: Optional auth password.
            key_prefix: Optional namespace prefix; must not contain
                ``@``.
            num_workers: Worker thread count; must be > 0.
            tls_enable: Whether to use TLS.
            database_id: Standalone DB id; ignored when
                ``cluster_mode=True``.
            request_timeout: Per-request timeout in seconds.
            connection_timeout: Initial connection timeout in seconds.
            max_capacity_gb: Aggregate capacity in GB for eviction; ``0``
                disables global eviction.
            ttl_seconds: Opt-in per-key TTL in seconds; must be a positive
                integer when set. ``None`` (default) writes keys with no
                expiry. Enables server-side time-based expiry (required by
                ``volatile-*`` policies). Do not combine with
                ``max_capacity_gb > 0`` (logs a warning; see the design
                doc's eviction section).

        Raises:
            ValueError: If any validation fails (empty
                ``startup_nodes``, non-positive ``num_workers``,
                negative capacity, non-positive ``ttl_seconds``, ``@`` in
                ``key_prefix``, etc.).
        """
        super().__init__()
        if not startup_nodes:
            raise ValueError("startup_nodes must contain at least one (host, port)")
        for node in startup_nodes:
            if not (
                isinstance(node, tuple)
                and len(node) == 2
                and isinstance(node[0], str)
                and node[0]
                and isinstance(node[1], int)
                and not isinstance(node[1], bool)
                and node[1] > 0
            ):
                raise ValueError(f"invalid startup_nodes entry: {node!r}")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int):
            raise ValueError("num_workers must be an integer")
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if _KEY_SEP in key_prefix:
            raise ValueError(
                f"key_prefix must not contain {_KEY_SEP!r}: {key_prefix!r}"
            )
        if not cluster_mode and len(startup_nodes) > 1:
            logger.warning(
                "Valkey: cluster_mode=False but %d startup_nodes provided; "
                "only the first (%s:%d) will be used",
                len(startup_nodes),
                startup_nodes[0][0],
                startup_nodes[0][1],
            )
        if cluster_mode and database_id is not None:
            logger.warning(
                "Valkey: database_id=%d is ignored in cluster_mode",
                database_id,
            )
            database_id = None
        if max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be >= 0")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        if connection_timeout <= 0:
            raise ValueError("connection_timeout must be > 0")
        if ttl_seconds is not None:
            # bool is an int subclass, so reject it (True would be 1 second).
            if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
                raise ValueError("ttl_seconds must be a positive integer or None")
            if ttl_seconds <= 0:
                raise ValueError("ttl_seconds must be a positive integer or None")
            if max_capacity_gb > 0:
                # Server-side expiry is not reported to LMCache, so its byte
                # accounting / LRU drift (see the eviction section in docs).
                logger.warning(
                    "Valkey: ttl_seconds set with max_capacity_gb > 0; "
                    "server-side expiry drifts LMCache eviction accounting. "
                    "Prefer max_capacity_gb=0 when using a TTL."
                )

        self.startup_nodes: list[tuple[str, int]] = startup_nodes
        self.cluster_mode: bool = cluster_mode
        self.username: str = username
        self.password: str = password
        self.key_prefix: str = key_prefix
        self.num_workers: int = num_workers
        self.tls_enable: bool = tls_enable
        self.database_id: Optional[int] = database_id
        self.request_timeout: float = request_timeout
        self.connection_timeout: float = connection_timeout
        self.max_capacity_gb: float = max_capacity_gb
        self.ttl_seconds: Optional[int] = ttl_seconds

    @classmethod
    def from_dict(cls, d: dict) -> "ValkeyL2AdapterConfig":
        """Build a config instance from a JSON-derived dict.

        Requires ``startup_nodes`` as a ``"host:port[,host:port...]"``
        string (see ``_parse_startup_nodes``); every other field is
        optional and falls back to its default.

        Args:
            d: Parsed JSON dict (must include ``type``: ``"valkey"``).

        Returns:
            A validated ``ValkeyL2AdapterConfig`` instance.

        Raises:
            ValueError: If required keys are missing or values are
                invalid.
        """
        startup_nodes = _parse_startup_nodes(d.get("startup_nodes"))

        cluster_mode = bool(d.get("cluster_mode", False))
        username = str(d.get("username", ""))
        password = str(d.get("password", ""))
        key_prefix = str(d.get("key_prefix", ""))

        num_workers_raw = d.get("num_workers", 8)
        if isinstance(num_workers_raw, bool) or not isinstance(num_workers_raw, int):
            raise ValueError("num_workers must be an integer")
        num_workers = num_workers_raw

        tls_enable = bool(d.get("tls_enable", False))

        database_id_raw = d.get("database_id")
        database_id = int(database_id_raw) if database_id_raw is not None else None

        request_timeout = float(d.get("request_timeout", DEFAULT_REQUEST_TIMEOUT_SECS))
        connection_timeout = float(
            d.get("connection_timeout", DEFAULT_CONNECTION_TIMEOUT_SECS)
        )

        max_capacity_raw = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_raw, (int, float)) or isinstance(
            max_capacity_raw, bool
        ):
            raise ValueError("max_capacity_gb must be a number")
        max_capacity_gb = float(max_capacity_raw)

        ttl_raw = d.get("ttl_seconds")
        if ttl_raw is None:
            ttl_seconds = None
        elif isinstance(ttl_raw, bool) or not isinstance(ttl_raw, int):
            raise ValueError("ttl_seconds must be a positive integer or omitted")
        else:
            ttl_seconds = ttl_raw

        return cls(
            startup_nodes=startup_nodes,
            cluster_mode=cluster_mode,
            username=username,
            password=password,
            key_prefix=key_prefix,
            num_workers=num_workers,
            tls_enable=tls_enable,
            database_id=database_id,
            request_timeout=request_timeout,
            connection_timeout=connection_timeout,
            max_capacity_gb=max_capacity_gb,
            ttl_seconds=ttl_seconds,
        )

    @classmethod
    def help(cls) -> str:
        """Return human-readable help describing the JSON config fields.

        Returns:
            A multi-line string used by the CLI to document the
            adapter's config schema.
        """
        return (
            "Valkey L2 adapter config fields:\n"
            "- startup_nodes (required): 'host:port' string, comma-"
            "separated for multiple seeds, e.g. 'h1:6379,h2:6379'. In "
            "standalone mode only the first entry is used.\n"
            "- cluster_mode (bool, default false): use "
            "GlideClusterClient when true.\n"
            "- username (str, default empty): auth username.\n"
            "- password (str, default empty): auth password.\n"
            "- key_prefix (str, default empty): deployment namespace "
            "joined with each key by '@'. Must not contain '@'.\n"
            "- num_workers (int, default 8): worker thread count "
            "(> 0).\n"
            "- tls_enable (bool, default false): enable TLS.\n"
            "- database_id (int, optional): standalone-only logical DB; "
            "ignored when cluster_mode=true.\n"
            "- request_timeout (float, default 5.0): per-request "
            "timeout (seconds).\n"
            "- connection_timeout (float, default 10.0): initial "
            "connection timeout (seconds).\n"
            "- max_capacity_gb (float, default 0): aggregate capacity "
            "for eviction; 0 disables global eviction. Recommended: set "
            "this > 0 and disable Valkey server-side eviction "
            "(maxmemory 0 / noeviction) so only LMCache evicts.\n"
            "- ttl_seconds (int, optional): opt-in per-key TTL in seconds; "
            "omit (default) for no expiry. Enables server-side time-based "
            "expiry (required by volatile-* policies). Do not combine with "
            "max_capacity_gb > 0."
        )


# ---------------------------------------------------------------------------
# Internal batch state
# ---------------------------------------------------------------------------


@dataclass
class _SubmitTaskBatchState:
    """Shared per-batch state used by per-key future callbacks.

    A batch holds one ``Future`` per key; each completion callback
    atomically updates ``per_key_ok[idx]`` and decrements ``remaining``
    under ``lock``. The last finisher (whose decrement drives
    ``remaining`` to zero) is responsible for publishing the result and
    signalling the event fd.

    ``sizes`` is only populated for store batches — load/lookup/delete
    batches leave it as an empty list because byte accounting is not
    needed in those paths.
    """

    task_id: L2TaskId
    remaining: int
    per_key_ok: list[bool]
    keys: list[ObjectKey]
    sizes: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def settle_unlaunched(self, launched: int) -> bool:
        """Account for keys that never got a future, after a launch error.

        Called when the per-key dispatch loop raised at index ``launched``:
        keys ``[launched, len(keys))`` will never run a completion callback,
        so their ``remaining`` share must be dropped here or the batch would
        never finalize. They keep their initial ``per_key_ok`` of ``False``.

        Takes ``lock`` and re-checks ``remaining``, so it races safely with
        the callbacks of the keys that did launch.

        Args:
            launched: Number of keys whose future was successfully
                submitted (i.e. the index at which dispatch raised).

        Returns:
            ``True`` if this call drove ``remaining`` to zero and the caller
            must therefore finalize the batch.
        """
        unlaunched = len(self.keys) - launched
        if unlaunched <= 0:
            return False
        with self.lock:
            self.remaining -= unlaunched
            return self.remaining == 0


class ValkeyL2Adapter(L2AdapterInterface):
    """
    L2 adapter that stores KV-cache chunks in Valkey (or Redis) via the
    glide sync client.

    See the module docstring for the design rationale and the list of
    capabilities this adapter provides.
    """

    def __init__(self, config: ValkeyL2AdapterConfig) -> None:
        """Initialize the adapter and warm up worker clients."""
        super().__init__(max_capacity_bytes=int(config.max_capacity_gb * (1024**3)))
        self._config: ValkeyL2AdapterConfig = config

        # Three distinct event fds (interface requirement).
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Result queues — popped (and cleared) by the L2 controller.
        self._completed_stores: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookups: dict[L2TaskId, Bitmap] = {}
        self._completed_loads: dict[L2TaskId, Bitmap] = {}

        # Client-side lock refcounts (in-flight lookups pin keys).
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)

        # Stored sizes, so delete can report accurate per-key byte deltas.
        # NOTE: grows unbounded under server-side eviction (delete is never
        # called); shared across remote adapters, best fixed in the base class.
        self._key_sizes: dict[ObjectKey, int] = {}

        self._next_task_id: L2TaskId = 0
        self._lock: threading.Lock = threading.Lock()

        # Shared glide client pool — owns clients, thread pool, copy-reduced
        # primitives, capability probing, and reliable shutdown.
        self._pool: ValkeyWorkerPool = ValkeyWorkerPool(
            addresses=config.startup_nodes,
            num_workers=config.num_workers,
            username=config.username,
            password=config.password,
            request_timeout=config.request_timeout,
            connection_timeout=config.connection_timeout,
            tls_enable=config.tls_enable,
            cluster_mode=config.cluster_mode,
            database_id=config.database_id,
            ttl_seconds=config.ttl_seconds,
        )

        self._closed: bool = False
        logger.info(
            "ValkeyL2Adapter ready: workers=%d cluster_mode=%s nodes=%s "
            "tls=%s buffer_get=%s key_prefix=%r",
            config.num_workers,
            config.cluster_mode,
            list(config.startup_nodes),
            config.tls_enable,
            self._pool.has_buffer_get,
            config.key_prefix,
        )

    # ---------------------------------------------------------------
    # Event Fd Interface
    # ---------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ---------------------------------------------------------------
    # Store Interface
    # ---------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking batch SET to Valkey.

        Each key is dispatched as an independent ``Future`` to the
        worker pool. When the final per-key future completes, the
        adapter publishes an ``L2StoreResult`` whose
        ``bytes_transferred`` reflects only the keys that actually
        wrote (partial failures are accounted correctly).

        If the adapter is closed, or the pool rejects a dispatch, the
        task completes immediately as a failure rather than hanging.

        Raises:
            ValueError: If ``keys`` and ``objects`` differ in length.
        """
        if len(keys) != len(objects):
            raise ValueError(
                "submit_store_task: keys and objects length mismatch "
                f"({len(keys)} vs {len(objects)})"
            )

        with self._lock:
            task_id = self._new_task_id()

        if self._closed:
            logger.warning(
                "submit_store_task on a closed ValkeyL2Adapter; failing task"
            )
            with self._lock:
                self._completed_stores[task_id] = L2StoreResult(False, 0)
            self._store_efd.notify()
            return task_id

        if not keys:
            with self._lock:
                self._completed_stores[task_id] = L2StoreResult(True, 0)
            self._store_efd.notify()
            return task_id

        wire_keys = [self._wire_key(k) for k in keys]
        sizes = [obj.get_size() for obj in objects]

        batch = _SubmitTaskBatchState(
            task_id=task_id,
            remaining=len(keys),
            per_key_ok=[False] * len(keys),
            keys=list(keys),
            sizes=sizes,
        )

        launched = 0
        try:
            for i in range(len(keys)):
                fut = self._pool.submit_set(wire_keys[i], objects[i].byte_array)
                # partial binds this key's batch+index; the future is passed
                # last by add_done_callback.
                fut.add_done_callback(functools.partial(self._on_store_done, batch, i))
                launched = i + 1
        except Exception as e:
            logger.error(
                "Valkey SET dispatch failed after %d/%d keys (task_id=%d): %s",
                launched,
                len(keys),
                task_id,
                e,
            )
            if batch.settle_unlaunched(launched):
                self._finalize_store(batch)

        return task_id

    def _on_store_done(
        self, batch: _SubmitTaskBatchState, idx: int, fut: Future
    ) -> None:
        """Per-key callback for a store batch; finalizes when last done.

        Runs in the worker thread that completed ``fut``. A cancelled or
        failed future is recorded as a per-key failure; the callback
        never re-raises (so it always decrements the batch counter and
        the batch can never hang).
        """
        if fut.cancelled():
            ok = False
            logger.warning(
                "Valkey SET cancelled (batch task_id=%d, key idx=%d)",
                batch.task_id,
                idx,
            )
        else:
            exc = fut.exception()
            ok = exc is None
            if not ok:
                logger.warning(
                    "Valkey SET failed (batch task_id=%d, key idx=%d): %s",
                    batch.task_id,
                    idx,
                    exc,
                )
        with batch.lock:
            batch.per_key_ok[idx] = ok
            batch.remaining -= 1
            done = batch.remaining == 0
        if done:
            self._finalize_store(batch)

    def _finalize_store(self, batch: _SubmitTaskBatchState) -> None:
        """Publish the final ``L2StoreResult`` for ``batch``.

        Accounting (``_notify_keys_stored``) is fired outside
        ``self._lock`` so a slow listener cannot stall the next submit.
        """
        all_ok = all(batch.per_key_ok)

        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        bytes_transferred = 0
        with self._lock:
            for k, sz, ok in zip(
                batch.keys, batch.sizes, batch.per_key_ok, strict=True
            ):
                if not ok:
                    continue
                stored_keys.append(k)
                if k in self._key_sizes:
                    stored_sizes.append(0)
                else:
                    self._key_sizes[k] = sz
                    stored_sizes.append(sz)
                    bytes_transferred += sz
            self._completed_stores[batch.task_id] = L2StoreResult(
                all_ok, bytes_transferred
            )

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)
        self._store_efd.notify()

    def pop_completed_store_tasks(
        self,
    ) -> dict[L2TaskId, L2StoreResult]:
        """Drain and return all completed store-task results.

        Returns:
            A mapping of ``L2TaskId`` to its ``L2StoreResult`` for every
            store task finished since the last call. The internal buffer is
            cleared, so each result is returned exactly once.
        """
        with self._lock:
            completed = self._completed_stores
            self._completed_stores = {}
        return completed

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> L2TaskId:
        """Submit a non-blocking batch EXISTS to Valkey.

        On completion the bitmap has bit ``i`` set iff key ``i`` was
        present, and the lock refcount for each present key is
        incremented so subsequent ``submit_unlock`` calls balance.

        If the adapter is closed, or the pool rejects a dispatch, the
        task completes immediately as an all-miss bitmap rather than
        hanging.

        Args:
            keys: Keys to look up and lock.
            layout_desc: The memory layout of the objects. Advisory hint,
                not used by the Valkey adapter.

        Returns:
            The assigned task id.
        """
        with self._lock:
            task_id = self._new_task_id()

        if self._closed:
            logger.warning(
                "submit_lookup_and_lock_task on a closed ValkeyL2Adapter; "
                "reporting all keys as missing"
            )
            with self._lock:
                self._completed_lookups[task_id] = Bitmap(len(keys))
            self._lookup_efd.notify()
            return task_id

        if not keys:
            with self._lock:
                self._completed_lookups[task_id] = Bitmap(0)
            self._lookup_efd.notify()
            return task_id

        wire_keys = [self._wire_key(k) for k in keys]
        batch = _SubmitTaskBatchState(
            task_id=task_id,
            remaining=len(keys),
            per_key_ok=[False] * len(keys),
            keys=list(keys),
        )
        launched = 0
        try:
            for i in range(len(keys)):
                fut = self._pool.submit_exists(wire_keys[i])
                fut.add_done_callback(functools.partial(self._on_lookup_done, batch, i))
                launched = i + 1
        except Exception as e:
            logger.error(
                "Valkey EXISTS dispatch failed after %d/%d keys (task_id=%d): %s",
                launched,
                len(keys),
                task_id,
                e,
            )
            if batch.settle_unlaunched(launched):
                self._finalize_lookup(batch)
        return task_id

    def _on_lookup_done(
        self, batch: _SubmitTaskBatchState, idx: int, fut: Future
    ) -> None:
        """Per-key callback for a lookup batch."""
        exists = False
        if fut.cancelled():
            logger.warning(
                "Valkey EXISTS cancelled (batch=%d, idx=%d)",
                batch.task_id,
                idx,
            )
        else:
            exc = fut.exception()
            if exc is None:
                try:
                    exists = bool(fut.result())
                except Exception as e:  # defensive — result() shouldn't raise here
                    logger.warning(
                        "Valkey EXISTS unexpected error (batch=%d, idx=%d): %s",
                        batch.task_id,
                        idx,
                        e,
                    )
            else:
                logger.warning(
                    "Valkey EXISTS failed (batch=%d, idx=%d): %s",
                    batch.task_id,
                    idx,
                    exc,
                )
        with batch.lock:
            batch.per_key_ok[idx] = exists
            batch.remaining -= 1
            done = batch.remaining == 0
        if done:
            self._finalize_lookup(batch)

    def _finalize_lookup(self, batch: _SubmitTaskBatchState) -> None:
        """Publish lookup bitmap and apply lock-refcount increments."""
        bitmap = Bitmap(len(batch.keys))
        accessed: list[ObjectKey] = []
        with self._lock:
            for i, (k, ok) in enumerate(zip(batch.keys, batch.per_key_ok, strict=True)):
                if ok:
                    bitmap.set(i)
                    self._locked_keys[k] += 1
                    accessed.append(k)
            self._completed_lookups[batch.task_id] = bitmap
        if accessed:
            self._notify_keys_accessed(accessed)
        self._lookup_efd.notify()

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Non-blockingly pop the result of a lookup-and-lock task.

        Args:
            task_id: Id of a previously submitted lookup-and-lock task.

        Returns:
            A ``Bitmap`` whose bits mark per-key success (key found and
            locked), or ``None`` if the task has not completed yet or its
            result was already retrieved (returns non-``None`` only once).
        """
        with self._lock:
            return self._completed_lookups.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Decrement the lock refcount for ``keys``.

        Missing keys are silently ignored — the L2 controller does not
        rely on unlock symmetry beyond refcount semantics.
        """
        with self._lock:
            for key in keys:
                count = self._locked_keys.get(key, 0)
                if count <= 1:
                    self._locked_keys.pop(key, None)
                else:
                    self._locked_keys[key] = count - 1

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking batch GET into ``objects`` buffers.

        Uses glide's buffer GET (copy-reduced) when available; otherwise
        falls back to copying through a ``bytes`` intermediate. In both
        paths a size mismatch is treated as a cache miss.

        If the adapter is closed, or the pool rejects a dispatch, the
        task completes immediately as an all-miss bitmap rather than
        hanging.

        Raises:
            ValueError: If ``keys`` and ``objects`` differ in length.
        """
        if len(keys) != len(objects):
            raise ValueError(
                "submit_load_task: keys and objects length mismatch "
                f"({len(keys)} vs {len(objects)})"
            )

        with self._lock:
            task_id = self._new_task_id()

        if self._closed:
            logger.warning(
                "submit_load_task on a closed ValkeyL2Adapter; "
                "reporting all keys as missing"
            )
            with self._lock:
                self._completed_loads[task_id] = Bitmap(len(keys))
            self._load_efd.notify()
            return task_id

        if not keys:
            with self._lock:
                self._completed_loads[task_id] = Bitmap(0)
            self._load_efd.notify()
            return task_id

        wire_keys = [self._wire_key(k) for k in keys]
        # ``sizes[i]`` is the expected chunk length; the load callback
        # treats any GET whose byte count differs as a cache miss (stale
        # or truncated value).
        expected_sizes = [obj.get_size() for obj in objects]
        batch = _SubmitTaskBatchState(
            task_id=task_id,
            remaining=len(keys),
            per_key_ok=[False] * len(keys),
            keys=list(keys),
            sizes=expected_sizes,
        )
        launched = 0
        try:
            for i in range(len(keys)):
                fut = self._pool.submit_get_into(wire_keys[i], objects[i].byte_array)
                fut.add_done_callback(functools.partial(self._on_load_done, batch, i))
                launched = i + 1
        except Exception as e:
            logger.error(
                "Valkey GET dispatch failed after %d/%d keys (task_id=%d): %s",
                launched,
                len(keys),
                task_id,
                e,
            )
            if batch.settle_unlaunched(launched):
                self._finalize_load(batch)
        return task_id

    def _on_load_done(
        self, batch: _SubmitTaskBatchState, idx: int, fut: Future
    ) -> None:
        """Per-key callback for a load batch.

        The pool returns the number of bytes available for the key (or
        ``GET_MISS``). A hit requires the byte count to match the
        expected chunk size, so a stale or truncated value is rejected
        as a cache miss.
        """
        loaded = False
        if fut.cancelled():
            logger.warning(
                "Valkey GET cancelled (batch=%d, idx=%d)",
                batch.task_id,
                idx,
            )
        else:
            exc = fut.exception()
            if exc is None:
                try:
                    nbytes = int(fut.result())
                    loaded = nbytes == batch.sizes[idx]
                    if nbytes >= 0 and not loaded:
                        logger.debug(
                            "Valkey GET size mismatch (batch=%d, idx=%d): "
                            "got %d bytes, expected %d",
                            batch.task_id,
                            idx,
                            nbytes,
                            batch.sizes[idx],
                        )
                except Exception as e:
                    logger.warning(
                        "Valkey GET unexpected error (batch=%d, idx=%d): %s",
                        batch.task_id,
                        idx,
                        e,
                    )
            else:
                logger.warning(
                    "Valkey GET failed (batch=%d, idx=%d): %s",
                    batch.task_id,
                    idx,
                    exc,
                )
        with batch.lock:
            batch.per_key_ok[idx] = loaded
            batch.remaining -= 1
            done = batch.remaining == 0
        if done:
            self._finalize_load(batch)

    def _finalize_load(self, batch: _SubmitTaskBatchState) -> None:
        """Publish the load result bitmap and notify access listeners."""
        bitmap = Bitmap(len(batch.keys))
        accessed: list[ObjectKey] = []
        for i, ok in enumerate(batch.per_key_ok):
            if ok:
                bitmap.set(i)
                accessed.append(batch.keys[i])
        with self._lock:
            self._completed_loads[batch.task_id] = bitmap
        if accessed:
            self._notify_keys_accessed(accessed)
        self._load_efd.notify()

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Non-blockingly pop the result of a load task.

        Args:
            task_id: Id of a previously submitted load task.

        Returns:
            A ``Bitmap`` whose bits mark per-key load success, or ``None``
            if the task has not completed yet or its result was already
            retrieved (returns non-``None`` only once).
        """
        with self._lock:
            return self._completed_loads.pop(task_id, None)

    # ---------------------------------------------------------------
    # Eviction Interface
    # ---------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Synchronously delete a batch of keys from Valkey.

        Locked keys (pinned by an in-flight lookup/load) are skipped —
        deleting a value mid-read would corrupt the load. The remaining
        keys are deleted; blocks until each per-key future completes or
        ``request_timeout`` is hit. Only keys whose DEL actually removed a
        value (glide returns ``1``) are reported via
        ``_notify_keys_deleted`` with their original stored size from
        ``_key_sizes``. A closed adapter deletes nothing.
        """
        if not keys:
            return

        if self._closed:
            logger.warning("delete on a closed ValkeyL2Adapter; ignoring")
            return

        # Skip keys currently pinned by a read; same convention as the
        # other L2 adapters (e.g. S3).
        with self._lock:
            deletable = [k for k in keys if self._locked_keys.get(k, 0) == 0]
        if not deletable:
            return

        wire_keys = [self._wire_key(k) for k in deletable]
        futures = [self._pool.submit_delete(wk) for wk in wire_keys]

        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for i, fut in enumerate(futures):
            try:
                deleted = bool(fut.result(timeout=self._config.request_timeout))
            except Exception as e:
                logger.warning("Valkey DEL failed for key idx %d: %s", i, e)
                continue
            if not deleted:
                continue
            deleted_keys.append(deletable[i])
            with self._lock:
                size = self._key_sizes.pop(deletable[i], 0)
            deleted_sizes.append(size)

        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``; the base
    # class maintains totals from ``_notify_keys_stored`` /
    # ``_notify_keys_deleted`` so no override is needed.

    def report_status(self) -> dict[str, Any]:
        """Return a JSON-serializable health/status snapshot."""
        usage = self.get_usage()
        status: dict[str, Any] = {
            "is_healthy": not self._closed,
            "type": "valkey",
            "cluster_mode": self._config.cluster_mode,
            "num_workers": self._config.num_workers,
            "startup_nodes": [
                {"host": h, "port": p} for h, p in self._config.startup_nodes
            ],
            "key_prefix": self._config.key_prefix,
            "tls_enable": self._config.tls_enable,
            "has_buffer_get": self._pool.has_buffer_get,
            "current_size_bytes": usage.total_bytes_used,
            "max_capacity_bytes": usage.total_capacity_bytes,
            "usage_fraction": usage.usage_fraction,
        }
        # In cluster mode also expose the server's real per-node memory
        # (``current_size_bytes`` above is LMCache's aggregate logical
        # count and cannot reveal a hot shard). Best-effort: an empty
        # dict means the INFO query failed or returned nothing.
        if self._config.cluster_mode:
            status["node_memory"] = self._pool.node_memory_usage()
        return status

    def close(self) -> None:
        """Close the worker pool (and its glide clients) and event fds.

        Idempotent so a double-close never submits to a shut-down
        executor. ``close`` is a lifecycle call from a single owner, not
        contended with ``submit_*`` — no lock is needed.
        """
        if self._closed:
            return
        self._closed = True
        self._pool.close()
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("ValkeyL2Adapter closed")

    def _new_task_id(self) -> L2TaskId:
        """Allocate the next task id. Caller must hold ``self._lock``."""
        tid = self._next_task_id
        self._next_task_id += 1
        return tid

    def _wire_key(self, key: ObjectKey) -> str:
        """Compute the on-wire string for ``key``.

        Layout: ``<key_prefix>@<_object_key_to_string(key)>`` when a
        prefix is configured, otherwise just the standard wire format
        (which already encodes ``cache_salt``).
        """
        base = _object_key_to_string(key)
        prefix = self._config.key_prefix
        return f"{prefix}{_KEY_SEP}{base}" if prefix else base


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------


def _create_valkey_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Factory entry-point registered under type name ``"valkey"``.

    ``l1_memory_desc`` is accepted for protocol parity with other
    adapters but is unused — Valkey stores raw bytes and has no need to
    co-register an L1 buffer with the backend.
    """
    if not isinstance(config, ValkeyL2AdapterConfig):
        raise TypeError(
            "_create_valkey_l2_adapter expected ValkeyL2AdapterConfig, "
            f"got {type(config).__name__}"
        )
    return ValkeyL2Adapter(config)


register_l2_adapter_type("valkey", ValkeyL2AdapterConfig)
register_l2_adapter_factory("valkey", _create_valkey_l2_adapter)
