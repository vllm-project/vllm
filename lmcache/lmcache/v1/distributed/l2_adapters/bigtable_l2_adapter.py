# SPDX-License-Identifier: Apache-2.0
"""
Cloud Bigtable L2 adapter.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional, cast
import asyncio
import threading

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# Third Party
from google.cloud.bigtable.data import (
    BigtableDataClientAsync,
    DeleteAllFromRow,
    RowMutationEntry,
    SetCell,
    row_filters,
)
import google.api_core.exceptions as google_exceptions
import google.auth

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.utils import TTLCache
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.bigtable_key_encoder import (
    BigtableL2KeyEncoder,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.storage_backend.connector.bigtable_config import (
    BigtablePluginConfig,
)
from lmcache.v1.storage_backend.connector.bigtable_sharder import (
    BigtablePayloadSharder,
)

logger = init_logger(__name__)


def _object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey to a deterministic row key name.

    Unsalted:
        <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>

    Salted (trailing ``cache_salt``):
        <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>
    """
    base = (
        f"{key.model_name}@{key.kv_rank:08x}"
        f"@{key.object_group_id:x}@{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _is_connection_error(exc: BaseException) -> bool:
    """Check if the given exception is a connection-class error."""
    return isinstance(
        exc,
        (
            google_exceptions.DeadlineExceeded,
            google_exceptions.ServiceUnavailable,
            TimeoutError,
            ConnectionError,
        ),
    )


def _extract_shards_from_row(row: Any, family_name: str) -> dict[str, bytes]:
    """Extracts sharded column qualifier to value map from a row object.
    Filters out stale shards from older writes by comparing timestamps.

    Args:
        row: The Bigtable row object.
        family_name: The column family name to filter.

    Returns:
        A dictionary mapping column qualifiers (as strings) to cell values.
    """
    shards_raw: dict[str, tuple[bytes, int]] = {}
    if row is None or not hasattr(row, "cells"):
        return {}

    # Helper to decode bytes or string qualifiers
    def decode_qual(qual):
        return (
            qual.decode("utf-8", errors="ignore") if isinstance(qual, bytes) else qual
        )

    if isinstance(row.cells, dict):
        cells_dict = row.cells.get(family_name, {})
        for qual_bytes, cell_list in cells_dict.items():
            if cell_list:
                qual_str = decode_qual(qual_bytes)
                # cell_list is sorted by timestamp descending, so cell_list[0] is latest
                shards_raw[qual_str] = (
                    cell_list[0].value,
                    cell_list[0].timestamp_micros,
                )
    else:
        for cell in row.cells:
            if cell.family == family_name:
                qual_str = decode_qual(cell.qualifier)
                # Keep the latest cell for this qualifier
                if (
                    qual_str not in shards_raw
                    or cell.timestamp_micros > shards_raw[qual_str][1]
                ):
                    shards_raw[qual_str] = (cell.value, cell.timestamp_micros)

    if not shards_raw:
        return {}

    # Find the maximum timestamp across all shards
    max_ts = max(ts for _, ts in shards_raw.values())

    # Discard shards older than max_ts by >5s (5,000,000 micros).
    # This prevents merging shards from different write operations.
    TOLERANCE_MICROS = 5 * 1000000
    shards: dict[str, bytes] = {}
    for qual, (val, ts) in shards_raw.items():
        if max_ts - ts <= TOLERANCE_MICROS:
            shards[qual] = val
        else:
            logger.warning(
                f"Discarding stale shard {qual} with timestamp {ts} "
                f"(max timestamp is {max_ts}, diff is {(max_ts - ts) / 1e6:.2f}s)"
            )

    return shards


def _prepare_bytes(blob: Any) -> bytes:
    if isinstance(blob, (bytes, bytearray)):
        return cast(bytes, blob)
    return bytes(blob)


def _prepare_and_shard(
    sharder: BigtablePayloadSharder, blob: Any
) -> tuple[bytes, dict[str, bytes]]:
    data_bytes = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    return cast(bytes, data_bytes), sharder.shard(cast(bytes, data_bytes))


class BigtableL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for Bigtable L2 Adapter."""

    def __init__(
        self,
        project_id: str,
        instance_id: str,
        table_name: str,
        app_profile_id: Optional[str] = None,
        read_timeout_sec: float = 0.2,
        write_timeout_sec: float = 0.5,
        exists_cache_ttl_seconds: float = 30.0,
        exists_cache_size: int = 10000,
        credentials_path: Optional[str] = None,
        max_retries: int = 3,
        max_chunk_size_mb: float = 90.0,
        family_name: str = "cf",
        column_name: str = "data",
        max_capacity_gb: float = 0,
        row_key_template: str = "{hash_prefix}@{model}@{rank}@{group}@{hash}@{salt}",
        layer_group_size: int = 10,
        num_layers: int = 32,
        kv_size: int = 2,
    ):
        super().__init__()
        self.project_id = project_id
        self.instance_id = instance_id
        self.table_name = table_name
        self.app_profile_id = app_profile_id
        self.read_timeout_sec = read_timeout_sec
        self.write_timeout_sec = write_timeout_sec
        self.exists_cache_ttl_seconds = exists_cache_ttl_seconds
        self.exists_cache_size = exists_cache_size
        self.credentials_path = credentials_path
        self.max_retries = max_retries
        self.max_chunk_size_mb = max_chunk_size_mb
        self.family_name = family_name
        self.column_name = column_name
        self.max_capacity_gb = max_capacity_gb
        self.row_key_template = row_key_template
        self.layer_group_size = layer_group_size
        self.num_layers = num_layers
        self.kv_size = kv_size

    @classmethod
    def from_dict(cls, d: dict) -> BigtableL2AdapterConfig:
        """Create a BigtableL2AdapterConfig from a configuration dictionary.

        Resolves common fields using BigtablePluginConfig, including pulling
        values from environment variables.

        Args:
            d: The configuration dictionary.

        Returns:
            The parsed BigtableL2AdapterConfig instance.

        Raises:
            ValueError: If max_capacity_gb is invalid.
        """
        # Resolve config via BigtablePluginConfig helper, which also handles env vars
        cfg = BigtablePluginConfig.from_extra_config(d)

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        row_key_template = (
            d.get("row_key_template")
            or d.get("bigtable_row_key_template")
            or cfg.row_key_template
        )
        if row_key_template in ("hash#model", "{hash}#{model}"):
            row_key_template = "{hash_prefix}@{model}@{rank}@{group}@{hash}@{salt}"

        # Map BigtablePluginConfig back to the L2 config parameters
        return cls(
            project_id=cfg.project_id,
            instance_id=cfg.instance_id,
            table_name=cfg.table_name,
            app_profile_id=cfg.app_profile_id,
            read_timeout_sec=cfg.read_timeout_sec,
            write_timeout_sec=cfg.write_timeout_sec,
            exists_cache_ttl_seconds=cfg.exists_cache_ttl_seconds,
            exists_cache_size=cfg.exists_cache_size,
            credentials_path=cfg.credentials_path,
            max_retries=cfg.max_retries,
            max_chunk_size_mb=cfg.max_chunk_size_mb,
            family_name=cfg.family_name,
            column_name=cfg.column_name,
            max_capacity_gb=float(max_capacity_gb),
            row_key_template=row_key_template,
            layer_group_size=int(d.get("layer_group_size", 10)),
            num_layers=int(d.get("num_layers", 32)),
            kv_size=int(d.get("kv_size", 2)),
        )

    @classmethod
    def help(cls) -> str:
        """Get help information for the configuration fields.

        Returns:
            A string listing all supported config fields and descriptions.
        """
        return (
            "Bigtable L2 adapter config fields:\n"
            "- bigtable_project_id (str): project ID (or BT_PROJECT_ID env)\n"
            "- bigtable_instance_id (str): instance ID (or BT_INSTANCE_ID env)\n"
            "- bigtable_table_name (str): table name (or BT_TABLE_NAME env)\n"
            "- bigtable_app_profile (str): optional app profile ID\n"
            "- bigtable_read_timeout_ms (float): read timeout in ms (default 200)\n"
            "- bigtable_write_timeout_ms (float): write timeout in ms (default 500)\n"
            "- exists_cache_ttl_seconds (float): TTL of exists cache (default 30)\n"
            "- exists_cache_size (int): Max size of exists cache (default 10000)\n"
            "- bigtable_credentials_path (str): path to GCP JSON service account file\n"
            "- bigtable_family_name (str): column family name (default 'cf')\n"
            "- bigtable_column_name (str): column qualifier (default 'data')\n"
            "- max_capacity_gb (float): max L2 capacity in GB (default 0 = disabled)\n"
            "- row_key_template (str): row key template configuration\n"
            "- layer_group_size (int): number of layers per shard (default 10)\n"
            "- num_layers (int): total layers in the KV cache model (default 32)\n"
            "- kv_size (int): kv size dimension multiplier (default 2)"
        )


class BigtableL2Adapter(L2AdapterInterface):
    """Bigtable-backed L2 adapter.

    Offloads operations to an asyncio event loop running on a background
    daemon thread.

    Locking: client-side refcount in ``_locked_keys``.
    Circuit breaker: disables the adapter after consecutive connection failures.
    """

    max_connection_failures = 3

    def __init__(self, config: BigtableL2AdapterConfig):
        """Initializes the instance.

        Args:
            config: Configuration for the Bigtable L2 adapter.
        """
        super().__init__(max_capacity_bytes=int(config.max_capacity_gb * (1024**3)))
        self._config = config
        self._family_name = config.family_name
        self._column_name_bytes = config.column_name.encode("utf-8")
        self._key_encoder = BigtableL2KeyEncoder(
            config.row_key_template,
            config.layer_group_size,
        )
        self._sharder = BigtablePayloadSharder(
            config.num_layers, config.layer_group_size, config.kv_size
        )

        # Eventfds for L2 adapter interface signaling
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Task tracking
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        # Client-side locking (refcount per ObjectKey)
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)

        # Thread-safe exists cache
        self._exists_cache = TTLCache(
            max_size=config.exists_cache_size,
            ttl_seconds=config.exists_cache_ttl_seconds,
        )

        # Store logical sizes of keys
        self._key_sizes: dict[ObjectKey, int] = {}
        self._object_size_cache: dict[str, int] = {}

        # Circuit breaker
        self._connection_failures = 0
        self._connection_disabled = False

        self._lock = threading.Lock()

        # Google async Bigtable client (initialized lazily)
        self._client: Optional[BigtableDataClientAsync] = None
        self._table = None

        # Dedicated background loop & thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="bigtable-l2-adapter-loop",
        )
        self._loop_thread.start()

        self._closed = False
        logger.info(
            "Initialized BigtableL2Adapter (project=%s, instance=%s, table=%s)",
            config.project_id,
            config.instance_id,
            config.table_name,
        )

    # ------------------------------------------------------------------
    # Lazy client initialization
    # ------------------------------------------------------------------

    async def _get_table(self):
        """Lazily build and return TableAsync instance."""
        if self._table is not None:
            return self._table

        # Resolve credentials
        if self._config.credentials_path:
            # Third Party
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                self._config.credentials_path
            )
        else:
            credentials, _ = google.auth.default()

        # Build client & get table reference
        self._client = BigtableDataClientAsync(
            project=self._config.project_id,
            credentials=credentials,
        )
        self._table = self._client.get_table(
            self._config.instance_id, self._config.table_name
        )
        return self._table

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """See base class."""
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """See base class."""
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """See base class."""
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """See base class."""
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_store_tasks[task_id] = L2StoreResult(False, 0)
                disabled = True
            else:
                disabled = False

        if disabled:
            self._store_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_store(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """See base class."""
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> L2TaskId:
        """See base class."""
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_lookup_tasks[task_id] = Bitmap(len(keys))
                disabled = True
            else:
                disabled = False

        if disabled:
            self._lookup_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(list(keys), task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """See base class."""
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """See base class."""
        with self._lock:
            for key in keys:
                if key not in self._locked_keys:
                    continue
                if self._locked_keys[key] <= 1:
                    del self._locked_keys[key]
                else:
                    self._locked_keys[key] -= 1

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """See base class."""
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_load_tasks[task_id] = Bitmap(len(keys))
                disabled = True
            else:
                disabled = False

        if disabled:
            self._load_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_load(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """See base class."""
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """See base class."""
        if not keys:
            return

        with self._lock:
            if self._connection_disabled:
                return
            deletable = [k for k in keys if self._locked_keys.get(k, 0) == 0]

        if not deletable:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._execute_delete(deletable),
            self._loop,
        )
        try:
            deleted_keys, deleted_sizes = fut.result(timeout=30.0)
        except Exception as e:
            logger.warning("BigtableL2Adapter delete failed: %s", e)
            return

        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ------------------------------------------------------------------
    # Cleanup & Status
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """See base class."""
        with self._lock:
            failures = self._connection_failures
            disabled = self._connection_disabled
        usage = self.get_usage()
        return {
            "is_healthy": self._loop_thread.is_alive() and not disabled,
            "type": "BigtableL2Adapter",
            "project_id": self._config.project_id,
            "instance_id": self._config.instance_id,
            "table_name": self._config.table_name,
            "connection_failures": failures,
            "connection_disabled": disabled,
            "current_size_bytes": usage.total_bytes_used,
            "max_capacity_bytes": usage.total_capacity_bytes,
        }

    def close(self) -> None:
        """See base class."""
        if self._closed:
            return
        self._closed = True

        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Close Bigtable client safely
            if self._client is not None:
                await self._client.close()

        if self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop).result(
                    timeout=5
                )
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join(timeout=5)
        try:
            self._loop.close()
        except Exception:
            pass

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("BigtableL2Adapter closed")

    # ------------------------------------------------------------------
    # Internal: Event Loop
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _record_connection_outcome(self, exc: Optional[BaseException]) -> None:
        """Update circuit breaker status based on operation outcome."""
        with self._lock:
            if exc is None:
                if self._connection_failures > 0:
                    logger.info("BigtableL2Adapter connection recovered")
                self._connection_failures = 0
                return
            if not _is_connection_error(exc):
                return
            self._connection_failures += 1
            logger.error(
                "BigtableL2Adapter connection error (%d/%d): %s",
                self._connection_failures,
                self.max_connection_failures,
                exc,
            )
            if self._connection_failures >= self.max_connection_failures:
                self._connection_disabled = True
                logger.error(
                    "BigtableL2Adapter disabled after %d consecutive failures",
                    self.max_connection_failures,
                )

    # ------------------------------------------------------------------
    # Internal Coroutines
    # ------------------------------------------------------------------

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Internal coroutine to write a batch of keys and objects to Bigtable.

        Applies sharding if enabled and filters out oversized payloads.

        Args:
            keys: The list of ObjectKeys to store.
            objects: The list of MemoryObjs corresponding to the keys.
            task_id: The identifier of the L2 store task.
        """
        success = True
        bytes_transferred = 0
        newly_stored_keys = []
        newly_stored_sizes = []
        last_error: Optional[BaseException] = None

        try:
            table = await self._get_table()
            kwargs = {}
            if self._config.app_profile_id:
                kwargs["app_profile_id"] = self._config.app_profile_id

            sharding_enabled = self._config.layer_group_size > 0

            indexed = []
            valid_keys_data = []

            for key, obj in zip(keys, objects, strict=True):
                row_key = self._key_encoder.encode_row_key(key)
                key_str = row_key.decode("utf-8")
                size = len(obj.byte_array)

                # Skip if already exists on-disk/remote
                if self._exists_cache.get(key_str):
                    continue

                if sharding_enabled:
                    limit_bytes = 240 * 1024 * 1024
                    if size > limit_bytes:
                        logger.warning(
                            f"Skipping write to Bigtable for key {key_str} "
                            f"because total payload size {size} bytes exceeds "
                            f"the absolute row size limit of 240 MB."
                        )
                        continue
                else:
                    limit_bytes = int(self._config.max_chunk_size_mb * 1024 * 1024)
                    if size > limit_bytes:
                        logger.warning(
                            f"Skipping write to Bigtable for key {key_str} "
                            f"because payload size {size} bytes exceeds the limit "
                            f"of {limit_bytes} bytes without sharding."
                        )
                        continue

                indexed.append((key, key_str, size))
                valid_keys_data.append((row_key, obj.byte_array))

            if indexed:
                if sharding_enabled:

                    async def write_sharded_key(rk, byte_array):
                        try:
                            data_bytes, shards = await self._loop.run_in_executor(
                                None, _prepare_and_shard, self._sharder, byte_array
                            )
                        except Exception as e:
                            return e
                        total_size = len(data_bytes)
                        if max(len(s) for s in shards.values()) > 90 * 1024 * 1024:
                            rk_str = rk.decode("utf-8", errors="ignore")
                            logger.warning(
                                f"Skipping write to Bigtable for key {rk_str} "
                                f"because a single shard exceeds the 90MB cell "
                                f"size limit."
                            )
                            return ValueError("Shard size exceeds cell limit")

                        # Standard
                        import time

                        timestamp = (time.time_ns() // 1000000) * 1000

                        if total_size < 150 * 1024 * 1024:
                            mutations = [
                                SetCell(
                                    self._family_name,
                                    qualifier.encode("utf-8"),
                                    shard_data,
                                    timestamp_micros=timestamp,
                                )
                                for qualifier, shard_data in shards.items()
                            ]
                            try:
                                await table.mutate_row(
                                    rk,
                                    mutations,
                                    operation_timeout=(self._config.write_timeout_sec),
                                    **kwargs,
                                )
                            except Exception as e:
                                return e
                            return None
                        else:
                            tasks = []
                            for qualifier, shard_data in shards.items():
                                tasks.append(
                                    table.mutate_row(
                                        rk,
                                        [
                                            SetCell(
                                                self._family_name,
                                                qualifier.encode("utf-8"),
                                                shard_data,
                                                timestamp_micros=timestamp,
                                            )
                                        ],
                                        operation_timeout=(
                                            self._config.write_timeout_sec
                                        ),
                                        **kwargs,
                                    )
                                )
                            results = await asyncio.gather(
                                *tasks, return_exceptions=True
                            )
                            for res in results:
                                if isinstance(res, BaseException):
                                    return res
                            return None

                    key_tasks = [
                        write_sharded_key(rk, ba) for rk, ba in valid_keys_data
                    ]
                    results = await asyncio.gather(*key_tasks, return_exceptions=True)
                else:
                    prepared_data = await asyncio.gather(
                        *[
                            self._loop.run_in_executor(None, _prepare_bytes, ba)
                            for _, ba in valid_keys_data
                        ]
                    )

                    MAX_BATCH_SIZE_BYTES = 30 * 1024 * 1024
                    current_batch: list[RowMutationEntry] = []
                    current_batch_size = 0
                    results = []

                    async def flush_unsharded_batch(batch):
                        if not batch:
                            return []
                        try:
                            res = await table.bulk_mutate_rows(
                                batch,
                                operation_timeout=self._config.write_timeout_sec,
                                **kwargs,
                            )
                            return res if res is not None else [None] * len(batch)
                        except Exception as e:
                            return [e] * len(batch)

                    for (rk, _), d in zip(valid_keys_data, prepared_data, strict=True):
                        blob_size = len(d)
                        if (
                            current_batch_size + blob_size > MAX_BATCH_SIZE_BYTES
                            and current_batch
                        ):
                            batch_results = await flush_unsharded_batch(current_batch)
                            results.extend(batch_results)
                            current_batch = []
                            current_batch_size = 0

                        entry = RowMutationEntry(
                            rk,
                            [SetCell(self._family_name, self._column_name_bytes, d)],
                        )
                        current_batch.append(entry)
                        current_batch_size += blob_size

                    if current_batch:
                        batch_results = await flush_unsharded_batch(current_batch)
                        results.extend(batch_results)

                for (key, key_str, size), result in zip(indexed, results, strict=True):
                    if result is not None:
                        success = False
                        last_error = (
                            result
                            if isinstance(result, BaseException)
                            else Exception(str(result))
                        )
                        logger.error(
                            "BigtableL2Adapter write failed for %s: %s",
                            key_str,
                            result,
                        )
                    else:
                        with self._lock:
                            is_new = key not in self._key_sizes
                            self._key_sizes[key] = size
                            self._object_size_cache[key_str] = size
                            self._exists_cache.put(key_str, True)
                        if is_new:
                            newly_stored_keys.append(key)
                            newly_stored_sizes.append(size)

                bytes_transferred = sum(newly_stored_sizes)

        except Exception as e:
            logger.exception("BigtableL2Adapter store failed: %s", e)
            success = False
            last_error = e

        self._record_connection_outcome(last_error)

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )

        if newly_stored_keys:
            self._notify_keys_stored(newly_stored_keys, newly_stored_sizes)
        self._store_efd.notify()

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        """Internal coroutine to verify existence of keys in Bigtable.

        Performs lightweight row reads (stripping values) and updates the local
        existence cache.

        Args:
            keys: The list of ObjectKeys to lookup.
            task_id: The identifier of the L2 lookup task.
        """
        bitmap = Bitmap(len(keys))
        futures = []
        indexed = []
        locked_by_me = []
        ghost_keys = []
        ghost_sizes = []

        try:
            table = await self._get_table()

            for i, key in enumerate(keys):
                row_key = self._key_encoder.encode_row_key(key)
                key_str = row_key.decode("utf-8")
                cached = self._exists_cache.get(key_str)
                if cached is True:
                    bitmap.set(i)
                    with self._lock:
                        self._locked_keys[key] += 1
                        locked_by_me.append(key)
                    continue
                elif cached is False:
                    continue

                # Read row using StripValueTransformerFilter for lightweight
                # existence check
                coro = table.read_row(
                    row_key,
                    row_filter=row_filters.StripValueTransformerFilter(True),
                    operation_timeout=self._config.read_timeout_sec,
                )
                futures.append(coro)
                indexed.append((i, key, key_str))

            if futures:
                results = await asyncio.gather(*futures, return_exceptions=True)
                last_error: Optional[BaseException] = None
                any_success = False

                for (idx, key, key_str), result in zip(indexed, results, strict=True):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    elif isinstance(result, BaseException):
                        last_error = result
                        logger.error(
                            "BigtableL2Adapter lookup failed for %s: %s",
                            key_str,
                            result,
                        )
                        continue

                    # If row is returned, it exists in Bigtable
                    if result is not None:
                        bitmap.set(idx)
                        any_success = True
                        self._exists_cache.put(key_str, True)
                        with self._lock:
                            self._locked_keys[key] += 1
                            locked_by_me.append(key)
                    else:
                        self._exists_cache.put(key_str, False)
                        with self._lock:
                            size = self._key_sizes.pop(key, None)
                            self._object_size_cache.pop(key_str, None)
                        if size is not None:
                            ghost_keys.append(key)
                            ghost_sizes.append(size)

                if last_error is None:
                    self._record_connection_outcome(None)
                elif not any_success:
                    self._record_connection_outcome(last_error)

        except (asyncio.CancelledError, Exception) as e:
            with self._lock:
                for key in locked_by_me:
                    if key in self._locked_keys:
                        if self._locked_keys[key] <= 1:
                            del self._locked_keys[key]
                        else:
                            self._locked_keys[key] -= 1
                self._completed_lookup_tasks[task_id] = bitmap
            self._lookup_efd.notify()
            if not isinstance(e, asyncio.CancelledError):
                logger.exception("BigtableL2Adapter lookup failed: %s", e)
                self._record_connection_outcome(e)
            raise e

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._lookup_efd.notify()

        # Update base access notifications
        accessed = [keys[i] for i in range(len(keys)) if bitmap.test(i)]
        if accessed:
            self._notify_keys_accessed(accessed)
        if ghost_keys:
            self._notify_keys_deleted(ghost_keys, ghost_sizes)

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Internal coroutine to load values from Bigtable into memory buffers.

        Reassembles sharded layer groups if sharding is enabled.

        Args:
            keys: The list of ObjectKeys to load.
            objects: The list of destination MemoryObjs.
            task_id: The identifier of the L2 load task.
        """
        bitmap = Bitmap(len(keys))
        futures = []
        indexed = []
        ghost_keys = []
        ghost_sizes = []

        try:
            table = await self._get_table()

            for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
                row_key = self._key_encoder.encode_row_key(key)
                key_str = row_key.decode("utf-8")

                coro = table.read_row(
                    row_key,
                    row_filter=row_filters.CellsColumnLimitFilter(1),
                    operation_timeout=self._config.read_timeout_sec,
                )
                futures.append(coro)
                indexed.append((i, key, key_str, obj))

            if futures:
                results = await asyncio.gather(*futures, return_exceptions=True)
                last_error: Optional[BaseException] = None
                any_success = False

                for (idx, key, key_str, obj), result in zip(
                    indexed, results, strict=True
                ):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    elif isinstance(result, BaseException):
                        last_error = result
                        logger.error(
                            "BigtableL2Adapter load failed for %s: %s",
                            key_str,
                            result,
                        )
                        continue

                    if result is None:
                        with self._lock:
                            size = self._key_sizes.pop(key, None)
                            self._object_size_cache.pop(key_str, None)
                        if size is not None:
                            ghost_keys.append(key)
                            ghost_sizes.append(size)
                        continue

                    shards = _extract_shards_from_row(result, self._family_name)
                    sharding_enabled = self._config.layer_group_size > 0
                    val: Optional[bytes] = None

                    if sharding_enabled:
                        try:
                            val = await self._loop.run_in_executor(
                                None, self._sharder.reassemble, shards
                            )
                        except ValueError as e:
                            logger.warning(
                                "Failed to reassemble sharded payload for %s: %s",
                                key_str,
                                e,
                            )
                            with self._lock:
                                size = self._key_sizes.pop(key, None)
                                self._object_size_cache.pop(key_str, None)
                            if size is not None:
                                ghost_keys.append(key)
                                ghost_sizes.append(size)
                            continue
                    else:
                        val = shards.get(self._config.column_name)

                    if val is None:
                        logger.warning(
                            f"Column {self._config.column_name} not "
                            f"found in row {key_str}"
                        )
                        with self._lock:
                            size = self._key_sizes.pop(key, None)
                            self._object_size_cache.pop(key_str, None)
                        if size is not None:
                            ghost_keys.append(key)
                            ghost_sizes.append(size)
                        continue

                    view = memoryview(obj.byte_array)
                    dst_buf = view.cast("B")
                    expected = len(dst_buf)
                    num_read = len(val)

                    if num_read != expected:
                        logger.warning(
                            "Incomplete read for %s: expected %d, got %d",
                            key_str,
                            expected,
                            num_read,
                        )
                        with self._lock:
                            size = self._key_sizes.pop(key, None)
                            self._object_size_cache.pop(key_str, None)
                        if size is not None:
                            ghost_keys.append(key)
                            ghost_sizes.append(size)
                        continue

                    # Safe and optimized memory copy
                    dst_buf[:num_read] = val
                    bitmap.set(idx)
                    any_success = True

                    # Update size tracking
                    with self._lock:
                        self._key_sizes[key] = num_read
                        self._object_size_cache[key_str] = num_read

                if last_error is None:
                    self._record_connection_outcome(None)
                elif not any_success:
                    self._record_connection_outcome(last_error)

        except Exception as e:
            logger.exception("BigtableL2Adapter load failed: %s", e)
            self._record_connection_outcome(e)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()
        if ghost_keys:
            self._notify_keys_deleted(ghost_keys, ghost_sizes)

    async def _execute_delete(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[int]]:
        """Run DELETE row mutations for each key."""
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []

        try:
            table = await self._get_table()
            entries = []
            indexed = []

            for key in keys:
                row_key = self._key_encoder.encode_row_key(key)
                key_str = row_key.decode("utf-8")

                entry = RowMutationEntry(row_key, DeleteAllFromRow())
                entries.append(entry)
                indexed.append((key, key_str))

            if entries:
                kwargs = {}
                if self._config.app_profile_id:
                    kwargs["app_profile_id"] = self._config.app_profile_id

                # Bulk mutate row deletions
                results = await table.bulk_mutate_rows(
                    entries,
                    operation_timeout=self._config.write_timeout_sec,
                    **kwargs,
                )

                if results is None:
                    results = [None] * len(entries)

                for (key, key_str), result in zip(indexed, results, strict=True):
                    if result is not None:
                        logger.error(
                            "BigtableL2Adapter delete failed for %s: %s",
                            key_str,
                            result,
                        )
                    else:
                        with self._lock:
                            sz = self._key_sizes.pop(key, None)
                            self._object_size_cache.pop(key_str, None)
                            self._exists_cache.invalidate(key_str)
                        deleted_keys.append(key)
                        deleted_sizes.append(sz if sz is not None else 0)

        except Exception as e:
            logger.exception("BigtableL2Adapter delete failed: %s", e)

        return deleted_keys, deleted_sizes


# ---------------------------------------------------------------------------
# Registry Registration
# ---------------------------------------------------------------------------

register_l2_adapter_type("bigtable", BigtableL2AdapterConfig)


def _create_bigtable_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    assert isinstance(config, BigtableL2AdapterConfig)
    return BigtableL2Adapter(config)


register_l2_adapter_factory("bigtable", _create_bigtable_adapter)
