# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import IntEnum, auto
from typing import Any, List, Optional, cast
import asyncio
import inspect

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, TTLCache
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# Local
from .bigtable_config import BigtablePluginConfig
from .bigtable_schema import BigtableSchema
from .bigtable_sharder import BigtablePayloadSharder

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


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


class BigtableConnector(RemoteConnector):
    """
    Native Bigtable remote connector integrated into LMCache.
    Tier Agnostic Design: serves all tier roles driven by per-instance configuration.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: LMCacheEngineConfig,
        plugin_name: Optional[str] = None,
    ):
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        extra_config = config.extra_config if config.extra_config is not None else {}
        self.cfg = BigtablePluginConfig.from_extra_config(extra_config, plugin_name)
        self.schema = BigtableSchema(
            self.cfg.row_key_template,
            self.cfg.family_name,
            self.cfg.column_name,
            layer_group_size=self.cfg.layer_group_size,
        )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.exists_cache = TTLCache(
            max_size=self.cfg.exists_cache_size,
            ttl_seconds=self.cfg.exists_cache_ttl_seconds,
        )

        # Independent gRPC client pool initialized lazily on first operation
        self._client = None
        self._table = None

        # Use native AsyncPQExecutor to run pure coroutines directly on the event loop
        self.pq_executor = AsyncPQExecutor(loop, max_workers=self.cfg.thread_pool_size)

        # Compute total layers dynamically
        num_layers = 0
        for shape in self.meta_shapes:
            if len(shape) >= 2:
                num_layers += shape[1]
            else:
                num_layers += 1
        if num_layers == 0:
            num_layers = 1

        kv_size = self.meta_shapes[0][0] if self.meta_shapes else 2
        self.sharder: Optional[BigtablePayloadSharder] = None
        if self.cfg.layer_group_size > 0:
            self.sharder = BigtablePayloadSharder(
                num_layers=num_layers,
                layer_group_size=self.cfg.layer_group_size,
                kv_size=kv_size,
            )

        logger.info(
            f"Initialized Bigtable remote connector for project={self.cfg.project_id}, "
            f"instance={self.cfg.instance_id}, table={self.cfg.table_name}"
        )

    def _get_client(self):
        """Lazy initialization of independent Bigtable async data client."""
        if self._client is not None:
            return self._client

        try:
            # Third Party
            from google.api_core import exceptions as google_exceptions
            from google.api_core.gapic_v1.client_info import ClientInfo
            from google.cloud.bigtable.data import BigtableDataClientAsync

            self.google_exceptions = google_exceptions

            logger.info(
                f"Lazily initializing Bigtable async client "
                f"for project={self.cfg.project_id}, "
                f"instance={self.cfg.instance_id}, "
                f"table={self.cfg.table_name}"
            )

            client_info = ClientInfo(user_agent="lmcache")

            if self.cfg.credentials_path:
                # Third Party
                from google.oauth2 import service_account
                import google.auth.exceptions

                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        self.cfg.credentials_path
                    )
                    self._client = BigtableDataClientAsync(
                        project=self.cfg.project_id,
                        credentials=credentials,
                        client_info=client_info,
                    )
                except (
                    OSError,
                    ValueError,
                    google.auth.exceptions.GoogleAuthError,
                ) as e:
                    logger.warning(
                        f"Failed to load credentials from "
                        f"{self.cfg.credentials_path} due to {e}. "
                        f"Falling back to Application Default Credentials."
                    )
                    self._client = BigtableDataClientAsync(
                        project=self.cfg.project_id,
                        client_info=client_info,
                    )
            else:
                self._client = BigtableDataClientAsync(
                    project=self.cfg.project_id,
                    client_info=client_info,
                )

            return self._client
        except Exception as e:
            logger.error(f"Failed to initialize Bigtable async client: {e}")
            raise

    def _get_table(self):
        """Lazy initialization and caching of TableAsync instance."""
        if self._table is not None:
            return self._table

        client = self._get_client()
        try:
            asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError(
                "TableAsync must be retrieved within an async event loop context."
            ) from e

        self._table = client.get_table(self.cfg.instance_id, self.cfg.table_name)
        return self._table

    def _get_row_filters_module(self):
        try:
            # Third Party
            from google.cloud.bigtable.data import row_filters

            return row_filters
        except ImportError:
            # Third Party
            from google.cloud.bigtable import row_filters

            return row_filters

    async def _exists_internal(self, key: CacheEngineKey) -> bool:
        key_str = key.to_string()
        cached_val = self.exists_cache.get(key_str)
        if cached_val is not None:
            return cached_val

        row_key = self.schema.get_row_key(key)

        row_filters = self._get_row_filters_module()
        row_filter = getattr(
            row_filters, "StripValueTransformerFilter", lambda flag: None
        )(True)

        retries = 0
        while True:
            try:
                kwargs = {}
                if row_filter is not None:
                    kwargs["row_filter"] = row_filter
                if self.cfg.app_profile_id:
                    kwargs["app_profile_id"] = self.cfg.app_profile_id

                table = self._get_table()
                row = await table.read_row(
                    row_key,
                    operation_timeout=self.cfg.read_timeout_sec,
                    **kwargs,
                )
                exists = row is not None
                self.exists_cache.put(key_str, exists)
                return exists
            except (
                self.google_exceptions.DeadlineExceeded,
                TimeoutError,
            ) as e:
                logger.warning(
                    f"Bigtable async timeout in exists: {e}. Treating as miss."
                )
                return False
            except (
                self.google_exceptions.PermissionDenied,
                self.google_exceptions.Unauthenticated,
            ) as e:
                logger.error(f"Bigtable permission/auth error in exists: {e}")
                raise
            except self.google_exceptions.NotFound as e:
                logger.error(f"Bigtable NotFound in exists: {e}")
                raise
            except self.google_exceptions.ResourceExhausted:
                if retries < self.cfg.max_retries:
                    sleep_time = 0.5 * (2**retries)
                    logger.warning(
                        f"Bigtable ResourceExhausted. Retrying in {sleep_time}s."
                    )
                    await asyncio.sleep(sleep_time)
                    retries += 1
                else:
                    logger.warning(
                        "Bigtable ResourceExhausted max retries reached in exists."
                    )
                    return False

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists_internal, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def _get_internal(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        row_key = self.schema.get_row_key(key)

        row_filters = self._get_row_filters_module()
        row_filter = getattr(row_filters, "CellsColumnLimitFilter", lambda n: None)(1)

        retries = 0
        while True:
            try:
                kwargs = {}
                if row_filter is not None:
                    kwargs["row_filter"] = row_filter
                if self.cfg.app_profile_id:
                    kwargs["app_profile_id"] = self.cfg.app_profile_id

                table = self._get_table()
                row = await table.read_row(
                    row_key,
                    operation_timeout=self.cfg.read_timeout_sec,
                    **kwargs,
                )
                if row is None:
                    self.exists_cache.put(key_str, False)
                    return None

                self.exists_cache.put(key_str, True)

                shards = _extract_shards_from_row(row, self.cfg.family_name)
                cell_value: Optional[bytes] = None
                if self.sharder is not None:
                    try:
                        loop = asyncio.get_running_loop()
                        cell_value = await loop.run_in_executor(
                            None, self.sharder.reassemble, shards
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to reassemble shards for key {key_str}: {e}"
                        )
                        return None
                else:
                    legacy_qual = (
                        self.cfg.column_name.decode("utf-8")
                        if isinstance(self.cfg.column_name, bytes)
                        else self.cfg.column_name
                    )
                    cell_value = shards.get(legacy_qual)

                if cell_value is None:
                    return None

                memory_obj = self.local_cpu_backend.allocate(
                    self.meta_shapes,
                    self.meta_dtypes,
                    self.meta_fmt,
                )
                if memory_obj is None:
                    logger.warning("Failed to allocate memory during Bigtable receive")
                    return None

                view = memory_obj.byte_array
                if not isinstance(view, memoryview):
                    view = memoryview(view)

                if isinstance(view.format, str) and view.format == "<B":
                    view = view.cast("B")

                if len(cell_value) > len(view):
                    logger.warning(
                        f"Bigtable cell size {len(cell_value)} exceeds "
                        f"allocated view size {len(view)}"
                    )
                    memory_obj.ref_count_down()
                    return None

                view[: len(cell_value)] = cell_value
                if len(cell_value) < len(view):
                    memory_obj = self.reshape_partial_chunk(memory_obj, len(cell_value))
                return memory_obj

            except (
                self.google_exceptions.DeadlineExceeded,
                TimeoutError,
            ) as e:
                logger.warning(f"Bigtable async timeout in get: {e}. Treating as miss.")
                return None
            except (
                self.google_exceptions.PermissionDenied,
                self.google_exceptions.Unauthenticated,
            ) as e:
                logger.error(f"Bigtable permission/auth error in get: {e}")
                raise
            except self.google_exceptions.NotFound as e:
                logger.error(f"Bigtable NotFound in get: {e}")
                raise
            except self.google_exceptions.ResourceExhausted:
                if retries < self.cfg.max_retries:
                    sleep_time = 0.5 * (2**retries)
                    logger.warning(
                        f"Bigtable ResourceExhausted. Retrying in {sleep_time}s."
                    )
                    await asyncio.sleep(sleep_time)
                    retries += 1
                else:
                    logger.warning(
                        "Bigtable ResourceExhausted max retries reached in get."
                    )
                    return None

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get_internal, key=key, priority=Priorities.GET
        )

    async def _put_internal(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        blob = memory_obj.byte_array
        blob_size_mb = len(blob) / (1024 * 1024)

        sharding_enabled = self.sharder is not None
        limit_mb = 240.0 if sharding_enabled else self.cfg.max_chunk_size_mb

        if blob_size_mb > limit_mb:
            logger.warning(
                f"Bigtable chunk size {blob_size_mb:.2f} MB exceeds "
                f"threshold {limit_mb} MB. "
                f"Skipping write to prevent hard failures."
            )
            return

        row_key = self.schema.get_row_key(key)
        loop = asyncio.get_running_loop()

        # Third Party
        from google.cloud.bigtable.data import SetCell

        if self.sharder is not None:
            data_bytes, shards = await loop.run_in_executor(
                None, _prepare_and_shard, self.sharder, blob
            )
            if max(len(s) for s in shards.values()) > 90 * 1024 * 1024:
                rk_str = row_key.decode("utf-8", errors="ignore")
                logger.warning(
                    f"Skipping write to Bigtable for key {rk_str} "
                    f"because a single shard exceeds the 90MB cell size limit."
                )
                return

            total_size = len(data_bytes)
            # Standard
            import time

            timestamp = (time.time_ns() // 1000000) * 1000

            if total_size < 150 * 1024 * 1024:
                mutations = [
                    SetCell(
                        self.cfg.family_name,
                        qual,
                        shard_data,
                        timestamp_micros=timestamp,
                    )
                    for qual, shard_data in shards.items()
                ]
                use_combined = True
            else:
                use_combined = False
        else:
            data_bytes = await loop.run_in_executor(None, _prepare_bytes, blob)
            mutations = [
                SetCell(self.cfg.family_name, self.cfg.column_name, data_bytes)
            ]
            use_combined = True

        retries = 0
        while True:
            try:
                kwargs = {}
                if self.cfg.app_profile_id:
                    kwargs["app_profile_id"] = self.cfg.app_profile_id

                table = self._get_table()
                if sharding_enabled and not use_combined:
                    await asyncio.gather(
                        *[
                            table.mutate_row(
                                row_key,
                                [
                                    SetCell(
                                        self.cfg.family_name,
                                        qual,
                                        shard_data,
                                        timestamp_micros=timestamp,
                                    )
                                ],
                                operation_timeout=self.cfg.write_timeout_sec,
                                **kwargs,
                            )
                            for qual, shard_data in shards.items()
                        ]
                    )
                else:
                    await table.mutate_row(
                        row_key,
                        mutations,
                        operation_timeout=self.cfg.write_timeout_sec,
                        **kwargs,
                    )
                self.exists_cache.put(key_str, True)
                logger.debug(
                    f"Successfully wrote "
                    f"{row_key.decode('utf-8', errors='ignore')} "
                    f"via async Bigtable"
                )
                return
            except (
                self.google_exceptions.DeadlineExceeded,
                TimeoutError,
            ) as e:
                logger.warning(f"Bigtable async timeout in put: {e}. Skipping write.")
                return
            except (
                self.google_exceptions.PermissionDenied,
                self.google_exceptions.Unauthenticated,
            ) as e:
                logger.error(f"Bigtable permission/auth error in put: {e}")
                raise
            except self.google_exceptions.NotFound as e:
                logger.error(f"Bigtable NotFound in put: {e}")
                raise
            except self.google_exceptions.ResourceExhausted:
                if retries < self.cfg.max_retries:
                    sleep_time = 0.5 * (2**retries)
                    logger.warning(
                        f"Bigtable ResourceExhausted. Retrying in {sleep_time}s."
                    )
                    await asyncio.sleep(sleep_time)
                    retries += 1
                else:
                    logger.warning(
                        "Bigtable ResourceExhausted max retries reached in put."
                    )
                    return

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put_internal,
            key=key,
            memory_obj=memory_obj,
            priority=Priorities.PUT,
        )

    def support_batched_get(self) -> bool:
        return True

    async def _batched_get_internal(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        # Third Party
        from google.cloud.bigtable.data import ReadRowsQuery

        row_keys: List[str | bytes] = [self.schema.get_row_key(k) for k in keys]

        row_filters = self._get_row_filters_module()
        row_filter = getattr(row_filters, "CellsColumnLimitFilter", lambda n: None)(1)

        query = ReadRowsQuery(row_keys=row_keys, row_filter=row_filter)

        retries = 0
        while True:
            try:
                kwargs = {}
                if self.cfg.app_profile_id:
                    kwargs["app_profile_id"] = self.cfg.app_profile_id

                table = self._get_table()
                rows_gen = await table.read_rows(
                    query=query,
                    operation_timeout=self.cfg.read_timeout_sec,
                    **kwargs,
                )

                row_dict = {}
                for row in rows_gen:
                    row_dict[row.row_key] = row

                memory_objs: List[Optional[MemoryObj]] = []
                for key, rk in zip(keys, row_keys, strict=True):
                    key_str = key.to_string()
                    row = row_dict.get(rk)
                    if row is None:
                        self.exists_cache.put(key_str, False)
                        memory_objs.append(None)
                        continue

                    self.exists_cache.put(key_str, True)

                    shards = _extract_shards_from_row(row, self.cfg.family_name)
                    cell_value: Optional[bytes] = None
                    if self.sharder is not None:
                        try:
                            loop = asyncio.get_running_loop()
                            cell_value = await loop.run_in_executor(
                                None, self.sharder.reassemble, shards
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to reassemble shards for key {key_str}: {e}"
                            )
                            memory_objs.append(None)
                            continue
                    else:
                        legacy_qual = (
                            self.cfg.column_name.decode("utf-8")
                            if isinstance(self.cfg.column_name, bytes)
                            else self.cfg.column_name
                        )
                        cell_value = shards.get(legacy_qual)

                    if cell_value is None:
                        memory_objs.append(None)
                        continue

                    memory_obj = self.local_cpu_backend.allocate(
                        self.meta_shapes,
                        self.meta_dtypes,
                        self.meta_fmt,
                    )
                    if memory_obj is None:
                        logger.warning(
                            "Failed to allocate memory during batched Bigtable receive"
                        )
                        memory_objs.append(None)
                        continue

                    view = memory_obj.byte_array
                    if not isinstance(view, memoryview):
                        view = memoryview(view)
                    if isinstance(view.format, str) and view.format == "<B":
                        view = view.cast("B")

                    if len(cell_value) > len(view):
                        logger.warning(
                            f"Bigtable cell size {len(cell_value)} "
                            f"exceeds allocated view size {len(view)}"
                        )
                        memory_obj.ref_count_down()
                        memory_objs.append(None)
                        continue

                    view[: len(cell_value)] = cell_value
                    if len(cell_value) < len(view):
                        memory_obj = self.reshape_partial_chunk(
                            memory_obj, len(cell_value)
                        )
                    memory_objs.append(memory_obj)

                return memory_objs

            except (
                self.google_exceptions.DeadlineExceeded,
                TimeoutError,
            ) as e:
                logger.warning(
                    f"Bigtable async timeout in batched_get: {e}. Treating as miss."
                )
                return [None] * len(keys)
            except (
                self.google_exceptions.PermissionDenied,
                self.google_exceptions.Unauthenticated,
            ) as e:
                logger.error(f"Bigtable permission/auth error in batched_get: {e}")
                raise
            except self.google_exceptions.NotFound as e:
                logger.error(f"Bigtable NotFound in batched_get: {e}")
                raise
            except self.google_exceptions.ResourceExhausted:
                if retries < self.cfg.max_retries:
                    sleep_time = 0.5 * (2**retries)
                    logger.warning(
                        f"Bigtable ResourceExhausted. Retrying in {sleep_time}s."
                    )
                    await asyncio.sleep(sleep_time)
                    retries += 1
                else:
                    logger.warning(
                        "Bigtable ResourceExhausted max retries reached in batched_get."
                    )
                    return [None] * len(keys)

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        return await self.pq_executor.submit_job(
            self._batched_get_internal, keys=keys, priority=Priorities.GET
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put_internal(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        # Third Party
        from google.cloud.bigtable.data import RowMutationEntry, SetCell

        current_batch: List[RowMutationEntry] = []
        current_batch_keys: List[str] = []
        current_batch_size = 0
        MAX_BATCH_SIZE_BYTES = 30 * 1024 * 1024  # 30MB safety limit

        table = self._get_table()
        kwargs = {}
        if self.cfg.app_profile_id:
            kwargs["app_profile_id"] = self.cfg.app_profile_id

        async def flush_batch(batch, batch_keys):
            if not batch:
                return
            retries = 0
            while True:
                try:
                    await table.bulk_mutate_rows(
                        batch,
                        operation_timeout=self.cfg.write_timeout_sec,
                        **kwargs,
                    )
                    for k_str in batch_keys:
                        self.exists_cache.put(k_str, True)
                    logger.debug(
                        f"Successfully batched put {len(batch)} rows via async Bigtable"
                    )
                    return
                except (
                    self.google_exceptions.DeadlineExceeded,
                    TimeoutError,
                ) as e:
                    logger.warning(
                        f"Bigtable async timeout in batched_put: {e}. Skipping."
                    )
                    return
                except (
                    self.google_exceptions.PermissionDenied,
                    self.google_exceptions.Unauthenticated,
                ) as e:
                    logger.error(f"Bigtable permission/auth error in batched_put: {e}")
                    raise
                except self.google_exceptions.NotFound as e:
                    logger.error(f"Bigtable NotFound in batched_put: {e}")
                    raise
                except self.google_exceptions.ResourceExhausted:
                    if retries < self.cfg.max_retries:
                        sleep_time = 0.5 * (2**retries)
                        logger.warning(f"Retrying Bigtable in {sleep_time}s.")
                        await asyncio.sleep(sleep_time)
                        retries += 1
                    else:
                        logger.warning(
                            "Bigtable ResourceExhausted max retries reached "
                            "in batched_put."
                        )
                        return
                except Exception as e:
                    logger.error(
                        f"Unexpected error in Bigtable "
                        f"_batched_put_internal flush: {e}",
                        exc_info=True,
                    )
                    raise

        for key, memory_obj in zip(keys, memory_objs, strict=True):
            if memory_obj is None:
                continue
            blob = memory_obj.byte_array
            blob_size = len(blob)
            blob_size_mb = blob_size / (1024 * 1024)

            sharding_enabled = self.sharder is not None
            limit_mb = 240.0 if sharding_enabled else self.cfg.max_chunk_size_mb

            if blob_size_mb > limit_mb:
                logger.warning(
                    f"Bigtable chunk size {blob_size_mb:.2f} MB exceeds "
                    f"threshold {limit_mb} MB. "
                    f"Skipping write for key {key.to_string()}."
                )
                continue

            if current_batch_size + blob_size > MAX_BATCH_SIZE_BYTES and current_batch:
                await flush_batch(current_batch, current_batch_keys)
                current_batch = []
                current_batch_keys = []
                current_batch_size = 0

            row_key = self.schema.get_row_key(key)
            loop = asyncio.get_running_loop()

            if self.sharder is not None:
                data_bytes, shards = await loop.run_in_executor(
                    None, _prepare_and_shard, self.sharder, blob
                )
                if max(len(s) for s in shards.values()) > 90 * 1024 * 1024:
                    logger.warning(
                        f"Skipping batched write to Bigtable for key {key.to_string()} "
                        f"because a single shard exceeds the 90MB cell size limit."
                    )
                    continue

                # Standard
                import time

                timestamp = (time.time_ns() // 1000000) * 1000

                if blob_size < 150 * 1024 * 1024:
                    mutations: List[Any] = [
                        SetCell(
                            self.cfg.family_name,
                            qualifier,
                            shard_data,
                            timestamp_micros=timestamp,
                        )
                        for qualifier, shard_data in shards.items()
                    ]
                    entry = RowMutationEntry(row_key, mutations)
                    current_batch.append(entry)
                    current_batch_keys.append(key.to_string())
                    current_batch_size += blob_size
                else:
                    for qualifier, shard_data in shards.items():
                        if (
                            current_batch_size + len(shard_data) > MAX_BATCH_SIZE_BYTES
                            and current_batch
                        ):
                            await flush_batch(current_batch, current_batch_keys)
                            current_batch = []
                            current_batch_keys = []
                            current_batch_size = 0

                        entry = RowMutationEntry(
                            row_key,
                            [
                                SetCell(
                                    self.cfg.family_name,
                                    qualifier,
                                    shard_data,
                                    timestamp_micros=timestamp,
                                )
                            ],
                        )
                        current_batch.append(entry)
                        current_batch_keys.append(key.to_string())
                        current_batch_size += len(shard_data)
            else:
                data_bytes = await loop.run_in_executor(None, _prepare_bytes, blob)
                entry = RowMutationEntry(
                    row_key,
                    [
                        SetCell(
                            self.cfg.family_name,
                            self.cfg.column_name,
                            data_bytes,
                        )
                    ],
                )
                current_batch.append(entry)
                current_batch_keys.append(key.to_string())
                current_batch_size += blob_size

        if current_batch:
            await flush_batch(current_batch, current_batch_keys)

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put_internal,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_contains_internal(self, keys: List[CacheEngineKey]) -> int:
        count = 0
        missing_keys = []
        for key in keys:
            val = self.exists_cache.get(key.to_string())
            if val is False:
                return count
            elif val is None:
                missing_keys = keys[count:]
                break
            count += 1

        if not missing_keys:
            return count

        # Third Party
        from google.cloud.bigtable.data import ReadRowsQuery

        row_keys_missing: List[str | bytes] = [
            self.schema.get_row_key(k) for k in missing_keys
        ]
        row_filters = self._get_row_filters_module()
        row_filter = getattr(
            row_filters, "StripValueTransformerFilter", lambda flag: None
        )(True)

        query = ReadRowsQuery(row_keys=row_keys_missing, row_filter=row_filter)

        retries = 0
        while True:
            try:
                kwargs = {}
                if self.cfg.app_profile_id:
                    kwargs["app_profile_id"] = self.cfg.app_profile_id

                table = self._get_table()
                rows_gen = await table.read_rows(
                    query=query,
                    operation_timeout=self.cfg.read_timeout_sec,
                    **kwargs,
                )

                existing_rk_set = set()
                for row in rows_gen:
                    existing_rk_set.add(row.row_key)

                for k, rk in zip(missing_keys, row_keys_missing, strict=True):
                    exists = rk in existing_rk_set
                    self.exists_cache.put(k.to_string(), exists)
                    if not exists:
                        return count
                    count += 1

                return count

            except (
                self.google_exceptions.DeadlineExceeded,
                TimeoutError,
            ) as e:
                logger.warning(f"Bigtable async timeout in batched_contains: {e}")
                return count
            except (
                self.google_exceptions.PermissionDenied,
                self.google_exceptions.Unauthenticated,
            ) as e:
                logger.error(f"Bigtable auth error in batched_contains: {e}")
                raise
            except self.google_exceptions.NotFound as e:
                logger.error(f"Bigtable NotFound in batched_contains: {e}")
                raise
            except self.google_exceptions.ResourceExhausted:
                if retries < self.cfg.max_retries:
                    sleep_time = 0.5 * (2**retries)
                    logger.warning(
                        f"Bigtable ResourceExhausted. Retrying in {sleep_time}s."
                    )
                    await asyncio.sleep(sleep_time)
                    retries += 1
                else:
                    logger.warning(
                        "Bigtable ResourceExhausted max retries reached "
                        "in batched_contains."
                    )
                    return count

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_contains_internal,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    def remove_sync(self, key: CacheEngineKey) -> bool:
        try:
            self.exists_cache.invalidate(key.to_string())
            # Third Party
            from google.cloud.bigtable.data import DeleteAllFromRow

            row_key = self.schema.get_row_key(key)

            kwargs = {}
            if self.cfg.app_profile_id:
                kwargs["app_profile_id"] = self.cfg.app_profile_id

            async def _do_remove():
                try:
                    table = self._get_table()
                    await table.mutate_row(
                        row_key,
                        DeleteAllFromRow(),
                        operation_timeout=self.cfg.write_timeout_sec,
                        **kwargs,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to remove key {key} from Bigtable in background: {e}"
                    )

            asyncio.run_coroutine_threadsafe(
                _do_remove(),
                self.loop,
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to schedule removal of key {key} from Bigtable: {e}"
            )
            return False

    async def _list_internal(self) -> List[str]:
        # Third Party
        from google.cloud.bigtable.data import ReadRowsQuery

        row_filters = self._get_row_filters_module()
        row_filter = getattr(
            row_filters, "StripValueTransformerFilter", lambda flag: None
        )(True)
        query = ReadRowsQuery(row_filter=row_filter)

        kwargs = {}
        if self.cfg.app_profile_id:
            kwargs["app_profile_id"] = self.cfg.app_profile_id

        table = self._get_table()
        rows_gen = await table.read_rows(
            query=query,
            operation_timeout=self.cfg.read_timeout_sec,
            **kwargs,
        )

        res = []
        for row in rows_gen:
            rk_str = row.row_key.decode("utf-8")
            parts_split = rk_str.split("#", 1)
            if len(parts_split) != 2:
                continue
            p1, p2 = parts_split
            if "@" in p1:
                fingerprint, chunk_hash_hex = p1, p2
            else:
                chunk_hash_hex, fingerprint = p1, p2
            parts = fingerprint.split("@")
            if len(parts) < 4:
                continue
            model_name = parts[0]
            world_size = parts[1]
            worker_id = parts[2]
            dtype_str = parts[3]

            std_str = (
                f"{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}@{dtype_str}"
            )
            if len(parts) > 4:
                std_str += "@" + "@".join(parts[4:])
            res.append(std_str)
        return res

    async def list(self) -> List[str]:
        return await self.pq_executor.submit_job(
            self._list_internal, priority=Priorities.GET
        )

    async def close(self):
        await self.pq_executor.shutdown_async(wait=False)
        self._table = None
        if getattr(self, "_client", None) is not None:
            try:
                res = self._client.close()
                if inspect.isawaitable(res):
                    await res
            except AttributeError:
                pass
        logger.info("Closed Bigtable connector cleanly.")

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        future = asyncio.run_coroutine_threadsafe(
            self.batched_async_contains("sync_lookup", keys), self.loop
        )
        return int(future.result())

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        try:
            client = self._get_client()
            kwargs = {}
            if self.cfg.app_profile_id:
                kwargs["app_profile_id"] = self.cfg.app_profile_id
            iterator = await client.execute_query(
                "SELECT 1;",
                self.cfg.instance_id,
                operation_timeout=self.cfg.read_timeout_sec,
                **kwargs,
            )
            async for _ in iterator:
                pass
            return 0
        except Exception as e:
            logger.warning(f"Bigtable ping failed: {e}")
            return 1
