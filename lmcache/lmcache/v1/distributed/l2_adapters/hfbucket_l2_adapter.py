# SPDX-License-Identifier: Apache-2.0
"""
Hugging Face Buckets L2 adapter for LMCache MP mode.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import asyncio
import os
import shutil
import tempfile
import threading
import time

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.memory_management import MemoryObj

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
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
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.storage_backend.connector.hfbucket_connector import (
    HFBucketClient,
    HFBucketClientInterface,
    HFBucketLocation,
    encode_hfbucket_object_name,
    parse_hfbucket_handle,
)

logger = init_logger(__name__)

# Use a separate temp root from non-MP HFBucket to avoid collisions.
_DEFAULT_DOWNLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "lmcache-hfbucket-mp"
_METADATA_CACHE_PRUNE_INTERVAL = 128


@dataclass(frozen=True)
class _CachedObjectMetadata:
    """Cached object size entry with expiration metadata."""

    size_bytes: int
    expires_at: float


class _PartialStoreFailure(RuntimeError):
    """Raised when a failed HFBucket batch store still wrote some objects."""

    def __init__(
        self,
        message: str,
        stored_keys: list[ObjectKey],
        stored_sizes: list[int],
    ) -> None:
        super().__init__(message)
        self.stored_keys = stored_keys
        self.stored_sizes = stored_sizes


def _object_key_to_string(key: ObjectKey) -> str:
    """Serialize an MP ``ObjectKey`` to the shared L2 object-name format.

    Unsalted keys use
    ``<model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>``. Salted
    keys append ``@<cache_salt>`` so tenants/users with identical token chunks
    do not collide in the backing bucket.
    """
    base = (
        f"{key.model_name}@{key.kv_rank:08x}"
        f"@{key.object_group_id:x}@{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _object_key_to_bucket_path(key: ObjectKey, location: HFBucketLocation) -> str:
    """Return the HFBucket object path for an MP object key."""
    encoded = encode_hfbucket_object_name(_object_key_to_string(key))
    if location.object_prefix:
        return f"{location.object_prefix}/{encoded}"
    return encoded


def _resolve_hf_token(token_env: str, token: str | None) -> str | None:
    """Resolve Hugging Face token from env-first adapter config."""
    env_token = os.environ.get(token_env, "") if token_env else ""
    if env_token:
        return env_token
    return token


def _get_path_info_path(path_info: object) -> str:
    """Read a Hugging Face path-info object's path field defensively."""
    path = getattr(path_info, "path", "")
    return path if isinstance(path, str) else ""


def _get_path_info_type(path_info: object) -> str:
    """Read a Hugging Face path-info object's type field defensively."""
    obj_type = getattr(path_info, "type", "")
    return obj_type if isinstance(obj_type, str) else ""


def _get_path_info_size(path_info: object) -> int:
    """Read a Hugging Face path-info object's size field defensively."""
    size = getattr(path_info, "size", 0)
    return size if isinstance(size, int) else 0


def _is_not_found_error(exc: Exception) -> bool:
    """Return whether an exception represents a missing bucket/object."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 404

    direct_status_code = getattr(exc, "status_code", None)
    if isinstance(direct_status_code, int):
        return direct_status_code == 404

    return "404" in str(exc)


class HFBucketL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for the HFBucket MP L2 adapter.

    Fields:
    - ``bucket_handle``: ``hf://buckets/<namespace>/<bucket>[/<prefix>]``.
    - ``token_env``: environment variable used to resolve the HF token.
    - ``token``: optional direct token fallback.
    - ``create_bucket_if_missing``: create the bucket lazily on first store.
    - ``download_tmp_dir``: root directory for temporary load downloads.
    - ``metadata_cache_ttl_secs``: TTL for path-size metadata cache.
    - ``num_workers``: worker threads for blocking Hugging Face API calls.
    - ``max_capacity_gb``: capacity used by inherited L2 usage accounting.
    """

    def __init__(
        self,
        bucket_handle: str,
        token_env: str = "HF_TOKEN",
        token: Optional[str] = None,
        create_bucket_if_missing: bool = False,
        download_tmp_dir: str = str(_DEFAULT_DOWNLOAD_TMP_DIR),
        metadata_cache_ttl_secs: float = 30.0,
        num_workers: int = 4,
        max_capacity_gb: float = 0.0,
    ) -> None:
        self.bucket_handle = bucket_handle
        self.bucket_location = parse_hfbucket_handle(bucket_handle)
        self.token_env = token_env
        self.token = token
        self.create_bucket_if_missing = create_bucket_if_missing
        self.download_tmp_dir = Path(download_tmp_dir)
        self.metadata_cache_ttl_secs = metadata_cache_ttl_secs
        self.num_workers = num_workers
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "HFBucketL2AdapterConfig":
        """Parse a config object from ``--l2-adapter`` JSON."""
        bucket_handle = d.get("bucket_handle")
        if not isinstance(bucket_handle, str) or not bucket_handle:
            raise ValueError("bucket_handle must be a non-empty string")

        token_env = d.get("token_env", "HF_TOKEN")
        if not isinstance(token_env, str):
            raise ValueError("token_env must be a string")

        token = d.get("token")
        if token is not None and not isinstance(token, str):
            raise ValueError("token must be a string")

        download_tmp_dir = d.get("download_tmp_dir", str(_DEFAULT_DOWNLOAD_TMP_DIR))
        if not isinstance(download_tmp_dir, str) or not download_tmp_dir:
            raise ValueError("download_tmp_dir must be a non-empty string")

        metadata_cache_ttl_secs = d.get("metadata_cache_ttl_secs", 30.0)
        if (
            not isinstance(metadata_cache_ttl_secs, (int, float))
            or isinstance(metadata_cache_ttl_secs, bool)
            or metadata_cache_ttl_secs < 0
        ):
            raise ValueError("metadata_cache_ttl_secs must be a non-negative number")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or isinstance(num_workers, bool):
            raise ValueError("num_workers must be a positive integer")
        if num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0.0)
        if (
            not isinstance(max_capacity_gb, (int, float))
            or isinstance(max_capacity_gb, bool)
            or max_capacity_gb < 0
        ):
            raise ValueError("max_capacity_gb must be a non-negative number")

        create_bucket_if_missing = d.get("create_bucket_if_missing", False)
        if not isinstance(create_bucket_if_missing, bool):
            raise ValueError("create_bucket_if_missing must be a boolean")

        cfg = cls(
            bucket_handle=bucket_handle,
            token_env=token_env,
            token=token,
            create_bucket_if_missing=create_bucket_if_missing,
            download_tmp_dir=download_tmp_dir,
            metadata_cache_ttl_secs=float(metadata_cache_ttl_secs),
            num_workers=num_workers,
            max_capacity_gb=float(max_capacity_gb),
        )
        cfg.eviction_config = cls._parse_eviction_config(d)
        return cfg

    @classmethod
    def help(cls) -> str:
        """Return CLI help text for this adapter type."""
        return (
            "HFBucket L2 adapter config fields:\n"
            "- bucket_handle (str, required): "
            "hf://buckets/<namespace>/<bucket>[/<prefix>]\n"
            "- token_env (str): env var for HF token (default HF_TOKEN)\n"
            "- token (str): direct token fallback\n"
            "- create_bucket_if_missing (bool): create bucket on first store\n"
            "- download_tmp_dir (str): temporary download root\n"
            "- metadata_cache_ttl_secs (float): metadata cache TTL\n"
            "- num_workers (int): blocking HF API worker threads\n"
            "- max_capacity_gb (float): capacity for get_usage (0 = disabled)\n"
            "- eviction (dict): optional, see L2AdapterConfigBase"
        )


class HFBucketL2Adapter(L2AdapterInterface):
    """Hugging Face Buckets backed MP L2 adapter."""

    def __init__(
        self,
        config: HFBucketL2AdapterConfig,
        bucket_client: HFBucketClientInterface | None = None,
    ) -> None:
        super().__init__(max_capacity_bytes=int(config.max_capacity_gb * (1024**3)))
        self._config = config
        self._bucket_location = config.bucket_location
        self._bucket_id = config.bucket_location.bucket_id
        self._object_prefix = config.bucket_location.object_prefix
        self._create_bucket_if_missing = config.create_bucket_if_missing
        self._metadata_cache_ttl_secs = config.metadata_cache_ttl_secs

        if bucket_client is None:
            token = _resolve_hf_token(config.token_env, config.token)
            self._bucket_client: HFBucketClientInterface = HFBucketClient(token=token)
        else:
            self._bucket_client = bucket_client

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)
        self._key_sizes: dict[ObjectKey, int] = {}
        self._metadata_cache: dict[str, _CachedObjectMetadata] = {}
        self._metadata_cache_updates = 0

        self._bucket_create_checked = False
        self._bucket_create_lock = threading.Lock()

        self._lock = threading.Lock()
        self._closed = False

        self._download_tmp_root = config.download_tmp_dir.expanduser()
        self._download_tmp_root.mkdir(parents=True, exist_ok=True)
        self._download_session_dir = Path(
            tempfile.mkdtemp(
                prefix="hfbucket-mp-",
                dir=self._download_tmp_root,
            )
        )

        self._executor = ThreadPoolExecutor(
            max_workers=config.num_workers,
            thread_name_prefix="hfbucket-l2",
        )
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="hfbucket-l2-adapter-loop",
        )
        self._loop_thread.start()

        logger.info(
            "Initialized HFBucketL2Adapter (bucket_id=%s prefix=%r "
            "workers=%d max_capacity_gb=%.2f)",
            self._bucket_id,
            self._object_prefix,
            config.num_workers,
            config.max_capacity_gb,
        )

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id_locked()
            if self._closed:
                self._completed_store_tasks[task_id] = L2StoreResult(False, 0)
                closed = True
            else:
                closed = False

        if closed:
            self._store_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_store(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id_locked()
            if self._closed:
                self._completed_lookup_tasks[task_id] = Bitmap(len(keys))
                closed = True
            else:
                closed = False

        if closed:
            self._lookup_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(list(keys), task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                if key not in self._locked_keys:
                    continue
                if self._locked_keys[key] <= 1:
                    del self._locked_keys[key]
                else:
                    self._locked_keys[key] -= 1

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id_locked()
            if self._closed:
                self._completed_load_tasks[task_id] = Bitmap(len(keys))
                closed = True
            else:
                closed = False

        if closed:
            self._load_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_load(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return

        with self._lock:
            if self._closed:
                return
            deletable = [key for key in keys if self._locked_keys.get(key, 0) == 0]

        if not deletable:
            return

        future = asyncio.run_coroutine_threadsafe(
            self._execute_delete(deletable),
            self._loop,
        )
        try:
            deleted_keys, deleted_sizes = future.result(timeout=30.0)
        except Exception as exc:
            logger.warning("HFBucketL2Adapter delete failed: %s", exc)
            return

        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    def report_status(self) -> dict:
        usage = self.get_usage()
        with self._lock:
            object_count = len(self._key_sizes)
            locked_key_count = len(self._locked_keys)
            closed = self._closed
        return {
            "is_healthy": self._loop_thread.is_alive() and not closed,
            "type": "HFBucketL2Adapter",
            "bucket_id": self._bucket_id,
            "object_prefix": self._object_prefix,
            "stored_object_count": object_count,
            "locked_key_count": locked_key_count,
            "current_size_bytes": usage.total_bytes_used,
            "max_capacity_bytes": usage.total_capacity_bytes,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _stop_tasks() -> None:
            tasks = [
                task
                for task in asyncio.all_tasks(self._loop)
                if task is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

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

        self._executor.shutdown(wait=True, cancel_futures=True)

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

        with self._lock:
            self._metadata_cache.clear()
            self._key_sizes.clear()
            self._locked_keys.clear()

        shutil.rmtree(self._download_session_dir, ignore_errors=True)
        logger.info("HFBucketL2Adapter closed")

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        try:
            stored_keys, stored_sizes = await self._loop.run_in_executor(
                self._executor,
                self._store_batch_sync,
                keys,
                objects,
            )
            success = True
        except _PartialStoreFailure as exc:
            logger.exception("HFBucketL2Adapter store task partially failed")
            stored_keys = exc.stored_keys
            stored_sizes = exc.stored_sizes
            success = False
        except Exception:
            logger.exception("HFBucketL2Adapter store task failed")
            stored_keys = []
            stored_sizes = []
            success = False

        bytes_transferred = sum(stored_sizes)
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(
                success,
                bytes_transferred,
            )

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)
        self._store_efd.notify()

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        try:
            sizes = await self._loop.run_in_executor(
                self._executor,
                self._resolve_object_sizes_sync,
                keys,
            )
        except Exception:
            logger.exception("HFBucketL2Adapter lookup task failed")
            sizes = [0] * len(keys)

        accessed: list[ObjectKey] = []
        with self._lock:
            for i, (key, size) in enumerate(zip(keys, sizes, strict=True)):
                if size <= 0:
                    continue
                bitmap.set(i)
                self._locked_keys[key] += 1
                accessed.append(key)
            self._completed_lookup_tasks[task_id] = bitmap

        self._lookup_efd.notify()
        if accessed:
            self._notify_keys_accessed(accessed)

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        try:
            bitmap = await self._loop.run_in_executor(
                self._executor,
                self._load_batch_sync,
                keys,
                objects,
            )
        except Exception:
            logger.exception("HFBucketL2Adapter load task failed")
            bitmap = Bitmap(len(keys))

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()

    async def _execute_delete(
        self,
        keys: list[ObjectKey],
    ) -> tuple[list[ObjectKey], list[int]]:
        return await self._loop.run_in_executor(
            self._executor,
            self._delete_batch_sync,
            keys,
        )

    def _store_batch_sync(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[list[ObjectKey], list[int]]:
        self._ensure_bucket_for_writes()

        additions: list[tuple[bytes, str]] = []
        indexed: list[tuple[ObjectKey, str, int]] = []
        for key, obj in zip(keys, objects, strict=True):
            object_path = _object_key_to_bucket_path(key, self._bucket_location)
            data = memoryview(obj.byte_array).cast("B").tobytes()
            additions.append((data, object_path))
            indexed.append((key, object_path, len(data)))

        if not additions:
            return [], []

        try:
            self._bucket_client.upload_files(self._bucket_id, additions)
        except Exception as exc:
            # Hugging Face batch writes are not transactional: a request can
            # write part of the batch and then fail. Fetch fresh backend
            # metadata, update accounting for objects that really landed, and
            # still report the submitted store task as failed.
            reconciled_keys, reconciled_sizes = self._reconcile_failed_store(indexed)
            raise _PartialStoreFailure(
                "HFBucket batch upload failed after partial reconciliation",
                reconciled_keys,
                reconciled_sizes,
            ) from exc

        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        with self._lock:
            for key, object_path, size in indexed:
                was_new = key not in self._key_sizes
                self._key_sizes[key] = size
                self._set_cached_object_size_locked(object_path, size)
                if was_new:
                    stored_keys.append(key)
                    stored_sizes.append(size)

        return stored_keys, stored_sizes

    def _resolve_object_sizes_sync(self, keys: list[ObjectKey]) -> list[int]:
        object_paths = [
            _object_key_to_bucket_path(key, self._bucket_location) for key in keys
        ]

        cached: dict[str, int] = {}
        unresolved_paths: list[str] = []
        with self._lock:
            for object_path in object_paths:
                cached_size = self._get_cached_object_size_locked(object_path)
                if cached_size is None:
                    unresolved_paths.append(object_path)
                else:
                    cached[object_path] = cached_size

        if unresolved_paths:
            fetched = self._fetch_object_sizes_sync(unresolved_paths)
            with self._lock:
                for object_path, size in fetched.items():
                    self._set_cached_object_size_locked(object_path, size)
            cached.update(fetched)

        return [cached.get(object_path, 0) for object_path in object_paths]

    def _fetch_object_sizes_sync(self, object_paths: list[str]) -> dict[str, int]:
        if not object_paths:
            return {}

        try:
            path_infos = self._bucket_client.get_paths_info(
                self._bucket_id,
                object_paths,
            )
        except Exception as exc:
            if _is_not_found_error(exc):
                return {object_path: 0 for object_path in object_paths}
            raise

        size_by_path: dict[str, int] = {}
        for path_info in path_infos:
            if _get_path_info_type(path_info) != "file":
                continue
            path = _get_path_info_path(path_info)
            if path:
                size_by_path[path] = _get_path_info_size(path_info)

        return {
            object_path: size_by_path.get(object_path, 0)
            for object_path in object_paths
        }

    def _load_batch_sync(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> Bitmap:
        bitmap = Bitmap(len(keys))
        object_paths = [
            _object_key_to_bucket_path(key, self._bucket_location) for key in keys
        ]

        batch_dir = Path(
            tempfile.mkdtemp(prefix="load-", dir=self._download_session_dir)
        )
        local_paths: list[tuple[int, Path]] = []
        files: list[tuple[str, str]] = []
        for index, object_path in enumerate(object_paths):
            local_path = batch_dir / f"{index}.bin"
            local_paths.append((index, local_path))
            files.append((object_path, str(local_path)))

        try:
            try:
                self._bucket_client.download_files(self._bucket_id, files)
            except Exception as exc:
                if not _is_not_found_error(exc):
                    logger.warning("Batch download from hfbucket raised: %s", exc)

            for index, local_path in local_paths:
                if not local_path.exists():
                    continue

                dst = memoryview(objects[index].byte_array).cast("B")
                file_size = local_path.stat().st_size
                if file_size != len(dst):
                    logger.error(
                        "Downloaded object %s has %d bytes, expected %d bytes; "
                        "rejecting load",
                        object_paths[index],
                        file_size,
                        len(dst),
                    )
                    with self._lock:
                        self._set_cached_object_size_locked(
                            object_paths[index],
                            file_size,
                        )
                    continue

                with local_path.open("rb") as f:
                    bytes_read = f.readinto(dst)
                if bytes_read != len(dst):
                    logger.error(
                        "Downloaded object %s read %d bytes, expected %d bytes; "
                        "rejecting load",
                        object_paths[index],
                        bytes_read,
                        len(dst),
                    )
                    with self._lock:
                        self._set_cached_object_size_locked(
                            object_paths[index],
                            bytes_read,
                        )
                    continue

                bitmap.set(index)
                with self._lock:
                    self._set_cached_object_size_locked(
                        object_paths[index],
                        file_size,
                    )

            return bitmap
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def _delete_batch_sync(
        self,
        keys: list[ObjectKey],
    ) -> tuple[list[ObjectKey], list[int]]:
        object_paths = [
            _object_key_to_bucket_path(key, self._bucket_location) for key in keys
        ]

        try:
            self._bucket_client.delete_files(self._bucket_id, object_paths)
        except Exception as exc:
            if not _is_not_found_error(exc):
                raise

        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        with self._lock:
            for key, object_path in zip(keys, object_paths, strict=True):
                size = self._key_sizes.pop(key, None)
                self._set_cached_object_size_locked(object_path, 0)
                deleted_keys.append(key)
                deleted_sizes.append(size if size is not None else 0)

        return deleted_keys, deleted_sizes

    def _ensure_bucket_for_writes(self) -> None:
        if not self._create_bucket_if_missing or self._bucket_create_checked:
            return

        with self._bucket_create_lock:
            if self._bucket_create_checked:
                return
            self._bucket_client.create_bucket(self._bucket_id)
            self._bucket_create_checked = True

    def _refresh_cached_sizes(self, keys: list[ObjectKey]) -> None:
        try:
            self._resolve_object_sizes_sync(keys)
        except Exception:
            logger.debug("Failed to refresh hfbucket object sizes", exc_info=True)

    def _reconcile_failed_store(
        self,
        indexed: list[tuple[ObjectKey, str, int]],
    ) -> tuple[list[ObjectKey], list[int]]:
        object_paths = [object_path for _, object_path, _ in indexed]
        try:
            sizes_by_path = self._fetch_object_sizes_sync(object_paths)
        except Exception:
            logger.debug("Failed to reconcile partial hfbucket store", exc_info=True)
            return [], []

        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        with self._lock:
            for key, object_path, _expected_size in indexed:
                size = sizes_by_path.get(object_path, 0)
                self._set_cached_object_size_locked(object_path, size)
                if size <= 0:
                    continue

                # Only notify net-new keys. Existing keys already contributed
                # to byte accounting, and cache objects should be fixed size.
                was_new = key not in self._key_sizes
                self._key_sizes[key] = size
                if was_new:
                    stored_keys.append(key)
                    stored_sizes.append(size)

        return stored_keys, stored_sizes

    def _get_cached_object_size_locked(self, object_path: str) -> int | None:
        entry = self._metadata_cache.get(object_path)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._metadata_cache.pop(object_path, None)
            return None
        return entry.size_bytes

    def _set_cached_object_size_locked(self, object_path: str, size: int) -> None:
        expires_at = time.monotonic() + self._metadata_cache_ttl_secs
        self._metadata_cache[object_path] = _CachedObjectMetadata(
            size_bytes=size,
            expires_at=expires_at,
        )
        self._metadata_cache_updates += 1
        if self._metadata_cache_updates % _METADATA_CACHE_PRUNE_INTERVAL == 0:
            self._prune_expired_cache_entries_locked(time.monotonic())

    def _prune_expired_cache_entries_locked(self, now: float) -> None:
        expired = [
            object_path
            for object_path, entry in self._metadata_cache.items()
            if entry.expires_at <= now
        ]
        for object_path in expired:
            self._metadata_cache.pop(object_path, None)


register_l2_adapter_type("hfbucket", HFBucketL2AdapterConfig)


def _create_hfbucket_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create an HFBucket L2 adapter from registry config."""
    return HFBucketL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("hfbucket", _create_hfbucket_l2_adapter)
