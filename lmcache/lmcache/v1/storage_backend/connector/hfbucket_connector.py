# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote, unquote
import asyncio
import os
import shutil
import tempfile
import time

# Third Party
from huggingface_hub import HfApi
from packaging.version import InvalidVersion, Version
import huggingface_hub

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

if TYPE_CHECKING:
    # Standard
    from collections.abc import Sequence
    import builtins

    # First Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

PLUGIN_TYPE = "hfbucket"
_HFBUCKET_HANDLE_PREFIX = "hf://buckets/"
_MIN_HUGGINGFACE_HUB_VERSION = Version("1.5.0")
_DEFAULT_DOWNLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "lmcache-hfbucket"
_METADATA_CACHE_PRUNE_INTERVAL = 128


@dataclass(frozen=True)
class HFBucketLocation:
    """Parsed Hugging Face bucket location.

    Args:
        bucket_id: Hugging Face bucket identifier in ``namespace/bucket`` format.
        object_prefix: Optional object prefix inside the bucket.
    """

    bucket_id: str
    object_prefix: str


@dataclass(frozen=True)
class HFBucketConnectorConfig:
    """Resolved HFBucket connector configuration.

    Args:
        plugin_name: Full LMCache plugin name, including any instance suffix.
        bucket_location: Parsed bucket handle and prefix information.
        token_env: Environment variable name used to resolve the access token.
        token: Optional direct token override used when ``token_env`` is unset.
        create_bucket_if_missing: Whether writes should lazily create the bucket.
        download_tmp_dir: Root directory for per-connector temporary downloads.
        metadata_cache_ttl_secs: TTL for cached object sizes and misses.
    """

    plugin_name: str
    bucket_location: HFBucketLocation
    token_env: str
    token: str | None
    create_bucket_if_missing: bool
    download_tmp_dir: Path
    metadata_cache_ttl_secs: float


@dataclass(frozen=True)
class _CachedObjectMetadata:
    """Cached object size entry with expiration metadata."""

    size_bytes: int
    expires_at: float


class HFBucketClientInterface(Protocol):
    """Protocol for synchronous Hugging Face bucket clients used by the connector."""

    def create_bucket(self, bucket_id: str) -> None:
        """Create a bucket if needed."""

    def bucket_info(self, bucket_id: str) -> object:
        """Fetch lightweight bucket metadata."""

    def get_paths_info(self, bucket_id: str, paths: Sequence[str]) -> list[object]:
        """Fetch exact metadata for bucket paths."""

    def list_tree(self, bucket_id: str, prefix: str) -> list[object]:
        """List bucket entries under the provided prefix."""

    def upload_files(
        self,
        bucket_id: str,
        add: Sequence[tuple[bytes, str]],
    ) -> None:
        """Upload files to a bucket in one batch call."""

    def download_files(
        self,
        bucket_id: str,
        files: Sequence[tuple[str, str]],
    ) -> None:
        """Download files from a bucket to local paths."""

    def delete_files(
        self,
        bucket_id: str,
        delete: Sequence[str],
    ) -> None:
        """Delete files from a bucket in one batch call."""


class HFBucketClient(HFBucketClientInterface):
    """Thin synchronous wrapper around ``huggingface_hub`` bucket APIs."""

    def __init__(self, token: str | None) -> None:
        _validate_huggingface_hub_support()
        self._api = HfApi(token=token)

    def create_bucket(self, bucket_id: str) -> None:
        """Create the bucket if it does not already exist."""
        self._api.create_bucket(bucket_id, exist_ok=True)

    def bucket_info(self, bucket_id: str) -> object:
        """Return bucket metadata for auth and health checks."""
        return self._api.bucket_info(bucket_id)

    def get_paths_info(self, bucket_id: str, paths: Sequence[str]) -> list[object]:
        """Return exact metadata entries for the requested paths."""
        if not paths:
            return []
        return list(self._api.get_bucket_paths_info(bucket_id, list(paths)))

    def list_tree(self, bucket_id: str, prefix: str) -> list[object]:
        """Return recursive bucket tree entries under ``prefix``."""
        return list(
            self._api.list_bucket_tree(bucket_id, recursive=True, prefix=prefix)
        )

    def upload_files(
        self,
        bucket_id: str,
        add: Sequence[tuple[bytes, str]],
    ) -> None:
        """Upload files in a single batch operation."""
        self._api.batch_bucket_files(bucket_id, add=list(add))

    def download_files(
        self,
        bucket_id: str,
        files: Sequence[tuple[str, str]],
    ) -> None:
        """Download files in a single batch operation."""
        self._api.download_bucket_files(bucket_id, files=list(files))

    def delete_files(
        self,
        bucket_id: str,
        delete: Sequence[str],
    ) -> None:
        """Delete files in a single batch operation."""
        self._api.batch_bucket_files(bucket_id, delete=list(delete))


class HFBucketConnector(RemoteConnector):
    """LMCache remote connector backed by Hugging Face Buckets."""

    def __init__(
        self,
        local_cpu_backend: LocalCPUBackend,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        connector_config: HFBucketConnectorConfig,
        bucket_client: HFBucketClientInterface | None = None,
    ) -> None:
        """Initialize the HFBucket connector.

        Args:
            local_cpu_backend: Local CPU backend used to allocate
                ``MemoryObj`` instances for downloaded chunks.
            config: LMCache engine config. ``save_unfull_chunk`` must be
                ``False``; an omitted ``save_chunk_meta`` is treated as
                ``False`` by this connector.
            metadata: LMCache engine metadata used to derive the expected
                full-chunk layout.
            connector_config: Resolved plugin-scoped configuration, including
                the bucket handle, token source, and metadata cache TTL.
            bucket_client: Optional test seam. When ``None`` (the default), a
                real ``HFBucketClient`` is constructed from the resolved
                token.

        Raises:
            ValueError: If ``save_chunk_meta`` is ``True`` or
                ``save_unfull_chunk`` is ``True``. This backend only supports
                full chunks without inline chunk metadata.
        """
        normalized_config = _normalize_save_chunk_meta_config(config)
        super().__init__(normalized_config, metadata)

        if self.save_chunk_meta:
            raise ValueError("save_chunk_meta must be False for hfbucket")
        if config.save_unfull_chunk:
            raise ValueError("save_unfull_chunk must be False for hfbucket")

        self.local_cpu_backend = local_cpu_backend
        self.plugin_name = connector_config.plugin_name
        self.bucket_id = connector_config.bucket_location.bucket_id
        self.object_prefix = connector_config.bucket_location.object_prefix
        self.create_bucket_if_missing = connector_config.create_bucket_if_missing
        self.metadata_cache_ttl_secs = connector_config.metadata_cache_ttl_secs

        if bucket_client is None:
            resolved_token = _resolve_hf_token(
                connector_config.token_env,
                connector_config.token,
            )
            self._bucket_client: HFBucketClientInterface = HFBucketClient(
                token=resolved_token
            )
        else:
            self._bucket_client = bucket_client

        self._metadata_cache: dict[str, _CachedObjectMetadata] = {}
        self._metadata_cache_lock = Lock()
        self._metadata_cache_updates = 0

        self._bucket_create_lock = Lock()
        self._bucket_create_checked = False

        self._download_tmp_root = connector_config.download_tmp_dir.expanduser()
        self._download_tmp_root.mkdir(parents=True, exist_ok=True)
        self._download_session_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{PLUGIN_TYPE}-{self.plugin_name.replace('.', '-')}-",
                dir=self._download_tmp_root,
            )
        )

        logger.info(
            "Initialized HFBucketConnector for bucket %s with prefix '%s'",
            self.bucket_id,
            self.object_prefix,
        )

    async def exists(self, key: CacheEngineKey) -> bool:
        """Return whether a full LMCache chunk exists for ``key``."""
        return await asyncio.to_thread(self.exists_sync, key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronously return whether a full LMCache chunk exists for ``key``."""
        object_size = self._get_object_size_bytes(key.to_string())
        return object_size == self.full_chunk_size_bytes

    async def get(self, key: CacheEngineKey) -> MemoryObj | None:
        """Retrieve the full chunk associated with ``key``."""
        memory_objs = await self.batched_get([key])
        return memory_objs[0]

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Store a full chunk in Hugging Face Buckets."""
        await self.batched_put([key], [memory_obj])

    async def list(self) -> builtins.list[str]:
        """List LMCache keys currently stored under this connector prefix."""
        return await asyncio.to_thread(self._list_sync)

    async def close(self) -> None:
        """Release connector-local resources and remove temporary downloads."""
        with self._metadata_cache_lock:
            self._metadata_cache.clear()
        await asyncio.to_thread(
            shutil.rmtree,
            self._download_session_dir,
            True,
        )

    def support_ping(self) -> bool:
        """Report support for lightweight connectivity checks."""
        return True

    async def ping(self) -> int:
        """Check basic access to the configured bucket."""
        return await asyncio.to_thread(self._ping_sync)

    def support_batched_put(self) -> bool:
        """Report support for batch uploads."""
        return True

    async def batched_put(
        self,
        keys: builtins.list[CacheEngineKey],
        memory_objs: builtins.list[MemoryObj],
    ) -> None:
        """Upload multiple full chunks in a single Hugging Face batch call."""
        if len(keys) != len(memory_objs):
            raise ValueError(
                "keys and memory_objs must have the same length for batched_put"
            )

        await asyncio.to_thread(self._batched_put_sync, keys, memory_objs)

    def support_batched_get(self) -> bool:
        """Report support for batch downloads."""
        return True

    async def batched_get(
        self,
        keys: builtins.list[CacheEngineKey],
    ) -> builtins.list[MemoryObj | None]:
        """Download multiple chunks while preserving input order."""
        if not keys:
            return []

        key_strings = [key.to_string() for key in keys]
        object_sizes = await asyncio.to_thread(self._resolve_object_sizes, key_strings)

        downloads: builtins.list[tuple[int, str]] = []
        results: builtins.list[MemoryObj | None] = [None] * len(keys)

        for index, (key_str, object_size) in enumerate(
            zip(key_strings, object_sizes, strict=False)
        ):
            if object_size == 0:
                continue

            if object_size != self.full_chunk_size_bytes:
                logger.error(
                    "Size mismatch for %s: bucket has %d bytes, expected %d bytes. "
                    "Rejecting the load because hfbucket only supports full chunks.",
                    key_str,
                    object_size,
                    self.full_chunk_size_bytes,
                )
                continue

            downloads.append((index, self._key_string_to_object_path(key_str)))

        downloaded_data = await asyncio.to_thread(self._download_objects, downloads)

        try:
            for index, data in downloaded_data.items():
                if data is None:
                    continue

                if len(data) != self.full_chunk_size_bytes:
                    key_str = key_strings[index]
                    logger.error(
                        "Downloaded object for %s has %d bytes, expected %d bytes. "
                        "Rejecting the load because hfbucket only supports full "
                        "chunks.",
                        key_str,
                        len(data),
                        self.full_chunk_size_bytes,
                    )
                    self._set_cached_object_size(key_str, len(data))
                    continue

                memory_obj = self.local_cpu_backend.allocate(
                    self.meta_shapes,
                    self.meta_dtypes,
                    self.meta_fmt,
                )
                if memory_obj is None:
                    logger.debug(
                        "Memory allocation failed while downloading from hfbucket."
                    )
                    continue

                try:
                    buffer = memory_obj.byte_array.cast("B")
                    if len(buffer) < len(data):
                        raise RuntimeError(
                            "Allocated buffer is smaller than downloaded "
                            "hfbucket object"
                        )
                    buffer[: len(data)] = data
                    results[index] = memory_obj
                except Exception:
                    memory_obj.ref_count_down()
                    raise
        except Exception:
            for existing in results:
                if existing is not None:
                    existing.ref_count_down()
            raise

        return results

    def support_batched_contains(self) -> bool:
        """Report support for synchronous prefix contains checks."""
        return True

    def batched_contains(self, keys: builtins.list[CacheEngineKey]) -> int:
        """Return the number of consecutive prefix keys that exist as full chunks."""
        key_strings = [key.to_string() for key in keys]
        object_sizes = self._resolve_object_sizes(key_strings)
        hit_count = 0
        for object_size in object_sizes:
            if object_size != self.full_chunk_size_bytes:
                return hit_count
            hit_count += 1
        return hit_count

    def support_batched_async_contains(self) -> bool:
        """Report support for async prefix contains checks."""
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: builtins.list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Asynchronously return the number of consecutive prefix hits."""
        del lookup_id
        del pin
        return await asyncio.to_thread(self.batched_contains, keys)

    def support_batched_get_non_blocking(self) -> bool:
        """Report support for non-blocking batch loads."""
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: builtins.list[CacheEngineKey],
    ) -> builtins.list[MemoryObj]:
        """Return the successful prefix of ``batched_get`` results."""
        del lookup_id
        results = await self.batched_get(keys)
        prefix_results: builtins.list[MemoryObj] = []
        found_failure = False
        for result in results:
            if found_failure:
                if result is not None:
                    result.ref_count_down()
                continue

            if result is None:
                found_failure = True
                continue

            prefix_results.append(result)
        return prefix_results

    def remove_sync(self, key: CacheEngineKey) -> bool:
        """Synchronously remove a bucket object for ``key``."""
        key_str = key.to_string()
        object_path = self._key_string_to_object_path(key_str)
        try:
            self._bucket_client.delete_files(self.bucket_id, [object_path])
        except Exception as exc:
            if _is_not_found_error(exc):
                self._set_cached_object_size(key_str, 0)
                return True
            logger.error("Failed to delete %s from hfbucket: %s", key_str, exc)
            return False

        self._set_cached_object_size(key_str, 0)
        return True

    def __repr__(self) -> str:
        return (
            f"<HFBucketConnector bucket_id={self.bucket_id} "
            f"prefix={self.object_prefix!r}>"
        )

    def _ping_sync(self) -> int:
        """Perform the synchronous bucket health check used by ``ping``."""
        try:
            self._bucket_client.bucket_info(self.bucket_id)
            return 0
        except Exception as exc:
            logger.warning("Failed to ping hfbucket %s: %s", self.bucket_id, exc)
            return 1

    def _batched_put_sync(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: Sequence[MemoryObj],
    ) -> None:
        """Upload all provided chunks using a single batch request."""
        self._ensure_bucket_for_writes()

        additions: builtins.list[tuple[bytes, str]] = []
        key_strings: builtins.list[str] = []
        for key, memory_obj in zip(keys, memory_objs, strict=True):
            key_str = key.to_string()
            self._validate_full_chunk_for_upload(key_str, memory_obj)
            additions.append(
                (
                    bytes(memory_obj.byte_array),
                    self._key_string_to_object_path(key_str),
                )
            )
            key_strings.append(key_str)

        try:
            self._bucket_client.upload_files(self.bucket_id, additions)
        except Exception:
            refreshed_sizes = self._fetch_object_sizes_sync(key_strings)
            for refreshed_key, refreshed_size in refreshed_sizes.items():
                self._set_cached_object_size(refreshed_key, refreshed_size)
            raise

        for key_str in key_strings:
            self._set_cached_object_size(key_str, self.full_chunk_size_bytes)

    def _ensure_bucket_for_writes(self) -> None:
        """Create the bucket on demand when configured to do so."""
        if not self.create_bucket_if_missing or self._bucket_create_checked:
            return

        with self._bucket_create_lock:
            if self._bucket_create_checked:
                return
            self._bucket_client.create_bucket(self.bucket_id)
            self._bucket_create_checked = True

    def _validate_full_chunk_for_upload(
        self,
        key_str: str,
        memory_obj: MemoryObj,
    ) -> None:
        """Reject partial or metadata-bearing uploads for the conservative MVP."""
        physical_size = memory_obj.get_physical_size()
        if physical_size != self.full_chunk_size_bytes:
            raise ValueError(
                f"Cannot upload {key_str}: chunk size {physical_size} bytes does not "
                f"match expected full chunk size {self.full_chunk_size_bytes} bytes. "
                "Partial/unfull chunks are not supported by hfbucket."
            )

    def _list_sync(self) -> builtins.list[str]:
        """Return LMCache key strings discovered under the configured prefix."""
        try:
            entries = self._bucket_client.list_tree(self.bucket_id, self.object_prefix)
        except Exception as exc:
            if _is_not_found_error(exc):
                return []
            raise

        keys: builtins.list[str] = []
        for entry in entries:
            if _get_object_type(entry) != "file":
                continue

            object_path = _get_object_path(entry)
            relative_path = self._relative_object_path(object_path)
            if relative_path is None or "/" in relative_path:
                continue

            key_str = decode_hfbucket_object_name(relative_path)
            keys.append(key_str)
            self._set_cached_object_size(key_str, _get_object_size(entry))

        return keys

    def _download_objects(
        self,
        downloads: Sequence[tuple[int, str]],
    ) -> dict[int, bytes | None]:
        """Download requested object paths and return their bytes by result index."""
        if not downloads:
            return {}

        batch_dir = Path(
            tempfile.mkdtemp(prefix="download-", dir=self._download_session_dir)
        )
        local_mappings: builtins.list[tuple[int, Path]] = []
        files: builtins.list[tuple[str, str]] = []
        for index, object_path in downloads:
            local_path = batch_dir / f"{index}.bin"
            local_mappings.append((index, local_path))
            files.append((object_path, str(local_path)))

        try:
            try:
                self._bucket_client.download_files(self.bucket_id, files)
            except Exception as exc:
                if not _is_not_found_error(exc):
                    logger.warning("Batch download from hfbucket raised: %s", exc)

            results: dict[int, bytes | None] = {}
            for index, local_path in local_mappings:
                if not local_path.exists():
                    results[index] = None
                    continue

                with open(local_path, "rb") as file_handle:
                    results[index] = file_handle.read()
            return results
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def _get_object_size_bytes(self, key_str: str) -> int:
        """Return the cached or fetched size for a specific LMCache key string."""
        cached_size = self._get_cached_object_size(key_str)
        if cached_size is not None:
            return cached_size

        object_sizes = self._fetch_object_sizes_sync([key_str])
        object_size = object_sizes.get(key_str, 0)
        self._set_cached_object_size(key_str, object_size)
        return object_size

    def _resolve_object_sizes(self, key_strings: Sequence[str]) -> builtins.list[int]:
        """Resolve cached and uncached object sizes while preserving order."""
        cached_results: dict[str, int] = {}
        unresolved: builtins.list[str] = []

        for key_str in key_strings:
            cached_size = self._get_cached_object_size(key_str)
            if cached_size is None:
                unresolved.append(key_str)
            else:
                cached_results[key_str] = cached_size

        if unresolved:
            fetched_sizes = self._fetch_object_sizes_sync(unresolved)
            for key_str, size in fetched_sizes.items():
                self._set_cached_object_size(key_str, size)
            cached_results.update(fetched_sizes)

        return [cached_results.get(key_str, 0) for key_str in key_strings]

    def _fetch_object_sizes_sync(
        self,
        key_strings: Sequence[str],
    ) -> dict[str, int]:
        """Fetch exact object sizes for ``key_strings`` in one metadata request."""
        if not key_strings:
            return {}

        object_paths = [
            self._key_string_to_object_path(key_str) for key_str in key_strings
        ]
        try:
            path_infos = self._bucket_client.get_paths_info(
                self.bucket_id, object_paths
            )
        except Exception as exc:
            if _is_not_found_error(exc):
                return {key_str: 0 for key_str in key_strings}
            raise

        # Match by path, not by request order: the HF API does not document
        # any ordering guarantee, and assuming positional correspondence can
        # silently cache zero sizes (= "missing") for existing objects when
        # the server reorders the response.
        size_by_path: dict[str, int] = {}
        for path_info in path_infos:
            path = _get_object_path(path_info)
            if path:
                size_by_path[path] = _extract_exact_object_size(
                    path_info,
                    expected_path=path,
                )

        return {
            key_str: size_by_path.get(object_path, 0)
            for key_str, object_path in zip(key_strings, object_paths, strict=False)
        }

    def _get_cached_object_size(self, key_str: str) -> int | None:
        """Return a live cache entry, pruning expired metadata opportunistically."""
        now = time.monotonic()
        with self._metadata_cache_lock:
            cache_entry = self._metadata_cache.get(key_str)
            if cache_entry is None:
                return None
            if cache_entry.expires_at <= now:
                self._metadata_cache.pop(key_str, None)
                return None
            return cache_entry.size_bytes

    def _set_cached_object_size(self, key_str: str, object_size: int) -> None:
        """Store an object size in the TTL cache."""
        expires_at = time.monotonic() + self.metadata_cache_ttl_secs
        with self._metadata_cache_lock:
            self._metadata_cache[key_str] = _CachedObjectMetadata(
                size_bytes=object_size,
                expires_at=expires_at,
            )
            self._metadata_cache_updates += 1
            if self._metadata_cache_updates % _METADATA_CACHE_PRUNE_INTERVAL == 0:
                self._prune_expired_cache_entries_locked(time.monotonic())

    def _prune_expired_cache_entries_locked(self, now: float) -> None:
        """Remove expired cache entries while holding ``_metadata_cache_lock``."""
        expired_keys: builtins.list[str] = [
            key_str
            for key_str, cache_entry in self._metadata_cache.items()
            if cache_entry.expires_at <= now
        ]
        for key_str in expired_keys:
            self._metadata_cache.pop(key_str, None)

    def _key_string_to_object_path(self, key_str: str) -> str:
        """Return the bucket object path for a serialized LMCache key."""
        encoded_key = encode_hfbucket_object_name(key_str)
        if self.object_prefix:
            return f"{self.object_prefix}/{encoded_key}"
        return encoded_key

    def _relative_object_path(self, object_path: str) -> str | None:
        """Return the path relative to the configured LMCache prefix."""
        normalized_path = object_path.strip("/")
        if not self.object_prefix:
            return normalized_path

        prefix_with_separator = f"{self.object_prefix}/"
        if not normalized_path.startswith(prefix_with_separator):
            return None
        return normalized_path[len(prefix_with_separator) :]


def parse_hfbucket_handle(bucket_handle: str) -> HFBucketLocation:
    """Parse a Hugging Face bucket handle into bucket id and prefix.

    Args:
        bucket_handle: Handle in ``hf://buckets/<namespace>/<bucket>/<prefix>`` form.

    Returns:
        Parsed bucket location containing the ``namespace/bucket`` bucket id and the
        optional LMCache object prefix.

    Raises:
        ValueError: If the handle is not a valid Hugging Face bucket handle.
    """
    normalized_handle = bucket_handle.strip()
    if not normalized_handle.startswith(_HFBUCKET_HANDLE_PREFIX):
        raise ValueError(
            "bucket_handle must start with 'hf://buckets/' for the hfbucket plugin"
        )

    path = normalized_handle[len(_HFBUCKET_HANDLE_PREFIX) :].strip("/")
    path_parts = [part for part in path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(
            "bucket_handle must be in the form "
            "'hf://buckets/<namespace>/<bucket>[/<prefix>]'"
        )

    bucket_id = f"{path_parts[0]}/{path_parts[1]}"
    object_prefix = "/".join(path_parts[2:])
    return HFBucketLocation(
        bucket_id=bucket_id,
        object_prefix=object_prefix,
    )


def resolve_hfbucket_connector_config(
    config: LMCacheEngineConfig,
    plugin_name: str,
) -> HFBucketConnectorConfig:
    """Resolve plugin-scoped configuration for the HFBucket connector.

    Args:
        config: LMCache engine config used to resolve ``extra_config`` values.
        plugin_name: Full plugin name, such as ``hfbucket`` or ``hfbucket.prod``.

    Returns:
        Parsed connector configuration for the requested plugin instance.

    Raises:
        ValueError: If required configuration keys are missing or malformed.
    """
    extra_config = config.extra_config or {}
    config_prefix = f"remote_storage_plugin.{plugin_name}"

    bucket_handle_obj = extra_config.get(f"{config_prefix}.bucket_handle")
    if not isinstance(bucket_handle_obj, str) or not bucket_handle_obj:
        raise ValueError(
            f"HFBucket connector '{plugin_name}' requires "
            f"'{config_prefix}.bucket_handle'"
        )

    token_env_obj = extra_config.get(f"{config_prefix}.token_env", "HF_TOKEN")
    token_env = token_env_obj if isinstance(token_env_obj, str) else "HF_TOKEN"

    token_obj = extra_config.get(f"{config_prefix}.token")
    token = token_obj if isinstance(token_obj, str) and token_obj else None

    create_bucket_if_missing = _coerce_bool(
        extra_config.get(f"{config_prefix}.create_bucket_if_missing", False)
    )
    download_tmp_dir_value = extra_config.get(
        f"{config_prefix}.download_tmp_dir",
        str(_DEFAULT_DOWNLOAD_TMP_DIR),
    )
    download_tmp_dir = Path(
        download_tmp_dir_value
        if isinstance(download_tmp_dir_value, str)
        else str(_DEFAULT_DOWNLOAD_TMP_DIR)
    )
    metadata_cache_ttl_secs = _coerce_float(
        extra_config.get(f"{config_prefix}.metadata_cache_ttl_secs", 30.0)
    )

    return HFBucketConnectorConfig(
        plugin_name=plugin_name,
        bucket_location=parse_hfbucket_handle(bucket_handle_obj),
        token_env=token_env,
        token=token,
        create_bucket_if_missing=create_bucket_if_missing,
        download_tmp_dir=download_tmp_dir,
        metadata_cache_ttl_secs=metadata_cache_ttl_secs,
    )


def encode_hfbucket_object_name(key_str: str) -> str:
    """Encode a serialized LMCache key into a reversible bucket object name."""
    return quote(key_str, safe="")


def decode_hfbucket_object_name(object_name: str) -> str:
    """Decode a reversible bucket object name back into an LMCache key string."""
    return unquote(object_name)


def _normalize_save_chunk_meta_config(
    config: LMCacheEngineConfig,
) -> LMCacheEngineConfig:
    """Clone config and default ``save_chunk_meta`` to ``False`` for hfbucket.

    The generic ``RemoteConnector`` base defaults ``save_chunk_meta`` to ``True``
    when the key is absent. The hfbucket connector is intentionally a full-chunk-only
    backend and treats an omitted key as ``False`` so users can configure it with only
    plugin-scoped fields.
    """
    if config.extra_config is not None and "save_chunk_meta" in config.extra_config:
        return config

    normalized_config = copy(config)
    normalized_extra_config = (
        dict(config.extra_config) if config.extra_config is not None else {}
    )
    normalized_extra_config["save_chunk_meta"] = False
    normalized_config.extra_config = normalized_extra_config
    return normalized_config


def _resolve_hf_token(token_env: str, token: str | None) -> str | None:
    """Resolve the Hugging Face access token from env-first configuration."""
    env_token = os.environ.get(token_env, "") if token_env else ""
    if env_token:
        return env_token
    return token


def _validate_huggingface_hub_support() -> None:
    """Raise a clear error when the installed ``huggingface_hub`` is too old.

    Assumes ``huggingface_hub`` is importable (guaranteed by the declared
    dependency). Validates that the version is recent enough to expose the
    Buckets APIs used by this connector.
    """
    version_value = getattr(huggingface_hub, "__version__", None)
    if isinstance(version_value, str):
        try:
            version = Version(version_value)
        except InvalidVersion:
            pass
        else:
            if version < _MIN_HUGGINGFACE_HUB_VERSION:
                raise _build_huggingface_hub_error(
                    "huggingface_hub version is too old for Buckets support; "
                    "install huggingface_hub>=1.5.0"
                )

    required_methods = (
        "batch_bucket_files",
        "download_bucket_files",
        "list_bucket_tree",
        "bucket_info",
        "create_bucket",
        "get_bucket_paths_info",
    )
    missing = [name for name in required_methods if not hasattr(HfApi, name)]
    if missing:
        raise _build_huggingface_hub_error(
            "huggingface_hub does not expose required Buckets APIs: "
            + ", ".join(missing)
        )


def _build_huggingface_hub_error(message: str) -> RuntimeError:
    """Create a consistent runtime error for unsupported ``huggingface_hub``."""
    return RuntimeError(
        f"{message}. LMCache HFBucket support requires huggingface_hub>=1.5.0."
    )


def _extract_exact_object_size(path_info: object, expected_path: str) -> int:
    """Extract a file size from an exact path info object."""
    if _get_object_type(path_info) != "file":
        return 0

    object_path = _get_object_path(path_info)
    if object_path and object_path != expected_path:
        return 0

    return _get_object_size(path_info)


def _get_object_path(path_info: object) -> str:
    """Safely read the ``path`` attribute from a Hugging Face path object."""
    path = getattr(path_info, "path", "")
    return path if isinstance(path, str) else ""


def _get_object_type(path_info: object) -> str:
    """Safely read the ``type`` attribute from a Hugging Face path object."""
    obj_type = getattr(path_info, "type", "")
    return obj_type if isinstance(obj_type, str) else ""


def _get_object_size(path_info: object) -> int:
    """Safely read the ``size`` attribute from a Hugging Face path object."""
    size_obj = getattr(path_info, "size", 0)
    return size_obj if isinstance(size_obj, int) else 0


def _is_not_found_error(exc: Exception) -> bool:
    """Return whether the exception represents a missing bucket or object."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 404

    direct_status_code = getattr(exc, "status_code", None)
    if isinstance(direct_status_code, int):
        return direct_status_code == 404

    return "404" in str(exc)


def _coerce_bool(value: object) -> bool:
    """Coerce config values that may arrive as bools or strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(value: object) -> float:
    """Coerce config values that may arrive as numeric strings."""
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"Expected float-compatible value, got {type(value).__name__}")
