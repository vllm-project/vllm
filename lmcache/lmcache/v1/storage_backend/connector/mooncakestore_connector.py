# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional, no_type_check
import asyncio
import json
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.system_detection import NUMADetector

logger = init_logger(__name__)

# TODO(baoloongmao): Remove these in the future
# Legacy positional-arg key order for old mooncake setup API
_LEGACY_SETUP_KEYS = [
    "local_hostname",
    "metadata_server",
    "global_segment_size",
    "local_buffer_size",
    "protocol",
    "device_name",
    "master_server_address",
]


# Legacy keys that must be passed as ``int`` to the old
# positional-arg ``store.setup()`` API (C++/pybind11).
_LEGACY_INT_KEYS = {"global_segment_size", "local_buffer_size"}

# Keys whose values may contain credentials and must be
# redacted when the setup dict is logged.
_SENSITIVE_SETUP_KEYS = {"metadata_server", "master_server_address"}


def _sanitize_setup_config(
    setup_config: Dict[str, str],
) -> Dict[str, str]:
    """Return a copy of *setup_config* with sensitive values
    redacted, safe for logging."""
    sanitized: Dict[str, str] = {}
    for k, v in setup_config.items():
        lower_k = k.lower()
        if (
            k in _SENSITIVE_SETUP_KEYS
            or "password" in lower_k
            or "token" in lower_k
            or "secret" in lower_k
            or "key" in lower_k
        ):
            sanitized[k] = "***REDACTED***"
        else:
            sanitized[k] = v
    return sanitized


def setup_mooncake_store(
    store: Any,
    config: "MooncakeStoreConfig",
) -> None:
    """Initialize a MooncakeDistributedStore instance.

    Calls ``store.setup()`` using the dict-based API introduced
    in Mooncake PR #1445 when available, and transparently falls
    back to the legacy positional-arg API otherwise.  This lets
    LMCache work with both new and old mooncake builds without
    requiring callers to know which API is present.

    Args:
        store: A MooncakeDistributedStore instance (already
            constructed but not yet setup).  Must expose a
            ``setup()`` method.
        config: A :class:`MooncakeStoreConfig` carrying the
            setup parameters in ``config.setup_config``.

    Returns:
        None.  ``store`` is mutated in place.

    Raises:
        Exception: Any exception raised by ``store.setup()``
            other than :class:`TypeError` is propagated to the
            caller (``TypeError`` is caught and used as the
            signal to fall back to the legacy API).
    """
    setup_dict = config.setup_config
    try:
        # New API (Mooncake PR #1445): setup(config: dict)
        store.setup(setup_dict)
        logger.info("Using dict-based setup API (new)")
    except TypeError:
        # Legacy API: setup with positional arguments.
        # Some keys (e.g. ``global_segment_size``) must be int.
        logger.info(
            "Dict-based setup not available, "
            "falling back to positional-arg API (legacy)"
        )
        args: List[Any] = []
        for k in _LEGACY_SETUP_KEYS:
            v: Any = setup_dict.get(k, "")
            if k in _LEGACY_INT_KEYS:
                try:
                    v = int(v) if v != "" else 0
                except (TypeError, ValueError):
                    v = 0
            args.append(v)
        store.setup(*args)


# Prefix for keys that should be forwarded to mooncake setup.
# e.g. ``mooncake_local_hostname`` -> ``local_hostname``
_MOONCAKE_PREFIX = "mooncake_"

# Keys that are consumed by LMCache only (never sent to mooncake).
_LMCACHE_ONLY_KEYS = {
    "transfer_timeout",
    "storage_root_dir",
    "prefer_local_alloc",
    "mooncake_transfer_timeout",
    "mooncake_storage_root_dir",
    "mooncake_prefer_local_alloc",
}

# Legacy keys that are forwarded without prefix (for compat).
_LEGACY_PASSTHROUGH_KEYS = {
    "local_hostname",
    "metadata_server",
    "global_segment_size",
    "local_buffer_size",
    "protocol",
    "device_name",
    "master_server_address",
}

# Default values for mooncake setup keys
_SETUP_DEFAULTS: Dict[str, str] = {
    "global_segment_size": "3355443200",
    "local_buffer_size": "1073741824",
    "protocol": "tcp",
    "device_name": "",
}


class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore.

    Mooncake setup keys are stored in ``setup_config`` dict
    and passed through to ``store.setup()`` as-is.  This
    means LMCache does **not** need to change when mooncake
    adds or removes setup keys.

    LMCache-only knobs (``transfer_timeout``,
    ``storage_root_dir``, ``prefer_local_alloc``) are kept
    as explicit attributes.
    """

    def __init__(
        self,
        setup_config: Dict[str, str],
        transfer_timeout: int = 1,
        storage_root_dir: str = "",
        prefer_local_alloc: bool = False,
    ):
        self.setup_config = dict(setup_config)
        self.transfer_timeout = transfer_timeout
        self.storage_root_dir = storage_root_dir
        self.prefer_local_alloc = prefer_local_alloc

    def __repr__(self) -> str:
        # Redact sensitive values to keep logs safe.
        return (
            "MooncakeStoreConfig("
            "setup_config=%r, "
            "transfer_timeout=%r, "
            "storage_root_dir=%r, "
            "prefer_local_alloc=%r)"
            % (
                _sanitize_setup_config(self.setup_config),
                self.transfer_timeout,
                self.storage_root_dir,
                self.prefer_local_alloc,
            )
        )

    # ----------------------------------------------------------
    # Deprecated attribute-style accessors for backward compat
    # ----------------------------------------------------------
    def _get_setup_key(self, key: str) -> str:
        return self.setup_config.get(key, "")

    def _set_setup_key(self, key: str, value: str) -> None:
        self.setup_config[key] = value

    @property
    def local_hostname(self) -> str:  # deprecated
        return self._get_setup_key("local_hostname")

    @property
    def metadata_server(self) -> str:  # deprecated
        return self._get_setup_key("metadata_server")

    @property
    def global_segment_size(self) -> str:  # deprecated
        return self._get_setup_key("global_segment_size")

    @property
    def local_buffer_size(self) -> str:  # deprecated
        return self._get_setup_key("local_buffer_size")

    @property
    def protocol(self) -> str:  # deprecated
        return self._get_setup_key("protocol")

    @property
    def device_name(self) -> str:  # deprecated
        return self._get_setup_key("device_name")

    @device_name.setter
    def device_name(self, value: str) -> None:  # deprecated
        self._set_setup_key("device_name", value)

    @property
    def master_server_address(self) -> str:  # deprecated
        return self._get_setup_key("master_server_address")

    @master_server_address.setter
    def master_server_address(  # deprecated
        self, value: str
    ) -> None:
        self._set_setup_key("master_server_address", value)

    # ----------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------
    @staticmethod
    def _build_setup_dict(
        raw: Dict[str, Any],
    ) -> Dict[str, str]:
        """Extract mooncake setup keys from *raw* config dict.

        Rules (evaluated in order):
        1. Keys in ``_LMCACHE_ONLY_KEYS`` are skipped.
        2. Keys starting with ``mooncake_`` have the prefix
           stripped and are forwarded (highest priority).
        3. Keys in ``_LEGACY_PASSTHROUGH_KEYS`` are forwarded
           as-is (backward compat, lower priority).
        4. All other keys are ignored.

        Missing keys with known defaults are filled in.
        """
        setup: Dict[str, str] = {}
        # Prefixed keys have higher priority than legacy keys;
        # collect them separately and apply last.
        prefixed_setup: Dict[str, str] = {}
        prefix_len = len(_MOONCAKE_PREFIX)

        for k, v in raw.items():
            if k in _LMCACHE_ONLY_KEYS or v is None:
                continue
            if k.startswith(_MOONCAKE_PREFIX):
                stripped = k[prefix_len:]
                if stripped:
                    prefixed_setup[stripped] = str(v)
            elif k in _LEGACY_PASSTHROUGH_KEYS:
                setup[k] = str(v)

        # Prefixed keys override legacy keys on conflict.
        setup.update(prefixed_setup)

        # Apply defaults for keys not provided
        for k, default_v in _SETUP_DEFAULTS.items():
            setup.setdefault(k, default_v)
        return setup

    @staticmethod
    def from_file(
        file_path: str,
    ) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            raw = json.load(fin)
        return MooncakeStoreConfig(
            setup_config=MooncakeStoreConfig._build_setup_dict(raw),
            transfer_timeout=raw.get("transfer_timeout", 1),
            storage_root_dir=raw.get("storage_root_dir", ""),
            prefer_local_alloc=raw.get("mooncake_prefer_local_alloc", False),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from MOONCAKE_CONFIG_PATH env var."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_file_path)

    @staticmethod
    def load_from_lmcache_config(
        config: "LMCacheEngineConfig",
    ) -> "MooncakeStoreConfig":
        """Load config from LMCacheEngineConfig.extra_config."""
        extra_config = config.extra_config
        if extra_config is None:
            raise ValueError("The extra config is not set.")
        return MooncakeStoreConfig(
            setup_config=MooncakeStoreConfig._build_setup_dict(extra_config),
            transfer_timeout=extra_config.get("transfer_timeout", 1),
            storage_root_dir=extra_config.get("storage_root_dir", ""),
            prefer_local_alloc=extra_config.get("mooncake_prefer_local_alloc", False),
        )


class MooncakestoreConnector(RemoteConnector):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_config: Optional[LMCacheEngineConfig],
        plugin_name: Optional[str] = None,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        try:
            # Third Party
            from mooncake.store import (
                MooncakeDistributedStore,
                ReplicateConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()
            config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
            if config_file_path is not None:
                self.config = MooncakeStoreConfig.from_file(config_file_path)
            elif lmcache_config is not None:
                self.config = MooncakeStoreConfig.load_from_lmcache_config(
                    lmcache_config
                )
            else:
                raise ValueError("MOONCAKE_CONFIG_PATH/lmcache_config must be provided")

            logger.info("Mooncake Configuration loaded. config: %s", self.config)

            # Check if storage_root_dir exists and set env var
            if self.config.storage_root_dir:
                os.environ["MOONCAKE_STORAGE_ROOT_DIR"] = self.config.storage_root_dir
                logger.info(
                    "Set MOONCAKE_STORAGE_ROOT_DIR to: %s",
                    self.config.storage_root_dir,
                )

            logger.info(
                "Setting up Mooncake store with setup_config: %s",
                _sanitize_setup_config(self.config.setup_config),
            )

            try:
                numa_mapping = getattr(
                    local_cpu_backend.memory_allocator, "numa_mapping", None
                )
                if numa_mapping is None and lmcache_config is not None:
                    numa_mapping = NUMADetector.get_numa_mapping(lmcache_config)

                if numa_mapping:
                    current_device_id = torch_dev.current_device()
                    gpu_to_numa = getattr(numa_mapping, "gpu_to_numa_mapping", {})
                    numa_id = gpu_to_numa.get(current_device_id)
                    logger.info(
                        f"NUMA mapping detected (pre-Mooncake setup): {gpu_to_numa}"
                    )
                    try:
                        # Third Party
                        from mooncake.store import bind_to_numa_node

                        if numa_id is not None:
                            bind_to_numa_node(numa_id)
                            logger.info(
                                f"GPU {current_device_id}, "
                                f"NUMA node {numa_id} binding done"
                            )
                        else:
                            logger.info(
                                f"NUMA mapping not found for GPU {current_device_id}"
                            )
                    except ImportError:
                        logger.warning(
                            "unable to import bind_to_numa_node from mooncake.store"
                        )
                else:
                    logger.info("NUMA mapping unavailable or disabled")
            except Exception as e:
                logger.warning(
                    f"Failed to determine NUMA mapping before Mooncake setup: {e}"
                )

            setup_mooncake_store(self.store, self.config)
            logger.info("Mooncake store setup completed successfully")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.registered_buffer_ptr = None

        # Initialize ReplicateConfig
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1

        # Set preferred_segment based on configuration
        if self.config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.store.get_hostname()

        # Register CPU buffer for zero-copy operations
        self._register_cpu_buffer()

        logger.info("MooncakeConnector initialized successfully.")

    def _register_cpu_buffer(self):
        """Register CPU buffer for zero-copy operations."""
        try:
            allocator = self.local_cpu_backend.memory_allocator
            if hasattr(allocator, "pin_allocator") and hasattr(
                allocator.pin_allocator, "buffer"
            ):
                buffer = allocator.pin_allocator.buffer
                self.registered_buffer_ptr = buffer.data_ptr()
                result = self.store.register_buffer(buffer.data_ptr(), buffer.numel())
                if result == 0:
                    logger.info(
                        f"Registered: {hex(buffer.data_ptr())}, {buffer.numel()} bytes"
                    )
                else:
                    logger.warning(f"Buffer registration failed: error={result}")
                    self.registered_buffer_ptr = None
            else:
                self.registered_buffer_ptr = None
        except Exception as e:
            logger.error(f"Buffer registration error: {e}")
            self.registered_buffer_ptr = None

    def _unregister_cpu_buffer(self):
        """Unregister CPU buffer."""
        if self.registered_buffer_ptr is not None:
            result = self.store.unregister_buffer(self.registered_buffer_ptr)
            if result == 0:
                logger.info(f"Unregistered buffer: {hex(self.registered_buffer_ptr)}")
            else:
                logger.warning(f"Buffer unregistration failed: error={result}")
            self.registered_buffer_ptr = None

    def support_batched_get(self) -> bool:
        """
        Check if the connector supports batched get

        Returns:
            True if batched get is supported, False otherwise
        """
        return True

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string())

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string())

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get operation - the only supported get method.
        Uses batch_get_into (with metadata) or batch_get_buffer (without metadata).
        """
        if not keys:
            return []

        # Check if we have metadata for zero-copy operations
        if self.save_chunk_meta:
            # Use legacy mode with metadata stored in remote
            return await self._batch_get_buffer(keys)
        else:
            # Use optimized mode with local metadata
            return await self._batch_get_into(keys)

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            if not self.store.is_exist(key.to_string()):
                break
            num_hit_counts += 1
        return num_hit_counts

    async def _batch_get_into(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Zero-copy batch get using batch_get_into when metadata is available locally.
        This is used when save_chunk_meta=False (metadata not stored remotely).
        """
        if not self.meta_shapes or not self.meta_dtypes or not self.meta_fmt:
            logger.error(
                f"Metadata required for batch_get_into but not available: "
                f"meta_shapes={self.meta_shapes}, "
                f"meta_dtypes={self.meta_dtypes}, "
                f"meta_fmt={self.meta_fmt}"
            )
            return [None] * len(keys)

        logger.debug(f"Using batch_get_into for {len(keys)} keys (zero-copy mode)")

        # Reserve a buffer for every requested chunk
        memory_objs: list[Optional[MemoryObj]] = []
        valid_idx: list[int] = []

        key_strs: list[str] = []
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []

        for i, _ in enumerate(keys):
            obj = self.local_cpu_backend.allocate(
                self.meta_shapes, self.meta_dtypes, self.meta_fmt
            )
            memory_objs.append(obj)
            if obj is not None and obj.raw_tensor is not None:
                valid_idx.append(i)

                # Prepare the argument lists for the C++ call
                key_strs.append(keys[i].to_string())
                buffer_ptrs.append(obj.data_ptr)
                buffer_sizes.append(obj.get_size())

        if not valid_idx:
            logger.warning("Batch-get aborted: unable to allocate any buffers.")
            return [None] * len(keys)

        try:
            # Single RPC call for multiple chunks
            logger.debug(f"Calling batch_get_into with {len(key_strs)} keys")
            bytes_read_list = await asyncio.to_thread(
                self.store.batch_get_into, key_strs, buffer_ptrs, buffer_sizes
            )
            logger.debug(f"batch_get_into returned: {bytes_read_list}")

            # Assemble the final result list
            results: list[Optional[MemoryObj]] = [None] * len(keys)

            for i, n_read in zip(valid_idx, bytes_read_list, strict=False):
                if n_read <= 0:
                    logger.warning(
                        f"batch_get_into failed for key {keys[i]} (code={n_read})"
                    )
                    memory_objs[i].ref_count_down()  # type: ignore
                    continue

                try:
                    results[i] = self.reshape_partial_chunk(
                        memory_objs[i],  # type: ignore
                        n_read,
                    )
                except Exception as exc:
                    logger.error(f"Reshape failed for key {keys[i]}: {exc}")
                    memory_objs[i].ref_count_down()  # type: ignore

            return results

        except Exception as exc:
            logger.error(f"batch_get_into threw exception: {str(exc)}")
            # Release any buffers we successfully allocated
            for i in valid_idx:
                memory_objs[i].ref_count_down()  # type: ignore
            return [None] * len(keys)

    async def _batch_get_buffer(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get using batch_get_buffer when metadata is stored remotely.
        This is used when save_chunk_meta=True (metadata stored with data).
        """
        key_strs = [key.to_string() for key in keys]

        try:
            buffers = await asyncio.to_thread(self.store.batch_get_buffer, key_strs)
        except Exception as e:
            logger.error(f"batch_get_buffer failed: {str(e)}")
            return [None] * len(keys)

        results: list[Optional[MemoryObj]] = []
        for i, buffer in enumerate(buffers):
            if buffer is None:
                logger.warning(f"Buffer {i} is None for key {key_strs[i]}")
                results.append(None)
                continue
            try:
                memory_obj = self._process_buffer_with_metadata(buffer)
                results.append(memory_obj)
            except Exception as e:
                logger.error(
                    f"Failed to process buffer {i} for key {key_strs[i]}: {str(e)}"
                )
                results.append(None)
        return results

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Single get method - NOT SUPPORTED.
        Use batched_get instead for all operations.
        """
        logger.error("Single get operation is not supported. Use batched_get instead.")
        raise NotImplementedError(
            "Single get is not supported. Use batched_get([key]) instead."
        )

    def _process_buffer_with_metadata(self, buffer: bytes) -> Optional[MemoryObj]:
        """
        Process buffer that contains metadata + data.
        Used when save_chunk_meta=True (metadata stored remotely).
        """
        retrieved_view = memoryview(buffer)
        metadata_bytes = retrieved_view[: self.remote_metadata_bytes]
        if metadata_bytes is None or len(metadata_bytes) != self.remote_metadata_bytes:
            return None

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shapes,
            metadata.dtypes,
            metadata.fmt,
        )
        assert len(retrieved_view) == metadata.length + self.remote_metadata_bytes

        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        if memory_obj.raw_tensor is not None:
            temp_tensor = torch.frombuffer(
                buffer,
                dtype=torch.uint8,
                offset=self.remote_metadata_bytes,
                count=metadata.length,
            )

            memory_obj.raw_tensor.copy_(temp_tensor)
            return memory_obj
        else:
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Put operation with metadata-consistent handling.
        Uses put_from (without metadata) or
        put_parts (with metadata) to match get behavior.
        """
        key_str = key.to_string()

        # Check metadata handling mode to match get behavior
        if self.save_chunk_meta:
            # Use put_parts with metadata stored remotely
            await self._put_with_metadata(key_str, memory_obj)
        else:
            # Use put_from without metadata (zero-copy)
            await self._put_without_metadata(key_str, memory_obj)

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ):
        """
        Batched put with clear split by metadata mode.
        - save_chunk_meta False: use Mooncake's batch_put_from (zero-copy).
        - save_chunk_meta True: no batch API; fall back to sequential put_parts.
        """
        if not keys:
            return

        if self.save_chunk_meta:
            await self._batched_put_with_metadata(keys, memory_objs)
        else:
            await self._batched_put_zero_copy(keys, memory_objs)

    async def _batched_put_zero_copy(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        key_strs = [k.to_string() for k in keys]
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []
        for obj in memory_objs:
            assert obj.raw_tensor is not None
            buffer_ptrs.append(obj.data_ptr)
            buffer_sizes.append(obj.get_size())

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.batch_put_from,
                    key_strs,
                    buffer_ptrs,
                    buffer_sizes,
                    self.replica_config,
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout during batch_put_from; some decoders may redo prefill."
            )

    async def _batched_put_with_metadata(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        for key, obj in zip(keys, memory_objs, strict=False):
            await self._put_with_metadata(key.to_string(), obj)

    async def _put_without_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Zero-copy put using put_from when metadata is not stored remotely.
        This is used when save_chunk_meta=False (matches _batch_get_into).
        """
        try:
            assert memory_obj.raw_tensor is not None
            buffer_ptr = memory_obj.data_ptr
            buffer_size = memory_obj.get_size()

            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_from,
                    key_str,
                    buffer_ptr,
                    buffer_size,
                    self.replica_config,
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when putting key {key_str} using put_from. "
                "Decode instance may redo prefill."
            )
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_from: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    async def _put_with_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Put using put_parts when metadata is stored remotely.
        This is used when save_chunk_meta=True (matches _batch_get_buffer).
        """
        try:
            # Serialize data and metadata
            kv_bytes = memory_obj.byte_array
            kv_shapes = memory_obj.get_shapes()
            kv_dtypes = memory_obj.get_dtypes()
            memory_format = memory_obj.get_memory_format()

            metadata_bytes = RemoteMetadata(
                len(kv_bytes), kv_shapes, kv_dtypes, memory_format
            ).serialize()
            assert len(metadata_bytes) == self.remote_metadata_bytes

            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_parts, key_str, metadata_bytes, kv_bytes
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when putting key {key_str} using put_parts. "
                "Decode instance may redo prefill."
            )
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_parts: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        # Unregister buffer before closing the store
        self._unregister_cpu_buffer()

        self.store.close()
        logger.info("Closed the mooncake store connection")
