# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import threading
import time

# Third Party
from maru import MaruConfig, MaruHandler
from maru_lmcache import CxlMemoryAdapter
import torch

# First Party
from lmcache import torch_device_type
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface

logger = init_logger(__name__)


class MaruBackend(AllocatorBackendInterface):
    """Maru CXL shared memory storage backend.

    Implements AllocatorBackendInterface with its own CxlMemoryAdapter.
    No LocalCPUBackend needed — data lives directly in CXL mmap memory.

    Put is async (Future): metadata registration via RPC.
    Get is sync: CXL memory direct read (no network I/O).

    Args:
        config: LMCache engine configuration. Must have maru_path set.
        metadata: LMCache engine metadata.
        loop: asyncio event loop for async put tasks.
        dst_device: Target device string (unused for CXL, kept for interface).
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = torch_device_type,
    ):
        super().__init__(dst_device=dst_device)

        if config.use_layerwise:
            raise NotImplementedError(
                "MaruBackend does not yet support layerwise KV cache."
            )

        # 1. Config
        self.config = config
        self.loop = loop

        self._full_chunk_size_bytes: int = get_size_bytes(
            metadata.get_shapes(), metadata.get_dtypes()
        )
        assert self._full_chunk_size_bytes % metadata.chunk_size == 0
        self._single_token_size: int = (
            self._full_chunk_size_bytes // metadata.chunk_size
        )

        self._mla_worker_id_as0_mode: bool = (
            config.get_extra_config_value(
                "remote_enable_mla_worker_id_as0", metadata.use_mla
            )
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )

        # 2. Handler
        self._handler = self._create_handler(config)

        # 3. Allocator
        self.memory_allocator = self.initialize_allocator(config, metadata)

        # 4. State
        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

    def __str__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _pool_size_gb_to_bytes(size_gb: float) -> int:
        """Convert pool size in GB to bytes."""
        return int(size_gb * 1024**3)

    # =========================================================================
    # Initialization helpers
    # =========================================================================

    def _create_handler(
        self,
        config: LMCacheEngineConfig,
    ) -> "MaruHandler":
        """Create and connect a MaruHandler.

        Args:
            config: LMCache engine configuration.

        Returns:
            Connected MaruHandler instance.

        Raises:
            RuntimeError: If MaruHandler connection fails.
        """
        assert config.maru_path is not None, "maru_path must be set for MaruBackend"

        # Convert maru:// scheme to tcp:// for ZMQ
        server_url = config.maru_path
        if server_url.startswith("maru://"):
            server_url = "tcp://" + server_url[len("maru://") :]

        extra = config.extra_config or {}
        maru_config = MaruConfig(
            server_url=server_url,
            instance_id=extra.get("maru_instance_id"),
            pool_size=self._pool_size_gb_to_bytes(config.maru_pool_size),
            chunk_size_bytes=self._full_chunk_size_bytes,
            auto_connect=False,
            timeout_ms=extra.get("maru_timeout_ms", 5000),
            use_async_rpc=extra.get("maru_use_async_rpc", True),
            max_inflight=extra.get("maru_max_inflight", 64),
            eager_map=extra.get("maru_eager_map", True),
        )

        handler = MaruHandler(maru_config)
        if not handler.connect():
            raise RuntimeError(f"Failed to connect MaruHandler to {config.maru_path}")
        logger.debug("[Maru] Connected to %s", config.maru_path)
        return handler

    # =========================================================================
    # AllocatorBackendInterface
    # =========================================================================

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> MemoryAllocatorInterface:
        """Create CxlMemoryAdapter backed by the connected handler.

        Args:
            config: LMCache engine configuration.
            metadata: LMCache engine metadata.

        Returns:
            CxlMemoryAdapter instance.
        """
        shapes = metadata.get_shapes()
        dtypes = metadata.get_dtypes()
        fmt = MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        chunk_size = self._handler.get_chunk_size()

        return CxlMemoryAdapter(
            handler=self._handler,
            shapes=shapes,
            dtypes=dtypes,
            fmt=fmt,
            chunk_size=chunk_size,
        )

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        """Returns the underlying CxlMemoryAdapter."""
        return self.memory_allocator

    def get_allocator_backend(self) -> "MaruBackend":
        """Returns self as the allocator backend."""
        return self

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate CXL-backed memory via CxlMemoryAdapter.

        Args:
            shapes: Tensor shape(s).
            dtypes: Tensor dtype(s).
            fmt: Memory format.
            eviction: Unused.
            busy_loop: Unused.

        Returns:
            MemoryObj backed by CXL memory, or None on failure.
        """
        obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if obj is not None:
            logger.debug(
                "[Maru] allocate rid=%d pid=%d",
                *CxlMemoryAdapter.decode_address(obj.metadata.address),
            )
        else:
            logger.debug("[Maru] allocate failed shapes=%s dtypes=%s", shapes, dtypes)
        return obj

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """Allocate multiple CXL-backed MemoryObjs.

        Args:
            shapes: Tensor shape(s) (same for each allocation).
            dtypes: Tensor dtype(s) (same for each allocation).
            batch_size: Number of allocations.
            fmt: Memory format.
            eviction: Unused.
            busy_loop: Unused.

        Returns:
            List of MemoryObj, or None if any allocation fails.
        """
        return self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    # =========================================================================
    # Put (async)
    # =========================================================================

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether key is in ongoing put tasks.

        Args:
            key: The cache key.

        Returns:
            True if the key has a pending put task.
        """
        with self.put_lock:
            return key in self.put_tasks

    @staticmethod
    def _create_immediate_empty_future() -> Future:
        """Create a Future that is already resolved with None."""
        f: Future = Future()
        f.set_result(None)
        return f

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """Submit a put task to register KV metadata with MaruServer.

        Data is already in CXL memory (zero-copy). This only registers
        the key -> location metadata via RPC.

        Args:
            key: The cache key.
            memory_obj: MemoryObj with data already written to CXL.
            on_complete_callback: Optional callback after registration.

        Returns:
            Future that completes when metadata is registered.
        """
        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return self._create_immediate_empty_future()

        assert memory_obj.tensor is not None

        # Keep CXL page alive: ref_count_down is only called on failure.
        # On success the ref is retained so the CXL memory is not reclaimed.
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_store(key, memory_obj, on_complete_callback),
            self.loop,
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        """Submit batched put tasks via single batch_store RPC.

        Args:
            keys: The cache keys.
            memory_objs: MemoryObjs with data already in CXL.
            transfer_spec: Unused.
            on_complete_callback: Optional per-key callback.

        Returns:
            List containing a single Future for the entire batch.
        """
        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return None

        for memory_obj in memory_objs:
            assert memory_obj.tensor is not None
            memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.update(keys)

        future = asyncio.run_coroutine_threadsafe(
            self._async_batch_store(list(keys), memory_objs, on_complete_callback),
            self.loop,
        )
        return [future]

    async def _async_store(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Register KV metadata with MaruServer (runs in event loop).

        Uses CxlMemoryAdapter.create_store_handle() to extract
        (region_id, page_index) from the MemoryObj's encoded address.

        Args:
            key: The cache key.
            memory_obj: MemoryObj backed by CXL memory.
            on_complete_callback: Optional callback after registration.
        """
        success = False
        try:
            allocator = self.memory_allocator
            assert isinstance(allocator, CxlMemoryAdapter)
            handle = allocator.create_store_handle(memory_obj)
            key_str = key.to_string()

            success = await asyncio.to_thread(self._handler.store, key_str, handle)

            logger.debug(
                "[Maru] store key=%s rid=%d pid=%d",
                key,
                handle.region_id,
                handle.page_index,
            )

        except Exception as e:
            logger.error("[Maru] store failed key=%s: %s", key, e)
            raise
        finally:
            with self.put_lock:
                self.put_tasks.discard(key)

            if not success:
                memory_obj.ref_count_down()

            if success and on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)

    async def _async_batch_store(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Register multiple KV metadata entries via single batch_store RPC."""
        results: Optional[list[bool]] = None
        try:
            allocator = self.memory_allocator
            assert isinstance(allocator, CxlMemoryAdapter)

            key_strs = [k.to_string() for k in keys]
            handles = [allocator.create_store_handle(m) for m in memory_objs]

            results = await asyncio.to_thread(
                self._handler.batch_store, key_strs, handles
            )
            if results is not None:
                logger.debug("[Maru] batch_store %d/%d ok", sum(results), len(results))
        except Exception as e:
            logger.error("[Maru] batch_store failed: %s", e)
            raise
        finally:
            with self.put_lock:
                self.put_tasks.difference_update(keys)

            # Release ref_count for failed stores
            for i, memory_obj in enumerate(memory_objs):
                succeeded = results is not None and i < len(results) and results[i]
                if not succeeded:
                    memory_obj.ref_count_down()

            if on_complete_callback is not None:
                for i, key in enumerate(keys):
                    if results is not None and i < len(results) and results[i]:
                        try:
                            on_complete_callback(key)
                        except Exception as e:
                            logger.warning(
                                "on_complete_callback failed for key %s: %s",
                                key,
                                e,
                            )

    # =========================================================================
    # Get (sync)
    # =========================================================================

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """Blocking get: read KV cache directly from CXL memory.

        Queries MaruServer for metadata, then returns a MemoryObj
        via CxlMemoryAdapter.get_by_location().

        Args:
            key: The cache key.

        Returns:
            MemoryObj backed by CXL memory, or None if not found.
        """
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        key_str = key.to_string()
        mem_info = self._handler.retrieve(key_str)
        if mem_info is None:
            logger.debug("[Maru] get_blocking miss key=%s", key)
            return None

        allocator = self.memory_allocator
        assert isinstance(allocator, CxlMemoryAdapter)

        memory_obj = allocator.get_by_location(
            region_id=mem_info.region_id,
            page_index=mem_info.page_index,
            actual_size=len(mem_info.view),
            single_token_size=self._single_token_size,
        )
        if memory_obj is None:
            logger.debug(
                "[Maru] get_blocking pool miss rid=%d pid=%d",
                mem_info.region_id,
                mem_info.page_index,
            )
            return None

        memory_obj.ref_count_up()

        logger.debug(
            "[Maru] get_blocking rid=%d pid=%d size=%d",
            mem_info.region_id,
            mem_info.page_index,
            len(mem_info.view),
        )
        return memory_obj

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Blocking batched get via single batch_retrieve RPC.

        Args:
            keys: The cache keys.

        Returns:
            List of MemoryObj (None for misses).
        """
        if self._mla_worker_id_as0_mode:
            keys = [k.with_new_worker_id(0) for k in keys]

        key_strs = [k.to_string() for k in keys]
        mem_infos = self._handler.batch_retrieve(key_strs)

        allocator = self.memory_allocator
        assert isinstance(allocator, CxlMemoryAdapter)

        results: List[Optional[MemoryObj]] = []
        for mem_info in mem_infos:
            if mem_info is None:
                results.append(None)
                continue
            memory_obj = allocator.get_by_location(
                region_id=mem_info.region_id,
                page_index=mem_info.page_index,
                actual_size=len(mem_info.view),
                single_token_size=self._single_token_size,
            )
            if memory_obj is None:
                results.append(None)
                continue
            memory_obj.ref_count_up()
            results.append(memory_obj)

        hits = sum(1 for r in results if r is not None)
        logger.debug("[Maru] batch_retrieve %d/%d hits", hits, len(results))
        return results

    # =========================================================================
    # Async lookup API (used by StorageManager.async_lookup_and_prefetch)
    # =========================================================================

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check how many prefix keys exist via single batch_exists RPC.

        Returns the count of contiguous keys starting from index 0
        that exist. Stops at first miss.

        Args:
            lookup_id: Unique request identifier.
            keys: Keys to check in prefix order.
            pin: If True, atomically check and pin via batch_pin RPC.

        Returns:
            Number of prefix-contiguous keys that exist.
        """
        return await asyncio.to_thread(self.batched_contains, keys, pin)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Non-blocking batched get via single batch_retrieve RPC.

        Uses handler.batch_retrieve() for a single RPC call, then
        resolves each MemoryInfo to a MemoryObj via CxlMemoryAdapter.
        Stops at first miss and returns the prefix.

        Args:
            lookup_id: Unique request identifier.
            keys: Keys to retrieve (already confirmed by contains).
            transfer_spec: Unused.

        Returns:
            List of MemoryObjs backed by CXL memory.
        """

        def _batch_get() -> list[MemoryObj]:
            if self._mla_worker_id_as0_mode:
                actual_keys = [k.with_new_worker_id(0) for k in keys]
            else:
                actual_keys = list(keys)

            key_strs = [k.to_string() for k in actual_keys]
            mem_infos = self._handler.batch_retrieve(key_strs)

            allocator = self.memory_allocator
            assert isinstance(allocator, CxlMemoryAdapter)

            results: list[MemoryObj] = []
            for mem_info in mem_infos:
                if mem_info is None:
                    break
                memory_obj = allocator.get_by_location(
                    region_id=mem_info.region_id,
                    page_index=mem_info.page_index,
                    actual_size=len(mem_info.view),
                    single_token_size=self._single_token_size,
                )
                if memory_obj is None:
                    break
                memory_obj.ref_count_up()
                memory_obj.pin()
                results.append(memory_obj)

            logger.debug(
                "[Maru] batch_get_non_blocking %d/%d hits", len(results), len(keys)
            )
            return results

        return await asyncio.to_thread(_batch_get)

    # =========================================================================
    # Contains / Pin / Unpin / Remove
    # =========================================================================

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists on MaruServer.

        Args:
            key: The cache key.
            pin: If True, atomically check existence and pin the entry
                 to protect it from eviction.

        Returns:
            True if key exists.
        """
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        key_str = key.to_string()
        if pin:
            return self._handler.pin(key_str)
        return self._handler.exists(key_str)

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check how many prefix keys exist via single batch_exists RPC.

        Args:
            keys: Keys to check in prefix order.
            pin: If True, atomically check and pin via
                 batch_pin RPC.

        Returns:
            Number of prefix-contiguous keys that exist.
        """
        if self._mla_worker_id_as0_mode:
            keys = [k.with_new_worker_id(0) for k in keys]

        key_strs = [k.to_string() for k in keys]
        if pin:
            results = self._handler.batch_pin(key_strs)
        else:
            results = self._handler.batch_exists(key_strs)
        num_hit = 0
        for exists in results:
            if not exists:
                break
            num_hit += 1
        return num_hit

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a key to prevent eviction on MaruServer.

        Increments the server-side pin_count.

        Args:
            key: The cache key.

        Returns:
            True if pinned successfully.
        """
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        return self._handler.pin(key.to_string())

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a key to allow eviction on MaruServer.

        Decrements the server-side pin_count. When pin_count reaches 0,
        the entry becomes eligible for eviction.

        Args:
            key: The cache key.

        Returns:
            True if unpinned successfully.
        """
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        return self._handler.unpin(key.to_string())

    def batched_unpin(self, keys: List[CacheEngineKey]) -> None:
        """Batch-unpin keys via single RPC.

        Decrements server-side pin_count for each key. When pin_count
        reaches 0, the entry becomes eligible for eviction.

        Args:
            keys: The cache keys to unpin.
        """
        if not keys:
            return
        if self._mla_worker_id_as0_mode:
            keys = [k.with_new_worker_id(0) for k in keys]
        key_strs = [k.to_string() for k in keys]
        self._handler.batch_unpin(key_strs)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a key from MaruServer.

        Args:
            key: The cache key.
            force: Whether to force removal.

        Returns:
            True if removed successfully.
        """
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        key_str = key.to_string()
        result = self._handler.delete(key_str)
        logger.debug("[Maru] remove key=%s success=%s", key, result)
        return result

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the backend and underlying MaruHandler."""
        while True:
            with self.put_lock:
                if not self.put_tasks:
                    break
            time.sleep(0.1)

        self.memory_allocator.close()
        self._handler.close()
        logger.info("MaruBackend closed.")
