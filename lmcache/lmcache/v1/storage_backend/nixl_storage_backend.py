# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from urllib.parse import quote as url_quote
import asyncio
import hashlib
import os
import threading
import time
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import nixl_prepped_dlist_handle as NixlDlistHandle

try:
    # Third Party
    from nixl._api import (
        nixl_thread_sync_t,
    )

    _NIXL_SYNC_MODE_SUPPORTED = True
    _NIXL_SYNC_MODE_DEFAULT = nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT
except ImportError:
    nixl_thread_sync_t = None  # type: ignore[assignment]
    _NIXL_SYNC_MODE_SUPPORTED = False
    _NIXL_SYNC_MODE_DEFAULT = None
# Third Party
from nixl._api import nixl_xfer_handle as NixlXferHandle
from nixl._api import (
    nixlBind,
)
import torch

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    _allocate_cpu_memory,
    _allocate_gpu_memory,
    _free_cpu_memory,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.path_sharder import PathSharder
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# POSIX permission mode for files created via ``os.open()`` with ``O_CREAT``.
# 0o644 = rw-r--r-- (owner read/write, group/others read-only).
DEFAULT_FILE_CREATE_MODE = 0o644

# Max concurrency for parallel S3 HEAD requests in batched_contains().
_CONTAINS_BATCH_SIZE = 16

# Max static b128 pool size: the slot index occupies 8 hex chars (32 bits) of
# the 32-hex-char (128-bit) name, so the index must be <= 0xffffffff.
B128_MAX_POOL_SIZE = 0x100000000  # 2**32


@dataclass
class NixlStorageConfig:
    buffer_size: int
    pool_size: int
    buffer_device: str
    backend: str
    backend_params: dict[str, str]
    dynamic_storage: bool
    enable_presence_cache: bool
    enable_async_put: bool
    use_direct_io: bool
    path: Union[str, List[str]]
    use_hugepages: bool
    enable_prog_thread: bool
    sync_mode: Optional[Any]  # nixl_thread_sync_t, None if unsupported
    presence_cache_only: bool
    path_sharding: str

    @staticmethod
    def validate_nixl_backend(backend: str, device: str) -> bool:
        device = device.split(":", 1)[0]
        if backend in ("GDS", "GDS_MT", "OBJ"):
            return device == "cpu" or device == "cuda"
        elif backend in ("POSIX", "HF3FS", "AZURE_BLOB", "DOCA_MEMOS"):
            return device == "cpu"
        else:
            return False

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ):
        assert config.nixl_buffer_device is not None
        if config.nixl_buffer_device != "cpu":
            assert config.nixl_buffer_size is not None

        extra_config = config.extra_config
        assert extra_config is not None
        assert extra_config.get("enable_nixl_storage")

        enable_presence_cache = extra_config.get("nixl_presence_cache", False)
        enable_async_put = extra_config.get("nixl_async_put", False)
        backend_params = dict(extra_config.get("nixl_backend_params", {}))
        use_direct_io = extra_config.get("use_direct_io", False)
        pool_size = extra_config.get("nixl_pool_size")

        backend = extra_config.get("nixl_backend")

        # Per-worker endpoint distribution: if nixl_endpoint_list is set,
        # override endpoint_override so each TP worker targets a different
        # object-storage endpoint (round-robin by local_worker_id).
        endpoint_list = extra_config.get("nixl_endpoint_list")
        if endpoint_list is not None and len(endpoint_list) == 0:
            raise ValueError("nixl_endpoint_list is set but empty")
        if backend == "OBJ" and endpoint_list:
            if "endpoint_override" in backend_params:
                logger.warning(
                    "nixl_endpoint_list is set; ignoring "
                    "nixl_backend_params.endpoint_override (%s)",
                    backend_params["endpoint_override"],
                )
            ep = endpoint_list[metadata.local_worker_id % len(endpoint_list)]
            if not ep.startswith(("http://", "https://")):
                raise ValueError(
                    f"nixl_endpoint_list entry {ep!r} is not a valid URL "
                    "(must start with 'http://' or 'https://')"
                )
            backend_params["endpoint_override"] = ep
            logger.info(
                "Worker %d using endpoint %s (from %d endpoints)",
                metadata.local_worker_id,
                ep,
                len(endpoint_list),
            )
        path = extra_config.get("nixl_path")
        use_hugepages = extra_config.get("nixl_use_hugepages", False)
        enable_prog_thread = extra_config.get("nixl_enable_prog_thread", True)
        sync_mode_str = extra_config.get("nixl_sync_mode", None)
        if sync_mode_str is not None and not _NIXL_SYNC_MODE_SUPPORTED:
            raise ValueError(
                "nixl_sync_mode is set in config but this NIXL version does not "
                "support the sync_mode argument in NixlAgentConfig "
                "(requires ai-dynamo/nixl#1501). Remove nixl_sync_mode from config "
                "or upgrade NIXL."
            )
        sync_mode = _NIXL_SYNC_MODE_DEFAULT
        if sync_mode_str is not None:
            attr_name = f"NIXL_THREAD_SYNC_{sync_mode_str.upper()}"
            if not hasattr(nixl_thread_sync_t, attr_name):
                raise ValueError(
                    f"Invalid nixl_sync_mode '{sync_mode_str}'. "
                    f"Valid values are the suffixes of NIXL_THREAD_SYNC_* "
                    f"in nixl_thread_sync_t."
                )
            sync_mode = getattr(nixl_thread_sync_t, attr_name)
        presence_cache_only = extra_config.get("nixl_presence_cache_only", False)

        path_sharding = extra_config.get("nixl_path_sharding", "by_gpu")

        assert pool_size is not None
        assert backend is not None
        assert use_direct_io in [False, True]

        dynamic_storage = pool_size == 0
        if dynamic_storage:
            assert not config.save_unfull_chunk, (
                "save_unfull_chunk should be set to False when using dynamic storage"
            )

        corrected_device = get_correct_device(
            config.nixl_buffer_device, metadata.worker_id
        )

        # align the buffer size to have the required alignment. In CPU mode the
        # pool is owned by LocalCPUBackend (sized by max_local_cpu_size) and the
        # nixl_buffer_size config field is unused; the buffer_size on the
        # resulting NixlStorageConfig is left as 0.
        if config.nixl_buffer_device == "cpu":
            buffer_size = 0
        else:
            align_bytes = get_size_bytes(
                [torch.Size(metadata.kv_shape)], [metadata.kv_dtype]
            )
            if config.nixl_buffer_size % align_bytes != 0:
                buffer_size = (
                    (config.nixl_buffer_size + align_bytes - 1) // align_bytes
                ) * align_bytes
                logger.warning(
                    f"Nixl buffer size {config.nixl_buffer_size} is not a multiple of "
                    f"align bytes {align_bytes}, auto aligned to {buffer_size}"
                )
                config.nixl_buffer_size = buffer_size
            else:
                buffer_size = config.nixl_buffer_size

        assert NixlStorageConfig.validate_nixl_backend(
            backend, config.nixl_buffer_device
        ), "Invalid NIXL backend & device combination"

        if backend in ("GDS", "GDS_MT", "POSIX", "HF3FS"):
            assert path is not None, f"nixl_path must be provided for {backend} backend"

        return NixlStorageConfig(
            buffer_size=buffer_size,
            pool_size=pool_size,
            buffer_device=corrected_device,
            backend=backend,
            backend_params=backend_params,
            dynamic_storage=dynamic_storage,
            enable_presence_cache=enable_presence_cache,
            enable_async_put=enable_async_put,
            use_direct_io=use_direct_io,
            path=path,
            use_hugepages=use_hugepages,
            enable_prog_thread=enable_prog_thread,
            sync_mode=sync_mode,
            presence_cache_only=presence_cache_only,
            path_sharding=path_sharding,
        )


class NixlDescPool(ABC):
    def __init__(self, size: int):
        self.lock = threading.Lock()
        self.size: int = size
        self.indices: List[int] = []
        self.indices.extend(reversed(range(size)))

    def get_num_available_descs(self) -> int:
        with self.lock:
            return len(self.indices)

    def pop(self) -> int:
        with self.lock:
            assert len(self.indices) > 0
            return self.indices.pop()

    def push(self, index: int):
        with self.lock:
            assert len(self.indices) < self.size
            self.indices.append(index)

    @abstractmethod
    def close(self):
        pass


class NixlFilePool(NixlDescPool):
    def __init__(
        self,
        size: int,
        sharder: PathSharder,
        use_direct_io: bool,
    ):
        super().__init__(size)
        self.fds: List[int] = []

        flags = os.O_CREAT | os.O_RDWR
        if use_direct_io:
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            else:
                logger.warning(
                    "use_direct_io is True, but O_DIRECT is not available on "
                    "this system. Falling back to buffered I/O."
                )
        base_path = sharder.selected

        for i in reversed(range(size)):
            filename = f"obj_{i}_{uuid.uuid4().hex[0:4]}.bin"
            tmp_path = os.path.join(base_path, filename)
            fd = os.open(tmp_path, flags)
            self.fds.append(fd)

    def close(self):
        # TODO: do we need to delete the files?
        with self.lock:
            assert len(self.fds) == self.size
            for fd in self.fds:
                os.close(fd)


class NixlObjectPool(NixlDescPool):
    def __init__(self, size: int, b128: bool = False) -> None:
        """Create a pool of ``size`` NIXL object slot names.

        Args:
            size: Number of object slots to pre-generate.
            b128: If True, generate 128-bit hex slot names (32 hex chars) for
                the DOCA_MEMOS backend instead of the default ``obj_{i}_{uuid}``
                names.

        Raises:
            ValueError: If ``b128`` is True and ``size`` exceeds
                ``B128_MAX_POOL_SIZE`` (the slot index would no longer fit in
                8 hex chars).
        """
        if b128 and size > B128_MAX_POOL_SIZE:
            raise ValueError(
                "b128 object pool supports at most 2**32 slots "
                f"(got {size}); the slot index must fit in 8 hex chars"
            )
        super().__init__(size)
        self.keys: List[str] = []

        for i in reversed(range(size)):
            if b128:
                # DOCA_MEMOS requires slot names that fit in 128 bits and are
                # hex-decodable (the NIXL plugin hex-decodes the name on the
                # other side). 8 hex chars (32 bits) encode the slot index for
                # in-pool uniqueness; 24 hex chars (96 bits) of randomness avoid
                # collisions across LMCache instances on the shared backend.
                key = f"{i:08x}{uuid.uuid4().hex[:24]}"
            else:
                key = f"obj_{i}_{uuid.uuid4().hex[0:4]}"
            self.keys.append(key)

    def close(self):
        pass


class SetPresenceCache:
    """Default presence cache using a thread-safe Python set."""

    def __init__(self) -> None:
        self._keys: set[int] = set()

    def add(self, key: int) -> None:
        self._keys.add(key)

    def discard(self, key: int) -> None:
        self._keys.discard(key)

    def contains(self, key: int) -> bool:
        return key in self._keys


PresenceCache = Union[SetPresenceCache]


@dataclass
class NixlDesc:
    device_id: int
    meta_info: str
    path: Optional[str] = None


def _close_file_descs(descs: List[NixlDesc]) -> None:
    """Best-effort close of the FDs in descs."""
    for d in descs:
        try:
            os.close(d.device_id)
        except OSError:
            pass


def _unlink_file_descs(descs: List[NixlDesc]) -> None:
    """
    Best-effort unlink of every desc whose ``path`` is set.

    Called only on FILE-write failure paths to remove the (empty or
    partially-written) file we created with ``O_CREAT``.
    """
    for d in descs:
        if d.path is None:
            continue
        try:
            os.unlink(d.path)
        except OSError:
            pass


@dataclass
class NixlKeyMetadata:
    index: int
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: Optional[MemoryFormat] = None
    pin_count: int = 0

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check if the related key can be evicted.
        """
        return not self.is_pinned


class NixlStorageAgent(ABC):
    agent_name: str
    nixl_agent: NixlAgent
    mem_reg_descs: nixlBind.nixlRegDList
    mem_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        enable_prog_thread: bool,
        sync_mode: Optional[Any] = None,
    ):
        buffer_ptr = allocator.buffer_ptr
        buffer_size = allocator.buffer_size
        page_size = allocator.align_bytes

        self.backend = backend
        self.agent_name = "NixlAgent_" + str(uuid.uuid4())
        nixl_conf_kwargs: dict[str, Any] = {
            "backends": [],
            "enable_prog_thread": enable_prog_thread,
        }
        if sync_mode is not None:
            nixl_conf_kwargs["sync_mode"] = sync_mode
        nixl_conf = NixlAgentConfig(**nixl_conf_kwargs)
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)
        self.nixl_agent.create_backend(backend, backend_params)

        device_id = torch_dev.current_device()
        self.init_mem_handlers(device, buffer_ptr, buffer_size, page_size, device_id)

    def init_mem_handlers(self, device, buffer_ptr, buffer_size, page_size, device_id):
        # Break the registration into page size chunks to ensure the maximum buffer size
        # of the underlying plugin is not exceeded
        reg_list = [
            (base_addr, page_size, device_id, "")
            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size)
        ]
        xfer_desc = [
            (base_addr, page_size, device_id)
            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size)
        ]

        if device == "cpu":
            mem_type = "DRAM"
        else:
            mem_type = "VRAM"

        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type=mem_type)
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            "", xfer_descs, mem_type=mem_type
        )

        self.mem_reg_descs = reg_descs
        self.mem_xfer_handler = xfer_handler

    def get_mem_to_storage_handle(
        self, mem_indices, storage_xfer_handler, storage_indices
    ) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.mem_xfer_handler,
            mem_indices,
            storage_xfer_handler,
            storage_indices,
        )

    def get_storage_to_mem_handle(
        self, mem_indices, storage_xfer_handler, storage_indices
    ) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "READ",
            self.mem_xfer_handler,
            mem_indices,
            storage_xfer_handler,
            storage_indices,
        )

    def post_blocking(self, handle: NixlXferHandle):
        state = self.nixl_agent.transfer(handle)

        # time.sleep here is acceptable for now:
        # - Dynamic backend writes: async_mode=True uses post_async + asyncio.sleep
        #   instead, so post_blocking is only reached with async_mode=False (sync puts).
        # - Dynamic backend reads: batched_get_non_blocking calls storage_to_mem
        #   synchronously, but async reads are broken regardless.
        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise
            if state != "DONE" and state != "ERR":
                time.sleep(0.001)

        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    def release_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle)

    @abstractmethod
    def close(self):
        pass


class NixlStaticStorageAgent(NixlStorageAgent):
    pool: NixlDescPool
    storage_reg_descs: nixlBind.nixlRegDList
    storage_xfer_descs: nixlBind.nixlXferDList
    storage_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        pool: NixlDescPool,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        enable_prog_thread: bool,
        sync_mode: Optional[Any] = None,
    ):
        super().__init__(
            allocator, device, backend, backend_params, enable_prog_thread, sync_mode
        )

        page_size = allocator.align_bytes

        if isinstance(pool, NixlFilePool):
            self.init_storage_handlers_file(page_size, pool.fds)
        elif isinstance(pool, NixlObjectPool):
            self.init_storage_handlers_object(page_size, pool.keys)
        else:
            raise TypeError(f"Unsupported pool type: {type(pool).__name__}")

    def init_storage_handlers_file(self, page_size, fds):
        reg_list = []
        xfer_desc = []
        for fd in fds:
            reg_list.append((0, page_size, fd, ""))
            xfer_desc.append((0, page_size, fd))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="FILE")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="FILE")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="FILE"
        )

        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def init_storage_handlers_object(self, page_size, keys):
        reg_list = []
        xfer_desc = []
        for i, key in enumerate(keys):
            reg_list.append((0, page_size, i, key))
            xfer_desc.append((0, page_size, i))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="OBJ"
        )

        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def get_mem_to_storage_handle(self, mem_indices, storage_indices) -> NixlXferHandle:  # type: ignore[override]
        return super().get_mem_to_storage_handle(
            mem_indices, self.storage_xfer_handler, storage_indices
        )

    def get_storage_to_mem_handle(self, mem_indices, storage_indices) -> NixlXferHandle:  # type: ignore[override]
        return super().get_storage_to_mem_handle(
            mem_indices, self.storage_xfer_handler, storage_indices
        )

    def close(self):
        self.nixl_agent.release_dlist_handle(self.storage_xfer_handler)
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.storage_reg_descs)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


class NixlDynamicStorageAgent(NixlStorageAgent):
    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        enable_prog_thread: bool,
        sync_mode: Optional[Any] = None,
    ):
        super().__init__(
            allocator, device, backend, backend_params, enable_prog_thread, sync_mode
        )

        if backend in ("OBJ", "AZURE_BLOB", "DOCA_MEMOS"):
            self.mem_type = "OBJ"
        else:
            self.mem_type = "FILE"

    def create_batched_storage_handler(self, descs: list[NixlDesc], page_size: int):
        reg_list = []
        xfer_desc = []

        for i in range(len(descs)):
            reg_list.append((0, page_size, descs[i].device_id, descs[i].meta_info))
            xfer_desc.append((0, page_size, descs[i].device_id))

        reg_descs = self.nixl_agent.register_memory(reg_list, self.mem_type)
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, self.mem_type)
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type=self.mem_type
        )
        return reg_descs, xfer_handler

    def post_async(self, handle: NixlXferHandle):
        """Non-blocking async post for WRITE operations."""
        state = self.nixl_agent.transfer(handle)
        return state

    def release_storage_handler(
        self,
        reg_descs: nixlBind.nixlRegDList,
        xfer_handler: NixlDlistHandle,
        descs: List[NixlDesc],
    ) -> None:
        """
        Release storage handler resources.

        :param reg_descs: Memory descriptors to deregister.
        :param xfer_handler: Transfer dlist handle to release.
        :param descs: Descriptors used for this transfer.
        """
        self.nixl_agent.release_dlist_handle(xfer_handler)
        self.nixl_agent.deregister_memory(reg_descs)
        if self.mem_type == "FILE":
            _close_file_descs(descs)

    def nixl_desc_exists(self, meta_info: str, path: str) -> bool:
        """
        Check whether a NIXL descriptor exists in storage.

        :param meta_info: Descriptor key (file basename for FILE backends,
            object key for OBJ backends).
        :param path: Directory for FILE backends; ignored for OBJ.
        :return: ``True`` if present, ``False`` otherwise.
        """
        if self.mem_type == "FILE":
            return os.path.exists(os.path.join(path, meta_info))

        reg_list = [(0, 0, 0, meta_info)]
        try:
            resp = self.nixl_agent.query_memory(
                reg_list, self.backend, mem_type=self.mem_type
            )
            # nixl api query_memory returns a list of nixlRegDesc
            if resp[0] is None:
                return False
            return True
        except Exception as exc:
            logger.warning(f"NIXL Desc {meta_info} query failed: {exc}")
            return False

    def batched_nixl_desc_exists(
        self, reg_list: List[tuple[int, int, int, str]], path: Optional[str] = None
    ) -> int:
        """Check if multiple descriptors exist from the start of ``reg_list``.

        :param reg_list: List of tuples ``(0, 0, 0, meta_info)`` where
            *meta_info* is the formatted object-key string.
        :param path: Directory for FILE backends (required for FILE; ignored
            for OBJ). FILE existence is resolved on the filesystem, since NIXL
            ``query_memory`` only answers for object stores.
        :return: Number of consecutive descriptors that exist from the
            start of the list.
        :raises ValueError: If ``path`` is ``None`` for a FILE backend.
        :raises: Errors from the underlying ``query_memory`` call (OBJ
            backends) are caught internally and logged as warnings; the
            method returns ``0`` in that case.
        """
        if not reg_list:
            return 0

        # FILE backends are not resolvable via NIXL ``query_memory`` (it only
        # answers for object stores), so reuse the single-key
        # ``nixl_desc_exists`` -- which already handles FILE via os.path.exists.
        # Without this, FILE/POSIX dynamic backends always return 0 here, so
        # every batched lookup misses and the cache is never reused.
        if self.mem_type == "FILE":
            if path is None:
                raise ValueError("path must be provided for FILE backends")
            consecutive_count = 0
            for _, _, _, meta_info in reg_list:
                if self.nixl_desc_exists(meta_info, path):
                    consecutive_count += 1
                else:
                    break
            return consecutive_count

        try:
            resp = self.nixl_agent.query_memory(
                reg_list, self.backend, mem_type=self.mem_type
            )
            # nixl api query_memory returns a list of nixlRegDesc
            # Count consecutive descriptors that exist (resp[i] is not None)
            consecutive_count = 0
            for reg_desc in resp:
                if reg_desc is not None:
                    consecutive_count += 1
                else:
                    break

            return consecutive_count
        except Exception as exc:
            logger.warning(f"NIXL batched query failed: {exc}")
            return 0

    def close(self):
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


class NixlStorageBackend(AllocatorBackendInterface, ABC):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the
    implementation.
    """

    def __init__(
        self,
        nixl_config: NixlStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
    ):
        """
        Initialize the Nixl storage backend.

        :param nixl_config: The Nixl storage configuration.
        :param config: The LMCache engine configuration.
        :param metadata: The LMCache metadata.
        :param loop: The asyncio event loop.
        :param local_cpu_backend: The LocalCPUBackend whose MixedMemoryAllocator
            (with use_paging=True) will be shared with this NIXL backend in CPU
            mode.  Must be provided (and non-None) when nixl_config.buffer_device
            is ``"cpu"``.  Ignored in GPU mode.
        :raises RuntimeError: In CPU mode, if *local_cpu_backend* is None, or if
            its allocator is not a MixedMemoryAllocator wrapping a
            PagedTensorMemoryAllocator.
        """
        super().__init__(dst_device=nixl_config.buffer_device)

        self.loop = loop
        self.key_lock = threading.RLock()

        self.progress_lock = threading.RLock()
        self.progress_set: Set[CacheEngineKey] = set()

        self.nixl_config = nixl_config
        self._local_cpu_backend: Optional["LocalCPUBackend"] = None

        if nixl_config.buffer_device != "cpu":
            # GPU mode: allocate own staging buffer now
            self.memory_allocator: PagedTensorMemoryAllocator = (
                self.initialize_allocator(config, metadata)
            )
        else:
            # CPU mode: share the LocalCPUBackend's PagedTensorMemoryAllocator
            if local_cpu_backend is None:
                raise RuntimeError(
                    "nixl_buffer_device=cpu requires a LocalCPUBackend staging buffer "
                    "(set max_local_cpu_size > 0)"
                )
            allocator = local_cpu_backend.get_memory_allocator()
            if not isinstance(allocator, MixedMemoryAllocator) or not isinstance(
                allocator.pin_allocator, PagedTensorMemoryAllocator
            ):
                raise RuntimeError(
                    "LocalCPUBackend must use MixedMemoryAllocator(use_paging=True) "
                    "when NIXL CPU mode is enabled. Ensure enable_nixl_storage + "
                    "nixl_buffer_device=cpu triggered the paged allocator path in "
                    "LocalCPUBackend.initialize_allocator()."
                )
            self.memory_allocator = allocator.pin_allocator
            self._local_cpu_backend = local_cpu_backend
            self.free_pinned_buffer = False
            logger.debug(
                "nixl_buffer_device=cpu: NIXL sharing LocalCPUBackend's pool "
                "(sized by max_local_cpu_size=%.2f GiB)",
                config.max_local_cpu_size,
            )

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> PagedTensorMemoryAllocator:
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )
        assert enable_nixl_storage

        corrected_device = get_correct_device(
            config.nixl_buffer_device,
            metadata.worker_id,
        )

        self.use_hugepages = self.nixl_config.use_hugepages
        self.buffer_size = config.nixl_buffer_size
        if corrected_device == "cpu":
            self.buffer = _allocate_cpu_memory(
                config.nixl_buffer_size, use_hugepages=self.use_hugepages
            )
            self.free_pinned_buffer = True
        else:
            if self.use_hugepages:
                logger.warning("Hugepages are not supported for GPU memory allocation")
                self.use_hugepages = False
            base_buffer, self.buffer = _allocate_gpu_memory(
                config.nixl_buffer_size, corrected_device
            )
            torch_dev.set_device(corrected_device)
            self.base_buffer = base_buffer  # Prevents early GC of the aligned tensor.
            self.free_pinned_buffer = False

        return PagedTensorMemoryAllocator(
            self.buffer,
            [torch.Size(metadata.kv_shape)],
            [metadata.kv_dtype],
            MemoryFormat.KV_2LTD,
        )

    def get_memory_allocator(self):
        if self._local_cpu_backend is not None:
            return self._local_cpu_backend.get_memory_allocator()
        return self.memory_allocator

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if busy_loop:
            logger.warning("NixlStorageBackend does not support busy loop for now")

        if self._local_cpu_backend is not None:
            return self._local_cpu_backend.allocate(
                shapes, dtypes, fmt, eviction=eviction, busy_loop=False
            )
        return self.memory_allocator.allocate(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("NixlStorageBackend does not support busy loop for now")

        if self._local_cpu_backend is not None:
            return self._local_cpu_backend.batched_allocate(
                shapes, dtypes, batch_size, fmt, eviction=eviction, busy_loop=False
            )
        return self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    def get_allocator_backend(self):
        if self._local_cpu_backend is not None:
            return self._local_cpu_backend
        return self

    @abstractmethod
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        pass

    @abstractmethod
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        pass

    @abstractmethod
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        pass

    @abstractmethod
    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        pass

    @abstractmethod
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        pass

    @abstractmethod
    def pin(self, key: CacheEngineKey) -> bool:
        pass

    @abstractmethod
    def unpin(self, key: CacheEngineKey) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @staticmethod
    def CreateNixlStorageBackend(
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        metadata: LMCacheMetadata,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
    ):
        """
        Create a Nixl backend with the given configuration.

        :param config: The LMCache engine configuration.
        :param loop: The asyncio event loop.
        :param metadata: The LMCache metadata.
        :param local_cpu_backend: The LocalCPUBackend whose MixedMemoryAllocator
            (with use_paging=True) will be shared with this NIXL backend in CPU
            mode.  Required when ``config.nixl_buffer_device == "cpu"``.

        :return: A NixlBackend instance.
        """
        # Create the Nixl config
        nixl_config = NixlStorageConfig.from_cache_engine_config(config, metadata)
        # Create the Nixl backend
        if nixl_config.dynamic_storage:
            return NixlDynamicStorageBackend(
                nixl_config, config, metadata, loop, local_cpu_backend
            )
        else:
            return NixlStaticStorageBackend(
                nixl_config, config, metadata, loop, local_cpu_backend
            )


class NixlStaticStorageBackend(NixlStorageBackend):
    def __init__(
        self,
        nixl_config: NixlStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
    ):
        super().__init__(nixl_config, config, metadata, loop, local_cpu_backend)

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.key_dict = self.cache_policy.init_mutable_mapping()

        self.pool = self.createPool(
            nixl_config.backend,
            nixl_config.pool_size,
            nixl_config.path,
            nixl_config.use_direct_io,
            nixl_config.path_sharding,
            f"cuda:{metadata.worker_id}",
        )
        assert self.pool is not None

        # In CPU mode self.memory_allocator was set by the base __init__
        # (from local_cpu_backend); in GPU mode it was allocated there too.
        # Either way it is ready here.
        self.agent: NixlStaticStorageAgent = NixlStaticStorageAgent(
            self.memory_allocator,
            self.pool,
            nixl_config.buffer_device,
            nixl_config.backend,
            nixl_config.backend_params,
            nixl_config.enable_prog_thread,
            nixl_config.sync_mode,
        )

    @staticmethod
    def createPool(
        backend: str,
        size: int,
        path: Union[str, List[str]],
        use_direct_io: bool,
        path_sharding: str,
        dst_device: str,
    ) -> NixlDescPool:
        """Create a NIXL descriptor pool with path sharding support.

        Args:
            backend: Backend type (e.g., "GDS", "POSIX", "OBJ").
            size: Pool size.
            path: Single path string or list of paths for sharding.
            use_direct_io: Whether to use direct I/O.
            path_sharding: Sharding strategy (e.g., "by_gpu").
            dst_device: Device string for path selection.

        Returns:
            NixlDescPool: The created descriptor pool.

        Raises:
            ValueError: If backend is unsupported or path is invalid.

        Note:
            When *path* is provided as a list, entries containing commas will be
            split when joined for PathSharder. Avoid commas in path entries to
            prevent unintended sharding.
        """

        if backend in ("GDS", "GDS_MT", "POSIX", "HF3FS"):
            if isinstance(path, list) and any("," in p for p in path):
                logger.warning(
                    "nixl_path entries contain commas; joining for PathSharder may "
                    "cause unintended sharding. Consider paths without commas or a "
                    "single comma-separated string."
                )
            sharder = PathSharder(
                raw_csv=path if isinstance(path, str) else ",".join(path),
                strategy=path_sharding,
                dst_device=dst_device,
                create_dirs=True,
            )
            return NixlFilePool(size, sharder, use_direct_io)
        elif backend in ("OBJ", "AZURE_BLOB", "DOCA_MEMOS"):
            return NixlObjectPool(size, b128=(backend == "DOCA_MEMOS"))
        else:
            raise ValueError(f"Unsupported NIXL backend: {backend}")

    def add_key_to_dict(
        self, key: CacheEngineKey, obj: MemoryObjMetadata, index: int
    ) -> None:
        with self.key_lock:
            assert key not in self.key_dict
            self.key_dict[key] = NixlKeyMetadata(
                shape=obj.shape,
                dtype=obj.dtype,
                fmt=obj.fmt,
                index=index,
            )
            self.cache_policy.update_on_put(key)

    async def mem_to_storage(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]

        storage_indices = []
        for i in range(len(keys)):
            index = self.pool.pop()
            storage_indices.append(index)
            self.add_key_to_dict(keys[i], mem_objs[i].meta, index)

        handle = self.agent.get_mem_to_storage_handle(mem_indices, storage_indices)
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)

        for key in keys:
            with self.progress_lock:
                self.progress_set.discard(key)

    def _collect_metadata_with_lock(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[NixlKeyMetadata]]:
        """
        Fast metadata collection with lock.
        Returns metadata for each key, None if key doesn't exist.
        """
        metadata_list: list[Optional[NixlKeyMetadata]] = []
        with self.key_lock:
            for key in keys:
                metadata = self.key_dict.get(key)
                if metadata is not None:
                    self.cache_policy.update_on_hit(key, self.key_dict)
                metadata_list.append(metadata)
        return metadata_list

    async def _nixl_transfer_async(
        self, metadata_list: list[Optional[NixlKeyMetadata]]
    ) -> list[Optional[MemoryObj]]:
        """
        Memory allocation and NIXL transfer without locks.
        Can run async and in parallel with other transfers.
        """
        obj_list: list[Optional[MemoryObj]] = []
        mem_indices = []
        storage_indices = []

        # Memory allocation outside lock
        for metadata in metadata_list:
            if metadata is None:
                obj_list.append(None)
                continue

            dtype = metadata.dtype
            shape = metadata.shape
            fmt = metadata.fmt
            assert dtype is not None
            assert shape is not None
            assert fmt is not None

            if self._local_cpu_backend is not None:
                obj = self._local_cpu_backend.allocate(
                    shape, dtype, fmt, eviction=True, busy_loop=False
                )
            else:
                obj = self.memory_allocator.allocate(shape, dtype, fmt)
            if obj is None:
                if self._local_cpu_backend is not None:
                    logger.warning(
                        "Failed to allocate from the NIXL/LocalCPUBackend "
                        "shared pool — all pages are pinned. Consider "
                        "increasing `max_local_cpu_size`."
                    )
                else:
                    logger.warning(
                        "Failed to allocate memory, consider increasing the "
                        "`nixl_buffer_size` value"
                    )
                break

            obj_list.append(obj)
            mem_indices.append(obj.meta.address)
            storage_indices.append(metadata.index)

        if not mem_indices:
            return obj_list

        handle = self.agent.get_storage_to_mem_handle(mem_indices, storage_indices)
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)

        return obj_list

    async def storage_to_mem(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        """
        Combined method: collect metadata with lock, then do NIXL transfer.
        """
        metadata_list = self._collect_metadata_with_lock(keys)
        return await self._nixl_transfer_async(metadata_list)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend.

        :return: True if the key exists, False otherwise
        """
        if self.exists_in_put_tasks(key):
            return False

        with self.key_lock:
            if key in self.key_dict:
                if pin:
                    self.key_dict[key].pin()
                return True
            else:
                return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.

        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        with self.progress_lock:
            return key in self.progress_set

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        :param on_complete_callback: Optional callback (not yet supported for
            NixlCacheBackend async operations).
        """
        # contains() reports in-flight puts as absent, so the store path can
        # re-submit a key whose put is still running.
        with self.key_lock, self.progress_lock:
            if not self.progress_set.isdisjoint(keys):
                kept_keys = []
                kept_objs = []
                for key, obj in zip(keys, memory_objs, strict=False):
                    if key not in self.progress_set:
                        kept_keys.append(key)
                        kept_objs.append(obj)
                if not kept_keys:
                    return
                keys, memory_objs = kept_keys, kept_objs

            available_descs = self.pool.get_num_available_descs()
            num_evict = len(keys) - available_descs
            if num_evict > 0:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.key_dict, num_candidates=num_evict
                )

                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Backend under pressure."
                    )
                    return None

                self.batched_remove(evict_keys, force=False)

            self.progress_set.update(keys)

        asyncio.run_coroutine_threadsafe(
            self.mem_to_storage(keys, memory_objs), self.loop
        )
        # TODO: Add callback support for async NIXL operations

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """

        future = asyncio.run_coroutine_threadsafe(self.storage_to_mem([key]), self.loop)

        obj_list = future.result()
        return obj_list[0] if obj_list else None

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.

        :return: a list of memory objects.
        """

        if not keys:
            return []

        future = asyncio.run_coroutine_threadsafe(self.storage_to_mem(keys), self.loop)

        obj_list = future.result()
        return obj_list

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        obj_list = await self.storage_to_mem(keys)
        for i, obj in enumerate(obj_list):
            if obj is None:
                for tail_obj in obj_list[i + 1 :]:
                    if tail_obj is not None:
                        tail_obj.ref_count_down()
                return cast(list[MemoryObj], obj_list[:i])
        return cast(list[MemoryObj], obj_list)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """

        with self.key_lock:
            metadata = self.key_dict.pop(key, None)
            if metadata is None:
                return False
            if force:
                self.cache_policy.update_on_force_evict(key)

        self.pool.push(metadata.index)
        return True

    def pin(self, key: CacheEngineKey) -> bool:
        with self.key_lock:
            if key in self.key_dict:
                self.key_dict[key].pin()
                return True
            else:
                return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.key_lock:
            if key in self.key_dict:
                self.key_dict[key].unpin()
                return True
            else:
                return False

    def close(self) -> None:
        """
        Close the storage backend.
        """
        if self.agent is not None:
            self.agent.close()
        self.pool.close()
        # In CPU mode the allocator is owned by LocalCPUBackend; do not close it here.
        if self._local_cpu_backend is None and self.memory_allocator is not None:
            self.memory_allocator.close()

        if self.free_pinned_buffer:
            _free_cpu_memory(
                self.buffer, self.buffer_size, use_hugepages=self.use_hugepages
            )


class NixlDynamicStorageBackend(NixlStorageBackend):
    def __init__(
        self,
        nixl_config: NixlStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
        cache_policy: Optional[PresenceCache] = None,
    ) -> None:
        super().__init__(nixl_config, config, metadata, loop, local_cpu_backend)

        self.async_mode = nixl_config.enable_async_put
        self.enable_presence_cache = nixl_config.enable_presence_cache
        self.presence_cache_only = nixl_config.presence_cache_only
        # The dynamic backend uses ``self.path`` directly as a single directory
        # (see ``_build_descs``/``key_exists``). Path sharding across multiple
        # paths is only supported for static pools via ``PathSharder``, so reject
        # a list here rather than silently mishandling it later.
        if isinstance(nixl_config.path, list):
            raise ValueError(
                "NixlDynamicStorageBackend (nixl_pool_size=0) does not support "
                "multiple nixl_path entries; provide a single path string. "
                "Path sharding across multiple paths is only available for "
                "static pools."
            )
        self.path: str = nixl_config.path
        self.direct_io_flag = 0
        if nixl_config.use_direct_io:
            if hasattr(os, "O_DIRECT"):
                self.direct_io_flag = os.O_DIRECT
            else:
                logger.warning(
                    "use_direct_io is True, but O_DIRECT is not available on "
                    "this system. Falling back to buffered I/O."
                )
        # DOCA_MEMOS needs object names that fit into 128 bits; other OBJ
        # backends use URL-safe names. See _format_object_key.
        self._use_b128_object_keys = nixl_config.backend == "DOCA_MEMOS"
        # Presence cache to reduce query_memory contains checks
        self.hit_counter = 0
        self.total_counter = 0
        self.key_presence_cache: Optional[PresenceCache] = None
        if self.enable_presence_cache:
            self.key_presence_cache = (
                cache_policy if cache_policy is not None else SetPresenceCache()
            )

        # Initialize metadata from config
        self.meta_shape: Optional[torch.Size] = None
        self.meta_dtype: Optional[torch.dtype] = None
        self.meta_fmt: Optional[MemoryFormat] = None
        self.init_chunk_meta(metadata)

        # Monotonically increasing counter for OBJ device_id values.
        # Each register/deregister cycle must use globally unique IDs to
        # avoid a race where an async PUT deregister erases a concurrent
        # GET registration in NIXL's devIdToObjKey_ map.
        self._device_id_counter = 0
        self._device_id_lock = threading.Lock()

        # In CPU mode self.memory_allocator was set by the base __init__
        # (from local_cpu_backend); in GPU mode it was allocated there too.
        # Either way it is ready here.
        self.agent: NixlDynamicStorageAgent = NixlDynamicStorageAgent(
            self.memory_allocator,
            nixl_config.buffer_device,
            nixl_config.backend,
            nixl_config.backend_params,
            nixl_config.enable_prog_thread,
            nixl_config.sync_mode,
        )

    def set_presence_cache(self, cache: PresenceCache) -> None:
        """Configure a custom cache policy for key presence tracking."""
        if self.enable_presence_cache:
            self.key_presence_cache = cache

    def _alloc_device_ids(self, count: int) -> list[int]:
        """Allocate ``count`` globally unique OBJ device_id values.

        The NIXL OBJ backend indexes registrations by device_id in a flat
        unordered_map with no reference counting.  If two concurrent
        operations (e.g. async PUT cleanup + sync GET) use the same
        device_id sequence (0, 1, 2, ...), the PUT deregister can erase
        the GET entry and cause ``prepXfer``/``postXfer`` to fail with
        NIXL_ERR_INVALID_PARAM.  Using a monotonic counter ensures every
        register/deregister cycle gets its own ID range.
        """
        with self._device_id_lock:
            start = self._device_id_counter
            self._device_id_counter += count
        return list(range(start, start + count))

    def _cache_contains(self, chunk_hash: int) -> bool:
        if not self.enable_presence_cache or self.key_presence_cache is None:
            return False
        found = self.key_presence_cache.contains(chunk_hash)
        self.hit_counter += 1 if found else 0
        self.total_counter += 1
        if self.total_counter % 100 == 0:
            logger.debug(f"Cache hit: {self.hit_counter} vs {self.total_counter}")
        return found

    def _cache_add(self, chunk_hash: int) -> None:
        if not self.enable_presence_cache or self.key_presence_cache is None:
            return
        self.key_presence_cache.add(chunk_hash)

    def _cache_discard(self, chunk_hash: int) -> None:
        if not self.enable_presence_cache or self.key_presence_cache is None:
            return
        self.key_presence_cache.discard(chunk_hash)

    def init_chunk_meta(
        self,
        metadata: Optional[LMCacheMetadata],
    ) -> None:
        """Initialize chunk metadata similar to base_connector.init_chunk_meta()"""
        if metadata is None:
            return

        self.meta_shape = torch.Size(
            [
                metadata.kv_shape[1],
                metadata.kv_shape[0],
                metadata.kv_shape[2],
                metadata.kv_shape[3] * metadata.kv_shape[4],
            ]
        )
        self.meta_dtype = metadata.kv_dtype
        self.meta_fmt = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )
        logger.info(
            f"Initialized nixl object backend metadata: "
            f"shape: {self.meta_shape}, "
            f"dtype: {self.meta_dtype}, "
            f"fmt: {self.meta_fmt}"
        )

    def _format_object_key(self, key: CacheEngineKey) -> str:
        """Format the NIXL object name for ``key`` per the configured backend."""
        if self._use_b128_object_keys:
            return self._format_object_key_b128(key)
        return self._format_object_key_url_safe(key)

    def _format_object_key_b128(self, key: CacheEngineKey) -> str:
        """
        Generate a 128-bit object key for the DOCA_MEMOS backend.

        Returns a 32-char hex string (sha256 truncated to 128 bits). Hex encoding
        is required because the key is passed via NIXL metadata as a string; the
        NIXL plugin hex-decodes it on the other side.
        """
        return hashlib.sha256(key.to_string().encode("utf-8")).hexdigest()[:32]

    def _format_object_key_url_safe(self, key: CacheEngineKey) -> str:
        """
        Generate a URL-safe object key for non-DOCA_MEMOS backends.

        Replaces slashes and @ signs with underscores, then percent-encodes
        the result for safe use as an object storage key.
        """
        key_str = key.to_string()
        # Replace slashes with underscores to make it safe for object storage/FS
        flat_key_str = key_str.replace("/", "_").replace("@", "_")
        # URL encode for safety
        return url_quote(flat_key_str, safe="")

    def _build_descs(
        self, keys: Sequence[CacheEngineKey], *, write: bool
    ) -> List[NixlDesc]:
        """
        Build NixlDescs for ``keys``. For FILE backends this opens one fd per
        key; the caller owns FD lifetime once this method returns successfully.
        On mid-loop failure, every already-opened fd is closed before the exception
        is re-raised (for write paths, files are also unlinked).

        :param write: If True, opens with O_CREAT | O_RDWR (mem_to_storage).
            If False, opens with O_RDONLY (storage_to_mem).
        """
        if self.agent.mem_type == "OBJ":
            device_ids = self._alloc_device_ids(len(keys))
            return [
                NixlDesc(device_id=device_ids[i], meta_info=self._format_object_key(k))
                for i, k in enumerate(keys)
            ]
        if self.agent.mem_type == "FILE":
            flags = (os.O_CREAT | os.O_RDWR) if write else os.O_RDONLY
            flags |= self.direct_io_flag
            mode_args = (DEFAULT_FILE_CREATE_MODE,) if write else ()
            descs: List[NixlDesc] = []
            try:
                for k in keys:
                    path = os.path.join(self.path, self._format_object_key(k))
                    fd = os.open(path, flags, *mode_args)
                    descs.append(
                        NixlDesc(
                            device_id=fd,
                            meta_info="",
                            path=path if write else None,
                        )
                    )
                return descs
            except OSError:
                _close_file_descs(descs)
                _unlink_file_descs(descs)
                raise
        # Already validated in validate_nixl_backend
        raise ValueError(f"unexpected mem_type: {self.agent.mem_type}")

    def _acquire_storage_handle(
        self,
        keys: Sequence[CacheEngineKey],
        mem_indices: List[int],
        storage_indices: Sequence[int],
        page_size: int,
        write: bool,
    ) -> Tuple[
        List[NixlDesc],
        nixlBind.nixlRegDList,
        NixlDlistHandle,
        NixlXferHandle,
    ]:
        """Open FDs, register the storage handler, and build the transfer handle.

        On any failure, releases everything already acquired (FDs, NIXL
        state, and any FILE-write files created with ``O_CREAT``) before
        re-raising, so the caller either gets a fully-acquired tuple or
        an exception with nothing leaked.
        """
        descs = self._build_descs(keys, write=write)
        try:
            reg_descs, xfer_handler = self.agent.create_batched_storage_handler(
                descs, page_size
            )
        except Exception:
            if self.agent.mem_type == "FILE":
                _close_file_descs(descs)
                _unlink_file_descs(descs)
            raise

        try:
            if write:
                handle = self.agent.get_mem_to_storage_handle(
                    mem_indices, xfer_handler, storage_indices
                )
            else:
                handle = self.agent.get_storage_to_mem_handle(
                    mem_indices, xfer_handler, storage_indices
                )
        except Exception:
            # release_storage_handler closes the FDs and releases the dlist.
            self.agent.release_storage_handler(reg_descs, xfer_handler, descs)
            if self.agent.mem_type == "FILE":
                _unlink_file_descs(descs)
            raise

        return descs, reg_descs, xfer_handler, handle

    def key_exists(self, key: CacheEngineKey) -> bool:
        meta_info = self._format_object_key(key)

        return self.agent.nixl_desc_exists(meta_info, self.path)

    def _allocate_for_read(
        self, keys: Sequence[CacheEngineKey]
    ) -> Tuple[Optional[List[MemoryObj]], List[int], List[int]]:
        """Allocate one MemoryObj per key.

        Returns ``(None, [], [])`` if any allocation fails, after freeing
        what was already taken; otherwise returns the allocated objects
        and the parallel mem/storage index lists.
        """
        assert self.meta_shape is not None
        assert self.meta_dtype is not None
        assert self.meta_fmt is not None
        obj_list: List[MemoryObj] = []
        mem_indices: List[int] = []
        storage_indices: List[int] = []
        for idx in range(len(keys)):
            if self._local_cpu_backend is not None:
                obj = self._local_cpu_backend.allocate(
                    self.meta_shape,
                    self.meta_dtype,
                    self.meta_fmt,
                    eviction=True,
                    busy_loop=False,
                )
            else:
                obj = self.memory_allocator.allocate(
                    self.meta_shape, self.meta_dtype, self.meta_fmt
                )
            if obj is None:
                if self._local_cpu_backend is not None:
                    logger.warning(
                        "Failed to allocate from the NIXL/LocalCPUBackend "
                        "shared pool — all pages are pinned. Consider "
                        "increasing `max_local_cpu_size`."
                    )
                else:
                    logger.warning(
                        "Failed to allocate memory, consider increasing the "
                        "`nixl_buffer_size` value"
                    )
                for obj in obj_list:
                    if obj is not None:
                        obj.ref_count_down()
                return None, [], []
            obj_list.append(obj)
            mem_indices.append(obj.meta.address)
            storage_indices.append(idx)
        return obj_list, mem_indices, storage_indices

    def storage_to_mem(
        self, keys: list[CacheEngineKey], pin: bool = False
    ) -> list[Optional[MemoryObj]]:
        page_size = self.memory_allocator.align_bytes
        start_time = time.time()

        obj_list, mem_indices, storage_indices = self._allocate_for_read(keys)
        if obj_list is None:
            return [None] * len(keys)

        try:
            descs, reg_descs, xfer_handler, handle = self._acquire_storage_handle(
                keys, mem_indices, storage_indices, page_size, write=False
            )
        except FileNotFoundError:
            # FILE backend: at least one key's file does not exist,
            # treat the whole batch as a miss.
            logger.warning("storage_to_mem: missing file in FILE backend")
            for obj in obj_list:
                if obj is not None:
                    obj.ref_count_down()
            for key in keys:
                self._cache_discard(key.chunk_hash)
            return [None] * len(keys)

        try:
            try:
                self.agent.post_blocking(handle)
                xfer_state = True
            except nixlBind.nixlBackendError as exc:
                logger.warning(f"Batch Transfer failed: {exc}")
                # Treat transfer failures (not found, timeout, etc.) as a
                # miss rather than raising and terminating the program.
                xfer_state = False
            finally:
                self.agent.release_handle(handle)
                self.agent.release_storage_handler(reg_descs, xfer_handler, descs)
        except Exception:
            # Acquisition or transfer raised; return the allocated MemoryObj
            # slots to the allocator so they aren't leaked.
            for obj in obj_list:
                if obj is not None:
                    obj.ref_count_down()
            raise

        if not xfer_state:
            for obj in obj_list:
                if obj is not None:
                    obj.ref_count_down()
            for key in keys:
                self._cache_discard(key.chunk_hash)
            return [None] * len(keys)

        for key in keys:
            self._cache_add(key.chunk_hash)
        duration = time.time() - start_time
        logger.debug(
            f"storage_to_mem for {len(keys)} objects size "
            f"{page_size * len(keys)} took {duration:.6f} seconds"
        )
        return cast(list[Optional[MemoryObj]], obj_list)

    async def _wait_for_transfer(
        self,
        handle: NixlXferHandle,
        initial_state: str,
        keys: Sequence[CacheEngineKey],
        storage_reg_descs: nixlBind.nixlRegDList,
        storage_xfer_handler: NixlDlistHandle,
        descs: List[NixlDesc],
        mem_objs: List[MemoryObj],
    ):
        """Asynchronously wait for transfer to complete without blocking.

        On any non-DONE outcome (``ERR`` or earlier exception), unlink any
        FILE-write files just created at the final key path so that
        ``contains()`` does not observe them as bogus hits.
        """
        state = ""
        try:
            state = initial_state
            while state != "DONE" and state != "ERR":
                state = self.agent.nixl_agent.check_xfer_state(handle)
                await asyncio.sleep(0.001)  # Avoid busy-waiting, yield to event loop
            if state == "ERR":
                raise RuntimeError("NIXL transfer failed")

        finally:
            # Release the handle after transfer completes (success or failure)
            self.agent.release_handle(handle)
            self.agent.release_storage_handler(
                storage_reg_descs, storage_xfer_handler, descs
            )

            if state == "DONE":
                for key in keys:
                    with self.progress_lock:
                        self.progress_set.discard(key)
                    self._cache_add(key.chunk_hash)
            elif self.agent.mem_type == "FILE":
                _unlink_file_descs(descs)

            for mem_obj in mem_objs:
                mem_obj.ref_count_down()

    async def mem_to_storage(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        if not keys:
            return

        page_size = self.memory_allocator.align_bytes
        storage_indices = range(len(keys))
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]

        descs, reg_descs, xfer_handler, handle = self._acquire_storage_handle(
            keys, mem_indices, storage_indices, page_size, write=True
        )

        if self.async_mode:
            self._submit_async_mem_to_storage(
                handle, keys, reg_descs, xfer_handler, descs, mem_objs
            )
        else:
            self._run_sync_mem_to_storage(
                handle, keys, reg_descs, xfer_handler, descs, page_size
            )

    def _submit_async_mem_to_storage(
        self,
        handle: NixlXferHandle,
        keys: Sequence[CacheEngineKey],
        reg_descs: nixlBind.nixlRegDList,
        xfer_handler: NixlDlistHandle,
        descs: List[NixlDesc],
        mem_objs: List[MemoryObj],
    ) -> None:
        """Post the transfer and hand cleanup off to a background task.

        On any failure before the task is scheduled, ownership has not
        transferred, so we release everything ourselves -- including any
        FILE-write files just created at the final key path.
        """
        try:
            initial_state = self.agent.post_async(handle)
            asyncio.create_task(
                self._wait_for_transfer(
                    handle,
                    initial_state,
                    keys,
                    reg_descs,
                    xfer_handler,
                    descs,
                    mem_objs,
                )
            )
        except Exception:
            self.agent.release_handle(handle)
            self.agent.release_storage_handler(reg_descs, xfer_handler, descs)
            if self.agent.mem_type == "FILE":
                _unlink_file_descs(descs)
            raise

    def _run_sync_mem_to_storage(
        self,
        handle: NixlXferHandle,
        keys: Sequence[CacheEngineKey],
        reg_descs: nixlBind.nixlRegDList,
        xfer_handler: NixlDlistHandle,
        descs: List[NixlDesc],
        page_size: int,
    ) -> None:
        start_time = time.time()
        try:
            self.agent.post_blocking(handle)
        except Exception:
            if self.agent.mem_type == "FILE":
                _unlink_file_descs(descs)
            raise
        finally:
            self.agent.release_handle(handle)
            self.agent.release_storage_handler(reg_descs, xfer_handler, descs)

        duration = time.time() - start_time
        logger.debug(
            f"mem_to_storage for {len(keys)} objects size "
            f"{page_size * len(keys)} took {duration:.3f} seconds"
        )
        for key in keys:
            with self.progress_lock:
                self.progress_set.discard(key)
            self._cache_add(key.chunk_hash)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.

        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        with self.progress_lock:
            return key in self.progress_set

    def _exists_in_put_tasks_or_cache(self, key: CacheEngineKey) -> tuple[bool, bool]:
        """Check whether key exists in put tasks or presence cache.

        This method only checks the local data structures and does not
        call the expensive key_exists operation.

        :param key: The key to check
        :return: Tuple of (found, result) where:
                - found: True if we determined the result locally
                - result: True if key exists, False if key doesn't exist
                  (in put tasks)
        """
        # Check if already in progress
        if self.exists_in_put_tasks(key):
            logger.debug(f"Key {key.chunk_hash:x} is in put tasks")
            return True, False

        # Check presence cache before issuing a query_memory call if not prefetching
        if self._cache_contains(key.chunk_hash):
            return True, True

        return False, False

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        Normally this checks local put-task state and the presence cache, then
        falls back to a NIXL ``query_memory`` (queryMem) call for keys not known
        locally; a hit from that call is added to the presence cache.

        When ``presence_cache_only`` is enabled (the ``nixl_presence_cache_only``
        config option), local put-task state and the presence cache are treated
        as authoritative: a key not known locally reports a miss and the queryMem
        call is skipped (DRAM-only metadata semantics).

        :param key: The key to check
        :param pin: Whether to pin the object in the backend
            (Not Implemented)

        :return: True if the key exists, False otherwise
        """
        # Check local data structures first
        found, local_result = self._exists_in_put_tasks_or_cache(key)
        if found:
            return local_result

        if self.presence_cache_only:
            return False

        xfer_state = self.key_exists(key)
        if xfer_state:
            self._cache_add(key.chunk_hash)

        return xfer_state

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check whether the keys are in the storage backend.

        Overrides the sequential base-class implementation to issue a
        single batched ``query_memory`` call for the keys that cannot
        be resolved from local data structures (put-task set and
        presence cache).

        When ``presence_cache_only`` is enabled (the ``nixl_presence_cache_only``
        config option), the batched queryMem call is skipped: the method returns
        the count of leading keys resolved from local data structures and treats
        the first locally-unknown key as a miss (DRAM-only metadata semantics).

        :param List[CacheEngineKey] keys: The keys of the MemoryObj.
        :param bool pin: Whether to pin the key (not implemented).
        :return: Number of contiguous hit chunks from the start of *keys*.
        :raises: No exceptions are raised. Errors from the underlying
            NIXL batched query are caught internally and logged as
            warnings.
        """
        if not keys:
            return 0

        # First, do fast sequential check of local data structures
        true_count = 0
        for key in keys:
            found, result = self._exists_in_put_tasks_or_cache(key)
            if found:
                if result:
                    true_count += 1
                else:
                    # Found in put tasks (False), stop the loop
                    return true_count
            else:
                # Not found locally, break to do expensive checks
                break

        # If we checked all keys locally, return the count
        if true_count == len(keys):
            return true_count

        # DRAM-only metadata semantics: keys not already in the presence cache
        # are treated as misses without issuing a query_memory call.
        if self.presence_cache_only:
            return true_count

        # For remaining keys, use the new batched_nixl_desc_exists method
        remaining_keys = keys[true_count:]
        reg_list = [(0, 0, 0, self._format_object_key(key)) for key in remaining_keys]

        # Use the agent's batched_nixl_desc_exists method
        consecutive_hits = self.agent.batched_nixl_desc_exists(reg_list, self.path)

        # Update cache for the hits and return total count
        for i in range(consecutive_hits):
            self._cache_add(remaining_keys[i].chunk_hash)

        return true_count + consecutive_hits

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not keys:
            return 0
        """
        Nixl API query_memory also supports batched query. However when there
        are hundreds of keys to be queried and keys in the first few places
        don't exist, the batched query has to be failed fast.
        Therefore we implement batched contains() in a managed thread pool,
        which fails fast when a key doesn't exist.
        """
        n = len(keys)
        idx = 0
        batch_size = _CONTAINS_BATCH_SIZE

        while idx < n:
            batch = keys[idx : idx + batch_size]
            tasks = [asyncio.to_thread(self.contains, k, pin) for k in batch]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            # Stop at the first miss
            for j, ok in enumerate(results):
                if not ok:
                    return idx + j
            idx += len(batch)

        return n

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        :param on_complete_callback: Optional callback invoked once per key
            after transfer completes. Only supported in sync mode (async_mode=False).
        """
        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key)

        if self.async_mode:
            for mem_obj in memory_objs:
                mem_obj.ref_count_up()
            asyncio.run_coroutine_threadsafe(
                self.mem_to_storage(keys, memory_objs), self.loop
            )
            # Note: callback not supported in async mode
        else:
            future = asyncio.run_coroutine_threadsafe(
                self.mem_to_storage(keys, memory_objs), self.loop
            )
            future.result()

            # Call completion callback for sync mode
            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            f"on_complete_callback failed for key {key}: {e}"
                        )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """
        obj_list = self.storage_to_mem([key], False)
        return obj_list[0]

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        A blocking function to get the kv cache from the storage backend.
        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :return: a list of memory objects.
        """
        if not keys:
            return []

        obj_list = self.storage_to_mem(keys, False)
        return obj_list

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        Non blocking interface to get the kv cache from the storage backend.
        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :return: a list of memory objects.
        """
        obj_list = self.storage_to_mem(keys, False)
        for i, obj in enumerate(obj_list):
            if obj is None:
                for tail_obj in obj_list[i + 1 :]:
                    if tail_obj is not None:
                        tail_obj.ref_count_down()
                return cast(list[MemoryObj], obj_list[:i])
        return cast(list[MemoryObj], obj_list)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        :param force: Whether to force removal (not used in this implementation)
        :return: True if the key is removed, False otherwise.
        """
        self._cache_discard(key.chunk_hash)
        if self.agent.mem_type == "FILE":
            try:
                os.unlink(os.path.join(self.path, self._format_object_key(key)))
            except FileNotFoundError:
                return False

        return True

    def pin(self, key: CacheEngineKey) -> bool:
        """
        Not implemented yet
        """
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        """
        Not implemented yet
        """
        return False

    def close(self) -> None:
        """
        Close the storage backend.
        """
        if self.agent is not None:
            self.agent.close()
        # In CPU mode the allocator is owned by LocalCPUBackend; do not close it here.
        if self._local_cpu_backend is None and self.memory_allocator is not None:
            self.memory_allocator.close()

        if self.free_pinned_buffer:
            _free_cpu_memory(
                self.buffer, self.buffer_size, use_hugepages=self.use_hugepages
            )
