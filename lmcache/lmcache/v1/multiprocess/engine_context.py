# SPDX-License-Identifier: Apache-2.0
"""Shared context and layout descriptor registry for engine modules."""

# Standard
from dataclasses import dataclass
from typing import TypedDict
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.gpu_connector.gds_context import (
    get_gds_context,
    initialize_gds_context,
)
from lmcache.v1.mp_observability.event_bus import EventBus, get_event_bus
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class ShmPoolInfo(TypedDict):
    """Shared-memory pool metadata returned during registration."""

    shm_name: str
    pool_size: int


@dataclass
class _LayoutDescEntry:
    """Stored layout descriptor and its active registration count."""

    layout_desc: MemoryLayoutDesc
    ref_count: int
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    """Cross-chunk attention windows of all object groups, in object-group
    order. Defaults to a single full-attention group."""


class LayoutDescRegistry:
    """Thread-safe registry mapping (model_name, world_size) to MemoryLayoutDesc.

    Modules write to this registry when KV caches are registered.
    Consumers (e.g. LookupModule) read from it to find layout descriptors
    for prefetch tasks. Multiple worker instances can share the same
    ``(model_name, world_size)`` entry, so the registry keeps the descriptor
    until the last matching registration is unregistered.
    """

    def __init__(self) -> None:
        # Key: (model_name, world_size) -> layout descriptor entry
        self._registry: dict[tuple[str, int], _LayoutDescEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model_name: str,
        world_size: int,
        layout_desc: MemoryLayoutDesc,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
    ) -> None:
        """Register a layout descriptor for a (model_name, world_size) pair.

        Re-registering the same pair increments the active registration
        count. The latest descriptor is retained for lookups.

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
            attn_desc: Cross-chunk attention windows of all object groups, in
                object-group order. Defaults to a single full-attention group.
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                self._registry[key] = _LayoutDescEntry(
                    layout_desc=layout_desc,
                    ref_count=1,
                    attn_desc=attn_desc,
                )
                return

            entry.layout_desc = layout_desc
            entry.attn_desc = attn_desc
            entry.ref_count += 1

    def unregister(self, model_name: str, world_size: int) -> None:
        """Unregister one layout descriptor registration for a pair.

        The descriptor is removed only when the last active registration for
        the pair is unregistered.

        Args:
            model_name: The model name.
            world_size: The world size.
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                return

            if entry.ref_count <= 1:
                self._registry.pop(key)
                return

            entry.ref_count -= 1

    def find(self, model_name: str, world_size: int) -> MemoryLayoutDesc | None:
        """Look up a layout descriptor by (model_name, world_size).

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The layout descriptor if found, otherwise None.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.layout_desc

    def find_attn_desc(self, model_name: str, world_size: int) -> AttnWindowDesc:
        """Look up the attention-window descriptor for a pair.

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The :class:`AttnWindowDesc` registered for the pair.

        Raises:
            ValueError: If no descriptor is registered for the pair. Callers
                must register the KV cache (which establishes the pair) before
                looking up its windows.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                raise ValueError(
                    f"No attention-window descriptor registered for model "
                    f"{model_name!r} with world size {world_size}"
                )
            return entry.attn_desc


class MPCacheServerContext:
    """Shared infrastructure for all engine modules.

    Holds the storage manager, token hasher, session manager, event bus,
    and layout descriptor registry. Modules receive this context at init
    and use it for shared operations.

    Args:
        storage_manager_config: Configuration for the storage manager.
        chunk_size: Chunk size for KV cache operations.
        hash_algorithm: Hash algorithm for token hashing.
        separate_object_groups: Whether to split kernel groups into one object
            group per sliding-window size at KV-cache registration. Default True.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ) -> None:
        self._chunk_size = chunk_size
        self._separate_object_groups = separate_object_groups
        self._full_sw_kv = full_sw_kv

        # Initialize the process-global GDS context.
        # No-op when GDS L1 is disabled (config is None).
        initialize_gds_context(storage_manager_config.l1_manager_config.gds_l1_config)

        self.shm_pool_info: ShmPoolInfo = self._compute_shm_pool_info(
            storage_manager_config
        )
        self._storage_manager = StorageManager(storage_manager_config)
        self._token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self._session_manager = SessionManager(self._token_hasher)
        self._event_bus = get_event_bus()
        self._layout_desc_registry = LayoutDescRegistry()

    def close(self) -> None:
        """
        Tear down the session manager, storage manager, and the process-global
        GDS context.
        """
        self._session_manager.close()
        self._storage_manager.close()
        # Tear down the GDS cuFile context (the shared slab + its handle).
        get_gds_context().close()

    @property
    def chunk_size(self) -> int:
        """Chunk size for KV cache operations."""
        return self._chunk_size

    @property
    def separate_object_groups(self) -> bool:
        """Whether to split kernel groups into per-sliding-window object groups."""
        return self._separate_object_groups

    @property
    def full_sw_kv(self) -> bool:
        """Whether sliding-window groups cache full per-chunk KV (no window cutting)."""
        return self._full_sw_kv

    @property
    def storage_manager(self) -> StorageManager:
        """The storage manager instance."""
        return self._storage_manager

    @property
    def token_hasher(self) -> TokenHasher:
        """The token hasher for computing chunk hashes."""
        return self._token_hasher

    @property
    def session_manager(self) -> SessionManager:
        """The session manager for request lifecycle tracking."""
        return self._session_manager

    @property
    def event_bus(self) -> EventBus:
        """The event bus for observability events."""
        return self._event_bus

    @property
    def layout_desc_registry(self) -> LayoutDescRegistry:
        """Registry mapping (model_name, world_size) to MemoryLayoutDesc."""
        return self._layout_desc_registry

    def resolve_obj_keys(
        self, key: IPCCacheServerKey, object_group_ids: list[int]
    ) -> list[list[ObjectKey]]:
        """Resolve per-object-group object keys from an IPC cache key.

        Uses the session manager to track token state and the token hasher
        to compute chunk hashes for the requested range.

        Args:
            key: IPC cache key describing model/session/token range.
            object_group_ids: Object group ids to produce keys for.

        Returns:
            The i-th element is the list of ObjectKeys for
            ``object_group_ids[i]``.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        return ipc_key_to_object_keys(key, chunk_hashes, object_group_ids)

    @staticmethod
    def _compute_shm_pool_info(
        storage_manager_config: StorageManagerConfig,
    ) -> ShmPoolInfo:
        """Compute normalized SHM pool metadata from storage config.

        Returns an empty pool (disabled SHM transport) when ``shm_name`` is
        empty or lazy memory mode is enabled. Otherwise strips any leading ``/``
        and ensures the name starts with ``lmcache_l1_pool_``.
        """
        mem_cfg = storage_manager_config.l1_manager_config.memory_config
        shm_name = mem_cfg.shm_name or ""
        if not shm_name or mem_cfg.use_lazy or mem_cfg.devdax_path:
            return {"shm_name": "", "pool_size": 0}
        bare = shm_name.lstrip("/")
        if not bare.startswith("lmcache_l1_pool_"):
            shm_name = f"lmcache_l1_pool_{bare}"
        return {"shm_name": shm_name, "pool_size": mem_cfg.size_in_bytes}
