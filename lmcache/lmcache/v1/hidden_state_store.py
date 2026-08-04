# SPDX-License-Identifier: Apache-2.0
"""HiddenStateStore: per-chunk hidden-state cache on a separate pinned pool.

This module owns hidden-state caching as a logically separate component from
the KV cache. It is constructed by :class:`~lmcache.v1.cache_engine.LMCacheEngine`
when ``config.enable_hidden_state_cache`` is True and exposed as
``engine.hidden_state_store``. See ``docs/design/v1/hidden_state_store.md``.
"""

# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.token_database import TokenDatabase

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.storage_manager import StorageManager

logger = init_logger(__name__)

_GIB = 1024**3

# Maximum LRU-evict-and-retry attempts in _alloc_chunk.  Each retry evicts one
# entry from this store's own pool (never from KV) before re-attempting the
# allocation, so the effective "wait" is exhausting our own LRU queue.
_HS_ALLOC_MAX_RETRIES = 8


class HiddenStateStore:
    """Stand-alone, chunk-aligned hidden-state cache.

    The store keeps one MemoryObj per (chunk-key, layer_idx) pair on
    its own pinned-CPU pool. Chunk keys are produced by the engine's
    TokenDatabase, so HS chunks share the exact
    CacheEngineKey as the corresponding KV chunks.

    Eviction is "lazy coupled": when a retrieve walks chunks in order, the
    store asks the bound StorageManager whether KV is still present
    for each key. If KV is gone, the orphan HS entry is dropped and the
    prefix ends there (KV evict -> HS evict). When the store's own pool is
    full, the store evicts its own LRU entry; it never evicts KV.

    Args:
        config: Engine config. Reads enable_hidden_state_cache,
            max_hidden_state_cpu_size (GB), and hidden_state_layers.
            Retrieval always uses prefix-strict assembly (stop at the first
            chunk missing KV or hidden state for the requested layer).
        token_database: The same TokenDatabase used by the engine,
            so chunk boundaries and keys match KV exactly.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        token_database: TokenDatabase,
    ) -> None:
        self._config = config
        self._token_database = token_database
        self._storage_manager: Optional["StorageManager"] = None

        size_bytes = int(config.max_hidden_state_cpu_size * _GIB)
        if size_bytes <= 0:
            raise ValueError(
                "max_hidden_state_cpu_size must be > 0 when "
                "enable_hidden_state_cache=True"
            )

        self._allocator = MixedMemoryAllocator(size_bytes, config=config)

        # CacheEngineKey -> {layer_idx: MemoryObj}. OrderedDict-like for LRU
        # via a separate _lru ordering (cheap, lets us update on access).
        self._chunks: Dict[CacheEngineKey, Dict[int, MemoryObj]] = {}
        self._lru: "OrderedDict[CacheEngineKey, None]" = OrderedDict()

        allowlist = config.hidden_state_layers
        self._layer_allowlist: Optional[set] = (
            set(allowlist) if allowlist is not None else None
        )

        logger.info(
            "HiddenStateStore initialized: pool=%.2f GB, layer_allowlist=%s",
            config.max_hidden_state_cpu_size,
            self._layer_allowlist,
        )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def bind_storage_manager(self, storage_manager: "StorageManager") -> None:
        """Attach the engine's :class:`StorageManager`.

        Required for coupled eviction. Without it, retrieve falls back to
        HS-only presence checks.
        """
        self._storage_manager = storage_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_hidden_states(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        hidden_states: torch.Tensor,
        *,
        layer_idx: int = 0,
        token_offset: int = 0,
    ) -> int:
        """Store hidden_states chunked under the same keys as KV.

        Args:
            token_ids: 1-D int tensor or list of the **full** token-ID prefix
                (same sequence used for KV storage so chunk boundaries and
                chunk keys align with KV exactly).
            hidden_states: 2-D tensor of shape
                ``[len(token_ids) - token_offset, hidden_dim]``
                (CPU or GPU). Will be moved to CPU/float32 internally.
                Corresponds to ``token_ids[token_offset:]``.
            layer_idx: Storage layer index. Defaults to 0 (main HS).
            token_offset: Number of leading tokens in ``token_ids`` that are
                **not** present in ``hidden_states`` because they were already
                cached in a prior incremental call.  Defaults to 0 (full
                sequence provided). Only chunks whose token range starts at or
                after ``token_offset`` are written; partially-covered chunks
                (``start < token_offset < end``) are skipped to keep each
                chunk atomic.

        Returns:
            Number of chunks stored (0 when filtered by allowlist or on
            allocation failure).

        Raises:
            ValueError: If ``token_offset`` is out of range, or if
                ``hidden_states`` has an unexpected shape.
            RuntimeError: If an allocated MemoryObj has no backing tensor
                (indicates an allocator bug).
        """
        if self._layer_allowlist is not None and layer_idx not in self._layer_allowlist:
            logger.debug(
                "HiddenStateStore: dropping layer_idx=%d (not in allowlist=%s)",
                layer_idx,
                self._layer_allowlist,
            )
            return 0

        if hidden_states.dim() != 2:
            raise ValueError(
                f"hidden_states must be 2-D [num_tokens, hidden_dim], "
                f"got shape {tuple(hidden_states.shape)}"
            )

        n_toks = (
            len(token_ids) if isinstance(token_ids, list) else int(token_ids.shape[0])
        )
        if not (0 <= token_offset <= n_toks):
            raise ValueError(
                f"token_offset ({token_offset}) must be in [0, len(token_ids) "
                f"({n_toks})]"
            )
        expected_rows = n_toks - token_offset
        if hidden_states.shape[0] != expected_rows:
            raise ValueError(
                f"hidden_states first dim ({hidden_states.shape[0]}) must equal "
                f"len(token_ids) - token_offset ({expected_rows})"
            )

        # Detach but keep the tensor on its original device
        hs_src = hidden_states.detach()
        hidden_dim = hs_src.shape[1]

        chunks = self._chunk(token_ids)
        stored = 0
        for start, end, key in chunks:
            # Skip chunks that are entirely before the provided hidden states,
            # or partially covered (keep chunks atomic with KV boundaries).
            if start < token_offset:
                continue

            existing = self._chunks.get(key)
            if existing is not None and layer_idx in existing:
                # Already cached for this layer; bump LRU.
                self._lru.pop(key, None)
                self._lru[key] = None
                continue

            n = end - start
            obj = self._alloc_chunk(n, hidden_dim)
            if obj is None:
                logger.warning(
                    "HiddenStateStore: out of HS pool memory after eviction; "
                    "stopping store at chunk start=%d",
                    start,
                )
                break
            tensor = obj.tensor
            if tensor is None:
                obj.ref_count_down()
                raise RuntimeError(
                    "HiddenStateStore: allocator returned MemoryObj with no "
                    "backing tensor"
                )

            tensor.copy_(hs_src[start - token_offset : end - token_offset])

            layer_map = self._chunks.setdefault(key, {})
            layer_map[layer_idx] = obj
            self._lru.pop(key, None)
            self._lru[key] = None
            stored += 1

        return stored

    def retrieve_hidden_states(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        *,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """Retrieve cached rows for layer_idx as a contiguous prefix.

        Walks chunks of token_ids in order. Stops at the first chunk
        where either KV is no longer present (lazy coupled-eviction cleanup)
        or HS for layer_idx is missing (prefix-strict).

        Returns:
            CPU float32 tensor of shape [num_cached_prefix_tokens, hidden_dim],
            or None if no chunk is cached.

        Raises:
            RuntimeError: If a cached MemoryObj has no backing tensor
                (indicates an allocator bug).
        """
        chunks = self._chunk(token_ids)
        out_rows: List[torch.Tensor] = []
        hit_keys: List[CacheEngineKey] = []
        for _, _, key in chunks:
            if not self._kv_present(key):
                # KV evicted -> drop HS for this key (coupled eviction) and stop.
                if key in self._chunks:
                    self._free_key(key)
                break

            layer_map = self._chunks.get(key)
            if layer_map is None or layer_idx not in layer_map:
                # HS missing for this chunk: prefix_strict stop.
                break

            obj = layer_map[layer_idx]
            tensor = obj.tensor
            if tensor is None:
                raise RuntimeError(
                    "HiddenStateStore: cached MemoryObj has no backing tensor"
                )
            out_rows.append(tensor)
            hit_keys.append(key)

        # evict the keys from the LRU
        for key in reversed(hit_keys):
            self._lru.pop(key, None)
            self._lru[key] = None

        if not out_rows:
            return None
        return torch.cat(out_rows, dim=0)

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    def num_cached_chunks(self) -> int:
        """Return the number of distinct chunk keys currently cached."""
        return len(self._chunks)

    def has_chunk(self, key: CacheEngineKey, layer_idx: int = 0) -> bool:
        """Return True if a chunk is cached for ``(key, layer_idx)``."""
        layer_map = self._chunks.get(key)
        return layer_map is not None and layer_idx in layer_map

    def drop_key(self, key: CacheEngineKey) -> bool:
        """Manually drop all HS layers for ``key``. Test/admin use."""
        if key not in self._chunks:
            return False
        self._free_key(key)
        return True

    def close(self) -> None:
        """Free every cached chunk and the underlying pinned pool."""
        for key in list(self._chunks.keys()):
            self._free_key(key)
        self._allocator.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk(
        self, token_ids: Union[torch.Tensor, List[int]]
    ) -> List["tuple[int, int, CacheEngineKey]"]:
        """Return [(start, end, key)] for token_ids using the engine TDB.

        The list is materialized so we can iterate it twice (key check and
        per-chunk copy) without re-running the hashing path.

        ``process_tokens`` is typed to yield ``CacheEngineKey | int`` (the int
        form is used by other callers); HiddenStateStore always runs it in the
        keyed mode, so the third element is a ``CacheEngineKey`` here.
        """
        return cast(
            "List[tuple[int, int, CacheEngineKey]]",
            list(self._token_database.process_tokens(tokens=token_ids)),
        )

    def _kv_present(self, key: CacheEngineKey) -> bool:
        """True if KV exists in any active backend, or unknown when SM unbound."""
        if self._storage_manager is None:
            # Without an SM we cannot distinguish "KV evicted" from "no KV";
            # treat as present so the prefix walk doesn't truncate spuriously.
            return True
        return self._storage_manager.contains(key) is not None

    def _alloc_chunk(self, n_tokens: int, hidden_dim: int) -> Optional[MemoryObj]:
        shape = torch.Size([n_tokens, hidden_dim])
        dtype = torch.float32
        for _ in range(_HS_ALLOC_MAX_RETRIES):
            obj = self._allocator.allocate(shape, dtype, MemoryFormat.HS_TD)
            if obj is not None:
                return obj
            # Pressure: drop our LRU entry and retry. Never touches KV.
            if not self._evict_one_lru():
                break
        return None

    def _evict_one_lru(self) -> bool:
        if not self._lru:
            return False
        key, _ = self._lru.popitem(last=False)
        self._free_key(key)
        logger.debug("HiddenStateStore: HS-only LRU evicted key=%s", key)
        return True

    def _free_key(self, key: CacheEngineKey) -> None:
        layer_map = self._chunks.pop(key, None)
        self._lru.pop(key, None)
        if not layer_map:
            return
        # ref_count_down is the public release path: it free()s the
        # underlying buffer when the count reaches zero (mirrors how
        # LocalCPUBackend.remove releases entries).
        for obj in layer_map.values():
            obj.ref_count_down()
