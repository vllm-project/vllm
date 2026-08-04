# SPDX-License-Identifier: Apache-2.0

"""Encoder Cache (EC) engine.

Mirrors the KV cache engine's layering, but each EC entry is keyed by one
multimodal hash and stores a single tensor of shape ``[num_tokens,
hidden_size]``. EC does not require token chunking, layerwise operations, or
paged gather/scatter.

See ``docs/design/v1/encoder-cache.md`` for the broader design.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
import hashlib

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager

logger = init_logger(__name__)

# Sentinel ``world_size``/``worker_id`` used in EC cache keys. Encoder outputs
# are replicated across tensor-parallel ranks (every rank produces the same
# tensor for a given mm_hash), so EC entries are deduplicated to a single
# logical "rank" on disk. Concurrent puts from multiple ranks land on the same
# key and are idempotent (identical contents).
_EC_KEY_WORLD_SIZE = 1
_EC_KEY_WORKER_ID = 0


def _stable_u64_from_str(s: str) -> int:
    """Hash an arbitrary string into a stable 64-bit unsigned int.

    Used to project arbitrary multimodal-hash strings (which may be opaque,
    not hex) into the ``chunk_hash: int`` field of :class:`CacheEngineKey`.
    """
    digest = hashlib.sha256(str(s).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class ECCacheEngine:
    """LMCache-backed engine for vLLM encoder cache (EC) tensors.

    The engine speaks tensors, not container objects: the vLLM-aware adapter
    is responsible for reading from / writing to vLLM's ``encoder_cache``
    dict. The engine itself only needs ``mm_hash`` and a tensor.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        encoder_dtype: torch.dtype,
    ) -> None:
        """Initialize the EC cache engine.

        Args:
            config: LMCache engine configuration; supplies storage backends.
            metadata: LMCache metadata describing model identity. Only
                ``model_name`` is used in the EC cache key; ``world_size``
                and ``worker_id`` are intentionally ignored (see module
                comment) to keep EC entries shared across TP ranks.
            encoder_dtype: dtype of the encoder output tensors. Used as the
                dtype field of the on-disk cache key, so it must be stable
                across processes that share the same EC cache. Decoupled
                from ``metadata.kv_dtype`` so that changing KV quantization
                does not invalidate EC entries.

        Raises:
            ValueError: if no non-allocator storage backend is configured.
        """
        self.config = config
        self.metadata = metadata
        self._model_name = metadata.model_name
        self._dtype = encoder_dtype

        self._event_manager = EventManager()
        self._storage_manager = StorageManager(
            config=config,
            metadata=metadata,
            event_manager=self._event_manager,
            lmcache_worker=None,
            async_lookup_server=None,
        )

        available_backends = self._storage_manager.get_non_allocator_backends()
        if len(available_backends) == 0:
            raise ValueError(
                "EC cache engine found no storage backends. Configure at least one "
                "backend (e.g. local_disk, remote_url, gds_path, nixl storage plugin)."
            )

        logger.info(
            "Initialized EC cache engine with storage backends=%s",
            available_backends,
        )

    def close(self) -> None:
        """Close EC storage resources and background workers."""
        if hasattr(self, "_storage_manager") and self._storage_manager is not None:
            self._storage_manager.close()

    def _make_cache_key(self, mm_hash: str) -> CacheEngineKey:
        return CacheEngineKey(
            model_name=self._model_name,
            world_size=_EC_KEY_WORLD_SIZE,
            worker_id=_EC_KEY_WORKER_ID,
            chunk_hash=_stable_u64_from_str(mm_hash),
            dtype=self._dtype,
            request_configs={},
        )

    def contains(self, mm_hash: str) -> bool:
        """Return whether encoder cache exists for the given multimodal hash."""
        key = self._make_cache_key(mm_hash)
        return self._storage_manager.contains(key) is not None

    def put(self, mm_hash: str, tensor: torch.Tensor) -> bool:
        """Store one encoder output tensor under ``mm_hash``.

        Args:
            mm_hash: multimodal-input identifier produced by vLLM.
            tensor: encoder output, shape ``[num_tokens, hidden_size]``,
                on any device. The engine copies it into a pinned CPU
                buffer; the caller's tensor is not retained.

        Returns:
            ``True`` if a store task was submitted to the storage manager;
            ``False`` if the underlying allocator could not provide a buffer
            (transient resource pressure). Never returns ``False`` for
            caller misuse — pass a real tensor.
        """
        key = self._make_cache_key(mm_hash)

        # Allocate via LMCache allocator (LocalCPUBackend) through StorageManager.
        # Preserve the source tensor dtype to avoid precision loss.
        mem_obj = self._storage_manager.allocate(
            shapes=tensor.shape,
            dtypes=tensor.dtype,
            fmt=MemoryFormat.EC_TD,
            eviction=True,
            busy_loop=False,
        )
        if mem_obj is None or mem_obj.tensor is None:
            logger.warning("EC allocate failed; skipping put for key %s", key)
            return False

        # Single copy: src -> pinned CPU buffer, handles device transfer + dtype cast.
        mem_obj.tensor.copy_(tensor)

        self._storage_manager.batched_put(
            [key],
            [mem_obj],
        )
        nbytes = tensor.element_size() * tensor.numel()
        logger.info("EC put: stored %d bytes for mm_hash=%s", nbytes, mm_hash)
        return True

    def get(self, mm_hash: str, device: str) -> Optional[torch.Tensor]:
        """Load the encoder output tensor for ``mm_hash`` if present.

        Args:
            mm_hash: multimodal-input identifier produced by vLLM.
            device: torch device string (e.g. ``"cuda"``, ``"cpu"``) onto
                which the returned tensor should reside.

        Returns:
            The encoder tensor on the requested device, or ``None`` on a
            cache miss. The returned tensor never aliases an LMCache-managed
            buffer — callers may keep it indefinitely.
        """
        key = self._make_cache_key(mm_hash)

        mem_objs = self._storage_manager.batched_get([key])
        mem_obj = mem_objs[0]
        if mem_obj is None:
            return None
        if mem_obj.tensor is None:
            mem_obj.ref_count_down()
            return None

        try:
            out = mem_obj.tensor.to(device=device)
            # Ensure the returned tensor doesn't alias the buffer we're about
            # to release: ``.to(same_device)`` is a no-op view, so clone.
            if out.data_ptr() == mem_obj.tensor.data_ptr():
                out = out.clone()
            return out
        finally:
            mem_obj.ref_count_down()
