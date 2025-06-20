# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping
from multiprocessing import Lock
from typing import TYPE_CHECKING, Any

from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde, SingleWriterShmObjectStorage, SingleWriterShmRingBuffer)
from vllm.envs import (VLLM_OBJECT_STORAGE_MAX_OBJECT_SIZE_MB,
                       VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
                       VLLM_OBJECT_STORAGE_SHM_BUFFER_SIZE_MB)
from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.multimodal.cache import MultiModalCache, MultiModalCacheItemMetadata
from vllm.multimodal.inputs import (MultiModalBatchedField,
                                    MultiModalFieldElem, MultiModalKwargsItem,
                                    NestedTensors)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

# The idea of multimodal input caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# For using LRU cache ShmMultiModalInputCache:
# -- P0:
#  - BaseMultiModalProcessor calls MultiModalHasher to get the `mm_hash` of
#    each input multi-modal item (e.g. image),
#  - BaseMultiModalProcessor processes the input items into `mm_kwargs`,
#    which are MultiModalKwargsItem instances that each correspond to an
#    input multi-modal item.
#  - MultiModalInputCacheClient accepts the `mm_kwargs` and corresponding
#    `mm_hash` for each item. It stores the `mm_hash` as keys and the size
#    of `mm_kwargs`, but not the `mm_kwargs` themselves, to avoid taking
#    up additional memory in P0.
#  - The `mm_hash` is always sent to P1.
#  - The corresponding `mm_kwargs` are only sent to P1 if they are not cached
#    in MultiModalInputCacheServer.
#
# -- P1:
#  - If the `mm_hash` is cached (i.e. `mm_kwargs` are not sent from P0),
#    MultiModalInputCacheServer retrieves the corresponding `mm_kwargs`.
#  - If the `mm_hash` is not cached (i.e. `mm_kwargs` are sent from P0),
#    MultiModalInputCacheServer stores `mm_kwargs` under the key `mm_hash`.
#  - Either way, the `mm_hash` and corresponding `mm_kwargs` are sent to
#    the engine for model execution.
#
# Both Client and Server must perform cache update and eviction based on the
# same item size. This ensures that the keys of MultiModalInputCacheClient
# and MultiModalInputCacheServer are mirrored, allowing us to determine in P0
# whether a key is cached in MultiModalInputCacheServer by querying
# MultiModalInputCacheClient without having to communicate with P1.
#
# For using ShmMultiModalInputCache:
# -- P0:
#  - MultiModalInputCacheClient accepts the `mm_kwargs` and corresponding
#    `mm_hash` for each item.
#  - It checks if the `mm_hash` is cached in MultiModalInputCacheServer.
#  - If not cached, store the `mm_kwargs` in shared memory based object
#    storage and replace `mm_kwargs` with its cache address and id.
#  - If cached, replace `mm_kwargs` with its cache address and id.
#
# -- P1:
#  - P1 is a no-op for ShmMultiModalInputCache, as the objects are only
#    read from the shared memory later in the executor.


class MultiModalInputCacheClient:
    """Used by P0 to check whether multi-modal kwargs are cached in P1."""

    def __init__(self, vllm_config: "VllmConfig",
                 mm_registry: MultiModalRegistry) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        self.enabled = mm_registry.enable_mm_input_cache(model_config)
        if self.enabled:
            if mm_registry.enable_mm_input_shm_cache(model_config):
                self.mm_cache: BaseMultiModalInputCache = \
                    ShmMultiModalInputCache(vllm_config, is_writer=True)
            else:
                self.mm_cache = LRUMultiModalInputCache(
                    model_config, mm_registry, MultiModalCacheItemMetadata)

    def get_and_update(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        if not self.enabled:
            return mm_kwargs
        return self.mm_cache.get_and_update_client(mm_kwargs, mm_hashes)

    def reset(self) -> None:
        self.mm_cache.reset()


class MultiModalInputCacheServer:
    """Used by P1 to avoid requiring past multi-modal kwargs from P0."""

    def __init__(self, vllm_config: "VllmConfig",
                 mm_registry: MultiModalRegistry) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        self.enabled = mm_registry.enable_mm_input_cache(model_config)
        if self.enabled:
            if mm_registry.enable_mm_input_shm_cache(model_config):
                self.mm_cache: BaseMultiModalInputCache = \
                    ShmMultiModalInputCache(vllm_config)
            else:
                self.mm_cache = LRUMultiModalInputCache(
                    model_config, mm_registry, Mapping[str, NestedTensors])

    def get_and_update(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        if not self.enabled:
            return mm_kwargs
        return self.mm_cache.get_and_update_server(mm_kwargs, mm_hashes)

    def reset(self) -> None:
        self.mm_cache.reset()


class BaseMultiModalInputCache(ABC):
    """Abstract base class for multimodal input cache implementations."""

    @abstractmethod
    def get_and_update_client(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        """Client-side cache update logic for P0."""
        pass

    @abstractmethod
    def get_and_update_server(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        """Server-side cache update logic for P1."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset/clear the cache."""
        pass


class LRUMultiModalInputCache(BaseMultiModalInputCache):
    """Base class for multimodal input cache operations."""

    def __init__(self, model_config: "ModelConfig",
                 mm_registry: MultiModalRegistry, cache_type: Any) -> None:
        self.mm_cache = MultiModalCache.get_lru_cache(
            model_config.get_mm_input_cache_gb(),
            cache_type,
        )

    def get_and_update_client(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        """Client-side cache update logic for P0."""

        assert len(mm_kwargs) == len(mm_hashes)

        out_mm_items = list[MultiModalKwargsItem]()
        for mm_item, mm_hash in zip(mm_kwargs, mm_hashes):
            if self.mm_cache.get(mm_hash) is not None:
                out_mm_items.append(mm_item.without_data())
            else:
                self.mm_cache[mm_hash] = \
                    MultiModalCacheItemMetadata.wraps(mm_item.require_data())
                out_mm_items.append(mm_item)

        return out_mm_items

    def get_and_update_server(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        """Server-side cache update logic for P1."""

        assert len(mm_kwargs) == len(mm_hashes)

        out_mm_items = list[MultiModalKwargsItem]()
        for mm_item, mm_hash in zip(mm_kwargs, mm_hashes):
            if (mm_data := mm_item.get_data()) is None:
                out_mm_items.append(mm_item.with_data(self.mm_cache[mm_hash]))
            else:
                self.mm_cache[mm_hash] = mm_data
                out_mm_items.append(mm_item)

        return out_mm_items

    def reset(self) -> None:
        self.mm_cache.clear()


class ShmMultiModalInputCache(BaseMultiModalInputCache):
    """Shared memory object cache for multimodal processing."""

    def __init__(self, vllm_config, is_writer: bool = False):
        self.world_size = vllm_config.parallel_config.world_size

        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=VLLM_OBJECT_STORAGE_SHM_BUFFER_SIZE_MB * 1024 *
            1024,
            name=VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
            create=is_writer,
        )
        self.mm_cache = SingleWriterShmObjectStorage(
            max_object_size=VLLM_OBJECT_STORAGE_MAX_OBJECT_SIZE_MB * 1024 *
            1024,
            n_readers=self.world_size,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
            reader_lock=Lock(),
        )

    def get_and_update_client(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        """Client-side cache update logic for P0."""

        assert len(mm_kwargs) == len(mm_hashes)

        out_mm_items = list[MultiModalKwargsItem]()
        for mm_item, mm_hash in zip(mm_kwargs, mm_hashes):
            try:
                address, monotonic_id = self.mm_cache.get_cached(mm_hash)
                # put mm_item in cache if not found
                if address is None:
                    address, monotonic_id = self.mm_cache.put(mm_hash, mm_item)
                address = MultiModalFieldElem(
                    modality=mm_item.modality,
                    key="address",
                    data=address,
                    field=MultiModalBatchedField(),
                )
                monotonic_id = MultiModalFieldElem(
                    modality=mm_item.modality,
                    key="monotonic_id",
                    data=monotonic_id,
                    field=MultiModalBatchedField(),
                )
                mm_item = MultiModalKwargsItem.from_elems(
                    [address, monotonic_id])

            except (ValueError, MemoryError) as e:
                # put may fail if the object is too large or
                # the cache is full.
                # In this case we log the error and keep the original mm_input.
                logger.debug("Failed to cache mm_input with hash %s: %s",
                             mm_hash, e)
            out_mm_items.append(mm_item)
        return out_mm_items

    def get_and_update_server(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        return mm_kwargs

    def reset(self) -> None:
        self.mm_cache.clear()
