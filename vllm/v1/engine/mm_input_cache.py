# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from multiprocessing import Lock
from typing import Optional

from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde, SingleWriterShmObjectStorage, SingleWriterShmRingBuffer)
from vllm.envs import (VLLM_MM_INPUT_CACHE_GIB,
                       VLLM_OBJECT_STORAGE_MAX_OBJECT_SIZE_MB,
                       VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
                       VLLM_OBJECT_STORAGE_SHM_BUFFER_SIZE_MB)
from vllm.logger import init_logger
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.processing import ProcessingCache
from vllm.utils import is_list_of

logger = init_logger(__name__)

# The idea of multimodal preprocessing caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# -- Client:
#  - BaseMultiModalProcessor to process MultiModalData into MultiModalKwargs
#    with built-in caching functionality, with mm_hash as its identifier.
#  - MirroredProcessingCache to keep track of the cached entries and
#    determine whether to send the MultiModalKwargs to P1.
#
# -- Server:
#  - MirroredProcessingCache to store the MultiModalKwargs from P0.
#
# The caching for both client and server is mirrored, and this allows us
# to avoid the serialization of "mm_inputs" (like pixel values) between
# client (=P0) and server (=P1) processes if the mm_hash is found in the client
# cache.
#
# LruObjectCache:
# For the LRU cache, the cache is replicated in both processes,
# both Client and Server must use the same cache size.
# This cache size is set by the environment
# variable VLLM_MM_INPUT_CACHE_GIB.
#
# ShmObjectCache:
# For the shared memory cache, the cache is created in the P0 process
# and shared with the P1 process.
# The cache size is set by the environment
# variable VLLM_OBJECT_STORAGE_MAX_OBJECT_SIZE_MB.
#
# Despite the FIFO nature of shared memory ring buffer, it should be more
# efficient than LruObjectCache in most cases, as it avoids memory replication
# and IPC overheads.


class MirroredProcessingCache:

    def __init__(self, vllm_config, is_writer: bool = False):
        """
        Args:
            vllm_config: The VLLM configuration object.
            is_writer: Whether this instance is the writer (P0) or reader (P1),
            only used for shared memory cache.
        """
        mm_config = vllm_config.model_config.multimodal_config
        disable_mm_preprocessor_cache = (
            mm_config is not None and mm_config.disable_mm_preprocessor_cache)
        self.use_cache = not disable_mm_preprocessor_cache

        if self.use_cache:
            if SingleWriterShmObjectStorage.is_enabled(mm_config):
                self.object_cache: BaseObjectCache = ShmObjectCache(
                    vllm_config, is_writer=is_writer)
            else:
                self.object_cache = LruObjectCache(vllm_config)

    def get_and_update_p0(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs
        else:
            return self.object_cache.get_and_update_p0(mm_inputs, mm_hashes)

    def get_and_update_p1(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs
        else:
            return self.object_cache.get_and_update_p1(mm_inputs, mm_hashes)

    def reset(self) -> bool:
        """
        Reset the cache, clearing all entries.
        Returns:
            bool: True if the cache was reset successfully.
        """
        return self.object_cache.reset() if self.use_cache else True


class BaseObjectCache(ABC):
    """Base class for object caches used in multimodal processing."""

    @abstractmethod
    def get_and_update_p0(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        """Get and update the cache for P0 (client)."""
        raise NotImplementedError()

    @abstractmethod
    def get_and_update_p1(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        """Get and update the cache for P1 (server)."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> bool:
        """Reset the cache, clearing all entries."""
        raise NotImplementedError()


class ShmObjectCache(BaseObjectCache):
    """Shared memory object cache for multimodal processing."""

    def __init__(self, vllm_config, is_writer: bool = False):
        self.world_size = vllm_config.parallel_config.world_size

        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=VLLM_OBJECT_STORAGE_SHM_BUFFER_SIZE_MB * 1024 *
            1024,
            name=VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
            create=is_writer,
            is_free_fn=SingleWriterShmObjectStorage.default_is_free_check,
        )
        self.mm_cache = SingleWriterShmObjectStorage(
            max_object_size=VLLM_OBJECT_STORAGE_MAX_OBJECT_SIZE_MB * 1024 *
            1024,
            n_readers=self.world_size,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
            reader_lock=Lock(),
        )

    def get_and_update_p0(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        full_mm_inputs = list[Optional[MultiModalKwargs]]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            try:
                address, monotonic_id = self.mm_cache.put(mm_hash, mm_input)
                mm_input = {
                    "address": address,
                    "monotonic_id": monotonic_id,
                }
            except (ValueError, MemoryError) as e:
                # put may fail if the object is too large or
                # the cache is full.
                # In this case we log the error and keep the original mm_input.
                logger.debug("Failed to cache mm_input with hash %s: %s",
                             mm_hash, e)
            full_mm_inputs.append(mm_input)
        return full_mm_inputs

    def get_and_update_p1(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        return mm_inputs

    def reset(self) -> bool:
        self.mm_cache.clear()
        return True


class LruObjectCache(BaseObjectCache):
    """LRU object cache for multimodal processing."""

    def __init__(self, vllm_config):
        self.mm_cache = ProcessingCache.get_lru_cache(VLLM_MM_INPUT_CACHE_GIB,
                                                      MultiModalKwargs)

    def get_and_update_p0(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        full_mm_inputs = list[Optional[MultiModalKwargs]]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if self.mm_cache.get(mm_hash) is not None:
                mm_input = None
            else:
                self.mm_cache[mm_hash] = mm_input

            full_mm_inputs.append(mm_input)

        return full_mm_inputs

    def get_and_update_p1(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        full_mm_inputs = list[MultiModalKwargs]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if mm_input is None:
                mm_input = self.mm_cache[mm_hash]
            else:
                self.mm_cache[mm_hash] = mm_input

            full_mm_inputs.append(mm_input)

        return full_mm_inputs

    def reset(self) -> bool:
        self.mm_cache.clear()
        return True
