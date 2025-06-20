# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from multiprocessing import Lock
from typing import Optional

from vllm.distributed.device_communicators.shm_object_storage import (
    SingleWriterShmObjectStorage, SingleWriterShmRingBuffer)
from vllm.envs import (VLLM_MM_INPUT_CACHE_GIB,
                       VLLM_OBJECT_STORAGE_MAX_OBJECT_SIZE_MB,
                       VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
                       VLLM_OBJECT_STORAGE_SHM_BUFFER_SIZE_MB)
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.processing import ProcessingCache
from vllm.utils import is_list_of

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

# Both Client and Server must use the same cache size
# (to perform mirrored caching). This cache size is set by the environment
# variable VLLM_MM_INPUT_CACHE_GIB.


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
                self.world_size = vllm_config.parallel_config.world_size

                ring_buffer = SingleWriterShmRingBuffer(
                    data_buffer_size=VLLM_OBJECT_STORAGE_SHM_BUFFER_SIZE_MB *
                    1024 * 1024,
                    name=VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
                    create=is_writer,
                    is_free_fn=SingleWriterShmObjectStorage.
                    default_is_free_check,
                )
                self.shm_cache = SingleWriterShmObjectStorage(
                    max_object_size=VLLM_OBJECT_STORAGE_MAX_OBJECT_SIZE_MB *
                    1024 * 1024,
                    n_readers=self.world_size,
                    ring_buffer=ring_buffer,
                    reader_lock=Lock(),
                )
                self.mm_cache_type = "shm"
            else:
                self.mm_cache = ProcessingCache.get_lru_cache(
                    VLLM_MM_INPUT_CACHE_GIB, MultiModalKwargs)
                self.mm_cache_type = "lru"

    def get_and_update_p0_shm(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        full_mm_inputs = list[Optional[MultiModalKwargs]]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            address, monotonic_id = self.shm_cache.put(mm_hash, mm_input)
            mm_input = {
                "address": address,
                "monotonic_id": monotonic_id,
            }
            full_mm_inputs.append(mm_input)
        return full_mm_inputs

    def get_and_update_p0_lru(
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

    def get_and_update_p0(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs
        elif self.mm_cache_type == "shm":
            return self.get_and_update_p0_shm(mm_inputs, mm_hashes)
        elif self.mm_cache_type == "lru":
            return self.get_and_update_p0_lru(mm_inputs, mm_hashes)
        else:
            raise ValueError(
                "No cache available. Please check the model configuration "
                "and ensure that the cache is properly initialized.")

    def get_and_update_p1_lru(
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

    def get_and_update_p1(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        if (not self.use_cache) or (self.mm_cache_type == "shm"):
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs
        elif self.mm_cache_type == "lru":
            return self.get_and_update_p1_lru(mm_inputs, mm_hashes)
        else:
            raise ValueError(
                "No cache available. Please check the model configuration "
                "and ensure that the cache is properly initialized.")

    def reset(self) -> bool:
        if hasattr(self, 'mm_cache'):
            self.mm_cache.clear()
        elif hasattr(self, 'shm_cache'):
            self.shm_cache.clear()

        return True
