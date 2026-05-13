# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Core abstractions for KV cache offloading in vLLM v1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NewType

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.worker.worker import OffloadingHandler

# `OffloadKey` identifies an offloaded block. It combines a block hash with
# its KV cache group index, encoded as raw bytes to avoid tuple GC overhead.
# Use the helper functions below to construct / decompose keys.
OffloadKey = NewType("OffloadKey", bytes)

logger = init_logger(__name__)


def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    """Pack a block hash and group index into an `OffloadKey`."""
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


def get_offload_block_hash(key: OffloadKey) -> bytes:
    """Extract the block hash from an `OffloadKey`."""
    return key[:-4]


def get_offload_group_idx(key: OffloadKey) -> int:
    """Extract the group index from an `OffloadKey`."""
    return int.from_bytes(key[-4:], "big", signed=False)


@dataclass
class ReqContext:
    req_id: str
    kv_transfer_params: dict[str, Any] | None = None


class LoadStoreSpec(ABC):
    """
    Abstract metadata that encapsulates information allowing a worker
    to load, and optionally also to store, blocks of KV data.
    """

    @staticmethod
    @abstractmethod
    def medium() -> str:
        """
        Returns a string representation of the medium type
        this store/load targets.
        """
        pass


@dataclass
class PrepareStoreOutput:
    keys_to_store: list[OffloadKey]
    store_spec: LoadStoreSpec
    evicted_keys: list[OffloadKey]


@dataclass
class OffloadingEvent:
    keys: list[OffloadKey]
    medium: str
    # True if blocks are removed, False if stored
    removed: bool


"""
OffloadingManager class for managing KV data offloading in vLLM v1

This class runs in the scheduler, tracks which blocks are offloaded
and their address.

The class provides the following primitives:
    lookup() - check whether a single block is offloaded and ready.
    prepare_load() - prepare given blocks to be read.
        The given blocks will be protected from eviction.
        This function returns a LoadSpec which encapsulates
        information required for performing the load.
    touch() - marks the give blocks as recently used. Can be used
        to track block's LRU. This function is separated from the
        prepare_load function to allow setting block recency even
        for blocks which do not need reading from the cache, such as
        blocks that are cached by the GPU prefix cache.
    complete_load() - mark blocks which were previously prepared to be
        loaded as done loading. This is to re-allow their eviction.
    prepare_store() - prepare the given blocks to be written.
        Returns a StoreSpec encapsulating offloading information,
        as well as a list of blocks that were evicted as a result.
    complete_store() - marks a previous store as completed.
        Following this call, the given blocks will become loadable.
"""


class OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Checks whether a single block is offloaded and ready to be read.

        Args:
            key: the key identifying the block to lookup.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            True if the block is offloaded and ready, False if not,
            or None if the lookup should be retried later.
            Returning None will delay the request handling by the vLLM
            scheduler.
        """
        pass

    @abstractmethod
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A LoadStoreSpec that can be used by a worker to locate and load
            the actual offloaded KV data.
        """
        pass

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).
        """
        return

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).
        """
        return

    @abstractmethod
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A PrepareStoreOutput indicating which blocks need storing,
            where to store them (LoadStoreSpec), and list of blocks that
            were evicted as a result.
            None is returned if the blocks cannot be stored.
        """
        pass

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If if_success is False, blocks that were not marked as stored will be
        removed.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).
            success: whether the blocks were stored successfully.
        """
        return

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Take the offloading events from the manager.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        return ()

    def shutdown(self) -> None:
        """Shutdown the manager and release any resources."""
        return


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing KV blocks from given block numbers.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.

    If there are multiple KV groups, the blocks are expected to be
    ordered by the group index.
    In that case, group_sizes[i] determines the number of blocks
    per the i-th KV group, and thus sum(group_sizes) == len(block_ids).
    group_sizes=None indicates a single KV group.

    If block_indices is given, each group (determined by group_sizes) of block IDs
    will correspond to logically contiguous blocks, e.g. blocks 5-10 of a some request.
    block_indices[i] will represent the block index of the first block in group #i.
    Thus, len(block_indices) == len(group_sizes) = number of KV cache groups.
    This information is required in order to support off/loading from offloaded blocks
    which are larger than GPU blocks.
    In such cases, the first GPU block per each group may be unaligned to the offloaded
    block size, and so knowing block_indices[i] allows the worker to correctly
    skip part of the first matching offloaded block.
    """

    def __init__(
        self,
        block_ids: list[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int],
    ):
        super().__init__(block_ids)
        assert sum(group_sizes) == len(block_ids)
        assert len(block_indices) == len(group_sizes)
        self.group_sizes: Sequence[int] = group_sizes
        self.block_indices: Sequence[int] = block_indices

    @staticmethod
    def medium() -> str:
        return "GPU"


@dataclass
class CanonicalKVCacheTensor:
    """
    A canonicalized KV cache tensor whose first dimension is num_blocks.

    For attention backends where the raw tensor has num_blocks at a
    non-leading physical dimension (e.g. FlashAttention's
    (2, num_blocks, ...) layout), the tensor is split so that each
    resulting CanonicalKVCacheTensor starts with (num_blocks, ...).
    """

    # The KV cache tensor with shape (num_blocks, ...)
    tensor: torch.Tensor
    # The (possibly padded) page size per block in bytes
    page_size_bytes: int


@dataclass
class CanonicalKVCacheRef:
    """
    Per-layer (or group of layers) reference to a specific (by index)
    CanonicalKVCacheTensor and records the un-padded page size used by that layer.
    """

    # Index into the list of CanonicalKVCacheTensor objects
    tensor_idx: int
    # The un-padded page size per block in bytes
    page_size_bytes: int


@dataclass
class CanonicalKVCaches:
    """
    Canonicalized block-level representation of the KV caches.

    Composed of:
        - Unique list of KV cache data tensors,
          each with shape (num_blocks, page_size_in_bytes) and int8 dtype.
        - Per-group data references of the tensors.
          i.e. how each KV cache group maps to the tensors.
    """

    # Ordered list of unique block tensors, each with shape
    # (num_blocks, ...).
    tensors: list[CanonicalKVCacheTensor]
    # Per-KV-cache-group list of data references that map each layer
    # in the group to the appropriate entry in the tensors list.
    group_data_refs: list[list[CanonicalKVCacheRef]]


class OffloadingSpec(ABC):
    """Spec for an offloading connector"""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        logger.warning(
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.extra_config = kv_transfer_config.kv_connector_extra_config

        parallel_config = vllm_config.parallel_config
        context_parallel_factor = (
            parallel_config.decode_context_parallel_size
            * parallel_config.prefill_context_parallel_size
        )

        # block size used by vLLM for hashing request tokens for the sake
        # of enabling prefix caching
        self.hash_block_size = (
            vllm_config.cache_config.block_size * context_parallel_factor
        )
        # gpu block size per group
        self.gpu_block_size: tuple[int, ...] = tuple(
            kv_cache_group.kv_cache_spec.block_size * context_parallel_factor
            for kv_cache_group in kv_cache_config.kv_cache_groups
        )

        for block_size in self.gpu_block_size:
            assert block_size % self.hash_block_size == 0, (
                f"gpu_block_size={block_size} not divisible by "
                f"hash_block_size={self.hash_block_size}. "
                f"Hybrid models (e.g. Mamba+Attention) need "
                f"--enable-prefix-caching to align block sizes."
            )

        # offloaded_block_size / gpu_block_size
        self.block_size_factor: int = 1

        offloaded_block_size = self.extra_config.get("block_size")
        if offloaded_block_size is not None:
            offloaded_block_size_int = int(offloaded_block_size)
            gpu_block_sizes = set(self.gpu_block_size)
            assert len(gpu_block_sizes) == 1, (
                "If 'block_size' is specified in kv_connector_extra_config, "
                "there must be at least one KV cache group, "
                "and all groups must have the same block size."
            )
            gpu_block_size = gpu_block_sizes.pop()

            assert offloaded_block_size_int % gpu_block_size == 0
            self.block_size_factor = offloaded_block_size_int // gpu_block_size

    @abstractmethod
    def get_manager(self) -> OffloadingManager:
        """
        Get an OffloadingManager that will be used
        by the scheduler-side offloading connector to track
        offloaded blocks and manage evictions.
        """
        pass

    @abstractmethod
    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """
        Get offloading handlers along with their respective src and dst types.

        Args:
            kv_caches: Canonicalized KV caches.

        Yields:
            Tuples of (src_type, dst_type, offloading_handler).
        """
        pass
