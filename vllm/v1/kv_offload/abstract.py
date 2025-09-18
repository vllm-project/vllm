# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OffloadingManager class for managing KV data offloading in vLLM v1

This class runs in the scheduler, tracks which blocks are offloaded
and their address.

The class provides the following primitives:
    lookup() - find the length of the maximal series of blocks,
        starting from the first one, that are all offloaded.
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

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from vllm.v1.core.kv_cache_utils import BlockHash


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
    block_hashes_to_store: list[BlockHash]
    store_spec: LoadStoreSpec
    block_hashes_evicted: list[BlockHash]


@dataclass
class OffloadingEvent:
    block_hashes: list[BlockHash]
    block_size: int
    medium: str
    # True if blocks are removed, False if stored
    removed: bool


class OffloadingManager(ABC):

    @abstractmethod
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        """
        Finds the length of the maximal series of blocks, starting from the
        first one, that are all offloaded.

        Args:
            block_hashes: the hashes identifying the blocks to lookup.

        Returns:
            An integer representing the maximal number of blocks that
            are currently offloaded.
        """
        pass

    @abstractmethod
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A LoadStoreSpec that can be used by a worker to locate and load
            the actual offloaded KV data.
        """
        pass

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        return

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        return

    @abstractmethod
    def prepare_store(
            self,
            block_hashes: Iterable[BlockHash]) -> Optional[PrepareStoreOutput]:
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A PrepareStoreOutput indicating which blocks need storing,
            where to store them (LoadStoreSpec), and list of blocks that
            were evicted as a result.
            None is returned if the blocks cannot be stored.
        """
        pass

    def complete_store(self,
                       block_hashes: Iterable[BlockHash],
                       success: bool = True):
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If if_success is False, blocks that were not marked as stored will be
        removed.

        Args:
            block_hashes: the hashes identifying the blocks.
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
