# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OffloadingManager class for managing KV data offloading in vLLM v1

This class runs in the scheduler, tracks which blocks are offloaded
and their address.

The class provides the following primitives:
    lookup() - find the length of the maximal series of blocks,
        starting from the first one, that are all offloaded.
    parepare_load() - prepare given blocks to be read.
        This given blocks will be protected from eviction.
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


class OffloadingManager(ABC):

    @abstractmethod
    def lookup(self, block_hashes: list[int]) -> int:
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
    def prepare_load(self, block_hashes: list[int]) -> LoadStoreSpec:
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A LoadSpec that can be used by a worker to
            locate and load the actual offloaded KV data.
        """
        pass

    @abstractmethod
    def touch(self, block_hashes: list[int]):
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        pass

    @abstractmethod
    def complete_load(self, block_hashes: list[int]):
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        pass

    @abstractmethod
    def prepare_store(
            self, block_hashes: list[int]) -> tuple[LoadStoreSpec, list[int]]:
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A tuple where the first is a StoreSpec that can be used by
            a worker to store KV data, and the second is a list of block
            hashes that were evicted by this store operation.
        """
        pass

    @abstractmethod
    def complete_store(self, block_hashes: list[int]):
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        pass
