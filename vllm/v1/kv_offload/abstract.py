# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NewType

# `OffloadKey` identifies an offloaded block. It combines a block hash with
# its KV cache group index, encoded as raw bytes to avoid tuple GC overhead.
# Use the helper functions below to construct / decompose keys.
OffloadKey = NewType("OffloadKey", bytes)


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
        keys: Sequence[OffloadKey],
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

    def touch(self, keys: Sequence[OffloadKey]):
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            keys: the keys identifying the blocks.
        """
        return

    def complete_load(self, keys: Iterable[OffloadKey]):
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            keys: the keys identifying the blocks.
        """
        return

    @abstractmethod
    def prepare_store(
        self,
        keys: Sequence[OffloadKey],
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

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True):
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If if_success is False, blocks that were not marked as stored will be
        removed.

        Args:
            keys: the keys identifying the blocks.
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
