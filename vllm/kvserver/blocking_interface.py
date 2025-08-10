# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import abc

import torch
from lmcache.integration.vllm.vllm_adapter import (init_lmcache_engine,
                                                   lmcache_get_config)
from lmcache.v1.cache_engine import LMCacheEngine

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockingKVInterface(abc.ABC):

    @abc.abstractmethod
    def register_kv_caches(self, rank: int, gpu_kv_caches: list[torch.Tensor]):
        """
        Register the GPU key-value caches.

        Args:
            gpu_kv_caches (list[torch.Tensor]): List of tensors representing 
                the kvcaches on the GPU.
        """
        pass

    @abc.abstractmethod
    def lookup(self, token_ids: list[int]) -> int:
        """
        Lookup the KV cache

        Args:
            token_ids (list[int]): List of token IDs to look up.

        Returns:
            int: The length of the matched prefix.
        """
        pass

    @abc.abstractmethod
    def offload(self, token_ids: list[int], block_ids: tuple[list[int], ...],
                skip_leading_tokens: int) -> None:
        """
        Offload the specified blocks to CPU.

        Args:
            token_ids (list[int]): List of token IDs corresponding to the 
                blocks.
            block_ids (tuple[list[int], ...]): Tuple of lists of block IDs to 
                offload.
            skip_leading_tokens (int): Number of leading tokens to skip during 
                offload.
        """
        pass

    @abc.abstractmethod
    def onload(self, token_ids: list[int], block_ids: tuple[list[int], ...],
               skip_leading_tokens: int) -> None:
        """
        Onload the specified blocks from CPU to GPU.

        Args:
            token_ids (list[int]): List of token IDs corresponding to the 
                blocks.
            block_ids (tuple[list[int], ...]): Tuple of lists of block IDs to 
                onload.
            skip_leading_tokens (int): Number of leading tokens to skip during 
                onload.
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        Close the KV interface and release resources.
        """
        pass


"""
Prototype implementation of BlockingKVInterface using LMCache
"""


class LMCacheBlockingKVMgr(BlockingKVInterface):

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig,
                 parallel_config: ParallelConfig,
                 scheduler_config: SchedulerConfig):
        self.world_size = parallel_config.world_size
        self.gpu_kv_caches: dict[int, list[torch.Tensor]] = {}
        self.lmcache_engines: dict[int, LMCacheEngine] = {}

        self.vllm_block_size = cache_config.block_size
        self.lmcache_chunk_size = lmcache_get_config().chunk_size

        for rank in range(self.world_size):
            lmcache_engine = init_lmcache_engine(
                model_config,
                parallel_config,
                cache_config,
                scheduler_config,
                engine_name=f"lmcache_vllm_blocking_{rank}",
            )
            self.lmcache_engines[rank] = lmcache_engine

        self.debug_offload_count = 0

    def _get_slot_mapping(self, token_ids: list[int],
                          block_ids: tuple[list[int], ...]) -> torch.Tensor:
        # Flatten block_ids
        block_ids = torch.tensor(block_ids[0], dtype=torch.long)
        num_blocks = block_ids.shape[0]

        # Convert to tensor
        block_size = self.vllm_block_size
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (block_offsets.reshape(
            (1, block_size)) + block_ids.reshape(
                (num_blocks, 1)) * block_size).flatten()

        # TODO: compatibility with multiple cuda devices
        return slot_mapping[:len(token_ids)].cuda()

    def register_kv_caches(self, rank: int, gpu_kv_caches: list[torch.Tensor]):
        if rank in self.gpu_kv_caches:
            raise ValueError(
                f"Rank {rank} has already registered its kv caches.")
        if rank > self.world_size:
            raise ValueError(
                f"Rank {rank} exceeds world size {self.world_size}.")

        self.gpu_kv_caches[rank] = gpu_kv_caches

    def lookup_internal(self, token_ids: list[int], pin: bool) -> int:
        lengths = []
        for i in range(self.world_size):
            length = self.lmcache_engines[0].lookup(token_ids, pin=pin)
            lengths.append(length)

        assert all(length == lengths[0] for length in lengths), \
            f"Mismatch in lookup lengths across ranks: {lengths}"

        return lengths[0]

    def lookup(self, token_ids: list[int]) -> int:
        return self.lookup_internal(token_ids, pin=False)

    def offload(self, token_ids: list[int], block_ids: tuple[list[int], ...],
                skip_leading_tokens: int) -> None:
        if len(block_ids) > 1:
            # Don't do for hybrid kv cache
            return

        # prepare tokens
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # prepare slot mapping
        slot_mapping = self._get_slot_mapping(token_ids, block_ids)

        if len(token_ids) > len(slot_mapping):
            token_ids = token_ids[:len(slot_mapping)]

        # prepare token mask
        token_mask = torch.ones_like(token_ids, dtype=torch.bool)
        skip_leading_tokens = (skip_leading_tokens // self.lmcache_chunk_size *
                               self.lmcache_chunk_size)
        token_mask[:skip_leading_tokens] = False

        for rank in range(self.world_size):
            engine = self.lmcache_engines[rank]
            engine.store(
                token_ids,
                mask=token_mask,
                kvcaches=self.gpu_kv_caches[rank],
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
            )

        self.debug_offload_count += 1
        logger.info("Finished offload #%d, offloaded %d tokens",
                    self.debug_offload_count,
                    len(token_ids) - skip_leading_tokens)

    def onload(self, token_ids: list[int], block_ids: tuple[list[int], ...],
               skip_leading_tokens: int) -> None:
        if len(block_ids) > 1:
            # Don't do for hybrid kv cache
            return

        matched_length = self.lookup_internal(token_ids, pin=False)

        # prepare tokens
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        token_ids = token_ids[:matched_length]

        # prepare slot mapping
        slot_mapping = self._get_slot_mapping(token_ids, block_ids)

        # prepare token mask
        token_mask = torch.ones_like(token_ids, dtype=torch.bool)
        skip_leading_tokens = (skip_leading_tokens // self.lmcache_chunk_size *
                               self.lmcache_chunk_size)
        token_mask[:skip_leading_tokens] = False

        for rank in range(self.world_size):
            engine = self.lmcache_engines[rank]
            engine.retrieve(
                token_ids,
                mask=token_mask,
                kvcaches=self.gpu_kv_caches[rank],
                slot_mapping=slot_mapping,
            )

    def close(self):
        for rank in range(self.world_size):
            engine = self.lmcache_engines[rank]
            engine.close()
        self.lmcache_engines.clear()
        self.gpu_kv_caches.clear()


def CreateKVInterface(
        model_config: ModelConfig, cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig) -> BlockingKVInterface:

    return LMCacheBlockingKVMgr(model_config=model_config,
                                cache_config=cache_config,
                                parallel_config=parallel_config,
                                scheduler_config=scheduler_config)
