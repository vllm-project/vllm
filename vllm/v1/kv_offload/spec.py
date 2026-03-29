# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.planner import HybridOffloadPlanner
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


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

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        logger.warning(
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.extra_config = kv_transfer_config.kv_connector_extra_config

        # block size used by vLLM for hashing request tokens for the sake
        # of enabling prefix caching
        self.hash_block_size = vllm_config.cache_config.block_size
        self.hash_function = get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )
        # TODO: Block hashes are computed from token IDs and a chain seed
        # (NONE_HASH), but do NOT incorporate model weights or config.
        # If model weights change (e.g. same model name, different revision
        # or fine-tune), stored KV cache files will still hash-match but
        # contain stale/wrong KV values — silent correctness corruption.
        # Fix: incorporate a model fingerprint (weights checksum, revision
        # hash, or tokenizer vocab hash) into the hash seed so that
        # weight changes invalidate the cache automatically.
        # gpu block size per group
        self.gpu_block_size: tuple[int, ...] = tuple(
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        )
        self.hybrid_planner: HybridOffloadPlanner | None = None
        self.hybrid_offload_enabled: bool = False
        self.group_hash_block_size: tuple[int, ...] = tuple(
            self.hash_block_size for _ in self.gpu_block_size
        )

        hybrid_chunk_size = self.extra_config.get("hybrid_chunk_size")
        if hybrid_chunk_size is not None:
            chunk_size_int = int(hybrid_chunk_size)
            # Warn about gpu_block_sizes that are not divisible by chunk_size,
            # as these groups cannot be split and will raise
            # first_hashable_chunk_idx, potentially making offloading
            # impossible for practical context lengths.
            for i, gbs in enumerate(self.gpu_block_size):
                if gbs > chunk_size_int and gbs % chunk_size_int != 0:
                    logger.warning(
                        "KV group %d has gpu_block_size=%d which is not "
                        "divisible by hybrid_chunk_size=%d. This group "
                        "cannot be split into chunks and will require "
                        "%d tokens before any offloading can occur. "
                        "Consider setting max_model_len to a multiple "
                        "of hybrid_chunk_size.",
                        i,
                        gbs,
                        chunk_size_int,
                        gbs,
                    )
            self.hybrid_planner = HybridOffloadPlanner(
                hash_block_size=self.hash_block_size,
                gpu_block_sizes=self.gpu_block_size,
                fixed_chunk_size=chunk_size_int,
            )
            self.hybrid_offload_enabled = True
            self.group_hash_block_size = self.hybrid_planner.offload_unit_sizes
            max_model_len = vllm_config.model_config.max_model_len
            if (
                self.hybrid_planner.first_hashable_chunk_idx * chunk_size_int
                >= max_model_len
            ):
                logger.error(
                    "Hybrid offloading is effectively disabled: "
                    "first_hashable_chunk_idx=%d requires %d tokens "
                    "but max_model_len=%d. No chunks can ever be "
                    "stored. Set max_model_len to a multiple of "
                    "hybrid_chunk_size=%d (e.g. %d).",
                    self.hybrid_planner.first_hashable_chunk_idx,
                    self.hybrid_planner.first_hashable_chunk_idx * chunk_size_int,
                    max_model_len,
                    chunk_size_int,
                    (max_model_len // chunk_size_int) * chunk_size_int,
                )
            else:
                logger.info(
                    "Hybrid offloading enabled: chunk_size=%d, "
                    "offload_unit_sizes=%s, "
                    "first_hashable_chunk_idx=%d, "
                    "min_tokens_for_offload=%d",
                    chunk_size_int,
                    self.hybrid_planner.offload_unit_sizes,
                    self.hybrid_planner.first_hashable_chunk_idx,
                    (self.hybrid_planner.first_hashable_chunk_idx + 1) * chunk_size_int,
                )
        else:
            for block_size in self.gpu_block_size:
                assert block_size % self.hash_block_size == 0

        self.offloaded_block_size: int = lcm(*self.gpu_block_size)
        self.block_size_factors: tuple[int, ...] = tuple(
            self.offloaded_block_size // block_size
            for block_size in self.gpu_block_size
        )

        offloaded_block_size = self.extra_config.get("block_size")
        if offloaded_block_size is not None:
            offloaded_block_size_int = int(offloaded_block_size)
            assert all(
                offloaded_block_size_int % gpu_block_size == 0
                for gpu_block_size in self.gpu_block_size
            ), (
                "If 'block_size' is specified in kv_connector_extra_config, "
                "it must be divisible by every KV cache group block size."
            )
            self.offloaded_block_size = offloaded_block_size_int
            self.block_size_factors = tuple(
                self.offloaded_block_size // block_size
                for block_size in self.gpu_block_size
            )

        if self.hybrid_offload_enabled:
            self.offloaded_block_size = int(hybrid_chunk_size)  # type: ignore[arg-type]
            self.block_size_factors = tuple(
                self.offloaded_block_size // block_size
                if self.offloaded_block_size % block_size == 0
                else 0
                for block_size in self.gpu_block_size
            )

    @property
    def requires_partial_group_offload(self) -> bool:
        return (
            self.hybrid_planner.requires_partial_group_offload_any
            if self.hybrid_planner is not None
            else False
        )

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
