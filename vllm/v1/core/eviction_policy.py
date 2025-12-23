# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reference: PagedEviction paper (arXiv:2509.04377v1)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Optional
import time
import torch

# Will be imported from kv_cache_utils
# from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass
class BlockMetadata:
    """
    Extended metadata for eviction decisions.

    This stores per-block statistics needed by eviction policies to make
    intelligent eviction decisions.

    Attributes:
        last_access_time: Timestamp of most recent access (for recency tracking)
        access_count: Number of times this block has been accessed (for frequency)
        utility_score: Computed utility score for eviction decisions
        is_cached: Whether this block is in the prefix cache
        cache_hit_count: Number of times this block was hit in prefix cache
    """

    last_access_time: float = 0.0
    access_count: int = 0
    utility_score: float = 0.0
    is_cached: bool = False
    cache_hit_count: int = 0


class EvictionPolicy(ABC):
    """
    Abstract base class for KV cache eviction policies.

    This interface defines the contract that all eviction policies must follow.
    It is designed to integrate seamlessly with vLLM's BlockPool without requiring
    modifications to CUDA attention kernels.

    The eviction policy is responsible for:
    1. Tracking block access patterns
    2. Computing block importance/utility scores
    3. Selecting which blocks to evict when memory is needed
    4. Maintaining any necessary metadata for decision-making
    """

    @abstractmethod
    def select_blocks_to_evict(
        self,
        num_blocks_needed: int,
        free_blocks: Sequence["KVCacheBlock"],  # type: ignore
        cached_blocks: dict[int, "KVCacheBlock"],  # type: ignore
    ) -> list["KVCacheBlock"]:  # type: ignore
        """
        Select blocks to evict from the free block queue.

        This is the core method where the eviction policy decides which blocks
        should be reused/evicted when new allocations are needed.

        Args:
            num_blocks_needed: Number of blocks to select for eviction
            free_blocks: Sequence of candidate blocks (all with ref_cnt == 0)
            cached_blocks: Map of block_id -> block for blocks in prefix cache

        Returns:
            List of blocks selected for eviction (length == num_blocks_needed)
        """
        pass

    @abstractmethod
    def update_access(
        self,
        blocks: Sequence["KVCacheBlock"],  # type: ignore
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update access patterns when blocks are touched (e.g., prefix cache hit).

        This method is called when blocks are accessed (typically during prefix
        cache hits via BlockPool.touch()). The policy should update its internal
        tracking structures (e.g., access counts, timestamps, etc.).

        Args:
            blocks: Blocks that were accessed
            timestamp: Optional timestamp (defaults to time.time())
        """
        pass

    @abstractmethod
    def on_block_allocated(self, block: "KVCacheBlock") -> None:  # type: ignore
        """
        Called when a block is allocated from the free queue.

        This hook allows the policy to initialize or update metadata when a block
        transitions from free to allocated state.

        Args:
            block: The block that was just allocated
        """
        pass

    @abstractmethod
    def on_block_freed(self, block: "KVCacheBlock") -> None:  # type: ignore
        """
        Called when a block is freed and returned to the free queue.

        This hook allows the policy to update metadata when a block transitions
        from allocated to free state.

        Args:
            block: The block that was just freed
        """
        pass

    @abstractmethod
    def compute_token_importance(
        self,
        key_tensor: "torch.Tensor",  # type: ignore
        value_tensor: "torch.Tensor",  # type: ignore
    ) -> "torch.Tensor":  # type: ignore
        """
        Compute importance scores for tokens based on Key and Value tensors.

        This implements the core importance metric from the PagedEviction paper.
        The paper uses: importance(i) = ||V_i||_2 / ||K_i||_2

        Args:
            key_tensor: Key tensor for tokens
            value_tensor: Value tensor for tokens

        Returns:
            Tensor of importance scores (one per token)
        """
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """
    Baseline LRU (Least Recently Used) eviction policy.

    This implements the current behavior in vLLM where blocks are evicted in
    FIFO order from the FreeKVCacheBlockQueue. This serves as a baseline for
    comparison with more sophisticated policies.

    The LRU policy simply relies on the natural ordering of the free block queue,
    so most methods are no-ops.
    """

    def select_blocks_to_evict(
        self,
        num_blocks_needed: int,
        free_blocks: Sequence["KVCacheBlock"],  # type: ignore
        cached_blocks: dict[int, "KVCacheBlock"],  # type: ignore
    ) -> list["KVCacheBlock"]:  # type: ignore
        """
        LRU eviction: Simply take the first N blocks from the free queue.

        The free queue is already maintained in LRU order by the FreeKVCacheBlockQueue,
        so we just take the first N blocks.
        """
        return list(free_blocks[:num_blocks_needed])

    def update_access(
        self,
        blocks: Sequence["KVCacheBlock"],  # type: ignore
        timestamp: Optional[float] = None,
    ) -> None:
        """
        LRU doesn't need extra tracking - queue ordering handles recency.

        The FreeKVCacheBlockQueue already maintains LRU ordering by appending
        freed blocks to the tail, so we don't need additional bookkeeping.
        """
        # No-op for LRU - queue ordering is sufficient
        pass

    def on_block_allocated(self, block: "KVCacheBlock") -> None:  # type: ignore
        """No additional tracking needed for LRU."""
        pass

    def on_block_freed(self, block: "KVCacheBlock") -> None:  # type: ignore
        """No additional tracking needed for LRU."""
        pass

    def compute_token_importance(
        self,
        key_tensor: "torch.Tensor",  # type: ignore
        value_tensor: "torch.Tensor",  # type: ignore
    ) -> "torch.Tensor":  # type: ignore
        """
        LRU doesn't use importance scores.
        """
        raise NotImplementedError


class PagedEvictionPolicy(EvictionPolicy):
    """
    PagedEviction: Block-wise KV cache eviction optimized for vLLM's PagedAttention.

    This implements the structured block-wise eviction strategy from the PagedEviction
    paper (arXiv:2509.04377v1). Key features:

    1. Token importance based on ||V||_2 / ||K||_2 ratio (no attention scores needed)
    2. Block-level eviction during decode phase (reduces fragmentation)
    3. Prefix cache awareness (preserves high-value cached blocks)
    4. Compatible with FlashAttention (no CUDA kernel modifications)

    The algorithm operates in two phases:
    - Prefill: Token-level eviction before block partitioning
    - Decode: Block-level eviction when current block becomes full
    """

    def __init__(
        self,
        recency_weight: float = 0.4,
        frequency_weight: float = 0.4,
        cache_weight: float = 0.2,
        time_decay: float = 0.95,
        enable_prefill_eviction: bool = True,
        enable_decode_eviction: bool = True,
    ):
        """
        Initialize PagedEviction policy with configurable parameters.

        Args:
            recency_weight: Weight for recency score in utility calculation
            frequency_weight: Weight for frequency score in utility calculation
            cache_weight: Weight for cache value in utility calculation
            time_decay: Exponential decay factor for recency (0 < decay < 1)
            enable_prefill_eviction: Whether to evict during prefill phase
            enable_decode_eviction: Whether to evict during decode phase
        """
        # General args
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.cache_weight = cache_weight
        self.time_decay = time_decay
        self.enable_prefill_eviction = enable_prefill_eviction
        self.enable_decode_eviction = enable_decode_eviction

        # Track metadata per block ID
        self.metadata: dict[int, BlockMetadata] = {}

        # Global statistics for normalization
        self.max_access_count: int = 1
        self.current_time: float = time.time()

        # Prefill/decode phase tracking
        self.is_prefill_phase: bool = True
        self.decode_block_counter: int = 0

        # Performance metrics
        self._metrics = {
            "selection_count": 0,
            "selection_total_time": 0.0,
            "selection_min_time": float("inf"),
            "selection_max_time": 0.0,
            "update_count": 0,
            "update_total_time": 0.0,
        }

    def select_blocks_to_evict(
        self,
        num_blocks_needed: int,
        free_blocks: Sequence["KVCacheBlock"],  # type: ignore
        cached_blocks: dict[int, "KVCacheBlock"],  # type: ignore
    ) -> list["KVCacheBlock"]:  # type: ignore
        """
        Select blocks for eviction based on utility scores.

        This implements the core block selection logic:
        1. Compute utility score for each candidate block
        2. Sort blocks by score (ascending = low utility evicted first)
        3. Return the N blocks with lowest scores
        """
        start = time.perf_counter()

        if not free_blocks:
            return []

        if num_blocks_needed >= len(free_blocks):
            return [b for b in free_blocks if not b.is_null and b.ref_cnt == 0]

        # XXX: if all blocks same score, can we check that easily and avoid the coming loop

        candidates = []
        for block in free_blocks:
            if block.is_null or block.ref_cnt > 0:
                continue
            score = self._calculate_eviction_score(block)
            candidates.append((score, block))

        candidates.sort(key=lambda x: x[0])  # ascending, lower score first
        blocks_to_evict = [block for _, block in candidates[:num_blocks_needed]]

        elapsed = time.perf_counter() - start
        self._metrics["selection_count"] += 1
        self._metrics["selection_total_time"] += elapsed
        self._metrics["selection_min_time"] = min(
            self._metrics["selection_min_time"], elapsed
        )
        self._metrics["selection_max_time"] = max(
            self._metrics["selection_max_time"], elapsed
        )

        return blocks_to_evict

    def _calculate_eviction_score(
        self,
        block: "KVCacheBlock",  # type: ignore
    ) -> float:
        """
        Calculate eviction score for a single block. Lower score = higher eviction priority.
        If metadata doesn't exist for a block, it should get a very low score (0.0) making it high priority for eviction.
        """

        metadata = self.metadata.get(block.block_id)

        if metadata is None:
            return 0.0  # Never accessed - Low score, high eviction priority

        time_since_access = self.current_time - metadata.last_access_time
        recency_score = self.time_decay**time_since_access

        frequency_score = metadata.access_count / max(self.max_access_count, 1)

        if block.block_hash is not None:
            cache_cost = 1.0 + metadata.cache_hit_count
        else:
            cache_cost = 0.0

        eviction_score = (
            recency_score * self.recency_weight
            + frequency_score * self.frequency_weight
            + cache_cost * self.cache_weight
        )

        return eviction_score

    def update_access(
        self,
        blocks: Sequence["KVCacheBlock"],  # type: ignore
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update access tracking when blocks are touched (prefix cache hit). This is called from BlockPool.touch() when blocks are accessed.
        XXX: Consider batching updates or sampling if profiling shows overhead.
        """
        start = time.perf_counter()

        if timestamp is None:
            timestamp = time.time()

        self.current_time = timestamp

        for block in blocks:
            metadata = self.metadata.setdefault(block.block_id, BlockMetadata())
            metadata.last_access_time = timestamp
            metadata.access_count += 1

            self.max_access_count = max(self.max_access_count, metadata.access_count)

            if block.block_hash is not None:
                metadata.is_cached = True
                metadata.cache_hit_count += 1

        elapsed = time.perf_counter() - start
        self._metrics["update_count"] += 1
        self._metrics["update_total_time"] += elapsed

    def on_block_allocated(self, block: "KVCacheBlock") -> None:  # type: ignore
        """
        Initialize metadata when block is allocated.
        """
        pass

    def on_block_freed(self, block: "KVCacheBlock") -> None:  # type: ignore
        """
        Update metadata when block is freed.
        """
        # XXX: This is complete removal. Could implement also a version that keeps history
        if block.block_id in self.metadata:
            del self.metadata[block.block_id]

    def compute_token_importance(
        self,
        key_tensor: "torch.Tensor",  # type: ignore
        value_tensor: "torch.Tensor",  # type: ignore
    ) -> "torch.Tensor":  # type: ignore
        """
        Compute per-token importance scores using Key and Value norms.
        """
        return torch.norm(value_tensor, p=2, dim=-1) / (
            torch.norm(key_tensor, p=2, dim=-1) + 1e-6
        )

    def compute_block_importance(
        self,
        key_tensor: "torch.Tensor",  # type: ignore
        value_tensor: "torch.Tensor",  # type: ignore
        block_size: int,
    ) -> "torch.Tensor":  # type: ignore
        """
        Compute per-block importance scores by aggregating token scores.
        """
        token_importance = self.compute_token_importance(key_tensor, value_tensor)

        num_tokens = token_importance.shape[0]
        num_blocks = num_tokens // block_size

        if num_blocks == 0:
            return torch.tensor([], dtype=token_importance.dtype)

        block_scores = token_importance[: num_blocks * block_size].view(
            num_blocks, block_size
        )
        return block_scores.mean(dim=-1)

    def get_stats(self) -> dict:
        """
        Return current policy statistics for monitoring.

        Returns:
            Dictionary with policy metrics.
        """
        stats = {
            "metadata_size": len(self.metadata),
            "max_access_count": self.max_access_count,
            # Configuration parameters
            "recency_weight": self.recency_weight,
            "frequency_weight": self.frequency_weight,
            "cache_weight": self.cache_weight,
            "time_decay": self.time_decay,
            "current_time": self.current_time,
            "is_prefill_phase": self.is_prefill_phase,
            # Performance metrics
            "selection_count": self._metrics["selection_count"],
            "selection_avg_time": (
                self._metrics["selection_total_time"]
                / max(1, self._metrics["selection_count"])
            ),
            "selection_min_time": self._metrics["selection_min_time"]
            if self._metrics["selection_count"] > 0
            else 0.0,
            "selection_max_time": self._metrics["selection_max_time"],
            "update_count": self._metrics["update_count"],
            "update_avg_time": (
                self._metrics["update_total_time"]
                / max(1, self._metrics["update_count"])
            ),
        }

        # Add only if we have metadata
        if self.metadata:
            access_counts = [m.access_count for m in self.metadata.values()]
            recency_times = [
                self.current_time - m.last_access_time for m in self.metadata.values()
            ]

            # Filter cached blocks
            cached_blocks = [m for m in self.metadata.values() if m.is_cached]
            cache_hits = [c.cache_hit_count for c in cached_blocks]

            stats.update(
                {
                    "avg_access_count": sum(access_counts) / len(access_counts),
                    "max_block_access_count": max(access_counts),
                    "min_block_access_count": min(access_counts),
                    "median_access_count": sorted(access_counts)[
                        len(access_counts) // 2
                    ],
                    "avg_time_since_access": sum(recency_times) / len(recency_times),
                    "max_time_since_access": max(recency_times),
                    "cached_blocks_tracked": len(cached_blocks),
                    "avg_cache_hit_count": sum(cache_hits) / len(cache_hits)
                    if cache_hits
                    else 0.0,
                    "max_cache_hit_count": max(cache_hits) if cache_hits else 0,
                }
            )

        return stats

    def evict_tokens_prefill(
        self,
        key_tensor: "torch.Tensor",  # type: ignore
        value_tensor: "torch.Tensor",  # type: ignore
        num_tokens_to_evict: int,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore
        """
        Evict tokens during prefill phase before block partitioning.
        """
        num_tokens = key_tensor.shape[0]

        if num_tokens_to_evict <= 0:
            return key_tensor, value_tensor

        if num_tokens_to_evict >= num_tokens:
            raise ValueError(
                f"Cannot evict {num_tokens_to_evict} tokens from {num_tokens} total"
            )

        token_importance = self.compute_token_importance(key_tensor, value_tensor)

        # Keep tokens with highest importance
        num_keep = num_tokens - num_tokens_to_evict
        _, keep_indices = torch.topk(token_importance, num_keep, largest=True).sort()[0]

        filtered_keys = key_tensor[keep_indices]
        filtered_values = value_tensor[keep_indices]

        return filtered_keys, filtered_values

    def should_evict_block_decode(
        self, current_sequence_length: int, block_size: int
    ) -> bool:
        """
        Determine if we should trigger block eviction during decode phase.
        """
        return current_sequence_length % block_size == 0

    def reset_for_new_request(self) -> None:
        """
        Reset policy state for a new request.
        """
        self.is_prefill_phase = True
        self.decode_block_counter = 0

    def transition_to_decode_phase(self) -> None:
        """
        Transition from prefill to decode phase.
        """
        self.is_prefill_phase = False
        self.decode_block_counter = 0
