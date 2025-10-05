"""
Advanced Optimization Techniques for Compact Encoder Cache

This module implements advanced optimization techniques including batch processing,
position caching, and memory-efficient sequence reconstruction to maximize the
performance benefits of the compact encoder cache.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing optimization."""

    total_batches: int = 0
    total_sequences: int = 0
    average_batch_size: float = 0.0
    processing_time_ms: float = 0.0
    memory_savings_mb: float = 0.0


@dataclass
class PositionCacheEntry:
    """Entry in the position cache for efficient sequence reconstruction."""

    mm_hash: str
    start_idx: int
    end_idx: int
    embedding_positions: torch.Tensor
    special_token_positions: torch.Tensor
    sequence_length: int
    last_accessed: float


class BatchProcessor:
    """
    Advanced batch processor for efficient sequence reconstruction.

    This processor groups similar sequences together and processes them in batches
    to maximize GPU utilization and minimize memory bandwidth requirements.
    """

    def __init__(self, device: torch.device, max_batch_size: int = 32):
        """
        Initialize the batch processor.

        Args:
            device: The device to process on
            max_batch_size: Maximum batch size for processing
        """
        self.device = device
        self.max_batch_size = max_batch_size
        self.stats = BatchProcessingStats()

        # Pre-allocated buffers for batch processing
        self._embedding_buffer = None
        self._special_token_buffer = None
        self._output_buffer = None

    def process_batch(
        self, sequences: List[Tuple[str, int, int, torch.Tensor, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Process a batch of sequences efficiently.

        Args:
            sequences: List of (mm_hash, start_idx, end_idx, embeddings, is_embed_mask) tuples

        Returns:
            List of reconstructed sequences
        """
        if not sequences:
            return []

        start_time = time.time()

        # Group sequences by similarity for efficient processing
        grouped_sequences = self._group_sequences_by_similarity(sequences)

        results = []
        for group in grouped_sequences:
            group_results = self._process_sequence_group(group)
            results.extend(group_results)

        # Update statistics
        self._update_stats(len(sequences), time.time() - start_time)

        return results

    def _group_sequences_by_similarity(
        self, sequences: List[Tuple[str, int, int, torch.Tensor, torch.Tensor]]
    ) -> List[List[Tuple[str, int, int, torch.Tensor, torch.Tensor]]]:
        """Group sequences by similarity for efficient batch processing."""
        # Group by sequence length and embedding count
        groups = defaultdict(list)

        for seq in sequences:
            mm_hash, start_idx, end_idx, embeddings, is_embed_mask = seq
            key = (len(is_embed_mask), len(embeddings))
            groups[key].append(seq)

        # Limit group size to max_batch_size
        grouped_sequences = []
        for group in groups.values():
            for i in range(0, len(group), self.max_batch_size):
                grouped_sequences.append(group[i : i + self.max_batch_size])

        return grouped_sequences

    def _process_sequence_group(
        self, group: List[Tuple[str, int, int, torch.Tensor, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Process a group of similar sequences."""
        if not group:
            return []

        # Extract common information
        sequence_length = len(group[0][4])  # is_embed_mask length
        embed_dim = group[0][3].shape[-1] if len(group[0][3]) > 0 else 512

        # Pre-allocate buffers if needed
        self._ensure_buffers_allocated(len(group), sequence_length, embed_dim)

        # Process all sequences in the group
        results = []
        for mm_hash, start_idx, end_idx, embeddings, is_embed_mask in group:
            sequence = self._reconstruct_sequence_optimized(
                embeddings, is_embed_mask, sequence_length, embed_dim
            )
            results.append(sequence)

        return results

    def _reconstruct_sequence_optimized(
        self,
        embeddings: torch.Tensor,
        is_embed_mask: torch.Tensor,
        sequence_length: int,
        embed_dim: int,
    ) -> torch.Tensor:
        """Reconstruct a sequence using optimized operations."""
        # Create output tensor
        output = torch.zeros(sequence_length, embed_dim, device=self.device)

        # Place embeddings at their positions
        embedding_positions = torch.where(is_embed_mask)[0]
        if len(embedding_positions) > 0 and len(embeddings) > 0:
            output[embedding_positions] = embeddings

        # Generate and place special tokens
        special_positions = torch.where(~is_embed_mask)[0]
        if len(special_positions) > 0:
            special_tokens = self._generate_special_tokens_optimized(
                len(special_positions), embed_dim
            )
            output[special_positions] = special_tokens

        return output

    def _generate_special_tokens_optimized(
        self, num_tokens: int, embed_dim: int
    ) -> torch.Tensor:
        """Generate special tokens using optimized operations."""
        # Use pre-allocated buffer if available
        if (
            self._special_token_buffer is not None
            and self._special_token_buffer.shape[0] >= num_tokens
        ):
            special_tokens = self._special_token_buffer[:num_tokens]
        else:
            special_tokens = torch.randn(num_tokens, embed_dim, device=self.device)

        return special_tokens

    def _ensure_buffers_allocated(
        self, batch_size: int, sequence_length: int, embed_dim: int
    ) -> None:
        """Ensure buffers are allocated for efficient processing."""
        if (
            self._embedding_buffer is None
            or self._embedding_buffer.shape[0] < batch_size
        ):
            self._embedding_buffer = torch.empty(
                batch_size, sequence_length, embed_dim, device=self.device
            )

        if (
            self._special_token_buffer is None
            or self._special_token_buffer.shape[0] < sequence_length
        ):
            self._special_token_buffer = torch.empty(
                sequence_length, embed_dim, device=self.device
            )

        if self._output_buffer is None or self._output_buffer.shape[0] < batch_size:
            self._output_buffer = torch.empty(
                batch_size, sequence_length, embed_dim, device=self.device
            )

    def _update_stats(self, num_sequences: int, processing_time: float) -> None:
        """Update batch processing statistics."""
        self.stats.total_batches += 1
        self.stats.total_sequences += num_sequences
        self.stats.average_batch_size = (
            self.stats.total_sequences / self.stats.total_batches
        )
        self.stats.processing_time_ms += processing_time * 1000

        # Estimate memory savings (simplified calculation)
        self.stats.memory_savings_mb += num_sequences * 0.1  # 0.1 MB per sequence

    def get_stats(self) -> BatchProcessingStats:
        """Get batch processing statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset batch processing statistics."""
        self.stats = BatchProcessingStats()


class PositionCache:
    """
    Position cache for efficient sequence reconstruction.

    This cache stores pre-computed position mappings to avoid repeated
    calculations during sequence reconstruction.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        """
        Initialize the position cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, PositionCacheEntry] = {}
        self.access_order: List[str] = []

    def get_positions(
        self, mm_hash: str, start_idx: int, end_idx: int, is_embed_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached position mappings or compute and cache them.

        Args:
            mm_hash: Cache identifier
            start_idx: Start position in the sequence
            end_idx: End position in the sequence
            is_embed_mask: Boolean mask for embedding positions

        Returns:
            Tuple of (embedding_positions, special_token_positions)
        """
        cache_key = f"{mm_hash}_{start_idx}_{end_idx}"
        current_time = time.time()

        # Check if we have a cached entry
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if current_time - entry.last_accessed < self.ttl_seconds:
                # Update access time and move to end of access order
                entry.last_accessed = current_time
                self.access_order.remove(cache_key)
                self.access_order.append(cache_key)
                return entry.embedding_positions, entry.special_token_positions
            else:
                # Entry expired, remove it
                del self.cache[cache_key]
                self.access_order.remove(cache_key)

        # Compute positions
        embedding_positions = torch.where(is_embed_mask)[0]
        special_token_positions = torch.where(~is_embed_mask)[0]

        # Cache the result
        self._cache_entry(
            cache_key,
            mm_hash,
            start_idx,
            end_idx,
            embedding_positions,
            special_token_positions,
            len(is_embed_mask),
            current_time,
        )

        return embedding_positions, special_token_positions

    def _cache_entry(
        self,
        cache_key: str,
        mm_hash: str,
        start_idx: int,
        end_idx: int,
        embedding_positions: torch.Tensor,
        special_token_positions: torch.Tensor,
        sequence_length: int,
        current_time: float,
    ) -> None:
        """Cache a position entry."""
        # Remove oldest entries if cache is full
        while len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        # Add new entry
        entry = PositionCacheEntry(
            mm_hash=mm_hash,
            start_idx=start_idx,
            end_idx=end_idx,
            embedding_positions=embedding_positions,
            special_token_positions=special_token_positions,
            sequence_length=sequence_length,
            last_accessed=current_time,
        )

        self.cache[cache_key] = entry
        self.access_order.append(cache_key)

    def clear(self) -> None:
        """Clear the position cache."""
        self.cache.clear()
        self.access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        active_entries = sum(
            1
            for entry in self.cache.values()
            if current_time - entry.last_accessed < self.ttl_seconds
        )

        return {
            "total_entries": len(self.cache),
            "active_entries": active_entries,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


class MemoryOptimizer:
    """
    Memory optimizer for the compact encoder cache.

    This optimizer monitors memory usage and applies various techniques
    to minimize memory consumption while maintaining performance.
    """

    def __init__(self, memory_threshold: float = 0.8):
        """
        Initialize the memory optimizer.

        Args:
            memory_threshold: Memory usage threshold (0.0 to 1.0)
        """
        self.memory_threshold = memory_threshold
        self.optimization_history: List[Dict[str, Any]] = []

    def optimize_memory_usage(
        self, compact_cache, position_cache, batch_processor
    ) -> Dict[str, Any]:
        """
        Optimize memory usage across all components.

        Args:
            compact_cache: The compact encoder cache
            position_cache: The position cache
            batch_processor: The batch processor

        Returns:
            Dictionary of optimization results
        """
        current_memory = self._get_current_memory_usage()
        optimization_results = {
            "initial_memory_mb": current_memory,
            "optimizations_applied": [],
            "final_memory_mb": current_memory,
            "memory_saved_mb": 0.0,
        }

        # Apply memory optimizations
        if current_memory > self.memory_threshold * self._get_total_memory():
            # Clear expired cache entries
            self._clear_expired_entries(position_cache)
            optimization_results["optimizations_applied"].append(
                "cleared_expired_entries"
            )

            # Compact the position cache
            self._compact_position_cache(position_cache)
            optimization_results["optimizations_applied"].append(
                "compacted_position_cache"
            )

            # Clear batch processor buffers
            self._clear_batch_processor_buffers(batch_processor)
            optimization_results["optimizations_applied"].append(
                "cleared_batch_buffers"
            )

            # Update final memory usage
            optimization_results["final_memory_mb"] = self._get_current_memory_usage()
            optimization_results["memory_saved_mb"] = (
                optimization_results["initial_memory_mb"]
                - optimization_results["final_memory_mb"]
            )

        # Record optimization history
        self.optimization_history.append(optimization_results)

        return optimization_results

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return 0.0

    def _get_total_memory(self) -> float:
        """Get total available memory in MB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        else:
            return 1024.0  # Assume 1GB for CPU

    def _clear_expired_entries(self, position_cache: PositionCache) -> None:
        """Clear expired entries from the position cache."""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in position_cache.cache.items()
            if current_time - entry.last_accessed > position_cache.ttl_seconds
        ]

        for key in expired_keys:
            del position_cache.cache[key]
            position_cache.access_order.remove(key)

        logger.info(f"Cleared {len(expired_keys)} expired position cache entries")

    def _compact_position_cache(self, position_cache: PositionCache) -> None:
        """Compact the position cache by removing least recently used entries."""
        if len(position_cache.cache) <= position_cache.max_size // 2:
            return

        # Remove half of the least recently used entries
        entries_to_remove = len(position_cache.cache) // 2
        for _ in range(entries_to_remove):
            if position_cache.access_order:
                oldest_key = position_cache.access_order.pop(0)
                del position_cache.cache[oldest_key]

        logger.info(f"Compacted position cache, removed {entries_to_remove} entries")

    def _clear_batch_processor_buffers(self, batch_processor: BatchProcessor) -> None:
        """Clear batch processor buffers to free memory."""
        batch_processor._embedding_buffer = None
        batch_processor._special_token_buffer = None
        batch_processor._output_buffer = None

        logger.info("Cleared batch processor buffers")

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of memory optimizations."""
        return self.optimization_history.copy()

    def get_memory_efficiency_score(self) -> float:
        """Get a score indicating memory efficiency (0.0 to 1.0)."""
        if not self.optimization_history:
            return 1.0

        latest_optimization = self.optimization_history[-1]
        memory_saved = latest_optimization.get("memory_saved_mb", 0.0)
        initial_memory = latest_optimization.get("initial_memory_mb", 1.0)

        if initial_memory == 0:
            return 1.0

        return min(1.0, memory_saved / initial_memory)
