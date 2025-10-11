"""
Token-Aware Scheduler for Compact Encoder Cache

This module implements a scheduler that is aware of token types (embeddings vs special tokens)
to efficiently handle the compact encoder cache format. The scheduler can distinguish between
actual embeddings that need to be fetched from the cache and special tokens that must be
generated on-demand.

Key features:
1. Token-type awareness for efficient sequence reconstruction
2. Batch processing of special token generation
3. Position tracking for accurate token insertion
4. Memory-efficient sequence handling
"""

import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TokenSequenceInfo:
    """Information about a token sequence for efficient processing."""

    mm_hash: str
    start_idx: int
    end_idx: int
    is_embed_mask: torch.Tensor
    embedding_count: int
    special_token_count: int
    metadata: Dict[str, Any]


@dataclass
class BatchTokenRequest:
    """Batch request for token processing."""

    sequences: List[TokenSequenceInfo]
    batch_size: int
    device: torch.device


class TokenAwareScheduler:
    """
    Scheduler that handles token-type awareness for compact encoder cache.

    This scheduler can efficiently process sequences that contain both cached
    embeddings and dynamically generated special tokens.
    """

    def __init__(self, compact_cache, enable_batch_processing: bool = True):
        """
        Initialize the token-aware scheduler.

        Args:
            compact_cache: The compact encoder cache instance
            enable_batch_processing: Whether to enable batch processing optimizations
        """
        self.compact_cache = compact_cache
        self.enable_batch_processing = enable_batch_processing
        self.batch_size_threshold = 8  # Minimum batch size for batch processing

        # Caches for performance optimization
        self.sequence_cache: Dict[str, torch.Tensor] = {}
        self.position_cache: Dict[str, torch.Tensor] = {}

    def process_sequences(
        self, sequence_infos: List[TokenSequenceInfo]
    ) -> List[torch.Tensor]:
        """
        Process multiple sequences with token-type awareness.

        Args:
            sequence_infos: List of sequence information to process

        Returns:
            List of reconstructed sequences
        """
        if not sequence_infos:
            return []

        # Group sequences by similarity for batch processing
        if (
            self.enable_batch_processing
            and len(sequence_infos) >= self.batch_size_threshold
        ):
            return self._process_sequences_batched(sequence_infos)
        else:
            return self._process_sequences_individual(sequence_infos)

    def _process_sequences_batched(
        self, sequence_infos: List[TokenSequenceInfo]
    ) -> List[torch.Tensor]:
        """Process sequences using batch optimizations."""
        # Group sequences by mm_hash for efficient cache access
        sequences_by_hash = defaultdict(list)
        for seq_info in sequence_infos:
            sequences_by_hash[seq_info.mm_hash].append(seq_info)

        results = []
        for mm_hash, hash_sequences in sequences_by_hash.items():
            # Process all sequences for this mm_hash together
            batch_results = self._process_hash_batch(mm_hash, hash_sequences)
            results.extend(batch_results)

        return results

    def _process_hash_batch(
        self, mm_hash: str, sequences: List[TokenSequenceInfo]
    ) -> List[torch.Tensor]:
        """Process all sequences for a specific mm_hash in batch."""
        if mm_hash not in self.compact_cache.cache:
            raise KeyError(f"Cache miss for {mm_hash}")

        entry = self.compact_cache.cache[mm_hash]

        # Extract all embedding slices needed
        embedding_slices = []
        special_token_requests = []

        for seq_info in sequences:
            is_embed_slice = entry.is_embed_mask[seq_info.start_idx : seq_info.end_idx]
            embedding_positions = torch.where(is_embed_slice)[0]

            if len(embedding_positions) > 0:
                # Extract embeddings for this sequence
                embedding_slice = entry.embeddings[embedding_positions]
                embedding_slices.append(embedding_slice)
            else:
                embedding_slices.append(
                    torch.empty(
                        0, entry.embeddings.shape[-1], device=entry.embeddings.device
                    )
                )

            # Collect special token generation requests
            special_positions = torch.where(~is_embed_slice)[0]
            if len(special_positions) > 0:
                special_token_requests.append(
                    {
                        "seq_info": seq_info,
                        "positions": special_positions,
                        "count": len(special_positions),
                    }
                )

        # Batch generate special tokens
        special_tokens_batch = self._batch_generate_special_tokens(
            mm_hash, special_token_requests, entry.metadata
        )

        # Reconstruct sequences
        results = []
        special_token_idx = 0

        for i, seq_info in enumerate(sequences):
            is_embed_slice = entry.is_embed_mask[seq_info.start_idx : seq_info.end_idx]
            embedding_slice = embedding_slices[i]

            # Get special tokens for this sequence
            if special_token_idx < len(special_tokens_batch):
                special_tokens = special_tokens_batch[special_token_idx]
                special_token_idx += 1
            else:
                special_tokens = torch.empty(
                    0, entry.embeddings.shape[-1], device=entry.embeddings.device
                )

            # Reconstruct the sequence
            reconstructed = self._reconstruct_sequence(
                embedding_slice, special_tokens, is_embed_slice
            )
            results.append(reconstructed)

        return results

    def _process_sequences_individual(
        self, sequence_infos: List[TokenSequenceInfo]
    ) -> List[torch.Tensor]:
        """Process sequences individually (fallback for small batches)."""
        results = []
        for seq_info in sequence_infos:
            try:
                sequence = self.compact_cache.retrieve_sequence(
                    seq_info.mm_hash, seq_info.start_idx, seq_info.end_idx
                )
                results.append(sequence)
            except KeyError as e:
                logger.error(f"Failed to retrieve sequence: {e}")
                # Return empty sequence as fallback
                results.append(torch.empty(0, dtype=torch.float32))

        return results

    def _batch_generate_special_tokens(
        self, mm_hash: str, special_token_requests: List[Dict], metadata: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """Batch generate special tokens for multiple sequences."""
        if not special_token_requests:
            return []

        # Group requests by token count for efficient batch processing
        requests_by_count = defaultdict(list)
        for req in special_token_requests:
            requests_by_count[req["count"]].append(req)

        special_tokens_batch = []

        for count, requests in requests_by_count.items():
            if count == 0:
                continue

            # Generate special tokens for all requests with this count
            batch_tokens = self._generate_special_tokens_batch(
                mm_hash, count, len(requests), metadata
            )

            # Distribute tokens to individual requests
            for i, req in enumerate(requests):
                start_idx = i * count
                end_idx = start_idx + count
                special_tokens_batch.append(batch_tokens[start_idx:end_idx])

        return special_tokens_batch

    def _generate_special_tokens_batch(
        self, mm_hash: str, token_count: int, batch_size: int, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate special tokens in batch for efficiency."""
        total_tokens = token_count * batch_size

        # Get device and embedding dimension from cache
        entry = self.compact_cache.cache[mm_hash]
        device = entry.embeddings.device
        embed_dim = entry.embeddings.shape[-1]

        # Generate all special tokens at once
        special_tokens = torch.randn(total_tokens, embed_dim, device=device)

        # Apply model-specific patterns
        if metadata.get("content_type") == "video":
            special_tokens = self._apply_video_patterns_batch(special_tokens, metadata)
        elif metadata.get("content_type") == "image":
            special_tokens = self._apply_image_patterns_batch(special_tokens, metadata)

        return special_tokens

    def _apply_video_patterns_batch(
        self, special_tokens: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply video-specific patterns to batch of special tokens."""
        # Apply timestamp patterns if available
        if "timestamps" in metadata:
            timestamps = metadata["timestamps"]
            for i, timestamp in enumerate(timestamps):
                if i < len(special_tokens):
                    # Apply timestamp-specific patterns
                    timestamp_bias = torch.full_like(
                        special_tokens[i], timestamp * 0.01
                    )
                    special_tokens[i] = special_tokens[i] + timestamp_bias

        return special_tokens

    def _apply_image_patterns_batch(
        self, special_tokens: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply image-specific patterns to batch of special tokens."""
        # Apply image-specific patterns
        if len(special_tokens) >= 2:
            # First token: image start marker
            start_bias = torch.full_like(special_tokens[0], 1.0)
            special_tokens[0] = special_tokens[0] + start_bias

            # Last token: image end marker
            end_bias = torch.full_like(special_tokens[-1], -1.0)
            special_tokens[-1] = special_tokens[-1] + end_bias

        return special_tokens

    def _reconstruct_sequence(
        self,
        embeddings: torch.Tensor,
        special_tokens: torch.Tensor,
        is_embed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a sequence by interleaving embeddings and special tokens."""
        sequence_length = len(is_embed_mask)
        embed_dim = (
            embeddings.shape[-1] if len(embeddings) > 0 else special_tokens.shape[-1]
        )
        device = embeddings.device if len(embeddings) > 0 else special_tokens.device

        # Create the reconstructed sequence
        reconstructed = torch.zeros(sequence_length, embed_dim, device=device)

        # Place embeddings at their positions
        embedding_positions = torch.where(is_embed_mask)[0]
        if len(embedding_positions) > 0 and len(embeddings) > 0:
            reconstructed[embedding_positions] = embeddings

        # Place special tokens at their positions
        special_positions = torch.where(~is_embed_mask)[0]
        if len(special_positions) > 0 and len(special_tokens) > 0:
            reconstructed[special_positions] = special_tokens

        return reconstructed

    def get_sequence_info(
        self, mm_hash: str, start_idx: int, end_idx: int
    ) -> TokenSequenceInfo:
        """Get information about a sequence for efficient processing."""
        if mm_hash not in self.compact_cache.cache:
            raise KeyError(f"Cache miss for {mm_hash}")

        entry = self.compact_cache.cache[mm_hash]
        is_embed_slice = entry.is_embed_mask[start_idx:end_idx]

        embedding_count = is_embed_slice.sum().item()
        special_token_count = len(is_embed_slice) - embedding_count

        return TokenSequenceInfo(
            mm_hash=mm_hash,
            start_idx=start_idx,
            end_idx=end_idx,
            is_embed_mask=is_embed_slice,
            embedding_count=embedding_count,
            special_token_count=special_token_count,
            metadata=entry.metadata,
        )

    def clear_caches(self) -> None:
        """Clear internal caches to free memory."""
        self.sequence_cache.clear()
        self.position_cache.clear()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            "sequence_cache_size": len(self.sequence_cache),
            "position_cache_size": len(self.position_cache),
            "batch_processing_enabled": self.enable_batch_processing,
            "batch_size_threshold": self.batch_size_threshold,
        }
