# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DraftProbsBuffer - Manages draft probability tensors for rejection sampling.

This module implements Option B from the temperature-aware draft sampling plan,
providing KV-cache-level management for draft probability tensors.

The key challenge: draft_probs are created during draft sampling but used later
during rejection sampling. The timing is unpredictable due to:
- Preemption: request may be paused
- Abort: request may be cancelled
- Finish: request may complete before drafts are verified

Reference: https://github.com/vllm-project/vllm/pull/16899
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DraftProbsSlot:
    """Metadata for a single request's draft probabilities."""

    request_id: str
    num_draft_tokens: int
    # Whether this slot has valid probs stored
    is_valid: bool = False


class DraftProbsBuffer:
    """Manages draft probability tensors across scheduling steps.

    Similar to KV cache management, this buffer stores draft probabilities
    created during draft sampling for later use in rejection sampling.

    Key operations:
    - store(): Save draft probs after draft sampling
    - get(): Retrieve draft probs for rejection sampling
    - clear(): Free slot when request finishes/aborts

    Memory layout:
    - probs_buffer: [max_batch_size, max_draft_len, vocab_size]
    - Each request gets one row in the first dimension

    Example:
        buffer = DraftProbsBuffer(max_batch_size=8, max_draft_len=4, vocab_size=32000)

        # During draft sampling
        draft_tokens, probs = sample_with_probs(logits, temperature)
        buffer.store(req_idx=0, draft_probs=probs, num_tokens=4)

        # During rejection sampling (later step)
        draft_probs = buffer.get(req_idx=0, num_tokens=4)
        acceptance = target_probs / draft_probs >= u

        # When request finishes
        buffer.clear(req_idx=0)
    """

    def __init__(
        self,
        max_batch_size: int,
        max_draft_len: int,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.max_batch_size = max_batch_size
        self.max_draft_len = max_draft_len
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype

        # Pre-allocate probability buffer
        # Shape: [max_batch_size, max_draft_len, vocab_size]
        self.probs_buffer = torch.zeros(
            (max_batch_size, max_draft_len, vocab_size),
            dtype=dtype,
            device=device,
        )

        # Track which slots have valid data
        self.valid_mask = torch.zeros(max_batch_size, dtype=torch.bool, device=device)

        # Track number of tokens stored per slot
        self.num_tokens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

        # Optional: request ID tracking for debugging
        self.request_ids: list[Optional[str]] = [None] * max_batch_size

    def store(
        self,
        req_idx: int,
        draft_probs: torch.Tensor,
        num_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Store draft probabilities for a request.

        Args:
            req_idx: Request slot index in batch
            draft_probs: Probability tensor [num_tokens, vocab_size]
            num_tokens: Number of draft tokens (inferred from draft_probs if None)
            request_id: Optional request ID for debugging
        """
        if num_tokens is None:
            num_tokens = draft_probs.shape[0]

        assert req_idx < self.max_batch_size
        assert num_tokens <= self.max_draft_len
        assert draft_probs.shape[-1] == self.vocab_size

        # Store probs in buffer
        self.probs_buffer[req_idx, :num_tokens] = draft_probs
        self.valid_mask[req_idx] = True
        self.num_tokens[req_idx] = num_tokens
        self.request_ids[req_idx] = request_id

    def store_batched(
        self,
        batch_size: int,
        draft_probs: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Store draft probabilities for entire batch.

        Args:
            batch_size: Number of requests in batch
            draft_probs: Probability tensor [batch_size, num_tokens, vocab_size]
            num_tokens: Number of draft tokens (same for all requests)
        """
        assert batch_size <= self.max_batch_size
        assert num_tokens <= self.max_draft_len

        self.probs_buffer[:batch_size, :num_tokens] = draft_probs
        self.valid_mask[:batch_size] = True
        self.num_tokens[:batch_size] = num_tokens

    def get(
        self,
        req_idx: int,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Retrieve draft probabilities for a request.

        Args:
            req_idx: Request slot index
            num_tokens: Number of tokens to retrieve (uses stored count if None)

        Returns:
            Probability tensor [num_tokens, vocab_size]
        """
        assert req_idx < self.max_batch_size

        if num_tokens is None:
            num_tokens = int(self.num_tokens[req_idx].item())

        return self.probs_buffer[req_idx, :num_tokens]

    def get_batched(
        self,
        batch_size: int,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Retrieve draft probabilities for entire batch.

        Args:
            batch_size: Number of requests
            num_tokens: Number of tokens to retrieve

        Returns:
            Probability tensor [batch_size, num_tokens, vocab_size]
        """
        assert batch_size <= self.max_batch_size

        if num_tokens is None:
            num_tokens = int(self.num_tokens[0].item())

        return self.probs_buffer[:batch_size, :num_tokens]

    def get_for_tokens(
        self,
        req_idx: int,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get probabilities for specific draft tokens.

        This is used in rejection sampling where we need P(draft_token).

        Args:
            req_idx: Request slot index
            draft_token_ids: [num_tokens] token IDs

        Returns:
            Probabilities [num_tokens] for each token
        """
        num_tokens = draft_token_ids.shape[0]
        probs = self.probs_buffer[req_idx, :num_tokens]
        # Gather the probability of each sampled token
        return probs.gather(dim=-1, index=draft_token_ids.unsqueeze(-1)).squeeze(-1)

    def get_batched_for_tokens(
        self,
        batch_size: int,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get probabilities for specific draft tokens (batched).

        Args:
            batch_size: Number of requests
            draft_token_ids: [batch_size, num_tokens] token IDs

        Returns:
            Probabilities [batch_size, num_tokens]
        """
        num_tokens = draft_token_ids.shape[1]
        probs = self.probs_buffer[:batch_size, :num_tokens]
        # Gather the probability of each sampled token
        token_ids_expanded = draft_token_ids.unsqueeze(-1)
        return probs.gather(dim=-1, index=token_ids_expanded).squeeze(-1)

    def clear(self, req_idx: int) -> None:
        """Clear slot when request finishes or is aborted.

        Args:
            req_idx: Request slot index to clear
        """
        self.valid_mask[req_idx] = False
        self.num_tokens[req_idx] = 0
        self.request_ids[req_idx] = None
        # Note: We don't zero the buffer for performance - just mark invalid

    def clear_batch(self, batch_size: int) -> None:
        """Clear all slots in batch.

        Args:
            batch_size: Number of slots to clear
        """
        self.valid_mask[:batch_size] = False
        self.num_tokens[:batch_size] = 0

    def is_valid(self, req_idx: int) -> bool:
        """Check if slot has valid data.

        Args:
            req_idx: Request slot index

        Returns:
            True if slot has valid draft probs
        """
        return bool(self.valid_mask[req_idx].item())

    def swap(self, i: int, j: int) -> None:
        """Swap two slots (for batch compaction).

        Args:
            i, j: Slot indices to swap
        """
        # Swap probs
        self.probs_buffer[i], self.probs_buffer[j] = (
            self.probs_buffer[j].clone(),
            self.probs_buffer[i].clone(),
        )
        # Swap metadata
        self.valid_mask[i], self.valid_mask[j] = (
            self.valid_mask[j].clone(),
            self.valid_mask[i].clone(),
        )
        self.num_tokens[i], self.num_tokens[j] = (
            self.num_tokens[j].clone(),
            self.num_tokens[i].clone(),
        )
        self.request_ids[i], self.request_ids[j] = (
            self.request_ids[j],
            self.request_ids[i],
        )

    @property
    def memory_bytes(self) -> int:
        """Get memory usage in bytes."""
        return self.probs_buffer.numel() * self.probs_buffer.element_size()

    def __repr__(self) -> str:
        valid_count = int(self.valid_mask.sum().item())
        mem_mb = self.memory_bytes / (1024 * 1024)
        return (
            f"DraftProbsBuffer(max_batch={self.max_batch_size}, "
            f"max_draft={self.max_draft_len}, vocab={self.vocab_size}, "
            f"valid={valid_count}, mem={mem_mb:.1f}MB)"
        )
