# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Drafting loops for speculative decoding - ported from TRT-LLM.

These wrappers wrap draft models and execute the drafting loop autoregressively,
enabling CUDA graph capture for the entire drafting process.

Based on: tensorrt_llm/_torch/speculative/drafting_loops.py
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.spec_tree_manager import SpecTreeManager


class BaseDraftingLoopWrapper(ABC, nn.Module):
    """Base class for drafting loop wrappers."""

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Execute the drafting loop.

        Returns:
            dict with 'new_draft_tokens' and optionally 'draft_logits'
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        top_k: int = 1,
    ) -> torch.Tensor:
        """Sample tokens from logits.

        Args:
            logits: [batch_size, vocab_size] or [batch_size * n, vocab_size]
            top_k: Number of tokens to sample per position

        Returns:
            Sampled token IDs [batch_size * top_k] or [batch_size * n * top_k]
        """
        raise NotImplementedError


class LinearDraftingLoopWrapper(BaseDraftingLoopWrapper):
    """Standard autoregressive drafting loop (1 token per step).

    This is the baseline that Eagle already uses in propose().
    """

    def __init__(
        self,
        max_draft_len: int,
        draft_model: nn.Module,
        device: torch.device,
    ):
        super().__init__()
        self.max_draft_len = max_draft_len
        self.draft_model = draft_model
        self.device = device

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Execute linear drafting loop."""
        batch_size = input_ids.shape[0]
        draft_tokens_list = []

        for _ in range(self.max_draft_len):
            # Run draft model
            logits = self.draft_model(
                input_ids=input_ids,
                positions=position_ids,
                hidden_states=hidden_states,
            )

            # Greedy sampling
            next_tokens = self.sample(logits)
            draft_tokens_list.append(next_tokens)

            # Update for next iteration
            input_ids = next_tokens
            position_ids = position_ids + 1

        # [batch_size, max_draft_len]
        draft_tokens = torch.stack(draft_tokens_list, dim=1)
        return {"new_draft_tokens": draft_tokens}

    def sample(
        self,
        logits: torch.Tensor,
        top_k: int = 1,
    ) -> torch.Tensor:
        """Greedy sampling (argmax)."""
        return logits.argmax(dim=-1)


class TreeDraftingLoopWrapper(BaseDraftingLoopWrapper):
    """Tree-based drafting loop for multi-candidate speculation.

    Uses SpecTreeManager for tree structure and top-k sampling
    at each node to generate multiple candidate paths.

    Ported from: tensorrt_llm/_torch/speculative/drafting_loops.py

    Key differences from TRT-LLM:
    - Uses Python fallback for extract_real_draft_tokens (no custom CUDA op)
    - Integrates with vLLM's TreeAttentionMetadataBuilder
    """

    def __init__(
        self,
        max_draft_len: int,
        max_total_draft_tokens: int,
        max_batch_size: int,
        draft_model: nn.Module,
        spec_tree_manager: SpecTreeManager,
        device: torch.device,
    ):
        super().__init__()
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_batch_size = max_batch_size
        self.draft_model = draft_model
        self.spec_tree_manager = spec_tree_manager
        self.device = device

        # Pre-allocate buffers for CUDA graph compatibility
        self.draft_tokens_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device=device,
        )
        self.position_ids_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Execute tree drafting loop.

        Args:
            input_ids: [batch_size] - input tokens
            position_ids: [batch_size] - position IDs
            hidden_states: [batch_size, hidden_size] - hidden states

        Returns:
            dict with:
            - new_draft_tokens: [batch_size, max_total_draft_tokens]
            - draft_logits: [batch_size, max_total_draft_tokens, vocab_size]
        """
        batch_size = input_ids.shape[0]
        max_top_k = self.spec_tree_manager.max_top_k

        # First layer: sample from initial logits
        logits = self.draft_model(
            input_ids=input_ids,
            positions=position_ids,
            hidden_states=hidden_states,
        )

        # Sample top-k tokens for first layer
        new_draft_tokens = self.sample(logits, top_k=max_top_k)
        self._extract_real_draft_tokens(
            cur_draft_idx=0,
            batch_size=batch_size,
            new_draft_tokens=new_draft_tokens,
        )

        # Prepare for subsequent layers
        self._prepare_for_generation(position_ids, batch_size)

        # Draft subsequent layers
        draft_logits = None
        for layer_idx in range(1, self.max_draft_len):
            # Run draft model on all tree nodes
            layer_input_ids = self.draft_tokens_buffer[:batch_size, :].reshape(-1)
            layer_position_ids = self.position_ids_buffer[:batch_size, :].reshape(-1)

            logits = self.draft_model(
                input_ids=layer_input_ids,
                positions=layer_position_ids,
            )

            # Sample top-k for this layer
            new_draft_tokens = self.sample(logits, top_k=max_top_k)
            self._extract_real_draft_tokens(
                cur_draft_idx=layer_idx,
                batch_size=batch_size,
                new_draft_tokens=new_draft_tokens,
            )

            if layer_idx == self.max_draft_len - 1:
                draft_logits = logits

        # Reshape outputs
        return_draft_tokens = self.draft_tokens_buffer[
            :batch_size, : self.max_total_draft_tokens
        ]

        return {
            "new_draft_tokens": return_draft_tokens,
            "draft_logits": draft_logits,
        }

    def sample(
        self,
        logits: torch.Tensor,
        top_k: int = 1,
    ) -> torch.Tensor:
        """Sample top-k tokens from logits.

        Args:
            logits: [n, vocab_size]
            top_k: Number of tokens to sample per position

        Returns:
            Token IDs [n * top_k]
        """
        if top_k == 1:
            return logits.argmax(dim=-1)

        # Top-k sampling
        indices = torch.topk(logits, k=top_k, dim=-1).indices  # [n, top_k]
        return indices.reshape(-1)

    def _extract_real_draft_tokens(
        self,
        cur_draft_idx: int,
        batch_size: int,
        new_draft_tokens: torch.Tensor,
    ):
        """Extract real draft tokens into buffer.

        Python fallback for TRT-LLM's extract_real_draft_tokens_op.
        """
        max_top_k = self.spec_tree_manager.max_top_k

        # Reshape tokens
        if cur_draft_idx == 0:
            # [batch_size * max_top_k] -> [batch_size, 1, max_top_k]
            tokens = new_draft_tokens.reshape(batch_size, 1, max_top_k)
        else:
            # [batch_size * (max_total + 1) * max_top_k] -> [batch_size, max_total+1, max_top_k]
            tokens = new_draft_tokens.reshape(
                batch_size, self.max_total_draft_tokens + 1, max_top_k
            )

        # Get gather indices for this layer
        gather_idx = self.spec_tree_manager.tokens_gather_idx_for_drafter_model[
            cur_draft_idx
        ]
        top_k_list = self.spec_tree_manager.top_k_list_cuda[cur_draft_idx]

        # Gather relevant tokens
        process_tokens = tokens[
            :, gather_idx, :
        ]  # [batch_size, num_process, max_top_k]
        process_tokens = process_tokens.reshape(-1, max_top_k)

        # Apply mask based on top_k list
        top_k_expanded = top_k_list.repeat(batch_size)
        col_indices = torch.arange(max_top_k, device=self.device).unsqueeze(0)
        mask = col_indices < top_k_expanded.unsqueeze(1)

        # Extract real tokens
        real_tokens = process_tokens[mask]
        real_tokens = real_tokens.reshape(batch_size, -1)

        # Write to buffer
        start = self.spec_tree_manager.draft_tokens_indices_cumsum[cur_draft_idx]
        end = self.spec_tree_manager.draft_tokens_indices_cumsum[cur_draft_idx + 1]
        self.draft_tokens_buffer[:batch_size, start:end] = real_tokens

    def _prepare_for_generation(
        self,
        position_ids: torch.Tensor,
        batch_size: int,
    ):
        """Prepare inputs for subsequent draft layers.

        Updates position_ids_buffer with tree position offsets.
        """
        # Get position offsets from tree manager
        pos_offsets = self.spec_tree_manager.spec_dec_position_offsets[0, 1:]

        # Compute position IDs for each tree node
        # position_ids: [batch_size] -> expand with tree offsets
        base_pos = position_ids.unsqueeze(1) + 1  # [batch_size, 1]
        self.position_ids_buffer[:batch_size, :-1] = (
            base_pos + pos_offsets.unsqueeze(0) - 1
        )
