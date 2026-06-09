# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SLEM (String-Level Exact Match) speculative decoding for heterogeneous vocabs.

Implements Algorithm 2 from:
  "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms
   for Heterogeneous Vocabularies" — Timor et al., ICML 2025.
  https://arxiv.org/abs/2502.05202
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class SlemMapper:
    """String-Level Exact Match mapper for heterogeneous-vocabulary spec decode.

    The draft model generates tokens using its full vocabulary (no intersection
    constraint). Tokens are decoded to text and re-tokenized for the target
    model. Verification uses exact matching at the token level.
    """

    def __init__(
        self,
        target_tokenizer,
        draft_tokenizer,
        device: torch.device,
    ):
        self.target_tokenizer = target_tokenizer
        self.draft_tokenizer = draft_tokenizer
        self.device = device

        self.target_eos_id: int = getattr(target_tokenizer, "eos_token_id", 0) or 0
        self.draft_eos_id: int = getattr(draft_tokenizer, "eos_token_id", 0) or 0

        logger.info(
            "SlemMapper initialized: target_vocab=%d, draft_vocab=%d",
            len(target_tokenizer.get_vocab()),
            len(draft_tokenizer.get_vocab()),
        )

    def draft_to_target_candidates(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode draft tokens to text, re-tokenize with target tokenizer.

        Args:
            draft_token_ids: [batch_size, max_draft_len] draft token IDs
            num_draft_tokens: [batch_size] valid draft token count per sequence

        Returns:
            target_candidate_ids: [batch_size, max_target_len] padded target IDs
            num_target_tokens: [batch_size] valid target token count per sequence
        """
        batch_size = draft_token_ids.shape[0]
        device = draft_token_ids.device

        all_target_ids: list[list[int]] = []
        max_target_len = 0

        for i in range(batch_size):
            n = int(num_draft_tokens[i].item())
            draft_ids_i = draft_token_ids[i, :n].tolist()

            text = self.draft_tokenizer.decode(draft_ids_i, skip_special_tokens=True)

            target_ids_i = self.target_tokenizer.encode(text, add_special_tokens=False)

            all_target_ids.append(target_ids_i)
            max_target_len = max(max_target_len, len(target_ids_i))

        if max_target_len == 0:
            max_target_len = 1

        target_candidate_ids = torch.full(
            (batch_size, max_target_len),
            self.target_eos_id,
            dtype=torch.long,
            device=device,
        )
        num_target_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, ids in enumerate(all_target_ids):
            length = len(ids)
            if length > 0:
                target_candidate_ids[i, :length] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
            num_target_tokens[i] = length

        return target_candidate_ids, num_target_tokens

    def verify_and_accept(
        self,
        target_candidate_ids: torch.Tensor,
        target_sampled_ids: torch.Tensor,
        num_target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact-match verification: find first mismatch, accept prefix + bonus.

        Args:
            target_candidate_ids: [batch_size, M] proposed target tokens
            target_sampled_ids: [batch_size, M+1] target model's greedy output
            num_target_tokens: [batch_size] valid candidate lengths

        Returns:
            accepted_ids: [batch_size, max_accepted] accepted target tokens
                (matched prefix + bonus token)
            accepted_lens: [batch_size] number of accepted tokens per sequence
        """
        batch_size, max_candidates = target_candidate_ids.shape
        device = target_candidate_ids.device

        # Compare candidates vs target samples at positions 0..M-1
        matches = target_candidate_ids == target_sampled_ids[:, :max_candidates]

        # Mask positions beyond valid length (treat as non-matching to stop)
        position_indices = torch.arange(max_candidates, device=device).unsqueeze(0)
        valid_mask = position_indices < num_target_tokens.unsqueeze(1)
        matches = matches & valid_mask

        # Count consecutive matches from position 0
        # A position matches only if all prior positions also match
        cumulative_match = matches.cumprod(dim=1)
        n_matches = cumulative_match.sum(dim=1)  # [batch_size]

        # Build output: matched prefix + bonus token
        max_accepted = int(n_matches.max().item()) + 1
        accepted_ids = torch.full(
            (batch_size, max_accepted),
            self.target_eos_id,
            dtype=torch.long,
            device=device,
        )
        accepted_lens = n_matches + 1  # +1 for bonus token

        for i in range(batch_size):
            n = int(n_matches[i].item())
            if n > 0:
                accepted_ids[i, :n] = target_candidate_ids[i, :n]
            # Bonus token: target's sample at the first mismatch position
            accepted_ids[i, n] = target_sampled_ids[i, n]

        return accepted_ids, accepted_lens

    def target_to_draft_input(
        self,
        accepted_target_ids: torch.Tensor,
        accepted_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert accepted target tokens back to draft vocab for continuation.

        Decodes accepted target tokens to text, re-encodes with draft tokenizer.

        Args:
            accepted_target_ids: [batch_size, max_accepted] accepted target tokens
            accepted_lens: [batch_size] valid lengths

        Returns:
            draft_input_ids: [batch_size, max_draft_len] draft token IDs
            num_draft_ids: [batch_size] valid lengths
        """
        batch_size = accepted_target_ids.shape[0]
        device = accepted_target_ids.device

        all_draft_ids: list[list[int]] = []
        max_draft_len = 0

        for i in range(batch_size):
            length = int(accepted_lens[i].item())
            target_ids_i = accepted_target_ids[i, :length].tolist()

            text = self.target_tokenizer.decode(target_ids_i, skip_special_tokens=True)

            draft_ids_i = self.draft_tokenizer.encode(text, add_special_tokens=False)

            all_draft_ids.append(draft_ids_i)
            max_draft_len = max(max_draft_len, len(draft_ids_i))

        if max_draft_len == 0:
            max_draft_len = 1

        draft_input_ids = torch.full(
            (batch_size, max_draft_len),
            self.draft_eos_id,
            dtype=torch.long,
            device=device,
        )
        num_draft_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, ids in enumerate(all_draft_ids):
            length = len(ids)
            if length > 0:
                draft_input_ids[i, :length] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
            num_draft_ids[i] = length

        return draft_input_ids, num_draft_ids
