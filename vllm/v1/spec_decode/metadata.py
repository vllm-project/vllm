# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


@dataclass
class MultiLayerEagleMetadata:
    # [batch_size]
    cached_len: torch.Tensor | None = None
    # [batch_size, layer_num]
    cached_token_ids: torch.Tensor | None = None
    # [batch_size, layer_num, hidden_size]
    cached_hidden_states: torch.Tensor | None = None
    # [batch_size, layer_num]
    cached_slot_mappings: torch.Tensor | None = None
    # [batch_size, layer_num]
    cached_positions: torch.Tensor | None = None

    @classmethod
    def make_dummy(
        cls,
        layer_num: int,
        hidden_size: int,
        device: torch.device,
    ) -> "MultiLayerEagleMetadata":
        cached_len = torch.zeros((1), dtype=torch.int64, device=device)
        cached_token_ids = torch.zeros((1, layer_num), dtype=torch.int32, device=device)
        cached_hidden_states = torch.zeros(
            (1, layer_num, hidden_size), dtype=torch.float32, device=device
        )
        cached_slot_mappings = torch.zeros(
            (1, layer_num), dtype=torch.int64, device=device
        )
        cached_positions = torch.zeros((1, layer_num), dtype=torch.int64, device=device)
        return cls(
            cached_len=cached_len,
            cached_token_ids=cached_token_ids,
            cached_hidden_states=cached_hidden_states,
            cached_slot_mappings=cached_slot_mappings,
            cached_positions=cached_positions,
        )
