# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.pooling_params import PoolingParams


@dataclass
class PoolingCursor:
    index: list[int]
    first: torch.Tensor  # GPU Tensor
    last: torch.Tensor  # GPU Tensor
    prompt_lens: torch.Tensor  # CPU Tensor
    num_scheduled_tokens: torch.Tensor  # CPU Tensor

    def __getitem__(self, indices: slice):
        return PoolingCursor(
            index=self.index[indices],
            first=self.first[indices],
            last=self.last[indices],
            prompt_lens=self.prompt_lens[indices],
            num_scheduled_tokens=self.num_scheduled_tokens[indices],
        )

    def is_partial_prefill(self):
        return len(
            self.prompt_lens == len(self.num_scheduled_tokens)
        ) and not torch.all(self.prompt_lens == self.num_scheduled_tokens)


@dataclass
class PoolingMetadata:
    """Tensors for pooling."""
    prompt_lens: torch.Tensor  # CPU Tensor
    prompt_token_ids: Optional[torch.Tensor]
    pooling_params: list[PoolingParams]
    pooling_cursor: Optional[PoolingCursor] = None

    def __getitem__(self, indices: slice):
        return PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None if self.prompt_token_ids is None else
            self.prompt_token_ids[indices],
            pooling_params=self.pooling_params[indices],
            pooling_cursor=None
            if self.pooling_cursor is None else self.pooling_cursor[indices],
        )

    def build_pooling_cursor(self, num_scheduled_tokens: list[int],
                             device: torch.device):
        self.pooling_cursor = build_pooling_cursor(num_scheduled_tokens,
                                                   self.prompt_lens.tolist(),
                                                   device)


def build_pooling_cursor(num_scheduled_tokens: list[int],
                         prompt_lens: list[int], device: torch.device):
    n_seq = len(num_scheduled_tokens)
    index = list(range(n_seq))
    prompt_lens = torch.tensor(prompt_lens, device="cpu")
    num_scheduled_tokens = torch.tensor(num_scheduled_tokens, device="cpu")
    cumsum = torch.zeros(n_seq + 1, dtype=torch.int64, device="cpu")
    torch.cumsum(num_scheduled_tokens, dim=0, out=cumsum[1:])
    first = torch.tensor(cumsum[:n_seq], device="cpu").to(device,
                                                          non_blocking=True)
    last = torch.tensor(cumsum[1:] - 1, device="cpu").to(device,
                                                         non_blocking=True)
    return PoolingCursor(index=index,
                         first=first,
                         last=last,
                         prompt_lens=prompt_lens,
                         num_scheduled_tokens=num_scheduled_tokens)
