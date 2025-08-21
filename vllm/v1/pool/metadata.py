# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.pooling_params import PoolingParams
from vllm.utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


@dataclass
class PoolingCursor:
    index: list[int]
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor

    def __getitem__(self, indices: slice):
        return PoolingCursor(
            index=self.index[indices],
            first_token_indices_gpu=self.first_token_indices_gpu[indices],
            last_token_indices_gpu=self.last_token_indices_gpu[indices],
            prompt_lens_cpu=self.prompt_lens_cpu[indices],
            num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],
        )

    def is_partial_prefill(self):
        return not torch.all(
            self.prompt_lens_cpu == self.num_scheduled_tokens_cpu)


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
                                                   self.prompt_lens, device)


def build_pooling_cursor(num_scheduled_tokens: list[int],
                         prompt_lens: torch.Tensor, device: torch.device):
    assert len(prompt_lens) == len(num_scheduled_tokens)

    n_seq = len(num_scheduled_tokens)
    index = list(range(n_seq))
    num_scheduled_tokens = torch.tensor(num_scheduled_tokens, device="cpu")
    cumsum = torch.zeros(n_seq + 1,
                         dtype=torch.int64,
                         pin_memory=pin_memory,
                         device="cpu")
    torch.cumsum(num_scheduled_tokens, dim=0, out=cumsum[1:])
    cumsum = cumsum.to(device, non_blocking=True)
    return PoolingCursor(index=index,
                         first_token_indices_gpu=cumsum[:n_seq],
                         last_token_indices_gpu=cumsum[1:] - 1,
                         prompt_lens_cpu=prompt_lens,
                         num_scheduled_tokens_cpu=num_scheduled_tokens)
