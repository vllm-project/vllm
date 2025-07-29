# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.pooling_params import PoolingParams


@dataclass
class PoolingMetadata:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_token_ids: Optional[torch.Tensor]
    pooling_params: list[PoolingParams]

    def __getitem__(self, indices: slice):
        return PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None if self.prompt_token_ids is None else
            self.prompt_token_ids[indices],
            pooling_params=self.pooling_params[indices],
        )
