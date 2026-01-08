# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata

PoolingFn = Callable[
    [torch.Tensor | list[torch.Tensor], PoolingMetadata],
    torch.Tensor | list[torch.Tensor],
]
ClassifierFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PoolingParamsUpdate:
    requires_token_ids: bool = False
    """Set this flag to enable `get_prompt_token_ids` for your pooler."""

    def apply(self, params: PoolingParams) -> None:
        params.requires_token_ids = self.requires_token_ids
