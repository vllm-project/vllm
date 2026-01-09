# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.pooling_params import PoolingParams

ClassifierFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PoolingParamsUpdate:
    requires_token_ids: bool = False
    """Set this flag to enable `get_prompt_token_ids` for your pooler."""

    def __or__(self, other: "PoolingParamsUpdate") -> "PoolingParamsUpdate":
        return PoolingParamsUpdate(
            requires_token_ids=self.requires_token_ids or other.requires_token_ids,
        )

    def apply(self, params: PoolingParams) -> None:
        params.requires_token_ids = self.requires_token_ids


__all__ = ["ClassifierFn", "PoolingParamsUpdate"]
