# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

from vllm.pooling_params import PoolingParams

_T = TypeVar("_T", bound=torch.Tensor | list[torch.Tensor])

ProjectorFn = Callable[[torch.Tensor], torch.Tensor]
ClassifierFn = Callable[[torch.Tensor], torch.Tensor]
ActivationFn = Callable[[_T], _T]


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


__all__ = ["ActivationFn", "ClassifierFn", "ProjectorFn", "PoolingParamsUpdate"]
