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
    """Set this flag to enable CPU prompt token IDs for your pooler."""
    requires_token_ids_gpu: bool = False
    """Set this flag to also enable model-device prompt token IDs."""

    def __or__(self, other: "PoolingParamsUpdate") -> "PoolingParamsUpdate":
        return PoolingParamsUpdate(
            requires_token_ids=self.requires_token_ids or other.requires_token_ids,
            requires_token_ids_gpu=(
                self.requires_token_ids_gpu or other.requires_token_ids_gpu
            ),
        )

    def apply(self, params: PoolingParams) -> None:
        params.requires_token_ids = self.requires_token_ids
        params.requires_token_ids_gpu = self.requires_token_ids_gpu


__all__ = ["ActivationFn", "ClassifierFn", "ProjectorFn", "PoolingParamsUpdate"]
