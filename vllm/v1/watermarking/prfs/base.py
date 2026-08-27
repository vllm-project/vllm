# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch

_UINT32_MASK = 2**32 - 1


class WatermarkPRF(ABC):
    @property
    @abstractmethod
    def max_context_width(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def uniform(self, contexts: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def uint32_to_uniform(values: torch.Tensor) -> torch.Tensor:
    mantissas = (values & _UINT32_MASK) >> 8
    return ((mantissas.to(torch.float64) + 1.0) / (2**24 + 1)).to(torch.float32)
