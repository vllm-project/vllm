# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch

RandomSampler: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class WatermarkSample:
    token_ids: torch.Tensor
    logits: torch.Tensor


class Watermarker(ABC):
    @property
    @abstractmethod
    def context_width(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sample: RandomSampler,
    ) -> WatermarkSample:
        raise NotImplementedError
