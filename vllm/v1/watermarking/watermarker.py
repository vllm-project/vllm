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


@dataclass(frozen=True)
class RandomSamplingState:
    expanded_idx_mapping: torch.Tensor
    temperatures: torch.Tensor
    seeds: torch.Tensor
    positions: torch.Tensor
    use_fp64: bool = False


class Watermarker(ABC):
    @property
    @abstractmethod
    def context_width(self) -> int:
        raise NotImplementedError

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sample: RandomSampler,
        skip_mask: torch.Tensor | None = None,
        sampling_state: RandomSamplingState | None = None,
    ) -> WatermarkSample:
        if skip_mask is not None:
            mixed = self._try_sample_mixed(logits, contexts, skip_mask, sampling_state)
            if mixed is not None:
                return mixed

        watermarked = self._sample_watermarked(logits, contexts)
        if skip_mask is None:
            return watermarked

        token_ids = torch.where(skip_mask, random_sample(logits), watermarked.token_ids)
        output_logits = watermarked.logits
        if output_logits is not logits:
            output_logits = torch.where(skip_mask.unsqueeze(-1), logits, output_logits)
        return WatermarkSample(token_ids, output_logits)

    @abstractmethod
    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise NotImplementedError

    def _try_sample_mixed(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor,
        sampling_state: RandomSamplingState | None,
    ) -> WatermarkSample | None:
        return None
