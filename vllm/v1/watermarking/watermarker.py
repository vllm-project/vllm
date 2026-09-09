# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


@dataclass(frozen=True)
class WatermarkSample:
    token_ids: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class RandomSampler:
    expanded_idx_mapping: torch.Tensor
    temperatures: torch.Tensor
    seeds: torch.Tensor
    positions: torch.Tensor
    use_fp64: bool = False

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return gumbel_sample(
            logits,
            self.expanded_idx_mapping,
            self.temperatures,
            self.seeds,
            self.positions,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=self.use_fp64,
        )


class Watermarker(ABC):
    @property
    @abstractmethod
    def context_width(self) -> int:
        raise NotImplementedError

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if skip_mask is not None:
            if random_sampler is None:
                raise ValueError("skip_mask requires a random sampler")
            mixed = self._try_sample_mixed(logits, contexts, skip_mask, random_sampler)
            if mixed is not None:
                return mixed

        watermarked = self._sample_watermarked(logits, contexts)
        if skip_mask is None:
            return watermarked

        assert random_sampler is not None
        token_ids = torch.where(
            skip_mask, random_sampler(logits), watermarked.token_ids
        )
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
        random_sampler: RandomSampler,
    ) -> WatermarkSample | None:
        return None


class SpeculativeVerification(str, Enum):
    STANDARD = "standard"
    WATERMARKED = "watermarked"


class AcceptanceRandomness(str, Enum):
    RANDOM = "random"
    KEYED = "keyed"


@runtime_checkable
class SupportsSpeculativeDecoding(Protocol):
    speculative_verification: SpeculativeVerification
    acceptance_randomness: AcceptanceRandomness

    def create_draft_watermarker(self) -> Watermarker: ...
