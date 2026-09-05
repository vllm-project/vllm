# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import torch

Context: TypeAlias = tuple[int, ...]


@dataclass(frozen=True)
class WatermarkDetection:
    score: float
    p_value: float
    num_scored_tokens: int
    is_watermarked: bool


class WatermarkDetector(ABC):
    def __init__(
        self,
        context_width: int,
        p_value_threshold: float,
        deduplicate_contexts: bool = False,
    ) -> None:
        if context_width < 1:
            raise ValueError("context_width must be positive")
        if not 0 < p_value_threshold < 1:
            raise ValueError("p_value_threshold must be between 0 and 1")
        self.context_width = context_width
        self.p_value_threshold = p_value_threshold
        self.deduplicate_contexts = deduplicate_contexts

    def detect(self, token_ids: list[int]) -> WatermarkDetection:
        contexts, targets = self._prepare_inputs(token_ids)
        num_scored_tokens = len(targets)
        if num_scored_tokens == 0:
            return WatermarkDetection(0.0, 1.0, 0, False)

        token_scores = self._score_tokens(contexts, targets)
        score = self._aggregate_scores(token_scores)
        p_value = self._get_p_value(score, num_scored_tokens)
        return WatermarkDetection(
            score=score,
            p_value=p_value,
            num_scored_tokens=num_scored_tokens,
            is_watermarked=p_value <= self.p_value_threshold,
        )

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts: list[Context] = []
        targets = []
        seen_contexts: set[Context] = set()
        prefix = [-1] * self.context_width
        for token_id in token_ids:
            context = tuple(prefix[-self.context_width :])
            # Reusing a context reuses one keyed random vector, so the resulting
            # token scores are not independent evidence.
            if not self.deduplicate_contexts or not _is_repeated_context(
                context, seen_contexts
            ):
                contexts.append(context)
                targets.append(token_id)
            prefix.append(token_id)

        return (
            torch.tensor(contexts, dtype=torch.int64).reshape(-1, self.context_width),
            torch.tensor(targets, dtype=torch.int64),
        )

    @abstractmethod
    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        raise NotImplementedError

    @abstractmethod
    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        raise NotImplementedError


def _is_repeated_context(context: Context, seen_contexts: set[Context]) -> bool:
    if context in seen_contexts:
        return True
    seen_contexts.add(context)
    return False
