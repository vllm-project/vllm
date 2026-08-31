# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gumbel-max watermark generation and detection primitives."""

import math
import warnings

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.detector import (
    WatermarkDetector,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)


def _gamma_survival_integer_shape(score: float, shape: int) -> float:
    """Return the survival probability of a Gamma(shape, 1) variable."""
    if shape <= 0 or score <= 0:
        return 1.0
    log_score = math.log(score)
    log_term = 0.0
    max_log_term = 0.0
    log_terms = [log_term]
    for index in range(1, shape):
        log_term += log_score - math.log(index)
        log_terms.append(log_term)
        max_log_term = max(max_log_term, log_term)
    log_sum = max_log_term + math.log(
        sum(math.exp(term - max_log_term) for term in log_terms)
    )
    return min(1.0, math.exp(min(0.0, -score + log_sum)))


class GumbelWatermarker(Watermarker):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        self.prf = prf
        self._context_width = context_width

    @property
    def context_width(self) -> int:
        return self._context_width

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        _random_sample: RandomSampler,
    ) -> WatermarkSample:
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

            return WatermarkSample(
                philox_gumbel_sample(logits, contexts, self.prf.key), logits
            )
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        uniforms = self.prf.uniform(contexts, vocabulary)
        uniforms = uniforms.clamp_min(torch.finfo(torch.float32).tiny)
        noise = -torch.log(-torch.log(uniforms))
        return WatermarkSample(torch.argmax(logits + noise, dim=-1), logits)


class GumbelWatermarkDetector(WatermarkDetector):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = prf

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        uniforms = self.prf.uniform(contexts, targets.unsqueeze(-1)).squeeze(-1)
        return -torch.log1p(-uniforms.to(torch.float64))

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _gamma_survival_integer_shape(score, num_scored_tokens)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()


def _validate_context_width(context_width: int) -> None:
    if context_width < 1:
        raise ValueError("context_width must be positive")
    if context_width > 16:
        warnings.warn(
            "context_width values greater than 16 reduce robustness to edits because "
            "each changed token affects more subsequent watermark contexts",
            stacklevel=3,
        )
