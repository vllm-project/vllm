# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text tournament watermark generation and detection primitives."""

import math

import torch

from vllm.config.watermarking import derive_watermark_key
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)


def synthid_text_logits(
    logits: torch.Tensor, contexts: torch.Tensor, prf: PhiloxPRF, depth: int
) -> torch.Tensor:
    """Return the log-probabilities after `depth` two-candidate tournament layers."""
    vocabulary = torch.arange(logits.shape[-1], device=logits.device)
    words = prf.words(contexts, vocabulary)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    for layer in range(depth):
        g_values = ((words >> layer) & 1).to(torch.float32)
        zero_mass = (probs * (1 - g_values)).sum(dim=-1, keepdim=True)
        one_mass = (probs * g_values).sum(dim=-1, keepdim=True)
        # 1 + g - G with G = one_mass / (zero_mass + one_mass), written so that a
        # G that rounds to 1 cannot make the g = 0 probabilities negative.
        probs = probs * (g_values + zero_mass / (zero_mass + one_mass))
    return probs.log()


def _binomial_survival(successes: int, trials: int) -> float:
    """Return P(X >= successes) for X ~ Binomial(trials, 1/2)."""
    counts = torch.arange(successes, trials + 1, dtype=torch.float64)
    log_pmf = (
        math.lgamma(trials + 1)
        - torch.lgamma(counts + 1)
        - torch.lgamma(trials - counts + 1)
        - trials * math.log(2)
    )
    return min(1.0, torch.logsumexp(log_pmf, dim=0).exp().item())


def _validate_depth(depth: int) -> None:
    if not 1 <= depth <= 32:
        raise ValueError("depth must be between 1 and 32")


class SynthIDTextWatermarker(Watermarker):
    def __init__(self, key: int, context_width: int = 4, depth: int = 30) -> None:
        _validate_depth(depth)
        self.prf = PhiloxPRF(derive_watermark_key(key, b"synthid_text"))
        self.depth = depth
        self._context_width = context_width

    @property
    def context_width(self) -> int:
        return self._context_width

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("SynthID-Text sampling requires a random sampler")
        if skip_mask is not None and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.synthid_text import synthid_text_sample

            token_ids = synthid_text_sample(
                logits,
                contexts,
                self.prf.key,
                self.depth,
                skip_mask,
                random_sampler.expanded_idx_mapping,
                random_sampler.temperatures,
                random_sampler.seeds,
                random_sampler.positions,
                use_fp64=random_sampler.use_fp64,
            )
            return WatermarkSample(token_ids, logits)

        watermarked = synthid_text_logits(logits, contexts, self.prf, self.depth)
        if skip_mask is not None:
            watermarked = torch.where(skip_mask.unsqueeze(-1), logits, watermarked)
        return WatermarkSample(random_sampler(watermarked), logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("SynthID-Text sampling requires a random sampler")


class SynthIDTextWatermarkDetector(WatermarkDetector):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        deduplicate_contexts: bool = True,
        depth: int = 30,
    ) -> None:
        _validate_depth(depth)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = PhiloxPRF(derive_watermark_key(key, b"synthid_text"))
        self.depth = depth

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        words = self.prf.words(contexts, targets.unsqueeze(-1))
        return ((words >> torch.arange(self.depth)) & 1).sum(dim=-1)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return float(token_scores.sum())

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _binomial_survival(round(score), self.depth * num_scored_tokens)
