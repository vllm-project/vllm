# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text watermark generation primitives."""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.prfs import PhiloxPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)


class SynthIDWatermarker(Watermarker):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 32,
        prf: WatermarkPRFName = "philox",
    ) -> None:
        if context_width < 1:
            raise ValueError("context_width must be positive")
        if not 1 <= depth <= 32:
            raise ValueError("SynthID depth must be between 1 and 32")
        selected_prf = create_prf(prf, key)
        if not isinstance(selected_prf, PhiloxPRF):
            raise ValueError("SynthID requires the Philox PRF")
        self.prf = selected_prf
        self._context_width = context_width
        self.depth = depth

    @property
    def context_width(self) -> int:
        return self._context_width

    @staticmethod
    def _update_scores(
        scores: torch.Tensor,
        words: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        """Reweight [batch, vocab] scores using bits from Philox words.

        Each Philox word provides up to 32 binary g-values for a candidate token.
        The SynthID reweighting is applied sequentially across these bits.

        Extract each bit plane lazily instead of materializing a
        [batch, vocab, depth] tensor. This keeps temporary memory proportional to
        [batch, vocab] while preserving the same depth-ordered reweighting.
        """
        probs = torch.softmax(scores, dim=1)

        for bit in range(depth):
            # Extract one binary g-value per candidate from the word: [B, V] -> [B, V].
            g = ((words >> bit) & 1).to(scores.dtype)
            # Sum each row's probability mass on g=1 candidates: [B, V] -> [B, 1].
            g_mass = (g * probs).sum(dim=1, keepdim=True)
            # Apply one SynthID reweighting step.
            probs = probs * (1 + g - g_mass)

        log_probs = torch.log(probs)
        return torch.where(
            torch.isfinite(log_probs),
            log_probs,
            torch.finfo(log_probs.dtype).min,
        )

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight logits using bits of one Philox word per candidate."""
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)

        # [B, context_width] and [V] produce one Philox word per candidate:
        # [B, V]. The individual SynthID bits are extracted lazily during
        # reweighting to avoid a [B, V, depth] intermediate tensor.
        words = self.prf.uint32(contexts, vocabulary)

        return self._update_scores(logits, words, self.depth)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("SynthID requires a random sampler")

        watermarked_logits = self.watermark_logits(logits, contexts)
        if skip_mask is not None:
            watermarked_logits = torch.where(
                skip_mask.unsqueeze(-1), logits, watermarked_logits
            )
        return WatermarkSample(
            token_ids=random_sampler(watermarked_logits),
            logits=watermarked_logits,
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("SynthID requires a random sampler")


def _binomial_survival(hits: int, trials: int) -> float:
    """Return P[Binomial(trials, 0.5) >= hits]."""
    if hits <= 0:
        return 1.0
    if hits > trials:
        return 0.0
    if hits <= trials // 2:
        return 1.0 - _binomial_survival(trials - hits + 1, trials)

    log_term = (
        math.lgamma(trials + 1)
        - math.lgamma(hits + 1)
        - math.lgamma(trials - hits + 1)
        - trials * math.log(2)
    )
    term = math.exp(log_term)
    total = term
    for count in range(hits, trials):
        term *= (trials - count) / (count + 1)
        total += term
    return min(1.0, total)


class SynthIDWatermarkDetector(WatermarkDetector):
    """Detect a surplus of Philox-derived SynthID bits in generated tokens.

    The score is the mean g value. The p-value uses an ideal-PRF
    Binomial(num_scored_tokens * depth, 0.5) null distribution.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 32,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        if not 1 <= depth <= 32:
            raise ValueError("SynthID depth must be between 1 and 32")
        selected_prf = create_prf(prf, key)
        if not isinstance(selected_prf, PhiloxPRF):
            raise ValueError("SynthID requires the Philox PRF")
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = selected_prf
        self.depth = depth

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # [N, context_width] and [N, 1] produce one word per token: [N].
        words = self.prf.uint32(contexts, targets.unsqueeze(-1)).squeeze(-1)
        bit_positions = torch.arange(self.depth, device=words.device)
        # [N, 1] and [depth] broadcast to binary g values shaped [N, depth].
        return (words.unsqueeze(-1) >> bit_positions) & 1

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.to(torch.float64).mean().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        trials = num_scored_tokens * self.depth
        return _binomial_survival(round(score * trials), trials)
