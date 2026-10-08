# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Red-green watermark generation and detection primitives."""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.gumbel import _validate_context_width
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)


def _binomial_survival(successes: int, trials: int, probability: float) -> float:
    """Return the survival probability of a Binomial random variable."""
    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(count + 1)
        - math.lgamma(trials - count + 1)
        + count * log_p
        + (trials - count) * log_q
        for count in range(successes, trials + 1)
    ]
    max_log_term = max(log_terms)
    log_sum = max_log_term + math.log(
        sum(math.exp(term - max_log_term) for term in log_terms)
    )
    return min(1.0, math.exp(log_sum))


def _validate_green_list(gamma: float) -> None:
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")


class RedGreenWatermarker(Watermarker):
    """Red-green list watermark of Kirchenbauer et al. (2023).

    The key and context select a green list holding the tokens whose PRF value is below
    ``gamma``. Sampling adds ``delta`` to the green tokens' logits, after
    temperature scaling and top-k/top-p filtering, and draws from the
    result with vLLM's random sampler.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        delta: float = 2.0,
        gamma: float = 0.25,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        if delta <= 0:
            raise ValueError("delta must be positive")
        _validate_green_list(gamma)
        self.prf = prf
        self._context_width = context_width
        self.delta = delta
        self.gamma = gamma

    @property
    def context_width(self) -> int:
        return self._context_width

    def green_list(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Boolean mask of each context's green tokens: [num_rows, vocab_size]."""
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        return self.prf.uniform(contexts, vocabulary) < self.gamma

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return float32 logits with ``delta`` added to the green tokens of
        every row not in ``skip_mask``."""
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.watermark import philox_green_bias

            return philox_green_bias(
                logits,
                contexts,
                self.prf.key,
                self.delta,
                self.gamma,
                skip_mask=skip_mask,
            )
        green = self.green_list(contexts, logits.shape[-1])
        if skip_mask is not None:
            green &= ~skip_mask.unsqueeze(-1)
        return logits.float() + self.delta * green

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("red-green watermarking requires a random sampler")
        if (
            isinstance(random_sampler, RandomSampler)
            and type(self.prf) is PhiloxPRF
            and logits.device.type == "cuda"
        ):
            from vllm.v1.worker.gpu.sample.watermark import philox_red_green_sample

            assert not random_sampler.is_drafting
            assert random_sampler.logits_cache is None
            token_ids = philox_red_green_sample(
                logits,
                contexts,
                self.prf.key,
                self.delta,
                self.gamma,
                random_sampler.expanded_idx_mapping,
                random_sampler.temperatures,
                random_sampler.seeds,
                random_sampler.positions,
                skip_mask=skip_mask,
                use_fp64=random_sampler.use_fp64,
            )
        else:
            token_ids = random_sampler(
                self.watermark_logits(logits, contexts, skip_mask)
            )
        # The logits before the watermark, as for Gumbel, so processed logprobs
        # do not reveal the green list.
        return WatermarkSample(token_ids, logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("red-green watermarking requires a random sampler")


class RedGreenWatermarkDetector(WatermarkDetector):
    """Detect a red-green watermark with a one-sided binomial test.

    Under the null hypothesis each scored token is green with probability
    ``gamma``, independently, so the number of green tokens is
    Binomial(num_scored_tokens, gamma).
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        gamma: float = 0.25,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        _validate_green_list(gamma)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = prf
        self.gamma = gamma

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        uniforms = self.prf.uniform(contexts, targets.unsqueeze(-1)).squeeze(-1)
        return (uniforms < self.gamma).to(torch.float64)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _binomial_survival(round(score), num_scored_tokens, self.gamma)
