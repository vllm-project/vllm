# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Binomial token scores and their detector, used by the chi-squared and
perplexity watermarks.

In the constrained optimization framework of Gloaguen et al. (2026),
https://arxiv.org/abs/2602.06754, the key and context give every token a score
drawn from Binomial(num_trials, 1/2), and sampling shifts probability toward
high-scoring tokens. Without the watermark the scores of the observed tokens are
independent, so detection tests their sum against
Binomial(num_trials * num_scored_tokens, 1/2).
"""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.gumbel import _validate_context_width
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import Watermarker

DEFAULT_NUM_TRIALS = 30


def binomial_thresholds(num_trials: int) -> torch.Tensor:
    """Return the Binomial(num_trials, 1/2) CDF at 0, ..., num_trials - 1.

    A token's score is the number of thresholds below its PRF uniform, which is
    Binomial(num_trials, 1/2) up to the uniform's 2**-24 resolution.
    """
    if not 1 <= num_trials <= 255:
        raise ValueError("num_trials must be between 1 and 255")
    cumulative = 0
    thresholds = []
    for count in range(num_trials):
        cumulative += math.comb(num_trials, count)
        thresholds.append(cumulative / 2**num_trials)
    return torch.tensor(thresholds, dtype=torch.float32)


def binomial_scores(uniforms: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Count the thresholds strictly below each uniform."""
    # torch.searchsorted starts OpenMP threads even for one text's tokens, and a
    # process that later forks, e.g. to start an engine, deadlocks in the child.
    return (uniforms.unsqueeze(-1) > thresholds).sum(dim=-1)


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


def copy_to_device(
    tensor: torch.Tensor,
    copies: dict[torch.device, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Return ``tensor`` on ``device``, copied once per device and cached in
    ``copies``. The copy comes from pinned memory, so it does not stall a
    sampling step."""
    copy = copies.get(device)
    if copy is None:
        copy = tensor
        if device.type != "cpu":
            copy = tensor.pin_memory().to(device, non_blocking=True)
        copies[device] = copy
    return copy


def _level_mass(
    log_probs: torch.Tensor, scores: torch.Tensor, num_levels: int
) -> torch.Tensor:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    log_probs = log_probs.masked_fill(log_probs.isnan(), float("-inf"))
    return log_probs.new_zeros(len(log_probs), num_levels).scatter_add_(
        1, scores, log_probs.exp()
    )


def _level_best(
    log_probs: torch.Tensor, scores: torch.Tensor, num_levels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    num_rows, vocab_size = log_probs.shape
    log_probs = log_probs.masked_fill(log_probs.isnan(), float("-inf"))
    best = log_probs.new_full((num_rows, num_levels), float("-inf")).scatter_reduce_(
        1, scores, log_probs, "amax"
    )
    vocabulary = torch.arange(vocab_size, device=log_probs.device).expand(num_rows, -1)
    candidates = torch.where(
        log_probs == best.gather(1, scores), vocabulary, vocab_size
    )
    token_ids = torch.full(
        (num_rows, num_levels), vocab_size, dtype=torch.int64, device=log_probs.device
    ).scatter_reduce_(1, scores, candidates, "amin")
    return best, token_ids.masked_fill(best == float("-inf"), 0)


def _least_probable_top_token(
    log_probs: torch.Tensor, scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    vocab_size = log_probs.shape[-1]
    log_probs = log_probs.masked_fill(log_probs.isnan(), float("-inf"))
    finite = log_probs > float("-inf")
    top_score = torch.where(finite, scores, -1).amax(dim=-1, keepdim=True)
    at_top = finite & (scores == top_score)
    least = torch.where(at_top, log_probs, float("inf")).amin(dim=-1, keepdim=True)
    vocabulary = torch.arange(vocab_size, device=log_probs.device)
    token_ids = torch.where(at_top & (log_probs == least), vocabulary, vocab_size)
    token_ids = token_ids.amin(dim=-1)
    return least.squeeze(-1), token_ids.masked_fill(token_ids == vocab_size, 0)


class BinomialScoreWatermarker(Watermarker):
    """Base for watermarks that favor tokens with high binomial scores.

    As in the reference implementation, a token is watermarked only once the
    output gives it a full context: the first ``context_width`` tokens of an
    output are sampled without the watermark, and the detector skips them.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        num_trials: int = DEFAULT_NUM_TRIALS,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        self.prf = prf
        self._context_width = context_width
        self.num_trials = num_trials
        self.thresholds = binomial_thresholds(num_trials)
        self._device_thresholds: dict[torch.device, torch.Tensor] = {}

    @property
    def context_width(self) -> int:
        return self._context_width

    def _thresholds_on(self, device: torch.device) -> torch.Tensor:
        return copy_to_device(self.thresholds, self._device_thresholds, device)

    def _skip_partial_contexts(
        self, contexts: torch.Tensor, skip_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Add the rows whose context reaches before the output, where it is
        padded with -1, to ``skip_mask``."""
        partial = (contexts < 0).any(dim=-1)
        return partial if skip_mask is None else skip_mask | partial

    def scores(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Score of every token in each row's context: [num_rows, vocab_size]."""
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        uniforms = self.prf.uniform(contexts, vocabulary)
        return binomial_scores(uniforms, self._thresholds_on(contexts.device))

    def level_mass(
        self, log_probs: torch.Tensor, contexts: torch.Tensor
    ) -> torch.Tensor:
        """Total probability of each row's tokens per score:
        [num_rows, num_trials + 1]."""
        scores = self.scores(contexts, log_probs.shape[-1])
        return _level_mass(log_probs, scores, self.num_trials + 1)

    def level_best(
        self, log_probs: torch.Tensor, contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Largest log-probability among each row's tokens per score (-inf
        without tokens) and the lowest token id attaining it (0 without tokens),
        each [num_rows, num_trials + 1]."""
        scores = self.scores(contexts, log_probs.shape[-1])
        return _level_best(log_probs, scores, self.num_trials + 1)

    def level_best_and_least_probable_top(
        self, log_probs: torch.Tensor, contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``level_best``, plus the log-probability and the lowest token id of
        the least probable token among those with the highest score, each
        [num_rows]."""
        scores = self.scores(contexts, log_probs.shape[-1])
        return (
            *_level_best(log_probs, scores, self.num_trials + 1),
            *_least_probable_top_token(log_probs, scores),
        )

    def add_level_bias(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        level_bias: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add ``level_bias[row, score]`` to every token's logit.

        Returns float32 logits; rows in ``skip_mask`` are returned unchanged.
        """
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.binomial_watermark import (
                philox_level_bias,
            )

            return philox_level_bias(
                logits,
                contexts,
                self.prf.key,
                self._thresholds_on(logits.device),
                level_bias,
                skip_mask=skip_mask,
            )
        bias = level_bias.gather(1, self.scores(contexts, logits.shape[-1]))
        if skip_mask is not None:
            bias = bias.masked_fill(skip_mask.unsqueeze(-1), 0.0)
        return logits.float() + bias


class BinomialScoreWatermarkDetector(WatermarkDetector):
    """Detect a chi-squared or perplexity watermark with a binomial test.

    Under the null hypothesis each scored token's score is
    Binomial(num_trials, 1/2), independently, so the summed score is
    Binomial(num_trials * num_scored_tokens, 1/2). The first ``context_width``
    tokens are not watermarked, so they are not scored.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        num_trials: int = DEFAULT_NUM_TRIALS,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = prf
        self.num_trials = num_trials
        self.thresholds = binomial_thresholds(num_trials)

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts, targets = super()._prepare_inputs(token_ids)
        full = (contexts >= 0).all(dim=-1)
        return contexts[full], targets[full]

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        uniforms = self.prf.uniform(contexts, targets.unsqueeze(-1)).squeeze(-1)
        thresholds = self.thresholds.to(uniforms.device)
        return binomial_scores(uniforms, thresholds).to(torch.float64)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _binomial_survival(
            round(score), self.num_trials * num_scored_tokens, 0.5
        )
