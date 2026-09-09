# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gumbel-max watermark generation and detection primitives."""

import math
import warnings

import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.detector import (
    WatermarkDetector,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsSpeculativeDecoding,
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

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
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

    def _try_sample_mixed(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor,
        random_sampler: RandomSampler,
    ) -> WatermarkSample | None:
        if type(self.prf) is not PhiloxPRF or logits.device.type != "cuda":
            return None

        from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

        token_ids = philox_gumbel_sample(
            logits,
            contexts,
            self.prf.key,
            skip_mask=skip_mask,
            expanded_idx_mapping=random_sampler.expanded_idx_mapping,
            temperatures=random_sampler.temperatures,
            seeds=random_sampler.seeds,
            positions=random_sampler.positions,
            use_fp64=random_sampler.use_fp64,
        )
        return WatermarkSample(token_ids, logits)


class DualKeyGumbelWatermarker(GumbelWatermarker, SupportsSpeculativeDecoding):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        alpha: float = 0.5,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.master_key = key
        self.prf_name = prf
        self.alpha = alpha
        super().__init__(derive_watermark_key(key, b"key_a"), context_width, prf)
        self.key_b_watermarker = GumbelWatermarker(
            derive_watermark_key(key, b"key_b"), context_width, prf
        )

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sample: RandomSampler,
    ) -> WatermarkSample:
        if self.alpha == 0:
            return super().sample(logits, contexts, random_sample)
        if self.alpha == 1:
            return self.key_b_watermarker.sample(logits, contexts, random_sample)

        key_a_sample = super().sample(logits, contexts, random_sample)
        key_b_sample = self.key_b_watermarker.sample(logits, contexts, random_sample)
        routing_logits = torch.tensor(
            [1 - self.alpha, self.alpha],
            dtype=torch.float32,
            device=logits.device,
        ).log()
        routing_logits = routing_logits.expand(logits.shape[0], -1)
        use_key_a = random_sample(routing_logits) == 0
        return WatermarkSample(
            torch.where(
                use_key_a,
                key_a_sample.token_ids,
                key_b_sample.token_ids,
            ),
            logits,
        )

    def create_draft_watermarker(self) -> Watermarker:
        return GumbelWatermarker(
            derive_watermark_key(self.master_key, b"key_a"),
            self.context_width,
            self.prf_name,
        )

    def create_target_watermarker(self) -> Watermarker:
        return GumbelWatermarker(
            derive_watermark_key(self.master_key, b"key_b"),
            self.context_width,
            self.prf_name,
        )


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


class DualKeyGumbelWatermarkDetector(GumbelWatermarkDetector):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        super().__init__(
            derive_watermark_key(key, b"key_a"),
            context_width,
            p_value_threshold,
            prf,
            deduplicate_contexts,
        )
        self.key_b_prf = create_prf(prf, derive_watermark_key(key, b"key_b"))

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            [
                -torch.log1p(
                    -prf.uniform(contexts, targets.unsqueeze(-1))
                    .squeeze(-1)
                    .to(torch.float64)
                )
                for prf in (self.prf, self.key_b_prf)
            ],
            dim=-1,
        )

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _gamma_survival_integer_shape(score * 2, num_scored_tokens * 2)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.mean(dim=-1).sum().item()


def _validate_context_width(context_width: int) -> None:
    if context_width < 1:
        raise ValueError("context_width must be positive")
    if context_width > 16:
        warnings.warn(
            "context_width values greater than 16 reduce robustness to edits because "
            "each changed token affects more subsequent watermark contexts",
            stacklevel=3,
        )
