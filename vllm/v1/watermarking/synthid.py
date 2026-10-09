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
    _validate_context_width,
)

# High stream word ("SYNT") keeps SynthID-Text off Gumbel-max's Philox streams.
_STREAM_DOMAIN = 0x53594E54 << 32


class SynthIDWatermarker(Watermarker):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 32,
        prf: WatermarkPRFName = "philox",
    ) -> None:
        _validate_context_width(context_width)
        if depth < 1:
            raise ValueError("SynthID-Text depth must be positive")
        selected_prf = create_prf(prf, key)
        if not isinstance(selected_prf, PhiloxPRF):
            raise ValueError("SynthID-Text requires the Philox PRF")
        self.prf = selected_prf
        self._context_width = context_width
        self.depth = depth

    @property
    def context_width(self) -> int:
        return self._context_width

    @staticmethod
    def _update_probs(
        probs: torch.Tensor,
        words: torch.Tensor,
        num_bits: int,
    ) -> torch.Tensor:
        """Apply up to 32 SynthID-Text tournament layers from one Philox word."""
        for bit in range(num_bits):
            # [B, V] -> [B, V]
            g = ((words >> bit) & 1).to(probs.dtype)

            # [B, V] -> [B, 1]
            g_mass = (g * probs).sum(dim=1, keepdim=True)

            probs = probs * (1 + g - g_mass)  # [B, V]

        return probs

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.watermark import synthid_watermark_logits

            # Masked tokens stay -inf, as below.
            return synthid_watermark_logits(
                logits, contexts, self.prf.key, self.depth, stream=_STREAM_DOMAIN
            )
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        probs = torch.softmax(logits, dim=1, dtype=torch.float32)  # [B, V]

        for start in range(0, self.depth, 32):
            stream = _STREAM_DOMAIN | start // 32
            num_bits = min(32, self.depth - start)

            # [B, C] and [V] -> [B, V]
            words = self.prf.uint32(
                contexts,
                vocabulary,
                stream=stream,
            )

            probs = self._update_probs(
                probs,
                words,
                num_bits,
            )

        log_probs = torch.log(probs)
        return torch.where(
            torch.isfinite(log_probs) | torch.isneginf(logits),
            log_probs,
            torch.finfo(log_probs.dtype).min,
        )

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("SynthID-Text requires a random sampler")

        watermarked_logits = self.watermark_logits(logits, contexts)
        if skip_mask is not None:
            watermarked_logits = torch.where(
                skip_mask.unsqueeze(-1), logits, watermarked_logits
            )
        return WatermarkSample(
            token_ids=random_sampler(watermarked_logits),
            logits=logits,
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("SynthID-Text requires a random sampler")


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
    """Detect a surplus of Philox-derived SynthID-Text bits in generated tokens.

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
        _validate_context_width(context_width)
        selected_prf = create_prf(prf, key)
        if not isinstance(selected_prf, PhiloxPRF):
            raise ValueError("SynthID-Text requires the Philox PRF")
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = selected_prf
        self.depth = depth

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        blocks = []

        for start in range(0, self.depth, 32):
            stream = _STREAM_DOMAIN | start // 32
            num_bits = min(32, self.depth - start)

            # [N, C] and [N, 1] -> [N, 1] -> [N]
            words = self.prf.uint32(
                contexts,
                targets.unsqueeze(-1),
                stream=stream,
            ).squeeze(-1)

            # [D_block]
            bit_positions = torch.arange(
                num_bits,
                device=words.device,
            )

            # [N, 1] and [D_block] -> [N, D_block]
            blocks.append((words.unsqueeze(-1) >> bit_positions) & 1)
        # [N, depth]
        return torch.cat(blocks, dim=-1)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.to(torch.float64).mean().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        trials = num_scored_tokens * self.depth
        return _binomial_survival(round(score * trials), trials)
