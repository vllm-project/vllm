# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text watermark generation primitives."""

import torch

from vllm.config.watermarking import WatermarkPRFName
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
        g_values: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight [batch, vocab] scores with [batch, vocab, depth] bits."""
        probs = torch.softmax(scores, dim=1)
        for depth in range(g_values.shape[-1]):
            g = g_values[:, :, depth]
            # Sum each row's probability mass on g=1 candidates: [B, V] -> [B, 1].
            g_mass = (g * probs).sum(dim=1, keepdim=True)
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
        # [B, context_width] and [V] produce one word per candidate: [B, V].
        words = self.prf.uint32(contexts, vocabulary)
        bit_positions = torch.arange(self.depth, device=logits.device)
        # Broadcast [B, V, 1] against [depth] to get [B, V, depth] g values.
        g_values = ((words.unsqueeze(-1) >> bit_positions) & 1).to(logits.dtype)
        return self._update_scores(logits, g_values)

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
