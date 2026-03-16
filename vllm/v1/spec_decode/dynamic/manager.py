# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.config.speculative import DynamicSpeculativeConfig


class DynamicSpeculativeDecodingManager:
    """Computes optimal number of draft tokens based on batch size and
    acceptance rate to maximize goodput = acceptance_length / ITL."""

    def __init__(
        self,
        dynamic_config: DynamicSpeculativeConfig,
        max_batch_size: int,
        num_speculative_tokens: int,
    ):
        self.dynamic_config = dynamic_config
        self.max_batch_size = max_batch_size
        self.num_speculative_tokens = num_speculative_tokens
        self.batch_stats = dynamic_config.batch_stats
        self.available_batch_sizes = sorted(dynamic_config.batch_stats.keys())
        self.steps = 0
        self.warmup_steps = dynamic_config.warmup_steps

        # Cumulative stats for online acceptance rate updates
        self.stats = SpecDecodingStats.new(num_speculative_tokens)

        self._validate_config()
        self._precompute_optimal_k()

    def _validate_config(self) -> None:
        cfg = self.dynamic_config
        assert self.num_speculative_tokens <= cfg.max_num_speculative_tokens, (
            "num_speculative_tokens must be <= max_num_speculative_tokens"
        )
        assert cfg.max_num_speculative_tokens == len(cfg.acceptance_rate_per_pos), (
            "max_num_speculative_tokens != len(acceptance_rate_per_pos)"
        )
        assert cfg.max_num_speculative_tokens > 0
        assert all(0.0 <= a <= 1.0 for a in cfg.acceptance_rate_per_pos)
        assert 1 in cfg.batch_stats, f"BS 1 not found in {cfg.batch_stats.keys()}"
        assert self.max_batch_size in cfg.batch_stats, (
            f"max BS {self.max_batch_size} not found in {cfg.batch_stats.keys()}"
        )

        for bs in self.available_batch_sizes:
            assert bs > 0
            assert 0 in cfg.batch_stats[bs], f"batch size {bs} must have K=0 stats"
            assert 1 in cfg.batch_stats[bs], f"batch size {bs} must have K=1 stats"

    def _precompute_optimal_k(self) -> None:
        """Precompute optimal K for all batch sizes 1..max_batch_size."""
        self._optimal_k_cache: dict[int, int] = {
            bs: self._compute_optimal_k(bs) for bs in range(1, self.max_batch_size + 1)
        }

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int) -> None:
        """Record draft/accept counts for online acceptance rate updates."""
        self.stats.observe_draft(num_draft_tokens, num_accepted_tokens)

    def step(self, batch_size: int) -> int:
        """Get optimal K for this step, updating acceptance rates if needed."""
        self.steps += 1

        # Update acceptance rates from online stats after warmup
        if self.steps > self.warmup_steps and self.stats.num_drafts > 0:
            acceptance_rates = self._compute_acceptance_rates()
            if acceptance_rates != self.dynamic_config.acceptance_rate_per_pos:
                self.dynamic_config.acceptance_rate_per_pos = acceptance_rates
                self._precompute_optimal_k()

        return self.get_optimal_k(batch_size)

    def _compute_acceptance_rates(self) -> list[float]:
        """Compute acceptance rate per position from cumulative stats."""
        rates = []
        for i in range(self.num_speculative_tokens):
            if self.stats.num_draft_tokens_per_pos[i] == 0:
                rates.append(0.0)
            else:
                rates.append(
                    self.stats.num_accepted_tokens_per_pos[i]
                    / self.stats.num_draft_tokens_per_pos[i]
                )
        return rates

    def get_optimal_k(self, batch_size: int) -> int:
        """Get precomputed optimal K for batch size."""
        assert 0 < batch_size <= self.max_batch_size
        return self._optimal_k_cache[batch_size]

    def _compute_optimal_k(self, batch_size: int) -> int:
        """Compute optimal K by maximizing goodput = AL / ITL."""
        batch_stats = self._interpolate_batch_stats(batch_size)
        acceptance_rates = self.dynamic_config.acceptance_rate_per_pos

        best_k = 0
        best_goodput = -1.0

        for k in range(self.dynamic_config.max_num_speculative_tokens + 1):
            # Acceptance length = 1 + sum of acceptance rates for positions 0..k-1
            acceptance_length = 1 + sum(acceptance_rates[:k])
            itl = self._interpolate_itl(batch_stats, k)
            goodput = acceptance_length / itl

            if goodput > best_goodput:
                best_goodput = goodput
                best_k = k

        return best_k

    def _interpolate_batch_stats(self, batch_size: int) -> dict[int, float]:
        """Get or interpolate ITL stats for a batch size."""
        if batch_size in self.batch_stats:
            return self.batch_stats[batch_size]

        # Find neighboring batch sizes
        smaller = [bs for bs in self.available_batch_sizes if bs < batch_size]
        larger = [bs for bs in self.available_batch_sizes if bs > batch_size]

        lo_bs = max(smaller) if smaller else self.available_batch_sizes[0]
        hi_bs = min(larger) if larger else self.available_batch_sizes[-1]

        if lo_bs == hi_bs:
            return self.batch_stats[lo_bs]

        # Linear interpolation
        ratio = (batch_size - lo_bs) / (hi_bs - lo_bs)
        lo_stats = self.batch_stats[lo_bs]
        hi_stats = self.batch_stats[hi_bs]

        return {k: lo_stats[k] + ratio * (hi_stats[k] - lo_stats[k]) for k in lo_stats}

    def _interpolate_itl(self, batch_stats: dict[int, float], k: int) -> float:
        """Get or interpolate ITL for a given K."""
        if k in batch_stats:
            return batch_stats[k]

        # Find neighboring K values
        lower_k = max(kk for kk in batch_stats if kk < k)
        upper_k = min(kk for kk in batch_stats if kk > k)

        ratio = (k - lower_k) / (upper_k - lower_k)
        lo, hi = batch_stats[lower_k], batch_stats[upper_k]
        return lo + ratio * (hi - lo)
