# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP-rank prefill alignment (PrefillDelayer).

Ported from SGLang's PrefillDelayer (via ATOM). Under data-parallel attention,
each rank pads its per-step batch up to the max token count across ranks, and
the MoE path all-gathers that padded batch every layer. When only some ranks
have a new prefill ready ("mixed" state), those prefill-sized batches inflate
work for every rank. This delays prefills on a rank until sibling ranks are also
ready, so dense prefills fire together (balanced) rather than straggling.

This class holds no distributed state: the cross-DP signals (how many ranks are
prefillable, and the total queued prefill tokens) are gathered by the engine
core's existing per-step DP sync and handed in. From them it decides FIRE/HOLD:
  - no rank prefillable            -> allow (nothing to coalesce or align).
  - ranks skewed (some prefillable,
    some not)                      -> hold (anti-skew alignment).
  - all ranks prefillable but the
    queued batch is underfull       -> hold (coalesce, Nagle-for-prefill).
  - all ranks prefillable and the
    queued batch is dense enough    -> allow (aligned + full).

``fill`` is ``pending_prefill_tokens / (prefillable_count * token_budget)`` — the
fraction of an aligned prefill forward's aggregate token capacity that is
currently queued. Firing only once ``fill >= target_fill`` coalesces many small
prefills (short prompts, chunk tails) into fewer dense forwards, so each prefill
forward carries useful work rather than a handful of tokens padded up to the
per-step batch. A hold is bounded by ``max_delay_passes`` consecutive steps OR
``max_delay_ms`` wall-clock, whichever comes first, then force-allow to bound
worst-case TTFT.
"""

from __future__ import annotations

import time

from vllm.logger import init_logger

logger = init_logger(__name__)


class PrefillDelayer:
    def __init__(
        self,
        dp_size: int,
        prefill_token_budget: int,
        target_fill: float = 0.9,
        max_delay_passes: int = 30,
        max_delay_ms: float = 5000.0,
    ):
        self.dp_size = dp_size
        # Per-forward prefill token budget (max_num_batched_tokens); sizes the
        # denominator of the fill fraction.
        self.prefill_token_budget = max(1, prefill_token_budget)
        # Coalescing target: fire once the queued prefill fills this fraction of
        # an aligned forward's aggregate capacity. Clamp to a usable range —
        # <= 0 disables coalescing (degrades to skew-only), > 1 is unreachable
        # since fill can exceed capacity only transiently.
        self.target_fill = min(1.0, max(0.0, target_fill))
        self.max_delay_passes = max_delay_passes
        self.max_delay_ms = max_delay_ms

        self._delayed_count: int = 0
        self._delay_start_ts: float = 0.0
        # Whether new prefills are currently being held back. Updated by
        # update_throttle() from the DP sync and read back in the next
        # schedule().
        self._throttle: bool = False

        logger.info(
            "PrefillDelayer initialized: dp_size=%d prefill_token_budget=%d "
            "target_fill=%.2f max_delay_passes=%d max_delay_ms=%.1f",
            dp_size,
            self.prefill_token_budget,
            self.target_fill,
            max_delay_passes,
            max_delay_ms,
        )

    @property
    def should_throttle(self) -> bool:
        """Whether new prefills should be deferred, per the last update_throttle()."""
        return self._throttle

    def update_throttle(
        self, prefillable_count: int, pending_prefill_tokens: int
    ) -> None:
        """Update the delay decision from the reduced cross-DP signals.

        Args:
            prefillable_count: Number of DP ranks that have a new prefill ready,
                summed across ranks by the engine core's DP sync.
            pending_prefill_tokens: Total queued prefill tokens across ranks
                (each rank capped at the token budget), from the same sync.

        Both inputs are reduced values, so every rank computes the identical
        decision.
        """
        self._throttle = not self._allow(prefillable_count, pending_prefill_tokens)

    def _allow(self, prefillable_count: int, pending_prefill_tokens: int) -> bool:
        # No rank has a prefill queued -> nothing to align or coalesce.
        if prefillable_count == 0:
            self.reset()
            return True

        # Fraction of an aligned prefill forward's aggregate token capacity that
        # is currently queued across the prefillable ranks.
        capacity = prefillable_count * self.prefill_token_budget
        fill = pending_prefill_tokens / capacity if capacity > 0 else 1.0
        aligned = prefillable_count == self.dp_size

        # Fire only when every rank is ready (aligned, so the MoE collective
        # stays balanced) AND the coalesced batch is dense enough to be worth a
        # forward. Otherwise hold: either ranks are skewed, or the aligned batch
        # is still underfull.
        if aligned and fill >= self.target_fill:
            self.reset()
            return True

        if self._delayed_count == 0:
            self._delay_start_ts = time.perf_counter()
            logger.info(
                "PrefillDelayer holding prefill: %d/%d ranks prefillable, "
                "fill=%.2f < target=%.2f; up to %d passes / %.1f ms.",
                prefillable_count,
                self.dp_size,
                fill,
                self.target_fill,
                self.max_delay_passes,
                self.max_delay_ms,
            )
        elapsed_ms = (time.perf_counter() - self._delay_start_ts) * 1000.0

        if (
            self._delayed_count < self.max_delay_passes
            and elapsed_ms < self.max_delay_ms
        ):
            self._delayed_count += 1
            return False

        # Timed out -> force allow to bound worst-case TTFT.
        logger.info(
            "PrefillDelayer force-allowing prefill after %d passes / %.1f ms "
            "(TTFT bound reached).",
            self._delayed_count,
            elapsed_ms,
        )
        self.reset()
        return True

    def reset(self) -> None:
        self._delayed_count = 0
        self._delay_start_ts = 0.0
        self._throttle = False
