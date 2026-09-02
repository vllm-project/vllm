# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP-rank prefill alignment (PrefillDelayer).

Ported from SGLang's PrefillDelayer (via ATOM). Under data-parallel attention,
each rank pads its per-step batch up to the max token count across ranks, and
the MoE path all-gathers that padded batch every layer. When only some ranks
have a new prefill ready ("mixed" state), those prefill-sized batches inflate
work for every rank, and a single prefill forces every rank off the fast decode
CUDA graph. This delays prefills on a rank until sibling ranks are also ready,
so dense prefills fire together (balanced) rather than straggling.

The delayer holds no distributed state: the cross-DP signals (how many ranks are
prefillable, queued prefill tokens, running/waiting counts, and a low-watermark
override) are reduced by the engine core's existing per-step DP sync and handed
in. From them it decides FIRE/HOLD and whether the next step is a *global-prefill
step* (all ranks defer decodes; prefill-ready ranks run a pure prefill, others
idle). Both decisions use only reduced inputs, so every rank agrees.

FIRE/HOLD:
  - no rank prefillable            -> allow (nothing to align or coalesce).
  - any rank under the KV low
    watermark                       -> allow (system underutilized).
  - ranks skewed (some prefillable,
    some not)                       -> hold (anti-skew alignment) until bound.
  - all ranks prefillable           -> SGLang slot-pressure / queue-length
    triggers (when ``max_prefill_bs`` > 0), else the token fill-fraction
    coalescing gate. Hold is bounded by ``max_delay_passes`` / ``max_delay_ms``.

Global-prefill step: emitted on a release step (not throttling, some rank
prefillable), capped at ``max_consecutive_prefill_steps`` consecutive steps so
held decodes cannot starve and a KV-blocked burst cannot spin on empty steps.
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
        max_running_requests: int = 0,
        target_fill: float = 0.9,
        max_delay_passes: int = 30,
        max_delay_ms: float = 5000.0,
        max_consecutive_prefill_steps: int = 4,
        max_prefill_bs: int = 0,
        queue_min_ratio: float = 0.0,
        coalesce_min_ranks: int = 0,
        idle_non_prefill_ranks: bool = False,
    ):
        self.dp_size = dp_size
        # Per-forward prefill token budget (max_num_batched_tokens); sizes the
        # denominator of the fill fraction.
        self.prefill_token_budget = max(1, prefill_token_budget)
        # Per-rank decode-batch capacity (max_num_seqs); sizes the slot-pressure
        # trigger's aggregate free-slot count.
        self.max_running_requests = max(0, max_running_requests)
        # Coalescing target for the token fill-fraction gate (used when the
        # SGLang request-count triggers are disabled, i.e. max_prefill_bs == 0).
        self.target_fill = min(1.0, max(0.0, target_fill))
        self.max_delay_passes = max_delay_passes
        self.max_delay_ms = max_delay_ms
        self.max_consecutive_prefill_steps = max(1, max_consecutive_prefill_steps)
        self.max_prefill_bs = max(0, max_prefill_bs)
        self.queue_min_ratio = max(0.0, queue_min_ratio)
        # Release prefills only once this many ranks are prefill-ready (coalesce
        # a dense cross-rank burst); 0 disables and keeps per-rank release.
        self.coalesce_min_ranks = min(max(0, coalesce_min_ranks), dp_size)
        # Whether decode-only ranks idle on a global-prefill step (off by
        # default: idling forgoes free decode without shortening the prefill).
        self.idle_non_prefill_ranks = idle_non_prefill_ranks

        self._delayed_count: int = 0
        self._delay_start_ts: float = 0.0
        # Whether new prefills are currently being held back. Updated by
        # update_throttle() from the DP sync and read back in the next
        # schedule().
        self._throttle: bool = False
        # Whether the next scheduling step is a global-prefill step (all ranks
        # defer decodes). Derived from the same reduced signals.
        self._is_global_prefill_step: bool = False
        # Consecutive global-prefill steps emitted; bounds decode starvation.
        self._consecutive_prefill_steps: int = 0
        self._last_prefillable_count: int = 0

        logger.info(
            "PrefillDelayer initialized: dp_size=%d prefill_token_budget=%d "
            "max_running_requests=%d target_fill=%.2f max_delay_passes=%d "
            "max_delay_ms=%.1f max_consecutive_prefill_steps=%d max_prefill_bs=%d "
            "queue_min_ratio=%.2f",
            dp_size,
            self.prefill_token_budget,
            self.max_running_requests,
            self.target_fill,
            max_delay_passes,
            max_delay_ms,
            self.max_consecutive_prefill_steps,
            self.max_prefill_bs,
            self.queue_min_ratio,
        )

    @property
    def should_throttle(self) -> bool:
        """Whether new prefills should be deferred, per the last update_throttle()."""
        return self._throttle

    @property
    def is_global_prefill_step(self) -> bool:
        """Whether the next schedule() is a global-prefill step (all ranks defer
        decodes so prefill-ready ranks run a pure prefill), per the last
        update_throttle()."""
        return self._is_global_prefill_step

    def update_throttle(
        self,
        prefillable_count: int,
        pending_prefill_tokens: int,
        token_watermark_force_allow: bool = False,
        running_count: int = 0,
        waiting_count: int = 0,
    ) -> None:
        """Update the delay decision from the reduced cross-DP signals.

        Args:
            prefillable_count: Number of DP ranks that have a new prefill ready,
                summed across ranks by the engine core's DP sync.
            pending_prefill_tokens: Total queued prefill tokens across ranks
                (each rank capped at the token budget), from the same sync.
            token_watermark_force_allow: True if any prefillable rank's KV-cache
                usage is below the low watermark (system underutilized).
            running_count: Total running (decode) requests across ranks.
            waiting_count: Total waiting requests across ranks.

        All inputs are reduced values, so every rank computes the identical
        decision.
        """
        self._last_prefillable_count = prefillable_count
        self._throttle = not self._allow(
            prefillable_count,
            pending_prefill_tokens,
            token_watermark_force_allow,
            running_count,
            waiting_count,
        )
        self._is_global_prefill_step = self._compute_global_prefill_step()

    def _compute_global_prefill_step(self) -> bool:
        # IDLE disabled -> never idle decode-only ranks; they decode for free.
        if not self.idle_non_prefill_ranks:
            return False
        # Not a release step, or nothing to prefill anywhere -> normal decode.
        if self._throttle or self._last_prefillable_count == 0:
            self._consecutive_prefill_steps = 0
            return False
        # Decode-progress bound: force a global-decode step after N consecutive
        # global-prefill steps so held decodes cannot starve and a KV-blocked
        # burst (all ranks empty) cannot spin on empty steps.
        if self._consecutive_prefill_steps >= self.max_consecutive_prefill_steps:
            self._consecutive_prefill_steps = 0
            return False
        self._consecutive_prefill_steps += 1
        return True

    def _allow(
        self,
        prefillable_count: int,
        pending_prefill_tokens: int,
        token_watermark_force_allow: bool,
        running_count: int,
        waiting_count: int,
    ) -> bool:
        # No rank has a prefill queued -> nothing to align or coalesce.
        if prefillable_count == 0:
            self.reset()
            return True

        # System underutilized -> holding only adds TTFT, so fire.
        if token_watermark_force_allow:
            self.reset()
            return True

        # Cross-rank coalescing: hold (steps run pure decode on all ranks) until
        # enough ranks are prefill-ready, then release them as one dense burst so
        # most steps stay pure-decode. Bounded by _hold to cap worst-case TTFT.
        if self.coalesce_min_ranks > 0:
            if prefillable_count >= self.coalesce_min_ranks:
                self.reset()
                return True
            return not self._hold(prefillable_count)

        aligned = prefillable_count == self.dp_size
        if aligned and not self._hold_aligned(
            pending_prefill_tokens, running_count, waiting_count
        ):
            self.reset()
            return True

        # Aligned-but-underfull, or skewed (some ranks prefillable, some not):
        # hold to align/coalesce, bounded to keep worst-case TTFT finite.
        return not self._hold(prefillable_count)

    def _hold_aligned(
        self, pending_prefill_tokens: int, running_count: int, waiting_count: int
    ) -> bool:
        """Whether to keep holding when all ranks are prefillable.

        With ``max_prefill_bs`` > 0, uses SGLang's slot-pressure and
        queue-length triggers to keep decode steps dominant; otherwise falls
        back to the token fill-fraction coalescing gate.
        """
        if self.max_prefill_bs > 0:
            max_running = self.dp_size * self.max_running_requests
            slot_condition = (max_running - running_count) < self.max_prefill_bs
            queue_condition = False
            if self.queue_min_ratio > 0.0 and running_count > 0:
                queue_min = min(
                    int(running_count * self.queue_min_ratio), self.max_prefill_bs
                )
                queue_condition = queue_min > 0 and waiting_count < queue_min
            return slot_condition or queue_condition

        capacity = self.dp_size * self.prefill_token_budget
        fill = pending_prefill_tokens / capacity if capacity > 0 else 1.0
        return fill < self.target_fill

    def _hold(self, prefillable_count: int) -> bool:
        """Advance the bounded hold. Returns True to keep holding, False once the
        pass/ms bound is reached (force-fire)."""
        if self._delayed_count == 0:
            self._delay_start_ts = time.perf_counter()
            logger.info(
                "PrefillDelayer holding prefill: %d/%d ranks prefillable; "
                "up to %d passes / %.1f ms.",
                prefillable_count,
                self.dp_size,
                self.max_delay_passes,
                self.max_delay_ms,
            )
        elapsed_ms = (time.perf_counter() - self._delay_start_ts) * 1000.0

        if (
            self._delayed_count < self.max_delay_passes
            and elapsed_ms < self.max_delay_ms
        ):
            self._delayed_count += 1
            return True

        logger.info(
            "PrefillDelayer force-allowing prefill after %d passes / %.1f ms "
            "(TTFT bound reached).",
            self._delayed_count,
            elapsed_ms,
        )
        self.reset()
        return False

    def reset(self) -> None:
        self._delayed_count = 0
        self._delay_start_ts = 0.0
        self._throttle = False
