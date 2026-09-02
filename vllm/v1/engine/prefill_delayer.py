# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP-rank prefill alignment (PrefillDelayer).

Under data-parallel attention every rank pads its per-step batch up to the max
token count across ranks, and a single prefill on any rank forces every rank off
the fast decode CUDA graph. When only some ranks have a new prefill ready
("skewed" state), those prefill-sized batches inflate work for every rank. The
PrefillDelayer holds new prefills on a rank until sibling ranks are also ready,
so dense prefills fire together (all ranks prefill the same step) and the other
steps are pure decode across all ranks (fast graph).

The delayer holds no distributed state: the cross-DP signals (how many ranks are
prefillable, queued prefill tokens, running decodes, and a few 0/1 booleans) are
reduced by the engine core's existing per-step DP sync and handed in. Every
signal is deterministic and tick based (no wall clock), so all ranks compute the
identical FIRE/HOLD decision and stay aligned.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)


class PrefillDelayer:
    """Decides whether to hold (throttle) new prefills to align DP ranks.

    A HOLD makes the engine core throttle prefills, which the scheduler turns
    into a pure-decode step on every rank; a FIRE lets aligned ranks admit their
    queued prefills together.
    """

    def __init__(
        self,
        dp_size: int,
        prefill_token_budget: int,
        target_fill: float = 0.9,
        ttft_max_ticks: int = 200,
        partial_max_ticks: int = 100,
        stall_ticks: int = 10,
    ):
        """Initialize the delayer.

        Args:
            dp_size: Number of data-parallel ranks to align.
            prefill_token_budget: Per-forward prefill token budget
                (``max_num_batched_tokens``); sizes the fill-fraction
                denominator.
            target_fill: Fire once the queued prefill tokens fill this fraction
                of an aligned forward's aggregate budget, in [0, 1].
            ttft_max_ticks: Max consecutive holds before force-firing, to bound
                worst-case TTFT.
            partial_max_ticks: Tighter hold bound applied while a chunked prefill
                is already in flight on some rank.
            stall_ticks: Fire once the queued prefill tokens stop growing for
                this many consecutive holds.
        """
        self.dp_size = dp_size
        self.prefill_token_budget = max(1, prefill_token_budget)
        self.target_fill = min(1.0, max(0.0, target_fill))
        self.ttft_max_ticks = max(1, ttft_max_ticks)
        self.partial_max_ticks = max(1, partial_max_ticks)
        self.stall_ticks = max(1, stall_ticks)

        self._throttle = False
        self._hold_ticks = 0
        self._prev_pending = 0
        self._stall_count = 0

        logger.info(
            "PrefillDelayer initialized: dp_size=%d prefill_token_budget=%d "
            "target_fill=%.2f ttft_max_ticks=%d partial_max_ticks=%d "
            "stall_ticks=%d",
            self.dp_size,
            self.prefill_token_budget,
            self.target_fill,
            self.ttft_max_ticks,
            self.partial_max_ticks,
            self.stall_ticks,
        )

    @property
    def should_throttle(self) -> bool:
        """Whether new prefills should be held, per the last update_throttle()."""
        return self._throttle

    def update_throttle(
        self,
        n_prefillable: int,
        pending_tokens: int,
        running_decode: int,
        kv_high: int,
        kv_low: int,
        has_partial: int,
        queue_hot: int,
    ) -> None:
        """Refresh the HOLD decision from the reduced cross-DP signals.

        Args:
            n_prefillable: Number of DP ranks that would admit a prefill next
                step (summed across ranks).
            pending_tokens: Total queued prefill tokens across ranks, each rank
                capped at the token budget.
            running_decode: Total running decode requests across ranks.
            kv_high: Non-zero if some rank's KV-cache usage is at the high
                watermark (cannot accumulate more).
            kv_low: Non-zero if some prefillable rank is under the KV low
                watermark (underutilized).
            has_partial: Non-zero if some rank has an in-flight chunked prefill.
            queue_hot: Non-zero if some rank's oldest waiting request breached
                the age SLA.

        All inputs are reduced values, so every rank computes the identical
        decision.
        """
        self._throttle = not self._allow(
            n_prefillable,
            pending_tokens,
            running_decode,
            kv_high,
            kv_low,
            has_partial,
            queue_hot,
        )

    def _allow(
        self,
        n_prefillable: int,
        pending_tokens: int,
        running_decode: int,
        kv_high: int,
        kv_low: int,
        has_partial: int,
        queue_hot: int,
    ) -> bool:
        if n_prefillable == 0:
            self.reset()
            return True
        if kv_high > 0:
            self.reset()
            return True
        if kv_low > 0:
            self.reset()
            return True
        if running_decode == 0:
            self.reset()
            return True
        if queue_hot > 0:
            self.reset()
            return True
        if self._hold_ticks >= self.ttft_max_ticks:
            self.reset()
            return True
        if has_partial > 0 and self._hold_ticks >= self.partial_max_ticks:
            self.reset()
            return True
        if n_prefillable < self.dp_size:
            return not self._hold()

        denom = n_prefillable * self.prefill_token_budget
        fill = pending_tokens / denom if denom > 0 else 1.0
        if fill >= self.target_fill:
            self.reset()
            return True
        if self._stalled(pending_tokens):
            self.reset()
            return True
        return not self._hold()

    def _hold(self) -> bool:
        self._hold_ticks += 1
        return True

    def _stalled(self, pending: int) -> bool:
        if pending > self._prev_pending:
            self._stall_count = 0
        else:
            self._stall_count += 1
        self._prev_pending = pending
        return self._stall_count >= self.stall_ticks

    def reset(self) -> None:
        self._hold_ticks = 0
        self._prev_pending = 0
        self._stall_count = 0
        self._throttle = False
