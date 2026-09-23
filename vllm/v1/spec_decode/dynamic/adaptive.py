# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-adaptive speculative length (K) for Dynamic SD.

The batch-size schedule in ``utils.py`` picks K from *load*. This controller
picks K from *drafter quality*: it tracks the unconditional acceptance rate of
each draft position from the scheduler's verification results and keeps only
the leading positions whose rate clears a threshold. Each extra draft position
costs a full drafter forward pass and returns ``acceptance[pos]`` expected
tokens, so positions below the threshold are net negative and are dropped.

When both mechanisms are configured the scheduler uses ``min(schedule_k,
adaptive_k)``: the schedule is a load cap, the controller a quality cap.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


class AcceptanceAdaptiveK:
    """Chooses num_speculative_tokens per scheduler step from observed acceptance.

    Args:
        max_num_speculative_tokens: Upper bound on K (the configured
            ``num_speculative_tokens``); KV/lookahead slots are reserved for it.
        threshold: A draft position is kept while its unconditional acceptance
            rate (accepted / drafted at that position) is >= this value.
        min_num_speculative_tokens: Lower bound on K. 0 lets the controller turn
            speculation off entirely when even position 0 is below threshold.
        window: Number of drafts the acceptance estimate averages over. The
            controller stays at ``max_num_speculative_tokens`` until this many
            drafts have been observed, then adapts using an exponential moving
            average with decay ``1 / window``.
        probe_interval: Every this many scheduler steps K is forced back to
            ``max_num_speculative_tokens`` for one step so that positions the
            controller has stopped drafting keep receiving fresh acceptance
            samples and can be re-enabled if the workload changes. 0 disables
            probing (K can then only shrink).
        hysteresis: A disabled position is only re-enabled once its rate
            reaches ``threshold + hysteresis``, so K does not flap step to step
            when a position hovers at the threshold.
    """

    def __init__(
        self,
        max_num_speculative_tokens: int,
        threshold: float,
        min_num_speculative_tokens: int = 0,
        window: int = 256,
        probe_interval: int = 64,
        hysteresis: float = 0.05,
    ) -> None:
        if max_num_speculative_tokens < 1:
            raise ValueError("max_num_speculative_tokens must be >= 1.")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        if not 0 <= min_num_speculative_tokens <= max_num_speculative_tokens:
            raise ValueError(
                "min_num_speculative_tokens must be in "
                "[0, max_num_speculative_tokens]."
            )
        if window < 1:
            raise ValueError("window must be >= 1.")
        if probe_interval < 0:
            raise ValueError("probe_interval must be >= 0.")
        if hysteresis < 0.0:
            raise ValueError("hysteresis must be >= 0.")
        self.max_k = max_num_speculative_tokens
        self.min_k = min_num_speculative_tokens
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.window = window
        self.probe_interval = probe_interval
        self._alpha = 1.0 / window
        # Per-position EMA of the acceptance indicator, only updated for
        # positions that were actually drafted in the observed step.
        self.acceptance_rate: list[float] = [0.0] * self.max_k
        self.num_observed_per_pos: list[int] = [0] * self.max_k
        # Positions start enabled; never-drafted positions have no estimate and
        # stay enabled so a fresh controller behaves like static K.
        self._enabled: list[bool] = [True] * self.max_k
        self.num_drafts = 0
        self.num_steps = 0
        self.current_k = self.max_k

    def observe(self, num_draft_tokens: int, num_accepted_tokens: int) -> None:
        """Record one verified draft (one request, one step)."""
        if num_draft_tokens <= 0:
            return
        num_draft_tokens = min(num_draft_tokens, self.max_k)
        num_accepted_tokens = min(num_accepted_tokens, num_draft_tokens)
        self.num_drafts += 1
        for pos in range(num_draft_tokens):
            accepted = 1.0 if pos < num_accepted_tokens else 0.0
            if self.num_observed_per_pos[pos] < self.window:
                # Plain mean while warming up so early samples are not
                # over-weighted by the EMA's fixed decay.
                count = self.num_observed_per_pos[pos]
                self.acceptance_rate[pos] = (
                    self.acceptance_rate[pos] * count + accepted
                ) / (count + 1)
            else:
                self.acceptance_rate[pos] += self._alpha * (
                    accepted - self.acceptance_rate[pos]
                )
            self.num_observed_per_pos[pos] += 1

    def recommended_k(self) -> int:
        """K implied by the current acceptance estimates (no probing, no warmup)."""
        k = 0
        for pos in range(self.max_k):
            if self.num_observed_per_pos[pos] > 0:
                rate = self.acceptance_rate[pos]
                if self._enabled[pos] and rate < self.threshold:
                    self._enabled[pos] = False
                elif not self._enabled[pos] and rate >= self.threshold + self.hysteresis:
                    self._enabled[pos] = True
            if not self._enabled[pos]:
                break
            k += 1
        return max(self.min_k, min(self.max_k, k))

    def next_num_speculative_tokens(self) -> int:
        """K to use for the scheduler step about to be issued."""
        self.num_steps += 1
        if self.num_drafts < self.window:
            return self.max_k
        if self.probe_interval and self.num_steps % self.probe_interval == 0:
            return self.max_k
        k = self.recommended_k()
        if k != self.current_k:
            logger.info(
                "Adaptive speculative decoding: num_speculative_tokens %d -> %d "
                "(per-position acceptance %s, threshold %.2f)",
                self.current_k,
                k,
                ", ".join(f"{rate:.2f}" for rate in self.acceptance_rate),
                self.threshold,
            )
            self.current_k = k
        return k


def possible_num_speculative_tokens(
    dense_schedule: list[int] | None,
    adaptive_min_k: int | None,
    max_k: int,
    max_batch_size: int,
) -> dict[int, int]:
    """Map every K the scheduler can emit to the widest batch size it can run at.

    Used to size CUDA graphs. With a batch-size schedule only, each batch size
    has exactly one K. With acceptance-adaptive K, any value in
    ``[adaptive_min_k, schedule_k(batch)]`` is possible at that batch size.
    """
    widest: dict[int, int] = {}
    for batch_size in range(1, max_batch_size + 1):
        cap = dense_schedule[batch_size] if dense_schedule is not None else max_k
        if adaptive_min_k is None:
            widest[cap] = batch_size
            continue
        for k in range(min(adaptive_min_k, cap), cap + 1):
            widest[k] = batch_size
    return widest
