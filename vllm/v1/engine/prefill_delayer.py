# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP-rank prefill alignment (PrefillDelayer).

Ported from SGLang's PrefillDelayer (via ATOM). Under data-parallel attention,
each rank pads its per-step batch up to the max token count across ranks, and
the MoE path all-gathers that padded batch every layer. When only some ranks
have a new prefill ready ("mixed" state), those prefill-sized batches inflate
work for every rank. This delays prefills on a rank until sibling ranks are also
ready, so dense prefills fire together (balanced) rather than straggling.

This class holds no distributed state: the cross-DP signal (how many ranks are
prefillable) is gathered by the engine core's existing per-step DP sync and
handed in. Given that count it derives a 3-way status:
  - "all"   -> every rank has a new prefill ready -> allow (aligned).
  - "none"  -> no rank has a prefill              -> allow (vacuous).
  - "mixed" -> only some ranks have prefill ready -> DELAY.
In "mixed", refuse the prefill for up to ``max_delay_passes`` consecutive steps
OR ``max_delay_ms`` wall-clock, whichever comes first, then force-allow to bound
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
        max_delay_passes: int = 30,
        max_delay_ms: float = 5000.0,
    ):
        self.dp_size = dp_size
        self.max_delay_passes = max_delay_passes
        self.max_delay_ms = max_delay_ms

        self._delayed_count: int = 0
        self._delay_start_ts: float = 0.0

        logger.info(
            "PrefillDelayer initialized: dp_size=%d max_delay_passes=%d "
            "max_delay_ms=%.1f",
            dp_size,
            max_delay_passes,
            max_delay_ms,
        )

    def should_allow_prefill(self, prefillable_count: int) -> bool:
        """Return True iff ranks may admit new prefills this step.

        Args:
            prefillable_count: Number of DP ranks that have a new prefill ready,
                summed across ranks by the engine core's DP sync. The decision
                is identical on every rank because the input is a reduced value.
        """
        # "all" and "none" are already aligned -> allow. Only "mixed" (some
        # ranks prefillable, some not) causes delay.
        if prefillable_count == 0 or prefillable_count == self.dp_size:
            self.reset()
            return True

        # status == "mixed" -> delay within budget.
        if self._delayed_count == 0:
            self._delay_start_ts = time.perf_counter()
        elapsed_ms = (time.perf_counter() - self._delay_start_ts) * 1000.0

        if (
            self._delayed_count < self.max_delay_passes
            and elapsed_ms < self.max_delay_ms
        ):
            self._delayed_count += 1
            return False

        # Timed out -> force allow to bound worst-case TTFT.
        self.reset()
        return True

    def reset(self) -> None:
        self._delayed_count = 0
        self._delay_start_ts = 0.0
