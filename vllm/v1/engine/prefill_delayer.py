# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP-rank prefill alignment (PrefillDelayer).

Ported from SGLang's PrefillDelayer (via ATOM). Under data-parallel attention,
each rank pads its per-step batch up to the max token count across ranks, and
the MoE path all-gathers that padded batch every layer. When only some ranks
have a new prefill ready ("mixed" state), those prefill-sized batches inflate
work for every rank. This delays prefills on a rank until sibling ranks are also
ready, so dense prefills fire together (balanced) rather than straggling.

Mechanism (per scheduler step):
  1. Each rank reports its local state via one CPU MAX all_reduce:
       (local_prefillable, watermark_force_allow).
  2. Derive a 3-way status across ranks:
       - "all"   -> every rank has a new prefill ready -> allow (aligned).
       - "none"  -> no rank has a prefill              -> allow (vacuous).
       - "mixed" -> only some ranks have prefill ready -> DELAY.
  3. In "mixed", refuse the prefill for up to ``max_delay_passes`` consecutive
     steps OR ``max_delay_ms`` wall-clock, whichever comes first, then
     force-allow to bound worst-case TTFT.
  4. Safety valve: if any rank's KV-cache usage drops below
     ``token_usage_low_watermark``, all ranks force-allow this step (don't delay
     while a GPU is idling).
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


class PrefillDelayer:
    def __init__(
        self,
        dp_size: int,
        cpu_group,
        max_delay_passes: int = 30,
        max_delay_ms: float = 5000.0,
        token_usage_low_watermark: float | None = None,
    ):
        self.dp_size = dp_size
        self.cpu_group = cpu_group
        self.max_delay_passes = max_delay_passes
        self.max_delay_ms = max_delay_ms
        self.token_usage_low_watermark = token_usage_low_watermark

        # 3-slot MAX-reduce buffer on CPU (gloo-friendly):
        #   slot 0 = local_prefillable      (MAX -> any rank prefillable)
        #   slot 1 = local_force            (MAX -> any rank forces allow)
        #   slot 2 = NOT local_prefillable  (MAX -> any rank lacks prefill)
        self._reduce_buf = torch.zeros(3, dtype=torch.int64, device="cpu")

        self._delayed_count: int = 0
        self._delay_start_ts: float = 0.0
        # Skip the first negotiation so the decode batch can build up on the
        # initial burst before we start aligning prefills.
        self._skip_first: bool = True

        logger.info(
            "PrefillDelayer initialized: dp_size=%d max_delay_passes=%d "
            "max_delay_ms=%.1f watermark=%s",
            dp_size,
            max_delay_passes,
            max_delay_ms,
            token_usage_low_watermark,
        )

    def should_allow_prefill(
        self,
        local_prefillable: bool,
        token_usage: float,
    ) -> bool:
        """Return True iff this rank may admit new prefills this step.

        Performs exactly one CPU all_reduce and must be called once per step on
        every DP rank (including idle/dummy steps) to stay in lockstep.

        Args:
            local_prefillable: This rank has a new prefill ready to admit.
            token_usage: Fraction of KV cache blocks in use, in [0, 1]. Used by
                the low-watermark safety valve.
        """
        force = (
            self.token_usage_low_watermark is not None
            and local_prefillable
            and token_usage < self.token_usage_low_watermark
        )

        self._reduce_buf[0] = 1 if local_prefillable else 0
        self._reduce_buf[1] = 1 if force else 0
        self._reduce_buf[2] = 0 if local_prefillable else 1
        dist.all_reduce(
            self._reduce_buf,
            op=dist.ReduceOp.MAX,
            group=self.cpu_group,
        )
        any_prefillable = int(self._reduce_buf[0].item()) > 0
        force_max = int(self._reduce_buf[1].item())
        any_not_prefillable = int(self._reduce_buf[2].item()) > 0

        # Any rank below the watermark forces all ranks to allow this step.
        if force_max > 0:
            self._reset_delay()
            return True

        # Skip the first call to maximize initial decode batch build-up.
        if self._skip_first:
            self._skip_first = False
            self._reset_delay()
            return True

        # Only "mixed" (some ranks prefillable, some not) causes delay.
        # "all" and "none" are already aligned -> allow.
        if not (any_prefillable and any_not_prefillable):
            self._reset_delay()
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
        self._reset_delay()
        return True

    def on_wave_boundary(self) -> None:
        """Reset delay state at a DP wave boundary.

        On a fresh wave the decode batch needs to build up again, so re-arm the
        skip-first behavior and clear any in-flight delay so the first prefill
        of the new wave is not held back.
        """
        self._skip_first = True
        self._reset_delay()

    def _reset_delay(self) -> None:
        self._delayed_count = 0
        self._delay_start_ts = 0.0
