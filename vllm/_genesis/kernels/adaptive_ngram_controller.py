# SPDX-License-Identifier: Apache-2.0
"""Adaptive ngram speculation controller — Genesis P77.

Ports SGLang's adaptive_spec_params.py (Apache-2.0) EMA + hysteresis logic
to vLLM's ngram speculation path. Adds auto-disable extension (per Nightjar
arXiv 2512.22420 + llama.cpp `--draft-p-min` style cutoff).

================================================================
PROBLEM
================================================================

vLLM ngram with fixed `num_speculative_tokens=K=3` wastes K forward passes
per accepted token when acceptance rate is low. On free-form text without
repetition, our acceptance is ~10-15% → effective decode is K=4 forward
passes per output token = ~4× slower than no-spec baseline.

Free-form bench: 46 tok/s vs MTP 127 tok/s vs no-spec ~150 tok/s
(theoretical).

================================================================
SOLUTION
================================================================

Track per-process EMA of accepted draft length over rolling window. Pick K
from a discrete set [0, 1, 3, 5] with hysteresis to prevent oscillation.
K=0 means "skip ngram entirely for this step" — equivalent to no-spec
mode, restoring baseline throughput on free-form text.

Algorithm (mirrors SGLang adaptive_spec_params.py:62-130):

    every batch:
        if batches_seen < WARMUP: return current_K
        if batches_seen % INTERVAL != 0: return current_K
        ema = (1 - alpha) * ema + alpha * batch_avg_accept_len
        for s in steps_sorted_ascending:
            if ema >= s - hysteresis_down: candidate = s
        current_K = candidate

================================================================
TUNABLE ENV
================================================================

GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1     # master switch
GENESIS_P77_STEPS="0,1,3,5"               # discrete K choices (0 = skip)
GENESIS_P77_EMA_ALPHA=0.2                 # smoothing factor
GENESIS_P77_WARMUP_BATCHES=10             # batches before tuning starts
GENESIS_P77_UPDATE_INTERVAL=5             # update K every N batches
GENESIS_P77_HYSTERESIS_DOWN=0.25          # bias against shrinking K
GENESIS_P77_HYSTERESIS_UP=0.0             # bias against growing K
GENESIS_P77_DISABLE_THRESHOLD=0.30        # accept rate below → drop to K=0
GENESIS_P77_PROBE_INTERVAL=100            # every N batches, force K=3 to retest
GENESIS_P77_LOG_EVERY=20                  # log K transitions every N batches

================================================================
THREAD SAFETY
================================================================

Single-process single-thread (vLLM scheduler is sequential). All state
in plain Python attrs — no locks needed. EMA / counters reset on engine
restart.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Algorithm port: SGLang Apache-2.0 (sgl-project/sglang).
Auto-disable extension: Nightjar arXiv 2512.22420.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.adaptive_ngram_controller")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_steps(name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(name, default)
    try:
        steps = tuple(sorted({int(x) for x in raw.split(",") if x.strip()}))
        return steps if steps else tuple(int(x) for x in default.split(","))
    except ValueError:
        return tuple(int(x) for x in default.split(","))


class AdaptiveNgramController:
    """Per-process adaptive K controller for ngram speculation.

    One instance per drafter (NgramProposer). Call `decide_K()` at top of
    propose(); call `update(accepted_lens)` after rejection_sample reports
    per-request accepted lengths.
    """

    def __init__(self) -> None:
        self.steps: tuple[int, ...] = _env_steps("GENESIS_P77_STEPS", "0,1,3,5")
        self.alpha: float = _env_float("GENESIS_P77_EMA_ALPHA", 0.2)
        self.warmup: int = _env_int("GENESIS_P77_WARMUP_BATCHES", 10)
        self.interval: int = _env_int("GENESIS_P77_UPDATE_INTERVAL", 5)
        self.hyst_down: float = _env_float("GENESIS_P77_HYSTERESIS_DOWN", 0.25)
        self.hyst_up: float = _env_float("GENESIS_P77_HYSTERESIS_UP", 0.0)
        self.disable_threshold: float = _env_float(
            "GENESIS_P77_DISABLE_THRESHOLD", 0.30,
        )
        self.probe_interval: int = _env_int("GENESIS_P77_PROBE_INTERVAL", 100)
        self.log_every: int = _env_int("GENESIS_P77_LOG_EVERY", 20)

        # Initial K = middle of the active steps (excluding 0 if present)
        non_zero = [s for s in self.steps if s > 0]
        self.current_K: int = non_zero[len(non_zero) // 2] if non_zero else 1
        self.ema: float = float(self.current_K)
        self.batches_seen: int = 0
        self.batches_since_update: int = 0
        self.batches_since_probe: int = 0
        self.batches_since_log: int = 0
        self.last_K: int = self.current_K
        self.transitions: int = 0

        # Telemetry
        self.total_accepted: int = 0
        self.total_drafted: int = 0
        self.disabled_steps: int = 0  # how many times we picked K=0

        log.info(
            "[Genesis P77] AdaptiveNgramController init: steps=%s alpha=%.2f "
            "warmup=%d interval=%d hyst_down=%.2f disable_thr=%.2f probe=%d "
            "initial_K=%d",
            self.steps, self.alpha, self.warmup, self.interval,
            self.hyst_down, self.disable_threshold, self.probe_interval,
            self.current_K,
        )

    def decide_K(self) -> int:
        """Return the K to use for the next propose() call.

        K=0 is a signal to caller: skip ngram entirely for this batch and
        return empty drafts (caller's NgramProposer should short-circuit).
        """
        return self.current_K

    def update(self, accepted_lens: list[int], drafted_lens: list[int]) -> None:
        """Record per-request acceptance from the most recent batch.

        Args:
            accepted_lens: list of accepted draft tokens per request (0 to K)
            drafted_lens: list of K (= num_speculative_tokens proposed) per req
        """
        if not accepted_lens:
            return  # all-greedy or no-spec-eligible batch

        n = len(accepted_lens)
        batch_avg = sum(accepted_lens) / n
        batch_drafted = sum(drafted_lens) / n if drafted_lens else self.current_K

        self.total_accepted += sum(accepted_lens)
        self.total_drafted += sum(drafted_lens) if drafted_lens else self.current_K * n

        self.batches_seen += 1
        self.batches_since_update += 1
        self.batches_since_probe += 1
        self.batches_since_log += 1

        # Warmup: don't tune until we have enough samples
        if self.batches_seen < self.warmup:
            return

        # Probe: every N batches, force a non-zero K to retest if workload
        # changed (e.g., user switched from free-form to tool-call mid-session)
        if (
            self.current_K == 0
            and self.probe_interval > 0
            and self.batches_since_probe >= self.probe_interval
        ):
            non_zero = [s for s in self.steps if s > 0]
            if non_zero:
                self.current_K = non_zero[len(non_zero) // 2]
                self.batches_since_probe = 0
                log.info(
                    "[Genesis P77] PROBE: K 0->%d (every %d batches) to retest "
                    "if workload acceptance recovered",
                    self.current_K, self.probe_interval,
                )
                self.last_K = self.current_K
                self.transitions += 1
                return

        # Update only every N batches (debounce)
        if self.batches_since_update < self.interval:
            return
        self.batches_since_update = 0

        # EMA
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * batch_avg

        # Auto-disable: if accept rate << K, drop to 0
        accept_rate = batch_avg / max(batch_drafted, 1.0)
        if (
            0 in self.steps
            and accept_rate < self.disable_threshold
            and self.current_K > 0
        ):
            # B4 fix v7.62.12: save OLD K as last_K BEFORE the transition
            # (mirrors hysteresis branch line 245). Previous behavior set
            # last_K = 0, breaking the "previous K" semantic and the log
            # message at line 219 used self.last_K which was already
            # overwritten in PRIOR disable cycles.
            _previous_K = self.current_K
            self.last_K = self.current_K
            self.current_K = 0
            self.disabled_steps += 1
            self.transitions += 1
            log.info(
                "[Genesis P77] DISABLE: K %d->0 (batch_avg_accept=%.2f, "
                "accept_rate=%.2f%% < %.2f%% threshold). EMA=%.2f. "
                "Will re-probe in %d batches.",
                _previous_K, batch_avg, accept_rate * 100,
                self.disable_threshold * 100, self.ema, self.probe_interval,
            )
            return

        # Hysteresis: pick highest step <= ema + hyst_up, requiring at least
        # ema - hyst_down to keep current up state (mirrors SGLang)
        candidate = self.steps[0]
        for s in self.steps:
            if self.ema >= s - self.hyst_down:
                candidate = s

        # Apply hyst_up bias: don't grow K unless ema clearly exceeds threshold
        if candidate > self.current_K and self.ema < (candidate + self.hyst_up):
            candidate = self.current_K

        if candidate != self.current_K:
            log.info(
                "[Genesis P77] TRANSITION: K %d->%d (ema=%.2f, batch_avg=%.2f, "
                "accept_rate=%.2f%%)",
                self.current_K, candidate, self.ema, batch_avg, accept_rate * 100,
            )
            self.last_K = self.current_K
            self.current_K = candidate
            self.transitions += 1

        # Periodic stats log
        if self.batches_since_log >= self.log_every:
            self.batches_since_log = 0
            overall_rate = self.total_accepted / max(self.total_drafted, 1)
            log.info(
                "[Genesis P77] STATS: batches=%d K=%d ema=%.2f overall_accept=%.2f%% "
                "transitions=%d disabled_steps=%d",
                self.batches_seen, self.current_K, self.ema,
                overall_rate * 100, self.transitions, self.disabled_steps,
            )

    def get_stats(self) -> dict:
        """Diagnostic snapshot for tests / observability."""
        return {
            "current_K": self.current_K,
            "ema": self.ema,
            "batches_seen": self.batches_seen,
            "transitions": self.transitions,
            "disabled_steps": self.disabled_steps,
            "total_accepted": self.total_accepted,
            "total_drafted": self.total_drafted,
            "overall_accept_rate": (
                self.total_accepted / max(self.total_drafted, 1)
            ),
        }


# ─── Module-level singleton (one controller per process) ────────────────────
_GLOBAL_CONTROLLER: Optional[AdaptiveNgramController] = None


def get_controller() -> AdaptiveNgramController:
    """Lazy-init the singleton controller."""
    global _GLOBAL_CONTROLLER
    if _GLOBAL_CONTROLLER is None:
        _GLOBAL_CONTROLLER = AdaptiveNgramController()
    return _GLOBAL_CONTROLLER


def reset_for_tests() -> None:
    """TESTS ONLY — clear singleton + re-read env."""
    global _GLOBAL_CONTROLLER
    _GLOBAL_CONTROLLER = None


def is_active() -> bool:
    """Returns True when env opt-in flag is set."""
    return os.environ.get(
        "GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K", ""
    ).strip().lower() in ("1", "true", "yes", "on")
