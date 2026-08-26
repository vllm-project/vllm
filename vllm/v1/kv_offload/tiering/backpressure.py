# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class BackpressurePolicy(ABC):
    """Decides what action to take when a detector signals pressure."""

    @abstractmethod
    def should_store(self, detector: BackpressureDetector) -> bool: ...

    @abstractmethod
    def on_store_skipped(self, num_blocks: int) -> None: ...

    @abstractmethod
    def pop_stores_dropped(self) -> tuple[int, int]:
        """Return and reset (stores_dropped, blocks_dropped)."""
        ...

    @abstractmethod
    def reset(self) -> None: ...


class DropStorePolicy(BackpressurePolicy):
    """Silently drop stores to pressured tiers."""

    def __init__(self):
        self._stores_dropped: int = 0
        self._blocks_dropped: int = 0

    def should_store(self, detector) -> bool:
        return not detector.is_under_pressure()

    def on_store_skipped(self, num_blocks) -> None:
        self._stores_dropped += 1
        self._blocks_dropped += num_blocks

    def pop_stores_dropped(self) -> tuple[int, int]:
        stores, blocks = self._stores_dropped, self._blocks_dropped
        self._stores_dropped = 0
        self._blocks_dropped = 0
        return stores, blocks

    def reset(self) -> None:
        self._stores_dropped = 0
        self._blocks_dropped = 0


class ThrottledDropPolicy(BackpressurePolicy):
    """Drop stores proportionally to pressure severity.

    Instead of dropping all stores when pressure is detected, the drop
    rate scales with how far the EMA exceeds the high watermark::

        overshoot = (ema - high_water) / high_water
        drop_rate = min(1.0, overshoot / ramp_factor)

    With the default ``ramp_factor=2.0``, stores ramp from 0% drop at
    the high watermark to 100% drop at 3x the high watermark.  This
    keeps secondary tiers populated under moderate pressure, reducing
    cache misses (and therefore recomputation / ITL regression) while
    still protecting TTFT under severe congestion.
    """

    def __init__(self, ramp_factor: float = 2.0):
        self._stores_dropped: int = 0
        self._blocks_dropped: int = 0
        self._call_count: int = 0
        self._ramp_factor = ramp_factor

    def should_store(self, detector: BackpressureDetector) -> bool:
        if not detector.is_under_pressure():
            return True
        ema = getattr(detector, "store_latency_ema", None)
        high = getattr(detector, "_high", None)
        if ema is None or high is None or high <= 0:
            return False
        overshoot = max(0.0, ema - high) / high
        drop_rate = min(1.0, overshoot / self._ramp_factor)
        if drop_rate <= 0:
            return True
        if drop_rate >= 1.0:
            return False
        self._call_count += 1
        period = max(2, round(1.0 / drop_rate))
        return (self._call_count % period) != 0

    def on_store_skipped(self, num_blocks: int) -> None:
        self._stores_dropped += 1
        self._blocks_dropped += num_blocks

    def pop_stores_dropped(self) -> tuple[int, int]:
        stores, blocks = self._stores_dropped, self._blocks_dropped
        self._stores_dropped = 0
        self._blocks_dropped = 0
        return stores, blocks

    def reset(self) -> None:
        self._stores_dropped = 0
        self._blocks_dropped = 0
        self._call_count = 0


class BackpressureDetector(ABC):
    """Observes store completion signals and determines pressure state."""

    def __init__(
        self,
        policy: BackpressurePolicy | None = None,
    ):
        self._policy = policy or ThrottledDropPolicy()

    @property
    def policy(self) -> BackpressurePolicy:
        return self._policy

    @abstractmethod
    def on_store_completed(self, elapsed_s: float, num_bytes: int) -> None: ...

    @abstractmethod
    def is_under_pressure(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...

    def update(self, submit_time: float, num_bytes: int) -> None:
        """Update pressure state from a completed store job.

        Args:
            submit_time: ``time.monotonic()`` when the job was submitted.
            num_bytes: Total bytes written (num_blocks * block_size_bytes).
        """
        if num_bytes <= 0:
            return
        elapsed = time.monotonic() - submit_time
        self.on_store_completed(elapsed, num_bytes)

    def should_store(self, num_blocks: int) -> bool:
        """Check policy and record skip if rejected."""
        if not self._policy.should_store(self):
            self._policy.on_store_skipped(num_blocks)
            return False
        return True

    @classmethod
    def default_config(
        cls, tier_type: str, *, locality: str | None = None
    ) -> dict | None:
        """Return default constructor kwargs for ``tier_type``, or None.

        Args:
            tier_type: The tier type string (e.g. ``"fs"``, ``"obj"``).
            locality: Optional locality hint from the tier config
                (``"LOCAL"`` or ``"REMOTE"``). When ``"REMOTE"``, tiers
                that would normally get local-storage watermarks (e.g.
                ``"fs"``) receive network watermarks instead.
        """
        return None

    @property
    def stats(self) -> dict[str, float]:
        return {}


class EMABackpressureDetector(BackpressureDetector):
    """EMA of store latency normalized by transfer size.

    The EMA tracks seconds per megabyte (s/MiB) so that the metric is
    comparable regardless of how many blocks are in a job or how large
    each block is.  Water marks are in the same unit.

    Default water marks are derived from fio benchmarks on the WDC H100
    cluster.  Two presets are provided:

      LOCAL (NVMe/SSD): NVMe sustains ~5 GB/s writes.
        high=0.005 s/MiB (~200 MB/s) catches severe congestion;
        low=0.001 s/MiB (~1 GB/s) requires meaningful recovery.

      NETWORK (CephFS, object store, ``obj``/``p2p`` tiers, or any
        tier with ``"locality": "REMOTE"``): CephFS sustains ~1.5 GB/s.
        high=0.020 s/MiB (~50 MB/s); low=0.005 s/MiB (~200 MB/s).

    ``obj`` and ``p2p`` tiers get NETWORK defaults automatically. An
    ``fs`` tier defaults to LOCAL (appropriate for NVMe/SSD); set
    ``"locality": "REMOTE"`` in the tier config for network-backed
    filesystems like CephFS.
    """

    _MIB = 1 << 20

    # EMA smoothing factor: higher values (→1) react faster to latency
    # spikes but are noisier; lower values (→0) smooth more but lag.
    DEFAULT_ALPHA = 0.3

    # Number of store completions to collect before seeding the EMA.
    # During warmup, pressure is never signalled; the EMA is initialized
    # to the mean of the warmup samples to avoid cold-start false positives.
    DEFAULT_WARMUP_COMPLETIONS = 3

    # When under pressure and no completions arrive, the EMA decays
    # toward zero so the detector can eventually recover.  The half-life
    # controls how quickly: after this many seconds of silence the EMA
    # halves.  With a 2 s half-life and high_water=0.005, an EMA frozen
    # at 0.010 reaches low_water=0.001 in ~6.6 s.
    DEFAULT_DECAY_HALF_LIFE_S = 2.0

    # After pressure clears, stores are throttled to 50% for this many
    # seconds to prevent a burst of stores from re-overwhelming the tier.
    DEFAULT_COOLDOWN_S = 1.0

    # Number of consecutive sub-low-watermark completions before the
    # detector enters a "healthy" fast-path that bypasses policy checks.
    _HEALTHY_THRESHOLD = 10

    LOCAL_HIGH_WATER_S = 0.005
    LOCAL_LOW_WATER_S = 0.001

    NETWORK_HIGH_WATER_S = 0.020
    NETWORK_LOW_WATER_S = 0.005

    _NETWORK_TIER_TYPES = frozenset({"obj"})

    @classmethod
    def default_config(
        cls, tier_type: str, *, locality: str | None = None
    ) -> dict | None:
        is_remote = locality is not None and locality.upper() == "REMOTE"
        if is_remote or tier_type in cls._NETWORK_TIER_TYPES:
            return {
                "high_water_s": cls.NETWORK_HIGH_WATER_S,
                "low_water_s": cls.NETWORK_LOW_WATER_S,
            }
        return {
            "high_water_s": cls.LOCAL_HIGH_WATER_S,
            "low_water_s": cls.LOCAL_LOW_WATER_S,
        }

    def __init__(
        self,
        high_water_s: float,
        low_water_s: float,
        alpha: float = DEFAULT_ALPHA,
        warmup_completions: int = DEFAULT_WARMUP_COMPLETIONS,
        decay_half_life_s: float = DEFAULT_DECAY_HALF_LIFE_S,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        policy: BackpressurePolicy | None = None,
    ):
        super().__init__(policy=policy)
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if low_water_s > high_water_s:
            raise ValueError(
                f"low_water_s ({low_water_s}) must be <= high_water_s ({high_water_s})"
            )

        self._alpha = alpha
        self._high = high_water_s
        self._low = low_water_s
        self._warmup_completions = warmup_completions
        self._decay_half_life_s = decay_half_life_s
        self._cooldown_s = cooldown_s
        # Exponential moving average of store latency in seconds.
        self._ema: float = 0.0
        # Whether the tier is currently considered under pressure.
        self._under_pressure: bool = False
        # Number of store completions observed (for warmup gating).
        self._completions: int = 0
        # Samples collected during warmup; used to seed the EMA.
        self._warmup_samples: list[float] = []
        # Monotonic timestamp of the last EMA update; used for idle decay.
        self._last_update: float = 0.0
        # Consecutive completions with EMA below low watermark.
        self._healthy_streak: int = 0
        # Monotonic timestamp when pressure was last cleared (for cooldown).
        self._pressure_cleared_at: float = 0.0
        self._cooldown_count: int = 0

    def _apply_idle_decay(self) -> None:
        """Decay the EMA toward zero based on elapsed idle time.

        Uses the formula ``ema * 0.5^(dt / half_life)`` so the EMA
        halves every ``decay_half_life_s`` seconds of silence.  This
        lets the detector recover from pressure when no stores are
        attempted (and therefore no completions arrive to update the
        EMA organically).
        """
        now = time.monotonic()
        dt = now - self._last_update
        if dt <= 0 or self._decay_half_life_s <= 0:
            return
        self._ema *= 0.5 ** (dt / self._decay_half_life_s)
        self._last_update = now

    def on_store_completed(self, elapsed_s: float, num_bytes: int) -> None:
        now = time.monotonic()
        s_per_mib = elapsed_s / (num_bytes / self._MIB)
        self._completions += 1
        if self._completions <= self._warmup_completions:
            self._warmup_samples.append(s_per_mib)
            if self._completions == self._warmup_completions:
                self._ema = sum(self._warmup_samples) / len(self._warmup_samples)
                self._last_update = now
            return
        self._ema = self._alpha * s_per_mib + (1 - self._alpha) * self._ema
        self._last_update = now
        if self._ema > self._high:
            self._under_pressure = True
            self._pressure_cleared_at = 0.0
            self._healthy_streak = 0
        elif self._ema < self._low:
            if self._under_pressure:
                self._pressure_cleared_at = now
                self._cooldown_count = 0
            self._under_pressure = False
            self._healthy_streak += 1
        else:
            self._healthy_streak = 0

    def is_under_pressure(self) -> bool:
        if self._under_pressure and self._completions >= self._warmup_completions:
            self._apply_idle_decay()
            if self._ema < self._low:
                self._under_pressure = False
                self._pressure_cleared_at = time.monotonic()
                self._cooldown_count = 0
        return self._under_pressure

    def is_healthy(self) -> bool:
        """True when the tier has been consistently fast.

        After ``_HEALTHY_THRESHOLD`` consecutive store completions with
        EMA below the low watermark (and no active pressure), the
        detector enters a healthy fast-path: ``should_store()`` returns
        ``True`` immediately, skipping the policy check entirely.
        """
        return (
            not self._under_pressure and self._healthy_streak >= self._HEALTHY_THRESHOLD
        )

    def should_store(self, num_blocks: int) -> bool:
        if self.is_healthy():
            return True
        if (
            not self._under_pressure
            and self._pressure_cleared_at > 0
            and time.monotonic() - self._pressure_cleared_at < self._cooldown_s
        ):
            self._cooldown_count += 1
            if self._cooldown_count % 2 == 0:
                self._policy.on_store_skipped(num_blocks)
                return False
        return super().should_store(num_blocks)

    def reset(self) -> None:
        self._ema = 0.0
        self._under_pressure = False
        self._completions = 0
        self._warmup_samples.clear()
        self._last_update = 0.0
        self._healthy_streak = 0
        self._pressure_cleared_at = 0.0
        self._cooldown_count = 0
        self._policy.reset()

    @property
    def store_latency_ema(self) -> float:
        return self._ema

    @store_latency_ema.setter
    def store_latency_ema(self, value: float) -> None:
        self._ema = value

    @property
    def stats(self) -> dict[str, float]:
        return {"store_latency_ema": self._ema}
