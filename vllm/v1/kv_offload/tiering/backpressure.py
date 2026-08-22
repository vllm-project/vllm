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


class BackpressureDetector(ABC):
    """Observes store completion signals and determines pressure state."""

    def __init__(
        self,
        policy: BackpressurePolicy | None = None,
    ):
        self._policy = policy or DropStorePolicy()

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

    LOCAL_HIGH_WATER_S = 0.005
    LOCAL_LOW_WATER_S = 0.001

    NETWORK_HIGH_WATER_S = 0.020
    NETWORK_LOW_WATER_S = 0.005

    _NETWORK_TIER_TYPES = frozenset({"obj", "p2p"})

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
        # Exponential moving average of store latency in seconds.
        self._ema: float = 0.0
        # Whether the tier is currently considered under pressure.
        self._under_pressure: bool = False
        # Number of store completions observed (for warmup gating).
        self._completions: int = 0
        # Samples collected during warmup; used to seed the EMA.
        self._warmup_samples: list[float] = []

    def on_store_completed(self, elapsed_s: float, num_bytes: int) -> None:
        s_per_mib = elapsed_s / (num_bytes / self._MIB)
        self._completions += 1
        if self._completions <= self._warmup_completions:
            self._warmup_samples.append(s_per_mib)
            if self._completions == self._warmup_completions:
                self._ema = sum(self._warmup_samples) / len(self._warmup_samples)
            return
        self._ema = self._alpha * s_per_mib + (1 - self._alpha) * self._ema
        if self._ema > self._high:
            self._under_pressure = True
        elif self._ema < self._low:
            self._under_pressure = False

    def is_under_pressure(self) -> bool:
        return self._under_pressure

    def reset(self) -> None:
        self._ema = 0.0
        self._under_pressure = False
        self._completions = 0
        self._warmup_samples.clear()
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
