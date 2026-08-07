# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import SecondaryTierManager


class BackpressureDetector(ABC):
    """Observes store completion signals and determines pressure state."""

    @abstractmethod
    def on_store_completed(self, elapsed_s: float) -> None: ...

    @abstractmethod
    def is_under_pressure(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...

    def update(self, submit_time: float) -> None:
        """Update pressure state from a completed store job's submit_time."""
        elapsed = time.monotonic() - submit_time
        self.on_store_completed(elapsed)

    @property
    def stats(self) -> dict[str, float]:
        return {}


class EMABackpressureDetector(BackpressureDetector):
    """EMA of store latency with high/low water mark hysteresis."""

    DEFAULT_ALPHA = 0.3
    DEFAULT_HIGH_WATER_S = 1.0
    DEFAULT_LOW_WATER_S = 0.5
    DEFAULT_WARMUP_COMPLETIONS = 3

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        high_water_s: float = DEFAULT_HIGH_WATER_S,
        low_water_s: float = DEFAULT_LOW_WATER_S,
        warmup_completions: int = DEFAULT_WARMUP_COMPLETIONS,
    ):
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

    def on_store_completed(self, elapsed_s: float) -> None:
        self._completions += 1
        if self._completions <= self._warmup_completions:
            self._warmup_samples.append(elapsed_s)
            if self._completions == self._warmup_completions:
                self._ema = sum(self._warmup_samples) / len(self._warmup_samples)
            return
        self._ema = self._alpha * elapsed_s + (1 - self._alpha) * self._ema
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

    @property
    def store_latency_ema(self) -> float:
        return self._ema

    @store_latency_ema.setter
    def store_latency_ema(self, value: float) -> None:
        self._ema = value

    @property
    def stats(self) -> dict[str, float]:
        return {"store_latency_ema": self._ema}


class BackpressurePolicy(ABC):
    """Decides what action to take for a store to a pressured tier."""

    @abstractmethod
    def should_store(
        self,
        tier_key: SecondaryTierManager,
        detector: BackpressureDetector,
    ) -> bool: ...

    @abstractmethod
    def on_store_skipped(
        self, tier_key: SecondaryTierManager, num_blocks: int
    ) -> None: ...

    @abstractmethod
    def pop_stores_dropped(self, tier_key: SecondaryTierManager) -> tuple[int, int]:
        """Return and reset (stores_dropped, blocks_dropped) for a tier."""
        ...

    @abstractmethod
    def reset(self) -> None: ...


class DropStorePolicy(BackpressurePolicy):
    """Silently drop stores to pressured tiers."""

    def __init__(self):
        self._stores_dropped: dict[SecondaryTierManager, int] = {}
        self._blocks_dropped: dict[SecondaryTierManager, int] = {}

    def should_store(self, tier_key, detector) -> bool:
        return not detector.is_under_pressure()

    def on_store_skipped(self, tier_key, num_blocks) -> None:
        self._stores_dropped[tier_key] = self._stores_dropped.get(tier_key, 0) + 1
        self._blocks_dropped[tier_key] = (
            self._blocks_dropped.get(tier_key, 0) + num_blocks
        )

    def pop_stores_dropped(self, tier_key) -> tuple[int, int]:
        stores = self._stores_dropped.get(tier_key, 0)
        blocks = self._blocks_dropped.get(tier_key, 0)
        self._stores_dropped[tier_key] = 0
        self._blocks_dropped[tier_key] = 0
        return stores, blocks

    def reset(self) -> None:
        self._stores_dropped.clear()
        self._blocks_dropped.clear()
