# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable


class BackpressureDetector(ABC):
    """Observes store completion signals and determines pressure state."""

    @abstractmethod
    def on_store_completed(self, elapsed_s: float) -> None: ...

    @abstractmethod
    def is_under_pressure(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    def stats(self) -> dict[str, float]:
        return {}


class EMABackpressureDetector(BackpressureDetector):
    """EMA of store latency with high/low water mark hysteresis."""

    DEFAULT_ALPHA = 0.3
    DEFAULT_HIGH_WATER_S = 1.0
    DEFAULT_LOW_WATER_S = 0.5

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        high_water_s: float = DEFAULT_HIGH_WATER_S,
        low_water_s: float = DEFAULT_LOW_WATER_S,
    ):
        self._alpha = alpha
        self._high = high_water_s
        self._low = low_water_s
        self._ema: float = 0.0
        self._under_pressure: bool = False

    def on_store_completed(self, elapsed_s: float) -> None:
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
        tier_key: Hashable,
        detector: BackpressureDetector,
    ) -> bool: ...

    @abstractmethod
    def on_store_skipped(self, tier_key: Hashable) -> None: ...

    @abstractmethod
    def get_stores_dropped(self, tier_key: Hashable) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class DropStorePolicy(BackpressurePolicy):
    """Silently drop stores to pressured tiers."""

    def __init__(self):
        self._stores_dropped: dict[Hashable, int] = {}

    def should_store(self, tier_key, detector) -> bool:
        return not detector.is_under_pressure()

    def on_store_skipped(self, tier_key) -> None:
        self._stores_dropped[tier_key] = self._stores_dropped.get(tier_key, 0) + 1

    def get_stores_dropped(self, tier_key) -> int:
        count = self._stores_dropped.get(tier_key, 0)
        self._stores_dropped[tier_key] = 0
        return count

    def reset(self) -> None:
        self._stores_dropped.clear()
