# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer row ranges shared by operator dispatch, warmup and kernel libraries."""

from bisect import bisect_right
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from torch.compiler import is_compiling

T = TypeVar("T")


class KernelPhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"
    VERIFY = "verify"
    UNKNOWN = "unknown"


def current_kernel_phase() -> KernelPhase:
    """Read host context lazily so low-level config imports stay independent."""
    from vllm.forward_context import get_kernel_phase

    return get_kernel_phase()


def m_in_range(m: int, *, start: int | None = None, stop: int | None = None) -> bool:
    """Match `[start, stop)` without rounding, padding or reading device data."""
    return (start is None or m >= start) and (stop is None or m < stop)


def select_nearest_m(points: Mapping[int, T], m: int) -> T:
    """Select a legacy nearest-point config with insertion-order tie breaking.

    For an exported interval table, use `MBucketTable.from_nearest(points)`.
    Keep the linear lookup for mutable legacy dictionaries on the hot path.
    """
    return points[min(points, key=lambda row: abs(row - m))]


@dataclass(frozen=True)
class MRange:
    start: int
    stop: int | None

    def __post_init__(self) -> None:
        if self.start < 0 or (self.stop is not None and self.stop <= self.start):
            raise ValueError(f"Invalid M range: {self}")

    def contains(self, m: int) -> bool:
        return m_in_range(m, start=self.start, stop=self.stop)


@dataclass(frozen=True)
class MBucketTable(Generic[T]):
    """A complete partition of nonnegative M with one value per interval.

    `stops=(17, 1025)` means `[0,17)`, `[17,1025)`, `[1025,infinity)`.
    Config values may identify implementations or their launch configurations.
    Construct tables during initialization, never during a kernel launch.
    """

    stops: tuple[int, ...]
    values: tuple[T, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.stops) + 1:
            raise ValueError("Each M interval needs one value")
        if any(a >= b for a, b in zip((0, *self.stops), self.stops)):
            raise ValueError("M stops must be positive and strictly increasing")

    def index(self, m: int) -> int:
        if m < 0:
            raise ValueError("M must be nonnegative")
        if isinstance(m, int) and not is_compiling():
            return bisect_right(self.stops, m)
        for index, stop in enumerate(self.stops):
            if m < stop:
                return index
        return len(self.stops)

    def select(self, m: int) -> T:
        return self.values[self.index(m)]

    def interval(self, m: int) -> MRange:
        index = self.index(m)
        return MRange(
            0 if index == 0 else self.stops[index - 1],
            self.stops[index] if index < len(self.stops) else None,
        )

    @classmethod
    def from_points(cls, points: Mapping[int, T], default: T) -> "MBucketTable[T]":
        """Compress equal adjacent points, preserving holes and the default.

        An isolated tuned M stays a singleton interval; this does not extend
        point measurements to unmeasured rows.
        """
        if any(m < 0 for m in points):
            raise ValueError("Tuned M must be nonnegative")
        boundaries = sorted({0, *(m for m in points), *(m + 1 for m in points)})
        stops: list[int] = []
        values = [points.get(0, default)]
        for start in boundaries[1:]:
            value = points.get(start, default)
            if value != values[-1]:
                stops.append(start)
                values.append(value)
        return cls(tuple(stops), tuple(values))

    @classmethod
    def from_nearest(cls, points: Mapping[int, T]) -> "MBucketTable[T]":
        """Convert nearest-point lookup to intervals, preserving insertion ties."""
        if not points or any(m < 0 for m in points):
            raise ValueError("Nearest lookup needs nonnegative tuned points")
        insertion_order = {m: index for index, m in enumerate(points)}
        rows = sorted(points)
        stops = tuple(
            (a + b + 1) // 2
            + int((a + b) % 2 == 0 and insertion_order[a] < insertion_order[b])
            for a, b in zip(rows, rows[1:])
        )
        return cls(stops, tuple(points[m] for m in rows))


@dataclass(frozen=True)
class KernelBucketKey:
    operator: str
    phase: KernelPhase
    interval: MRange
    specialization: tuple[Hashable, ...]
    axis: str = "M"


@dataclass(frozen=True)
class PhaseBucketTable(Generic[T]):
    """Separate execution phases; shared graphs and warmup use `fallback`.

    A missing phase never borrows another phase's implementation. The caller
    supplies the phase of this operation, which can differ from a mixed batch.
    """

    operator: str
    fallback: MBucketTable[T]
    prefill: MBucketTable[T] | None = None
    decode: MBucketTable[T] | None = None
    mixed: MBucketTable[T] | None = None
    verify: MBucketTable[T] | None = None
    axis: str = "M"
    _phase_independent: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_phase_independent",
            all(
                table is None or table is self.fallback
                for table in (self.prefill, self.decode, self.mixed, self.verify)
            ),
        )

    def select_current(self, m: int) -> T:
        """Skip context lookup when every phase uses the same fallback table."""
        if self._phase_independent:
            return self.fallback.select(m)
        return self.select(m, phase=current_kernel_phase())

    def table(self, phase: KernelPhase) -> MBucketTable[T]:
        if not isinstance(phase, KernelPhase):
            raise TypeError("Pass an explicit KernelPhase")
        if phase == KernelPhase.UNKNOWN:
            return self.fallback
        return getattr(self, phase.value) or self.fallback

    def select(self, m: int, *, phase: KernelPhase) -> T:
        return self.table(phase).select(m)

    def key(
        self, m: int, *, phase: KernelPhase, specialization: tuple[Hashable, ...]
    ) -> KernelBucketKey:
        if phase == KernelPhase.UNKNOWN:
            raise ValueError("An Agent Kernel key needs a known execution phase")
        return KernelBucketKey(
            self.operator,
            phase,
            self.table(phase).interval(m),
            specialization,
            self.axis,
        )


def phase_from_counts(
    *, prefill_tokens: int, decode_tokens: int, has_draft_tokens: bool = False
) -> KernelPhase:
    """Classify host execution metadata, independently of M and graph padding."""
    if prefill_tokens < 0 or decode_tokens < 0:
        raise ValueError("Execution token counts must be nonnegative")
    if prefill_tokens and decode_tokens:
        return KernelPhase.MIXED
    if prefill_tokens:
        return KernelPhase.PREFILL
    if decode_tokens:
        return KernelPhase.VERIFY if has_draft_tokens else KernelPhase.DECODE
    return KernelPhase.UNKNOWN
