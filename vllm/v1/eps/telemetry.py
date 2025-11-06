# ABOUTME: EPS telemetry containers for per-step counters.
# ABOUTME: Aggregates unique-block statistics for EigenPage Summaries.

from dataclasses import dataclass, asdict, fields
from typing import Iterable, Sequence


def blocks_to_groups(num_blocks: int, group_blocks: int) -> int:
    if num_blocks <= 0:
        return 0
    return (num_blocks + group_blocks - 1) // group_blocks


def collect_unique_blocks(
    rows: Sequence[Sequence[int]],
    counts: Sequence[int],
) -> set[int]:
    unique: set[int] = set()
    for row, count in zip(rows, counts):
        take = int(count)
        if take <= 0:
            continue
        unique.update(int(b) for b in row[:take])
    return unique


@dataclass
class EpsStepCounters:
    layers: int = 0
    groups_total: int = 0
    groups_kept: int = 0
    blocks_total: int = 0
    blocks_kept: int = 0
    unique_blocks_total: int = 0
    unique_blocks_kept: int = 0
    kv_bytes_total: int = 0
    kv_bytes_kept: int = 0
    pages_total: int = 0
    pages_visited: int = 0
    pages_skipped: int = 0
    pages_unique_total: int = 0
    pages_unique_kept: int = 0
    eps_prepass_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def groups_dropped(self) -> int:
        return self.groups_total - self.groups_kept

    @property
    def blocks_dropped(self) -> int:
        return self.blocks_total - self.blocks_kept

    @property
    def unique_blocks_dropped(self) -> int:
        return self.unique_blocks_total - self.unique_blocks_kept

    @property
    def kv_bytes_saved(self) -> int:
        return self.kv_bytes_total - self.kv_bytes_kept


    def add_from(self, other: "EpsStepCounters") -> None:
        if other is None:
            return
        for field in fields(EpsStepCounters):
            name = field.name
            value = getattr(other, name)
            setattr(self, name, getattr(self, name) + value)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
