# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class AttentionHeadPartition:
    """Per-rank attention/GQA head mapping.

    The local Q heads are tensor-parallel sharded evenly. The local KV heads
    are the global KV heads touched by those local Q heads. This preserves GQA
    semantics when a Q shard crosses a KV-group boundary.
    """

    total_num_heads: int
    total_num_kv_heads: int
    tp_size: int
    tp_rank: int

    def __post_init__(self) -> None:
        if self.total_num_heads <= 0:
            raise ValueError("total_num_heads must be positive")
        if self.total_num_kv_heads <= 0:
            raise ValueError("total_num_kv_heads must be positive")
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                f"tp_rank must be in [0, {self.tp_size}), got {self.tp_rank}"
            )
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError(
                f"total_num_heads ({self.total_num_heads}) must be divisible "
                f"by tp_size ({self.tp_size})"
            )
        if self.total_num_heads % self.total_num_kv_heads != 0:
            raise ValueError(
                f"total_num_heads ({self.total_num_heads}) must be divisible "
                f"by total_num_kv_heads ({self.total_num_kv_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"local num_heads ({self.num_heads}) must be divisible by "
                f"local num_kv_heads ({self.num_kv_heads})"
            )

    @property
    def q_per_kv(self) -> int:
        return self.total_num_heads // self.total_num_kv_heads

    @property
    def num_heads(self) -> int:
        return self.total_num_heads // self.tp_size

    @property
    def q_start(self) -> int:
        return self.tp_rank * self.num_heads

    @property
    def q_end(self) -> int:
        return self.q_start + self.num_heads

    @property
    def q_head_indices(self) -> tuple[int, ...]:
        return tuple(range(self.q_start, self.q_end))

    @property
    def kv_start(self) -> int:
        return self.q_start // self.q_per_kv

    @property
    def kv_end(self) -> int:
        return (self.q_end + self.q_per_kv - 1) // self.q_per_kv

    @property
    def unique_kv_head_indices(self) -> tuple[int, ...]:
        return tuple(range(self.kv_start, self.kv_end))

    @property
    def q_to_unique_kv_head_indices(self) -> tuple[int, ...]:
        return tuple(q // self.q_per_kv for q in self.q_head_indices)

    @property
    def local_q_per_kv_slot(self) -> int:
        """Number of local Q heads represented by each local KV slot.

        Current attention backends map local Q heads to local KV heads by
        fixed-size contiguous groups. When a TP Q range cuts across global GQA
        groups, we represent a single global KV head by multiple local KV slots.
        """

        slot_size = 0
        for rank in range(self.tp_size):
            q_start = rank * self.num_heads
            q_end = q_start + self.num_heads
            kv_start = q_start // self.q_per_kv
            kv_end = (q_end + self.q_per_kv - 1) // self.q_per_kv
            for kv_idx in range(kv_start, kv_end):
                overlap_start = max(q_start, kv_idx * self.q_per_kv)
                overlap_end = min(q_end, (kv_idx + 1) * self.q_per_kv)
                count = overlap_end - overlap_start
                if count > 0:
                    slot_size = count if slot_size == 0 else gcd(slot_size, count)
        return slot_size

    @property
    def kv_head_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        for kv_idx in self.unique_kv_head_indices:
            overlap_start = max(self.q_start, kv_idx * self.q_per_kv)
            overlap_end = min(self.q_end, (kv_idx + 1) * self.q_per_kv)
            slot_count = (overlap_end - overlap_start) // self.local_q_per_kv_slot
            indices.extend([kv_idx] * slot_count)
        return tuple(indices)

    @property
    def num_kv_heads(self) -> int:
        return len(self.kv_head_indices)

    @property
    def q_to_local_kv_indices(self) -> tuple[int, ...]:
        return tuple(i // self.local_q_per_kv_slot for i in range(self.num_heads))

    @property
    def has_overlapping_kv_partition(self) -> bool:
        ideal = self.total_num_kv_heads / self.tp_size
        return self.num_kv_heads != ideal


def make_attention_head_partition(
    *,
    total_num_heads: int,
    total_num_kv_heads: int,
    tp_size: int,
    tp_rank: int,
) -> AttentionHeadPartition:
    return AttentionHeadPartition(
        total_num_heads=total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
