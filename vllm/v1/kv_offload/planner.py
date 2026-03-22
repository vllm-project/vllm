# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass


@dataclass(frozen=True)
class HybridOffloadPlanner:
    """Plan fixed-size external offload units for hybrid KV groups.

    This planner is intentionally pure and scheduler-facing. It does not assume
    that every group can be transferred as a native full GPU block. When a
    group's offload unit is smaller than its GPU block size, that group needs
    partial-state transfer support in the worker/backend.
    """

    hash_block_size: int
    gpu_block_sizes: tuple[int, ...]
    fixed_chunk_size: int

    def __post_init__(self) -> None:
        if self.hash_block_size <= 0:
            raise ValueError("hash_block_size must be positive")
        if self.fixed_chunk_size <= 0:
            raise ValueError("fixed_chunk_size must be positive")
        if self.fixed_chunk_size % self.hash_block_size != 0:
            raise ValueError(
                "fixed_chunk_size must be divisible by hash_block_size"
            )
        if not self.gpu_block_sizes:
            raise ValueError("gpu_block_sizes must be non-empty")
        if any(block_size <= 0 for block_size in self.gpu_block_sizes):
            raise ValueError("gpu_block_sizes must be positive")
        if any(block_size % self.hash_block_size != 0 for block_size in self.gpu_block_sizes):
            raise ValueError("All gpu_block_sizes must be divisible by hash_block_size")

    @property
    def offload_unit_sizes(self) -> tuple[int, ...]:
        units: list[int] = []
        for gpu_block_size in self.gpu_block_sizes:
            if gpu_block_size <= self.fixed_chunk_size:
                units.append(gpu_block_size)
            elif gpu_block_size % self.fixed_chunk_size == 0:
                units.append(self.fixed_chunk_size)
            else:
                units.append(gpu_block_size)
        return tuple(units)

    @property
    def requires_partial_group_offload(self) -> tuple[bool, ...]:
        return tuple(
            unit_size < gpu_block_size
            for unit_size, gpu_block_size in zip(
                self.offload_unit_sizes, self.gpu_block_sizes
            )
        )

    @property
    def requires_partial_group_offload_any(self) -> bool:
        return any(self.requires_partial_group_offload)

    @property
    def group_hash_factors(self) -> tuple[int, ...]:
        return tuple(
            unit_size // self.hash_block_size for unit_size in self.offload_unit_sizes
        )

    def storeable_prefix_tokens(self, request_tokens: int) -> int:
        if request_tokens <= 0:
            return 0
        per_group_covered_tokens = tuple(
            (request_tokens // unit_size) * unit_size
            for unit_size in self.offload_unit_sizes
        )
        return self.loadable_prefix_tokens(per_group_covered_tokens)

    def loadable_prefix_tokens(self, group_covered_tokens: tuple[int, ...]) -> int:
        if len(group_covered_tokens) != len(self.gpu_block_sizes):
            raise ValueError("group_covered_tokens must match gpu_block_sizes length")
        if any(tokens < 0 for tokens in group_covered_tokens):
            raise ValueError("group_covered_tokens must be non-negative")

        common_prefix = min(group_covered_tokens, default=0)
        return common_prefix - (common_prefix % self.hash_block_size)
