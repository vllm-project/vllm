# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Common types for CPU KV offload.

Provides backward-compatible :class:`CPULoadStoreSpec` (legacy packed-row
addressing) and the new :class:`CompactCPULoadStoreSpec` / :class:`CompactCPUAddress`
for per-payload-class compact CPU storage.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, LoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_ALLOCATED_BYTES = "vllm:kv_offload_cpu_allocated_bytes"
    CPU_FREE_BYTES = "vllm:kv_offload_cpu_free_bytes"
    CPU_LARGEST_FREE_EXTENT_BYTES = "vllm:kv_offload_cpu_largest_free_extent_bytes"
    CPU_FRAGMENTATION_RATIO = "vllm:kv_offload_cpu_fragmentation_ratio"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """Spec for loading/storing a KV block to CPU memory (legacy packed rows)."""


@dataclass(frozen=True)
class CompactCPUAddressSpan:
    """One physical span backing a logical compact CPU payload."""

    byte_offset: int
    logical_length: int
    allocated_length: int

    def __post_init__(self) -> None:
        if self.byte_offset < 0:
            raise ValueError("span byte_offset must be non-negative")
        if self.logical_length <= 0:
            raise ValueError("span logical_length must be positive")
        if self.allocated_length < self.logical_length:
            raise ValueError("span allocated_length must cover logical_length")


@dataclass(frozen=True)
class CompactCPUAddress:
    """Byte-granularity address of one compact offload block in a CPU extent.

    Each offload *key* (one OffloadKey) has exactly one ``CompactCPUAddress``.

    Fields
    ------
    byte_offset:
        Byte offset from the start of this group's CPU extent (pool base).
    logical_length:
        Real (un-padded) payload bytes for this offload block.
    allocated_length:
        Actually allocated bytes (logical_length + alignment padding).
    group_idx:
        Index of the KV cache group this block belongs to.
    spans:
        Optional physical sub-spans for multi-page allocations.  When
        empty, the address describes a single contiguous extent.
    """

    byte_offset: int
    logical_length: int
    allocated_length: int
    group_idx: int = 0
    spans: tuple[CompactCPUAddressSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.group_idx < 0:
            raise ValueError(f"group_idx must be non-negative, got {self.group_idx}")
        if self.byte_offset < 0:
            raise ValueError(
                f"byte_offset must be non-negative, got {self.byte_offset}"
            )
        if self.logical_length <= 0:
            raise ValueError(
                f"logical_length must be positive, got {self.logical_length}"
            )
        if self.allocated_length < self.logical_length:
            raise ValueError(
                f"allocated_length ({self.allocated_length}) must be >= "
                f"logical_length ({self.logical_length})"
            )
        if self.spans:
            if self.byte_offset != self.spans[0].byte_offset:
                raise ValueError("byte_offset must match the first physical span")
            if sum(span.logical_length for span in self.spans) != self.logical_length:
                raise ValueError(
                    "physical spans must cover the logical payload exactly"
                )
            if (
                sum(span.allocated_length for span in self.spans)
                != self.allocated_length
            ):
                raise ValueError("physical spans must cover allocated_length exactly")
            physical_intervals = sorted(
                (span.byte_offset, span.byte_offset + span.allocated_length)
                for span in self.spans
            )
            if any(
                right_start < left_end
                for (_, left_end), (right_start, _) in zip(
                    physical_intervals, physical_intervals[1:]
                )
            ):
                raise ValueError("physical spans must not overlap")

    @property
    def physical_spans(self) -> tuple[CompactCPUAddressSpan, ...]:
        if self.spans:
            return self.spans
        return (
            CompactCPUAddressSpan(
                self.byte_offset, self.logical_length, self.allocated_length
            ),
        )


class CompactCPULoadStoreSpec(LoadStoreSpec):
    """Per-payload-class compact CPU addressing.

    Carries one :class:`CompactCPUAddress` per offload key, co-indexed
    with the keys.  Does not use ``block_ids`` (no fake IDs).
    """

    def __init__(self, compact_addresses: list[CompactCPUAddress]):
        self.compact_addresses: list[CompactCPUAddress] = list(compact_addresses)

    def __repr__(self) -> str:
        return f"CompactCPULoadStoreSpec({self.compact_addresses})"
