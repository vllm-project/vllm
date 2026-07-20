# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure accounting helpers for compact CPU KV offload layouts.

The current CPU backend allocates one full packed GPU row for every offload
key.  These helpers describe that representation and the compact alternative
without changing allocation or transfer behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export neutral compact geometry symbols so existing callers (tests,
# kv_cache_interface, kv_cache_utils) continue to find them at the
# cpu.compact_accounting import path.
from vllm.v1.kv_offload.compact_geometry import (
    CompactGroupCharge,
    KVCacheConfig,
    _layer_specs,
    _real_page_size_bytes,
    build_compact_group_charges,
)

__all__ = [
    "CompactGroupCharge",
    "CompactLayoutAccounting",
    "GroupCompactAccounting",
    "GroupCompactSlice",
    "PackedSlotAccounting",
    "build_compact_group_charges",
    "build_compact_layout_accounting",
]


@dataclass(frozen=True)
class PackedSlotAccounting:
    """One physical slice in a packed GPU manager-block row."""

    offset_bytes: int
    padded_bytes_per_gpu_block: int
    shared_by: tuple[str, ...]


@dataclass(frozen=True)
class GroupCompactSlice:
    """One ordered slice descriptor within a group's compact transfer plan.

    Describes one packed GPU slot (contiguous byte range within a packed
    GPU block row) attributed to a single KV cache layer within the group.
    ``real_bytes_per_gpu_block`` and ``padded_bytes_per_gpu_block`` are
    per-GPU-block per-rank values (before ``block_size_factor`` scaling).

    Order within ``GroupCompactAccounting.slices`` matches the group's
    canonical ``layer_names`` order so that compact transfer can reconstruct
    slices from the collapsed packed whole-row ref without relying on
    ``CanonicalKVCaches.group_data_refs`` (which collapses to whole row on
    the active packed registration path).
    """

    offset_bytes: int
    real_bytes_per_gpu_block: int
    padded_bytes_per_gpu_block: int
    layer_name: str
    shared_by: tuple[str, ...]


@dataclass(frozen=True)
class GroupCompactAccounting:
    """Compact payload charge for one scheduler KV group.

    ``slices`` is ordered by the group's canonical ``layer_names`` and
    provides the full per-slice provenance needed for compact transfer
    planning.  Aggregate ``compact_real_bytes_per_rank`` and
    ``compact_padded_bytes_per_rank`` equal
    ``sum(s.real_bytes_per_gpu_block for s in slices) * block_size_factor``
    and ``sum(s.padded_bytes_per_gpu_block for s in slices) *
    block_size_factor`` respectively.
    """

    group_idx: int
    native_block_tokens: int
    slices: tuple[GroupCompactSlice, ...]
    compact_real_bytes_per_rank: int
    compact_padded_bytes_per_rank: int

    @property
    def slot_offsets(self) -> tuple[int, ...]:
        """Packed GPU row offsets for each slice in layer-names order."""
        return tuple(s.offset_bytes for s in self.slices)


@dataclass(frozen=True)
class CompactLayoutAccounting:
    """Proposed compact group charges (packed-row representation).

    Excludes dead fields from the final product:
    ``packed_row_stride_bytes_per_rank``, ``current_slot_bytes_server``,
    ``current_num_slots``, and the ``compact_common_prefix_capacity_tokens``
    method (a DeepSeek-specific capacity predictor that depends on bounded-tail
    reservation, which is deferred to a later phase).
    """

    world_size: int
    block_size_factor: int
    cpu_budget_bytes: int
    groups: tuple[GroupCompactAccounting, ...]


def _packed_slots(
    kv_cache_config: KVCacheConfig,
) -> tuple[int, tuple[PackedSlotAccounting, ...]]:
    tensors = kv_cache_config.kv_cache_tensors
    if not tensors or not any(t.block_stride for t in tensors):
        raise ValueError("compact packed accounting requires a packed KV layout")
    if not all(t.block_stride for t in tensors):
        raise ValueError("mixed packed and unpacked KV tensors are unsupported")

    strides = {t.block_stride for t in tensors}
    if len(strides) != 1:
        raise ValueError("packed KV tensors must share one block stride")
    stride = strides.pop()

    by_offset: dict[int, set[str]] = {}
    for tensor in tensors:
        if tensor.offset < 0 or tensor.offset >= stride:
            raise ValueError("packed tensor offset is outside its block stride")
        by_offset.setdefault(tensor.offset, set()).update(tensor.shared_by)

    offsets = sorted(by_offset)
    slots = []
    for idx, offset in enumerate(offsets):
        end = offsets[idx + 1] if idx + 1 < len(offsets) else stride
        if end <= offset:
            raise ValueError("packed tensor offsets must define positive slices")
        slots.append(
            PackedSlotAccounting(
                offset_bytes=offset,
                padded_bytes_per_gpu_block=end - offset,
                shared_by=tuple(sorted(by_offset[offset])),
            )
        )
    if sum(slot.padded_bytes_per_gpu_block for slot in slots) != stride:
        raise ValueError("packed slots do not cover the packed row")
    return stride, tuple(slots)


def build_compact_layout_accounting(
    kv_cache_config: KVCacheConfig,
    *,
    world_size: int,
    block_size_factor: int,
    cpu_budget_bytes: int,
) -> CompactLayoutAccounting:
    """Derive per-group compact charges from runtime packed-tensor specs."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if block_size_factor <= 0:
        raise ValueError("block_size_factor must be positive")
    if cpu_budget_bytes <= 0:
        raise ValueError("cpu_budget_bytes must be positive")

    _stride, slots = _packed_slots(kv_cache_config)

    # Build layer-to-slot lookup for ordered iteration.
    layer_to_slot: dict[str, PackedSlotAccounting] = {}
    for slot in slots:
        for layer in slot.shared_by:
            if layer in layer_to_slot:
                raise ValueError(
                    f"Layer {layer} appears in multiple packed slots "
                    f"({layer_to_slot[layer].offset_bytes} and {slot.offset_bytes})"
                )
            layer_to_slot[layer] = slot

    groups: list[GroupCompactAccounting] = []
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        inner_specs = _layer_specs(group.kv_cache_spec)
        slices: list[GroupCompactSlice] = []
        real_bytes = 0
        seen_layers: set[str] = set()
        for layer_name in group.layer_names:
            slot = layer_to_slot.get(layer_name)
            if slot is None:
                raise ValueError(
                    f"KV group {group_idx} layer {layer_name} has no packed slot"
                )
            if layer_name in seen_layers:
                raise ValueError(
                    f"KV group {group_idx} layer {layer_name} appears "
                    "multiple times in layer_names"
                )
            seen_layers.add(layer_name)
            layer_spec = inner_specs.get(layer_name, group.kv_cache_spec)
            layer_real_bytes = _real_page_size_bytes(layer_spec)
            if layer_real_bytes > slot.padded_bytes_per_gpu_block:
                raise ValueError(
                    f"KV layer {layer_name} real payload {layer_real_bytes} "
                    f"exceeds packed slot {slot.padded_bytes_per_gpu_block}"
                )
            slices.append(
                GroupCompactSlice(
                    offset_bytes=slot.offset_bytes,
                    real_bytes_per_gpu_block=layer_real_bytes,
                    padded_bytes_per_gpu_block=slot.padded_bytes_per_gpu_block,
                    layer_name=layer_name,
                    shared_by=slot.shared_by,
                )
            )
            real_bytes += layer_real_bytes
        if not slices:
            raise ValueError(f"KV group {group_idx} has no packed tensor slices")

        padded_bytes = sum(s.padded_bytes_per_gpu_block for s in slices)
        native_tokens = group.kv_cache_spec.block_size
        groups.append(
            GroupCompactAccounting(
                group_idx=group_idx,
                native_block_tokens=native_tokens,
                slices=tuple(slices),
                compact_real_bytes_per_rank=real_bytes * block_size_factor,
                compact_padded_bytes_per_rank=padded_bytes * block_size_factor,
            )
        )

    return CompactLayoutAccounting(
        world_size=world_size,
        block_size_factor=block_size_factor,
        cpu_budget_bytes=cpu_budget_bytes,
        groups=tuple(groups),
    )
