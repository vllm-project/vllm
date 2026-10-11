# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Literal, cast

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MLAAttentionSpec

from .config import MooncakeCompressionConfig, QuantAxis

KVPart = Literal[0, 1]
ALIGNMENT = 256


@dataclass(frozen=True)
class QuantSection:
    layer_name: str
    layer_index: int
    kv_part: KVPart
    precision: str
    head_start: int
    head_count: int
    num_levels: int
    axis: QuantAxis
    raw_offset: int
    u8_offset: int
    raw_shape: tuple[int, int, int, int]
    aux_min_offset: int
    aux_scale_offset: int
    aux_shape: tuple[int, ...]


@dataclass(frozen=True)
class LayerPartLayout:
    layer_name: str
    layer_index: int
    kv_part: KVPart
    group_index: int
    block_size: int
    num_heads: int
    head_size: int
    dtype: torch.dtype
    raw_offset: int
    raw_bytes: int
    u8_offset: int
    u8_bytes: int
    sections: tuple[QuantSection, ...]


@dataclass(frozen=True)
class CompressionLayout:
    layer_parts: tuple[LayerPartLayout, ...]
    raw_bytes_by_part: tuple[int, int]
    u8_bytes_by_part: tuple[int, int]
    packed_bytes_by_part: tuple[int, int]
    aux_bytes_by_part: tuple[int, int]
    arena_bytes: int
    # Two reusable raw scratch regions live in arena A.  The packed u8 stream
    # starts at u8_base in arena B so later layer gathers cannot overwrite it.
    scratch_stride: int
    u8_base: int
    aux_arena_bytes: int
    fa_group_indices: frozenset[int]
    index_rows_by_group: dict[int, int] = field(default_factory=dict)
    aggregate: bool = False

    def parts(self, kv_part: int) -> tuple[LayerPartLayout, ...]:
        return tuple(part for part in self.layer_parts if part.kv_part == kv_part)

    @property
    def raw_tail_capacity(self) -> int:
        if self.aggregate:
            return 0
        # FP16/BF16, at most block_size - 1 valid tokens per layer and KV part.
        return round_up(
            max(
                sum(
                    (part.block_size - 1) * part.num_heads * part.head_size * 2
                    for part in self.parts(kv_part)
                )
                for kv_part in (0, 1)
            ),
            ALIGNMENT,
        )

    def raw_tail_bytes(self, kv_part: KVPart, tokens: dict[int, int]) -> int:
        return sum(
            (tokens[part.group_index] % part.block_size)
            * part.num_heads
            * part.head_size
            * 2
            for part in self.parts(kv_part)
        )

    def has_body(self, tokens: dict[int, int]) -> bool:
        return any(
            tokens[part.group_index] >= part.block_size for part in self.parts(0)
        )


def _reduction_shape(
    axis: QuantAxis, shape: tuple[int, int, int, int]
) -> tuple[int, ...]:
    blocks, tokens, heads, head_size = shape
    if axis == "channel":
        return (1, 1, heads, head_size)
    if axis == "token":
        return (blocks, tokens, 1, 1)
    return (1, 1, 1, 1)


def _split_counts(
    config: MooncakeCompressionConfig,
    layer_index: int,
    num_heads: int,
    low_head_counts: dict[int, int],
    num_model_layers: int,
) -> tuple[int, int]:
    if config.split_type == "head":
        low = low_head_counts[layer_index]
        return low, num_heads - low
    low_layer_count = round(num_model_layers * 0.33)
    is_low = low_layer_count > 0 and layer_index >= num_model_layers - low_layer_count
    return (num_heads, 0) if is_low else (0, num_heads)


def build_compression_layout(
    specs: dict[str, KVCacheSpec],
    config: MooncakeCompressionConfig,
    low_head_counts: dict[int, int],
    num_model_layers: int,
    group_indices: dict[str, int] | None = None,
) -> CompressionLayout:
    group_indices = group_indices or {}
    fa_specs = [
        (name, spec)
        for name, spec in specs.items()
        if isinstance(spec, FullAttentionSpec)
    ]
    fa_specs.sort(key=lambda item: extract_layer_index(item[0]))
    if not fa_specs:
        raise ValueError("Mooncake compression requires FullAttention KV caches")
    if len({spec.block_size for _, spec in fa_specs}) != 1:
        raise ValueError(
            "Mooncake compression requires a common logical block size "
            "across FullAttention cache groups."
        )
    for layer_name, spec in fa_specs:
        if isinstance(spec, MLAAttentionSpec):
            raise ValueError("Mooncake compression does not support MLA KV caches")
        if config.uses_transformer and any(
            size < 1 or size & (size - 1) for size in (spec.head_size, spec.head_size_v)
        ):
            raise ValueError("Hadamard compression requires power-of-two head sizes")
        if spec.dtype is not torch.bfloat16 or spec.dtype.itemsize != 2:
            raise ValueError(
                "Mooncake compression currently supports only BF16 KV caches; "
                f"layer {layer_name} uses {spec.dtype}."
            )

    raw_offsets = [0, 0]
    u8_offsets = [0, 0]
    aux_offsets = [0, 0]
    worst_aux_offsets = [0, 0]
    layer_parts: list[LayerPartLayout] = []
    fa_groups: set[int] = set()
    chunk_blocks = config.logical_blocks_per_chunk

    # Compression processes one layer at a time.  Reserve two reusable raw
    # buffers instead of giving every layer a raw offset; this keeps the
    # compact u8 stream disjoint from all future gather writes.
    scratch_stride = round_up(
        max(
            chunk_blocks * spec.block_size * spec.num_kv_heads * head_size * 2
            for _, spec in fa_specs
            for head_size in (spec.head_size, spec.head_size_v)
        ),
        ALIGNMENT,
    )
    u8_base = round_up(2 * scratch_stride, ALIGNMENT)

    for layer_name, base_spec in fa_specs:
        spec = base_spec
        layer_index = extract_layer_index(layer_name)
        group_index = group_indices.get(layer_name, 0)
        fa_groups.add(group_index)
        low_heads, high_heads = _split_counts(
            config,
            layer_index,
            spec.num_kv_heads,
            low_head_counts,
            num_model_layers,
        )

        for kv_part, head_size, axis, low_levels, high_levels in (
            (
                0,
                spec.head_size,
                config.axis_key,
                config.low_key_num_levels,
                config.high_key_num_levels,
            ),
            (
                1,
                spec.head_size_v,
                config.axis_value,
                config.low_value_num_levels,
                config.high_value_num_levels,
            ),
        ):
            raw_offset = 0
            u8_offset = u8_offsets[kv_part]
            layer_numel = chunk_blocks * spec.block_size * spec.num_kv_heads * head_size
            layer_raw_bytes = layer_numel * 2
            sections: list[QuantSection] = []
            section_raw_offset = raw_offset
            section_u8_offset = u8_offset
            head_start = 0
            for precision, head_count, levels in (
                ("low", low_heads, low_levels),
                ("high", high_heads, high_levels),
            ):
                if head_count == 0:
                    continue
                shape = (
                    chunk_blocks,
                    spec.block_size,
                    head_count,
                    head_size,
                )
                aux_shape = _reduction_shape(axis, shape)
                aux_nbytes = prod(aux_shape) * 2
                min_offset = round_up(aux_offsets[kv_part], ALIGNMENT)
                scale_offset = round_up(min_offset + aux_nbytes, ALIGNMENT)
                aux_offsets[kv_part] = scale_offset + aux_nbytes
                worst_aux_numel = max(
                    chunk_blocks * spec.block_size,
                    head_count * head_size,
                    1,
                )
                worst_aux_nbytes = worst_aux_numel * 2
                worst_min_offset = round_up(worst_aux_offsets[kv_part], ALIGNMENT)
                worst_scale_offset = round_up(
                    worst_min_offset + worst_aux_nbytes, ALIGNMENT
                )
                worst_aux_offsets[kv_part] = worst_scale_offset + worst_aux_nbytes
                section_numel = prod(shape)
                sections.append(
                    QuantSection(
                        layer_name=layer_name,
                        layer_index=layer_index,
                        kv_part=cast(KVPart, kv_part),
                        precision=precision,
                        head_start=head_start,
                        head_count=head_count,
                        num_levels=levels,
                        axis=axis,
                        raw_offset=section_raw_offset,
                        u8_offset=section_u8_offset,
                        raw_shape=shape,
                        aux_min_offset=min_offset,
                        aux_scale_offset=scale_offset,
                        aux_shape=aux_shape,
                    )
                )
                section_raw_offset += section_numel * 2
                section_u8_offset += section_numel
                head_start += head_count

            layer_parts.append(
                LayerPartLayout(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    kv_part=cast(KVPart, kv_part),
                    group_index=group_index,
                    block_size=spec.block_size,
                    num_heads=spec.num_kv_heads,
                    head_size=head_size,
                    dtype=spec.dtype,
                    raw_offset=raw_offset,
                    raw_bytes=layer_raw_bytes,
                    u8_offset=u8_offset,
                    u8_bytes=layer_numel,
                    sections=tuple(sections),
                )
            )
            raw_offsets[kv_part] += layer_raw_bytes
            u8_offsets[kv_part] += layer_numel

    raw_by_part = (raw_offsets[0], raw_offsets[1])
    u8_by_part = (u8_offsets[0], u8_offsets[1])
    aux_by_part = (
        round_up(aux_offsets[0], ALIGNMENT),
        round_up(aux_offsets[1], ALIGNMENT),
    )
    aggregate = "aggregate" in config.pipeline
    uses_quantizer = "quantizer" in config.pipeline and not aggregate
    packed_by_part = u8_by_part if uses_quantizer else raw_by_part
    if aggregate:
        arena_bytes = round_up(max(*raw_by_part), ALIGNMENT)
    else:
        arena_bytes = round_up(
            max(2 * scratch_stride, u8_base + max(*packed_by_part)), ALIGNMENT
        )
    # K and V are encoded as separate sub-chunks and reuse the same slot.
    worst_aux_bytes = max(round_up(value, ALIGNMENT) for value in worst_aux_offsets)
    aux_arena_bytes = 0 if (aggregate or not uses_quantizer) else 2 * worst_aux_bytes
    return CompressionLayout(
        layer_parts=tuple(layer_parts),
        raw_bytes_by_part=raw_by_part,
        u8_bytes_by_part=u8_by_part,
        packed_bytes_by_part=packed_by_part,
        aux_bytes_by_part=aux_by_part,
        arena_bytes=arena_bytes,
        scratch_stride=scratch_stride,
        u8_base=u8_base,
        aux_arena_bytes=aux_arena_bytes,
        fa_group_indices=frozenset(fa_groups),
        index_rows_by_group={
            group_index: row for row, group_index in enumerate(sorted(fa_groups))
        },
        aggregate=aggregate,
    )
