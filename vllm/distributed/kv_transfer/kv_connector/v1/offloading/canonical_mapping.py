# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derivation of canonical page mappings for KV offloading.

The only place in the offloading stack that reasons about parallelism
(TP/DCP/PCP/PP); everything downstream consumes byte mappings. The canonical
page of a layer is the full offloaded block without parallelism: all KV
heads, all block_size * dcp * pcp tokens, in the worker's page encoding. A
certified layer whose physical page layout is not recognized places its pages
whole by KV head shard; uncertifiable layers get an opaque fallback mapping
(fail closed). A chunk row holds one KV cache group's canonical pages, layer
by layer in model order, so each pipeline stage fills the slots of its own
layers.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.layer_layout import (
    LayerObjectLayout,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Version of the canonical byte format; bump on any layout change.
# 2: a chunk row holds one group's layers in model order at fixed offsets.
CANONICAL_FORMAT_VERSION = 2


def canonical_format_id(kv_cache_layout: str) -> str:
    """Identity of the canonical byte format, for namespacing persisted KV.
    Canonical pages keep the worker's KV layout family, so the id couples the
    format version with that family; consumers must match it exactly. The family
    keeps its historical NHD/HND spelling so ids stay stable for KV persisted
    before the layout enum existed."""
    layout = KVCacheLayout[kv_cache_layout]
    legacy = {KVCacheLayout.LBNHC: "nhd", KVCacheLayout.LBHNC: "hnd"}
    family = legacy.get(layout, layout.name.lower())
    return f"v{CANONICAL_FORMAT_VERSION}-{family}"


@dataclass(frozen=True, slots=True)
class _RankContext:
    """Sharding parameters of one worker rank within the offload group."""

    tp_size: int
    dcp_size: int
    pcp_size: int
    interleave: int
    total_kv_heads: int
    rank: int

    @property
    def cp_size(self) -> int:
        return self.dcp_size * self.pcp_size

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def total_cp_rank(self) -> int:
        pcp_rank = self.rank // self.tp_size
        return pcp_rank * self.dcp_size + self.tp_rank % self.dcp_size


@dataclass(frozen=True, slots=True)
class ByteRegion:
    """A byte region within a page that repeats once per token."""

    local_offset: int
    canonical_offset: int
    bytes_per_token: int
    canonical_token_stride: int


def _coalesce_runs(runs: list[CopyRun]) -> tuple[CopyRun, ...]:
    """Collapse contiguous fragments within and across runs to minimize the
    number of copy ops (e.g. a single-rank mapping becomes one whole-page run).
    """
    out: list[CopyRun] = []
    for run in runs:
        if (
            run.num_fragments > 1
            and run.local_stride == run.fragment_size
            and run.canonical_stride == run.fragment_size
        ):
            size = run.fragment_size * run.num_fragments
            run = CopyRun(run.local_offset, run.canonical_offset, size, 1, size, size)
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.num_fragments == 1
            and run.num_fragments == 1
            and prev.local_offset + prev.fragment_size == run.local_offset
            and prev.canonical_offset + prev.fragment_size == run.canonical_offset
        ):
            size = prev.fragment_size + run.fragment_size
            out[-1] = CopyRun(
                prev.local_offset, prev.canonical_offset, size, 1, size, size
            )
        else:
            out.append(run)
    return tuple(out)


def _local_to_canonical_token(local_idx: int, ctx: _RankContext) -> int:
    """Canonical position of one of this rank's local token indices."""
    chunk, pos_in_chunk = divmod(local_idx, ctx.interleave)
    return (chunk * ctx.cp_size + ctx.total_cp_rank) * ctx.interleave + pos_in_chunk


def _interleave_cp_tokens(
    regions: list[ByteRegion],
    num_tokens: int,
    ctx: _RankContext,
) -> tuple[CopyRun, ...]:
    """Place each region's num_tokens rows at their canonical token positions,
    one run per chunk of interleaved tokens."""
    runs: list[CopyRun] = []
    for region in regions:
        if ctx.cp_size == 1:
            runs.append(
                CopyRun(
                    region.local_offset,
                    region.canonical_offset,
                    region.bytes_per_token,
                    num_tokens,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                )
            )
            continue
        for chunk_start in range(0, num_tokens, ctx.interleave):
            canonical_token = _local_to_canonical_token(chunk_start, ctx)
            runs.append(
                CopyRun(
                    region.local_offset + chunk_start * region.bytes_per_token,
                    region.canonical_offset
                    + canonical_token * region.canonical_token_stride,
                    region.bytes_per_token,
                    ctx.interleave,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                )
            )
    return _coalesce_runs(runs)


def _packed_kv_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """K and V adjacent per (token, head), in NHD or HND stride order."""
    bs, heads = spec.block_size, spec.num_kv_heads
    elem = kv_cache.element_size()
    head_elems = 2 * spec.head_size
    if heads * bs * head_elems * elem != spec.real_page_size_bytes:
        return None
    _, head_stride, token_stride, inner_stride = kv_cache.stride()
    if inner_stride != 1:
        return None
    head_bytes = head_elems * elem
    token_row_bytes = heads * head_bytes

    if head_stride == head_elems and token_stride == heads * head_elems:  # NHD
        return [
            ByteRegion(
                local_offset=0,
                canonical_offset=head_shard * token_row_bytes,
                bytes_per_token=token_row_bytes,
                canonical_token_stride=num_head_shards * token_row_bytes,
            )
        ]

    if head_stride == bs * head_elems and token_stride == head_elems:  # HND
        canonical_span = bs * cp_size  # canonical tokens per offloaded block
        return [
            ByteRegion(
                local_offset=head * bs * head_bytes,
                canonical_offset=(head_shard * heads + head)
                * canonical_span
                * head_bytes,
                bytes_per_token=head_bytes,
                canonical_token_stride=head_bytes,
            )
            for head in range(heads)
        ]
    return None


def _split_kv_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """K and V in separate page halves, in NHD or HND stride order."""
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    elem = kv_cache.element_size()
    if 2 * bs * heads * head_size * elem != spec.real_page_size_bytes:
        return None
    _, half_stride, token_stride, head_stride, inner_stride = kv_cache.stride()
    if inner_stride != 1 or half_stride != bs * heads * head_size:
        return None
    head_bytes = head_size * elem
    token_row_bytes = heads * head_bytes
    canonical_span = bs * cp_size  # canonical tokens per offloaded block

    if token_stride == heads * head_size and head_stride == head_size:  # NHD
        canonical_half_bytes = canonical_span * num_head_shards * token_row_bytes
        return [
            ByteRegion(
                local_offset=half * bs * token_row_bytes,
                canonical_offset=half * canonical_half_bytes
                + head_shard * token_row_bytes,
                bytes_per_token=token_row_bytes,
                canonical_token_stride=num_head_shards * token_row_bytes,
            )
            for half in range(2)  # K, then V
        ]

    if head_stride == bs * head_size and token_stride == head_size:  # HND
        local_half_bytes = bs * heads * head_bytes
        total_heads = num_head_shards * heads
        return [
            ByteRegion(
                local_offset=half * local_half_bytes + head * bs * head_bytes,
                canonical_offset=(half * total_heads + head_shard * heads + head)
                * canonical_span
                * head_bytes,
                bytes_per_token=head_bytes,
                canonical_token_stride=head_bytes,
            )
            for half in range(2)
            for head in range(heads)
        ]
    return None


def _attention_byte_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    num_blocks: int,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """Byte regions of an attention page, given this rank's head shard.
    None when the physical layout is not recognized (fail closed)."""
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    if tuple(kv_cache.shape) == (num_blocks, heads, bs, 2 * head_size):
        return _packed_kv_regions(kv_cache, spec, head_shard, num_head_shards, cp_size)
    if tuple(kv_cache.shape) == (num_blocks, 2, bs, heads, head_size):
        return _split_kv_regions(kv_cache, spec, head_shard, num_head_shards, cp_size)
    return None


def _head_shards(spec: AttentionSpec, ctx: _RankContext) -> tuple[int, int] | None:
    """(Number of distinct KV head shards, ranks holding each shard) of a
    non-MLA attention layer, or None when TP does not split its heads evenly."""
    total, tp = ctx.total_kv_heads, ctx.tp_size
    if spec.num_kv_heads != max(1, total // tp):
        return None
    if total >= tp:
        if total % tp:
            return None
        num_head_shards, replication = tp, 1
    else:
        if tp % total:
            return None
        num_head_shards, replication = total, tp // total
    # DCP shards tokens across ranks holding replicated KV
    if replication % ctx.dcp_size:
        return None
    return num_head_shards, replication


def _certified_page_size(spec: KVCacheSpec, ctx: _RankContext) -> int | None:
    """Canonical page size of a layer whose spec passes certification, or
    None. The physical layout is checked separately, per worker."""
    if not isinstance(spec, AttentionSpec):
        return None
    bs = spec.block_size
    page = spec.real_page_size_bytes
    if ctx.cp_size > 1 and (ctx.interleave > bs or bs % ctx.interleave):
        return None
    if isinstance(spec, MLAAttentionSpec):
        # TP-replicated latent; CP shards its tokens across the DCP groups
        if (
            spec.tokens_per_state != 1
            or page % bs
            or ctx.tp_size % ctx.dcp_size
            or spec.kv_quant_mode.is_per_token_head
        ):
            return None
        return ctx.cp_size * page
    if spec.kv_quant_mode.is_per_token_head:
        return None
    shards = _head_shards(spec, ctx)
    if shards is None:
        return None
    return ctx.cp_size * shards[0] * page


def _layer_mapping(
    spec: KVCacheSpec,
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    num_blocks: int,
    ctx: _RankContext,
) -> CanonicalPageMapping | None:
    """Certified mapping for one layer at one rank, or None (fail closed)."""
    canonical_page = _certified_page_size(spec, ctx)
    if canonical_page is None:
        return None
    assert isinstance(spec, AttentionSpec)
    bs = spec.block_size
    page = spec.real_page_size_bytes

    if isinstance(spec, MLAAttentionSpec):
        row = page // bs
        return CanonicalPageMapping(
            canonical_page_size_bytes=canonical_page,
            local_page_size_bytes=page,
            runs=_interleave_cp_tokens([ByteRegion(0, 0, row, row)], bs, ctx),
            num_writers=ctx.tp_size // ctx.dcp_size,
            writer_index=ctx.tp_rank // ctx.dcp_size,
            parallelism_agnostic=ctx.cp_size == 1,
        )

    if not isinstance(kv_cache, torch.Tensor):
        return None
    shards = _head_shards(spec, ctx)
    assert shards is not None
    num_head_shards, replication = shards
    head_shard = ctx.tp_rank // replication
    regions = _attention_byte_regions(
        kv_cache, spec, num_blocks, head_shard, num_head_shards, ctx.cp_size
    )
    if regions is None:
        return None
    return CanonicalPageMapping(
        canonical_page_size_bytes=canonical_page,
        local_page_size_bytes=page,
        runs=_interleave_cp_tokens(regions, bs, ctx),
        num_writers=replication // ctx.dcp_size,
        writer_index=(ctx.tp_rank % replication) // ctx.dcp_size,
        parallelism_agnostic=ctx.cp_size == 1,
    )


def _head_shard_fallback_mapping(
    spec: AttentionSpec, ctx: _RankContext
) -> CanonicalPageMapping:
    """Fallback for a certified attention layer whose physical page layout is
    not recognized: place the rank's page whole at the position of its KV head
    shard and CP rank, so it fills the certified canonical page size. Ranks
    holding the same heads and tokens take turns writing it."""
    shards = _head_shards(spec, ctx)
    assert shards is not None
    num_head_shards, replication = shards
    page = spec.real_page_size_bytes
    position = (ctx.tp_rank // replication) * ctx.cp_size + ctx.total_cp_rank
    run = CopyRun(0, position * page, page, 1, page, page)
    return CanonicalPageMapping(
        canonical_page_size_bytes=ctx.cp_size * num_head_shards * page,
        local_page_size_bytes=page,
        runs=(run,),
        num_writers=replication // ctx.dcp_size,
        writer_index=(ctx.tp_rank % replication) // ctx.dcp_size,
        parallelism_agnostic=False,
    )


def _opaque_fallback_mapping(
    page_size_bytes: int, num_ranks: int, rank: int
) -> CanonicalPageMapping:
    """Fallback: place the worker's page whole at a worker-exclusive offset."""
    run = CopyRun(
        0, rank * page_size_bytes, page_size_bytes, 1, page_size_bytes, page_size_bytes
    )
    return CanonicalPageMapping(
        canonical_page_size_bytes=num_ranks * page_size_bytes,
        local_page_size_bytes=page_size_bytes,
        runs=(run,),
        num_writers=1,
        writer_index=0,
        parallelism_agnostic=False,
    )


def _run_intervals(runs: tuple[CopyRun, ...], canonical: bool) -> list[tuple[int, int]]:
    intervals = []
    for run in runs:
        offset = run.canonical_offset if canonical else run.local_offset
        stride = run.canonical_stride if canonical else run.local_stride
        for i in range(run.num_fragments):
            start = offset + i * stride
            intervals.append((start, start + run.fragment_size))
    return sorted(intervals)


def _is_exact_partition(intervals: list[tuple[int, int]], size: int) -> bool:
    return (
        bool(intervals)
        and intervals[0][0] == 0
        and intervals[-1][1] == size
        and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    )


def _verify_tiling(layer_name: str, per_rank: list[CanonicalPageMapping]) -> None:
    """Whichever ranks a block elects as writers must tile the canonical page
    exactly once, and each rank's runs must cover exactly its local page."""
    size = per_rank[0].canonical_page_size_bytes
    num_writers = per_rank[0].num_writers
    for mapping in per_rank:
        assert mapping.canonical_page_size_bytes == size
        assert mapping.num_writers == num_writers
        local = _run_intervals(mapping.runs, canonical=False)
        assert _is_exact_partition(local, mapping.local_page_size_bytes), (
            f"runs do not cover the local page of layer {layer_name}"
        )
    for block_id in range(num_writers):
        stored: list[tuple[int, int]] = []
        for mapping in per_rank:
            if mapping.is_writer(block_id):
                stored += _run_intervals(mapping.runs, canonical=True)
        stored.sort()
        assert _is_exact_partition(stored, size), (
            f"writers of block {block_id} do not tile the canonical page "
            f"of layer {layer_name}"
        )


def _unpadded_page_size(spec: KVCacheSpec) -> int | None:
    if isinstance(spec, AttentionSpec):
        return spec.unpadded_page_size_bytes
    if isinstance(spec, MambaSpec):
        return replace(spec, page_size_padded=None).page_size_bytes
    return None


def _rank_context(vllm_config: "VllmConfig", rank: int) -> _RankContext:
    """Sharding parameters of one rank of a pipeline stage's TP x PCP grid."""
    parallel_config = vllm_config.parallel_config
    return _RankContext(
        tp_size=parallel_config.tensor_parallel_size,
        dcp_size=parallel_config.decode_context_parallel_size,
        pcp_size=parallel_config.prefill_context_parallel_size,
        interleave=parallel_config.cp_kv_cache_interleave_size,
        total_kv_heads=vllm_config.model_config.get_total_num_kv_heads(),
        rank=rank,
    )


def canonical_page_size(vllm_config: "VllmConfig", spec: KVCacheSpec) -> int | None:
    """Canonical page size of a layer, from its spec alone: the size of every
    mapping `derive_canonical_mappings` gives it, certified or fallback. None
    for a layer without a canonical page."""
    ctx = _rank_context(vllm_config, rank=0)
    page = _certified_page_size(spec, ctx)
    if page is not None:
        return page
    unpadded_page = _unpadded_page_size(spec)
    if unpadded_page is None:
        return None
    return ctx.tp_size * ctx.pcp_size * unpadded_page


def canonical_row_layouts(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    group_ids: Iterable[int],
    blocks_per_chunk: int,
) -> dict[int, LayerObjectLayout]:
    """The layout of a canonical chunk row for each given group.

    A row holds the chunk's canonical pages of one group's layers, on every
    pipeline stage, back to back in model order, so every worker and the
    scheduler compute the same layout under any parallel config.
    """
    layouts: dict[int, LayerObjectLayout] = {}
    for group_id in group_ids:
        layer_sizes: list[tuple[str, int]] = []
        for layer_name, spec in kv_cache_config.pipeline_layer_specs(group_id).items():
            page = canonical_page_size(vllm_config, spec)
            if page is not None:
                layer_sizes.append((layer_name, page * blocks_per_chunk))
        layouts[group_id] = LayerObjectLayout.pack(layer_sizes)
    return layouts


def derive_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
) -> dict[str, CanonicalPageMapping]:
    """Per-layer canonical page mappings for this worker.

    Empty when the workers are not a TP x PCP grid per pipeline stage; layers
    absent from the result have no canonical representation.
    """
    parallel_config = vllm_config.parallel_config
    group_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    if (
        parallel_config.world_size
        != group_size * parallel_config.pipeline_parallel_size
    ):
        return {}

    def ctx(rank: int) -> _RankContext:
        return _rank_context(vllm_config, rank)

    # Workers are ordered by PP rank, then by PCP and TP rank.
    my_rank = parallel_config.rank % group_size
    num_blocks = kv_cache_config.num_blocks

    mappings: dict[str, CanonicalPageMapping] = {}
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        for layer_name, spec in kv_cache_group.kv_cache_specs.items():
            per_rank: list[CanonicalPageMapping] = []
            for rank in range(group_size):
                mapping = _layer_mapping(
                    spec, kv_caches.get(layer_name), num_blocks, ctx(rank)
                )
                if mapping is None:
                    break
                per_rank.append(mapping)
            if len(per_rank) != group_size:
                if _certified_page_size(spec, ctx(0)) is not None:
                    # The spec is certified but the physical page layout is
                    # not: keep the canonical page size its row slot has.
                    assert isinstance(spec, AttentionSpec)
                    per_rank = [
                        _head_shard_fallback_mapping(spec, ctx(rank))
                        for rank in range(group_size)
                    ]
                else:
                    page = _unpadded_page_size(spec)
                    if page is None:
                        continue
                    per_rank = [
                        _opaque_fallback_mapping(page, group_size, rank)
                        for rank in range(group_size)
                    ]
            _verify_tiling(layer_name, per_rank)
            mappings[layer_name] = per_rank[my_rank]
    return mappings
