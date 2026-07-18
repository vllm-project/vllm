# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derivation of canonical page mappings for KV offloading.

The only place in the offloading stack that reasons about parallelism
(TP/DCP/PCP); everything downstream consumes byte mappings. The canonical
page of a layer is the full offloaded block without parallelism: all KV
heads, all block_size * dcp * pcp tokens, in the worker's page encoding.
Uncertifiable layers get a rank-private mapping (fail closed).
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, MappedRun

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Version of the canonical byte format; bump on any layout change
CANONICAL_SCHEMA_VERSION = 1


def canonical_schema_id() -> str:
    """Identity of the canonical byte format, for namespacing persisted KV.
    Canonical pages keep the worker's KV layout family, so the id couples the
    schema version with that family; consumers must match it exactly."""
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    return f"v{CANONICAL_SCHEMA_VERSION}-{get_kv_cache_layout().lower()}"


@dataclass(frozen=True)
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


def _coalesce_runs(runs: list[MappedRun]) -> tuple[MappedRun, ...]:
    """Collapse contiguous fragments within and across runs to minimize the
    number of copy ops (e.g. a single-rank mapping becomes one whole-page run).
    """
    out: list[MappedRun] = []
    for run in runs:
        if (
            run.num_fragments > 1
            and run.local_stride == run.fragment_size
            and run.canonical_stride == run.fragment_size
        ):
            size = run.fragment_size * run.num_fragments
            run = MappedRun(run.local_offset, run.canonical_offset, size, 1, size, size)
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.num_fragments == 1
            and run.num_fragments == 1
            and prev.local_offset + prev.fragment_size == run.local_offset
            and prev.canonical_offset + prev.fragment_size == run.canonical_offset
        ):
            size = prev.fragment_size + run.fragment_size
            out[-1] = MappedRun(
                prev.local_offset, prev.canonical_offset, size, 1, size, size
            )
        else:
            out.append(run)
    return tuple(out)


def _chunk_runs(
    channels: list[tuple[int, int, int, int]],
    num_tokens: int,
    ctx: _RankContext,
) -> tuple[MappedRun, ...]:
    """Place each channel's num_tokens rows: local token l of CP rank c is
    canonical token ((l // I) * cp + c) * I + l % I. A channel is
    (local_base, canonical_base, local_row, canonical_row)."""
    runs: list[MappedRun] = []
    interleave, cp = ctx.interleave, ctx.cp_size
    for local_base, canonical_base, local_row, canonical_row in channels:
        if cp == 1:
            runs.append(
                MappedRun(
                    local_base,
                    canonical_base,
                    local_row,
                    num_tokens,
                    local_row,
                    canonical_row,
                )
            )
            continue
        for chunk in range(num_tokens // interleave):
            canonical_token = (chunk * cp + ctx.total_cp_rank) * interleave
            runs.append(
                MappedRun(
                    local_base + chunk * interleave * local_row,
                    canonical_base + canonical_token * canonical_row,
                    local_row,
                    interleave,
                    local_row,
                    canonical_row,
                )
            )
    return _coalesce_runs(runs)


def _attention_channels(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    num_blocks: int,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[tuple[int, int, int, int]] | None:
    """Byte channels of an attention page, given this rank's head shard.

    Recognizes packed KV (num_blocks, heads, block_size, 2 * head_size) and
    split KV (num_blocks, 2, block_size, heads, head_size), in NHD or HND
    stride order. None when the layout is ambiguous (fail closed)."""
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    elem = kv_cache.element_size()
    page = spec.real_page_size_bytes
    content = 2 * head_size
    span = bs * cp_size  # canonical tokens per offloaded block

    if tuple(kv_cache.shape) == (num_blocks, heads, bs, content):
        if heads * bs * content * elem != page:
            return None
        s = kv_cache.stride()
        if s[3] != 1:
            return None
        row = heads * content * elem
        if s[1] == content and s[2] == heads * content:  # NHD: token-major
            return [(0, head_shard * row, row, num_head_shards * row)]
        if s[1] == bs * content and s[2] == content:  # HND: head-major
            g = content * elem
            return [
                (
                    j * bs * g,
                    (head_shard * heads + j) * span * g,
                    g,
                    g,
                )
                for j in range(heads)
            ]
        return None

    if tuple(kv_cache.shape) != (num_blocks, 2, bs, heads, head_size):
        return None
    if 2 * bs * heads * head_size * elem != page:
        return None
    s = kv_cache.stride()
    if s[4] != 1 or s[1] != bs * heads * head_size:
        return None
    k_size = bs * heads * head_size * elem
    if s[2] == heads * head_size and s[3] == head_size:  # NHD
        row = heads * head_size * elem
        return [
            (
                region * bs * row,
                region * span * num_head_shards * row + head_shard * row,
                row,
                num_head_shards * row,
            )
            for region in range(2)  # K, then V
        ]
    if s[3] == bs * head_size and s[2] == head_size:  # HND
        f = head_size * elem
        total_heads = num_head_shards * heads
        return [
            (
                region * k_size + j * bs * f,
                (region * total_heads + head_shard * heads + j) * span * f,
                f,
                f,
            )
            for region in range(2)
            for j in range(heads)
        ]
    return None


def _layer_mapping(
    spec: KVCacheSpec,
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    num_blocks: int,
    ctx: _RankContext,
) -> CanonicalPageMapping | None:
    """Certified mapping for one layer at one rank, or None (fail closed)."""
    if not isinstance(spec, AttentionSpec):
        return None
    bs = spec.block_size
    page = spec.real_page_size_bytes
    if ctx.cp_size > 1 and (ctx.interleave > bs or bs % ctx.interleave):
        return None

    if isinstance(spec, MLAAttentionSpec):
        # TP-replicated latent; CP shards its tokens; first DCP group writes
        if spec.compress_ratio != 1 or page % bs:
            return None
        row = page // bs
        runs = _chunk_runs([(0, 0, row, row)], bs, ctx)
        return CanonicalPageMapping(
            canonical_page_size_bytes=ctx.cp_size * page,
            local_page_size_bytes=page,
            store_runs=runs if ctx.tp_rank < ctx.dcp_size else (),
            load_runs=runs,
            parallel_invariant=ctx.cp_size == 1,
        )

    if spec.kv_quant_mode.is_per_token_head or not isinstance(kv_cache, torch.Tensor):
        return None
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

    head_shard = ctx.tp_rank // replication
    channels = _attention_channels(
        kv_cache, spec, num_blocks, head_shard, num_head_shards, ctx.cp_size
    )
    if channels is None:
        return None
    runs = _chunk_runs(channels, bs, ctx)
    # One writer among ranks holding identical bytes
    contributor = (ctx.tp_rank % replication) // ctx.dcp_size == 0
    return CanonicalPageMapping(
        canonical_page_size_bytes=ctx.cp_size * num_head_shards * page,
        local_page_size_bytes=page,
        store_runs=runs if contributor else (),
        load_runs=runs,
        parallel_invariant=ctx.cp_size == 1,
    )


def _rank_private_mapping(
    page_size_bytes: int, num_ranks: int, rank: int
) -> CanonicalPageMapping:
    """Fallback: place the worker's page whole at a worker-exclusive offset."""
    run = MappedRun(
        0, rank * page_size_bytes, page_size_bytes, 1, page_size_bytes, page_size_bytes
    )
    return CanonicalPageMapping(
        canonical_page_size_bytes=num_ranks * page_size_bytes,
        local_page_size_bytes=page_size_bytes,
        store_runs=(run,),
        load_runs=(run,),
        parallel_invariant=False,
    )


def _run_intervals(
    runs: tuple[MappedRun, ...], canonical: bool
) -> list[tuple[int, int]]:
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


def _verify_mappings(layer_name: str, per_rank: list[CanonicalPageMapping]) -> None:
    """All ranks' store runs must tile the canonical page exactly once, and
    each rank's load runs must cover exactly its local page."""
    size = per_rank[0].canonical_page_size_bytes
    store_intervals: list[tuple[int, int]] = []
    for mapping in per_rank:
        assert mapping.canonical_page_size_bytes == size
        store_intervals += _run_intervals(mapping.store_runs, canonical=True)
        local = _run_intervals(mapping.load_runs, canonical=False)
        assert _is_exact_partition(local, mapping.local_page_size_bytes), (
            f"load runs do not cover the local page of layer {layer_name}"
        )
    store_intervals.sort()
    assert _is_exact_partition(store_intervals, size), (
        f"store runs do not tile the canonical page of layer {layer_name}"
    )


def _unpadded_page_size(spec: KVCacheSpec) -> int | None:
    if isinstance(spec, AttentionSpec):
        return spec.real_page_size_bytes
    if isinstance(spec, MambaSpec):
        return replace(spec, page_size_padded=None).page_size_bytes
    return None


def derive_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
) -> dict[str, CanonicalPageMapping]:
    """Per-layer canonical page mappings for this worker.

    Empty when the worker group is not exactly the TP x PCP grid; layers
    absent from the result have no canonical representation.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    group_size = tp_size * pcp_size
    if parallel_config.world_size != group_size:
        return {}

    def ctx(rank: int) -> _RankContext:
        return _RankContext(
            tp_size=tp_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pcp_size=pcp_size,
            interleave=parallel_config.cp_kv_cache_interleave_size,
            total_kv_heads=vllm_config.model_config.get_total_num_kv_heads(),
            rank=rank,
        )

    my_rank = parallel_config.rank
    num_blocks = kv_cache_config.num_blocks

    mappings: dict[str, CanonicalPageMapping] = {}
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        group_kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
            per_layer_specs = group_kv_cache_spec.kv_cache_specs
        else:
            per_layer_specs = {}
        for layer_name in kv_cache_group.layer_names:
            spec = per_layer_specs.get(layer_name, group_kv_cache_spec)
            per_rank: list[CanonicalPageMapping] = []
            for rank in range(group_size):
                mapping = _layer_mapping(
                    spec, kv_caches.get(layer_name), num_blocks, ctx(rank)
                )
                if mapping is None:
                    break
                per_rank.append(mapping)
            if len(per_rank) != group_size:
                page = _unpadded_page_size(spec)
                if page is None:
                    continue
                per_rank = [
                    _rank_private_mapping(page, group_size, rank)
                    for rank in range(group_size)
                ]
            _verify_mappings(layer_name, per_rank)
            mappings[layer_name] = per_rank[my_rank]
    return mappings
