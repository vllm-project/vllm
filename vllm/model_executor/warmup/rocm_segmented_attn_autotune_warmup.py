# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-only, persistent launch tuning for segmented ROCm attention."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack

import torch
from filelock import FileLock

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops import segmented_attention, segmented_attention_config
from vllm.v1.attention.ops.segmented_attention import (
    _segmented_attention_reduce,
    _segmented_attention_stage,
    compile_segmented_attention,
    run_segmented_attention,
    segmented_workspace_shapes,
    supports_segmented_native_fp8,
)
from vllm.v1.attention.ops.segmented_attention_config import (
    KV_UPCAST_MODES,
    MAX_QUERY_LEN,
    KVUpcastMode,
    SegmentedAttentionConfig,
    SegmentedConfigOverrides,
    SegmentedLaunchConfig,
    segmented_kv_upcast_mode,
    segmented_query_capacity,
    select_segmented_config,
)

ConfigKey = tuple[tuple[str, object], ...]


class _ConfigRecord(TypedDict):
    workload: Sequence[int]
    best: SegmentedAttentionConfig


class _ConfigTable(TypedDict):
    identity: dict[str, Any]
    records: list[_ConfigRecord]


class _ConfigComparison(TypedDict):
    config: SegmentedAttentionConfig
    us: float
    paired_speedup_vs_default: float
    paired_warm_speedup_vs_default: NotRequired[float]
    paired_balanced_speedup_vs_default: NotRequired[float]
    rejected_warm_regression: NotRequired[bool]
    rejected_cold_regression: NotRequired[bool]


if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_TABLES: dict[tuple, _ConfigTable] = {}
_QUERY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)
_SEQUENCE_BUCKETS = (128, 512, 2048, 8192, 32768, 131072, 262144)
_COMPILE_WORKERS = 2
_CONDITION_ROUNDS = 2
_SCREEN_ROUNDS = 5
_UPCAST_GEOMETRY_SHORTLIST = 8
_FINALIST_COUNT = 3
_FINAL_ROUNDS = 7
_MIN_PROMOTION_SPEEDUP = 1.02
_MAX_WARM_SLOWDOWN = 1.02
_WARM_GRAPH_CALLS = 2
_MAX_WARM_GRAPH_CALLS = 256
_WARM_TARGET_US = 1000


def _key(
    device,
    dtype,
    kv_dtype,
    heads,
    kv_heads,
    dim,
    page,
    scale,
    sliding_window=-1,
    causal=True,
    has_sinks=False,
):
    return (
        torch.device(device).index,
        dtype,
        kv_dtype,
        heads,
        kv_heads,
        dim,
        page,
        scale,
        sliding_window,
        causal,
        has_sinks,
    )


def _normalized_config(
    config: SegmentedLaunchConfig, batch: int, query_len: int, heads: int, dim: int
) -> SegmentedAttentionConfig:
    qcap = segmented_query_capacity(query_len)
    result: SegmentedAttentionConfig = {
        **config,
        "kv_upcast": segmented_kv_upcast_mode(config),
    }
    result.pop("native_fp8_upcast", None)
    result.pop("native_fp8_pack", None)
    result.setdefault("waves_per_eu", 2)
    result.setdefault("reduce_d", 64 if batch * qcap * heads < 64 else dim)
    result.setdefault("reduce_warps", 4)
    if dim == 512:
        result.setdefault("pv_split", False)
        result.setdefault("prefix_fast", False)
        result.setdefault("qk_pipeline", None)
        result.setdefault("qk_unroll", None)
    return result


def _candidate_configs(
    default: SegmentedLaunchConfig,
    batch: int,
    query_len: int,
    heads: int,
    dim: int,
    seq_len: int | None = None,
    *,
    native_fp8: bool = False,
    legacy_default: SegmentedLaunchConfig | None = None,
) -> list[SegmentedAttentionConfig]:
    """Search conversion modes jointly with each eligible launch geometry."""
    base = _normalized_config(default, batch, query_len, heads, dim)
    if not native_fp8:
        base["kv_upcast"] = "normal"
    bases = [base]
    if legacy_default is not None:
        legacy = _normalized_config(legacy_default, batch, query_len, heads, dim)
        legacy["kv_upcast"] = base["kv_upcast"]
        if legacy != base:
            bases.append(legacy)
    launches: list[SegmentedAttentionConfig] = []
    for seed in bases:
        launches.extend(
            _launch_candidate_configs(seed, batch, query_len, heads, dim, seq_len)
        )
        if native_fp8:
            if dim == 512 and query_len > 2:
                for bn in (32, 64):
                    for warps in (4, 8):
                        launches.append(
                            {
                                **seed,
                                "bm": 16,
                                "bn": bn,
                                "bk": 64,
                                "warps": warps,
                                "stages": 1,
                                "waves_per_eu": 2,
                                "pv_split": False,
                                "qk_pipeline": None,
                                "qk_unroll": None,
                            }
                        )
            elif dim in (128, 256) and query_len >= 64:
                for bm, bn, waves in ((32, 64, 2), (64, 64, 6), (64, 128, 6)):
                    launches.append(
                        {
                            **seed,
                            "bm": bm,
                            "bn": bn,
                            "bk": 64,
                            "warps": 8,
                            "stages": 1,
                            "waves_per_eu": waves,
                        }
                    )
    launches = list({_config_key(c): c for c in launches}.values())
    normal_modes: tuple[KVUpcastMode, ...] = ("normal",)
    return [
        {**config, "kv_upcast": mode}
        for config in launches
        for mode in (
            dict.fromkeys((config["kv_upcast"], *KV_UPCAST_MODES))
            if native_fp8
            else normal_modes
        )
    ]


def _geometry_key(config: Mapping[str, object]) -> ConfigKey:
    return tuple(sorted((k, v) for k, v in config.items() if k != "kv_upcast"))


def _initial_search_configs(
    configs: list[SegmentedAttentionConfig],
) -> list[SegmentedAttentionConfig]:
    """Screen native4 geometries while retaining the static incumbent."""
    if not any(c["kv_upcast"] == "native4" for c in configs):
        return configs
    return [
        configs[0],
        *(c for c in configs if c["kv_upcast"] == "native4" and c != configs[0]),
    ]


def _upcast_expansion_configs(
    configs: list[SegmentedAttentionConfig], comparisons: list[_ConfigComparison]
) -> list[SegmentedAttentionConfig]:
    """Expand conversion modes on promising or unmeasurable geometries."""
    initial = _initial_search_configs(configs)
    attempted = {_config_key(c) for c in initial}
    if len(attempted) == len(configs):
        return []
    anchors = sorted(
        (r for r in comparisons if r["config"]["kv_upcast"] == "native4"),
        key=lambda r: r["paired_balanced_speedup_vs_default"],
        reverse=True,
    )
    measured = {_geometry_key(r["config"]) for r in anchors}
    geometries = {_geometry_key(c) for c in configs} - measured
    geometries.add(_geometry_key(configs[0]))
    geometries.update(
        _geometry_key(r["config"]) for r in anchors[:_UPCAST_GEOMETRY_SHORTLIST]
    )
    return [
        c
        for c in configs
        if _geometry_key(c) in geometries and _config_key(c) not in attempted
    ]


def _launch_candidate_configs(
    default: SegmentedLaunchConfig,
    batch: int,
    query_len: int,
    heads: int,
    dim: int,
    seq_len: int | None = None,
) -> list[SegmentedAttentionConfig]:
    """Search launch tiles and split counts within a bounded workspace."""
    base = _normalized_config(default, batch, query_len, heads, dim)
    max_splits = base["splits"]
    long_d64_decode = (
        dim == 64
        and batch == 1
        and query_len == 1
        and seq_len is not None
        and seq_len >= 131072
    )
    if long_d64_decode:
        max_splits = 512
    elif dim == 64 and query_len <= 512 and seq_len is not None and seq_len >= 4096:
        max_splits = min(64, 2 * max_splits)
    candidates: list[SegmentedAttentionConfig] = []
    seen = set()

    def add(**updates: Unpack[SegmentedConfigOverrides]) -> None:
        config: SegmentedAttentionConfig = {**base, **updates}
        if (
            config["splits"] > max_splits
            or dim % config["bk"]
            or config["reduce_d"] > dim
        ):
            return
        identity = tuple(sorted(config.items()))
        if identity not in seen:
            seen.add(identity)
            candidates.append(config)

    if long_d64_decode:
        add()
        for splits in (128, 256, 512, 64):
            for bn, warps, stages in (
                (32, 1, 1),
                (32, 2, 1),
                (32, 2, 2),
                (64, 1, 1),
                (64, 2, 1),
                (64, 2, 2),
            ):
                add(
                    bm=16,
                    bn=bn,
                    bk=64,
                    splits=splits,
                    warps=warps,
                    stages=stages,
                )
        return candidates

    split_choices = [base["splits"]]
    if max_splits > base["splits"]:
        split_choices.append(max_splits)
    for divisor in (2, 4):
        split_choices.append(max(1, base["splits"] // divisor))
    split_choices.append(1)
    for splits in split_choices:
        add(splits=splits)

    if dim == 512:
        # Keep the wide PV accumulator distributed over enough threads. Search
        # QK chunks separately instead of inheriting the full-width decode tile.
        add(prefix_fast=not base["prefix_fast"])
        add(pv_split=False, qk_pipeline=None, qk_unroll=None)
        add(qk_unroll=2)
        add(qk_unroll=4)
        if base["splits"] > 1:
            for reduce_d in (64, 128, dim):
                add(reduce_d=reduce_d)
            for warps in (1, 2):
                add(reduce_d=64, reduce_warps=warps)
        add(waves_per_eu=4)
        add(
            bm=32,
            bn=64,
            bk=64,
            warps=8,
            stages=2,
            waves_per_eu=4,
            pv_split=False,
            qk_pipeline=None,
            qk_unroll=None,
        )
        add(
            bm=32,
            bn=64,
            bk=128,
            warps=8,
            stages=2,
            waves_per_eu=4,
            pv_split=False,
            qk_pipeline=None,
            qk_unroll=None,
        )
        if query_len > 2:
            for bn in (32, 64):
                for pipeline in (None, 3):
                    add(
                        bm=64,
                        bn=bn,
                        bk=64,
                        warps=8,
                        stages=1,
                        waves_per_eu=6,
                        pv_split=True,
                        qk_pipeline=pipeline,
                        qk_unroll=None,
                    )
        if query_len <= 2:
            tiles = (
                (16, 32, 64, 4, 1),
                (16, 64, 64, 4, 1),
                (16, 32, 64, 8, 1),
                (16, 64, 64, 8, 1),
                (16, 32, 128, 8, 1),
                (16, 64, 128, 8, 1),
            )
        else:
            tiles = (
                (32, 32, 64, 8, 1),
                (32, 64, 64, 8, 1),
                (64, 32, 64, 8, 1),
                (64, 64, 64, 8, 1),
                (32, 32, 128, 8, 1),
                (32, 64, 64, 8, 2),
            )
        for splits in (base["splits"], max(1, base["splits"] // 2)):
            for bm, bn, bk, warps, stages in tiles:
                for waves in (2, 6) if splits == base["splits"] else (2,):
                    add(
                        bm=bm,
                        bn=bn,
                        bk=bk,
                        splits=splits,
                        warps=warps,
                        stages=stages,
                        waves_per_eu=waves,
                        pv_split=False,
                        qk_pipeline=None,
                        qk_unroll=None,
                    )
        return candidates

    tile_variants: tuple[tuple[int, int, int, int, int, int], ...] = (
        (16, 32, dim, 4, 1, 2),
        (16, 64, min(128, dim), 4, 1, 2),
        (32, 32, 64, 4, 2, 2),
        (32, 64, 64, 4, 1, 2),
        (64, 32, min(128, dim), 4, 1, 6),
        (128, 32, min(128, dim), 8, 1, 6),
    )
    if dim == 64:
        tile_variants += (
            (16, 16, 64, 4, 1, 2),
            (32, 16, 64, 4, 2, 2),
            (64, 32, 64, 4, 1, 2),
            (64, 64, 64, 4, 1, 2),
            (128, 32, 64, 4, 1, 2),
            (128, 64, 64, 4, 1, 2),
        )
    for bm, bn, bk, warps, stages, waves in tile_variants:
        add(
            bm=bm,
            bn=bn,
            bk=bk,
            warps=warps,
            stages=stages,
            waves_per_eu=waves,
        )

    half_splits = max(1, base["splits"] // 2)
    for bm, bn, bk, warps, stages, waves in tile_variants[:3]:
        add(
            bm=bm,
            bn=bn,
            bk=bk,
            splits=half_splits,
            warps=warps,
            stages=stages,
            waves_per_eu=waves,
        )

    if dim == 64:
        for splits in (max_splits, half_splits):
            for bm, bn, bk, warps, stages, waves in tile_variants[2:]:
                add(
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    splits=splits,
                    warps=warps,
                    stages=stages,
                    waves_per_eu=waves,
                )

    add(waves_per_eu=6)
    add(stages=2 if base["stages"] == 1 else 1)
    add(reduce_d=64 if base["reduce_d"] != 64 else dim)
    return candidates


def _identity(
    device,
    dtype,
    kv_dtype,
    heads,
    kv_heads,
    dim,
    page,
    scale,
    max_tokens,
    max_len,
    max_seqs,
    sliding_window,
    causal,
    max_query_len,
    has_sinks=False,
    physical_max_len=None,
):
    properties = torch.cuda.get_device_properties(device)
    from triton._C.libtriton import get_cache_invalidating_env_vars

    source = Path(__file__)
    kernel = Path(segmented_attention.__file__)
    return {
        "schema": 8,
        "gpu": properties.name,
        "arch": properties.gcnArchName,
        "compute_units": properties.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "compiler_env": get_cache_invalidating_env_vars(),
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "selector_sha256": hashlib.sha256(
            Path(segmented_attention_config.__file__).read_bytes()
        ).hexdigest(),
        "tuner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dtype": str(dtype),
        "kv_dtype": str(kv_dtype),
        "heads": heads,
        "kv_heads": kv_heads,
        "dim": dim,
        "page": page,
        "scale": scale,
        "max_tokens": max_tokens,
        "max_len": max_len,
        "max_seqs": max_seqs,
        "sliding_window": sliding_window,
        "causal": causal,
        "max_query_len": max_query_len,
        "has_sinks": has_sinks,
        "physical_max_len": physical_max_len,
        "query_buckets": list(_QUERY_BUCKETS),
        "batch_buckets": list(_BATCH_BUCKETS),
        "sequence_buckets": list(_SEQUENCE_BUCKETS),
        "compile_workers": _COMPILE_WORKERS,
        "condition_rounds": _CONDITION_ROUNDS,
        "screen_rounds": _SCREEN_ROUNDS,
        "finalist_count": _FINALIST_COUNT,
        "final_rounds": _FINAL_ROUNDS,
        "min_promotion_speedup": _MIN_PROMOTION_SPEEDUP,
        "warm_cache_guard": True,
        "kv_upcast_modes": list(KV_UPCAST_MODES)
        if supports_segmented_native_fp8(dtype, kv_dtype)
        else ["normal"],
        "kv_upcast_search": "native4_geometry_shortlist",
        "kv_upcast_geometry_shortlist": _UPCAST_GEOMETRY_SHORTLIST,
        "promotion_objective": "warm_cold_geomean",
        "graph_internal_events": True,
        "max_warm_slowdown": _MAX_WARM_SLOWDOWN,
        "warm_graph_calls": _WARM_GRAPH_CALLS,
        "max_warm_graph_calls": _MAX_WARM_GRAPH_CALLS,
        "warm_target_us": _WARM_TARGET_US,
    }


def _memory_budget(device):
    free, total = torch.accelerator.get_memory_info(device)
    available = int(min(free // 2, total // 4))
    return 1 << (available.bit_length() - 1) if available else 0


def _query_lengths(batch, query_len, max_tokens):
    if batch * query_len <= max_tokens:
        return [query_len] * batch
    return [min(query_len, max_tokens - batch + 1)] + [1] * (batch - 1)


def _scratch_bytes(
    dtype,
    kv_dtype,
    heads,
    kv_heads,
    dim,
    page,
    max_tokens,
    workload,
    physical_max_len=None,
):
    batch, query_len, seq_len = workload
    queries = _query_lengths(batch, query_len, max_tokens)
    tokens = sum(queries)
    scratch_seq_len = (
        max(seq_len, physical_max_len)
        if physical_max_len is not None and batch == 1 and query_len <= 8
        else seq_len
    )
    blocks = batch * math.ceil(scratch_seq_len / page)
    cache = 2 * blocks * page * kv_heads * dim * kv_dtype.itemsize
    tensors = tokens * dim * (4 * heads + 2 * kv_heads) * dtype.itemsize
    default = select_segmented_config(
        batch, query_len, seq_len, heads, kv_heads, dim, kv_dtype.itemsize == 1
    )
    candidates = _candidate_configs(
        default,
        batch,
        query_len,
        heads,
        dim,
        seq_len=seq_len,
        native_fp8=dtype == torch.bfloat16 and kv_dtype == torch.float8_e4m3fn,
        legacy_default=segmented_attention._select_segmented_launch_config(
            batch, query_len, seq_len, heads, kv_heads, dim, kv_dtype.itemsize == 1
        ),
    )
    shapes = segmented_workspace_shapes(
        batch,
        segmented_query_capacity(query_len),
        heads,
        kv_heads,
        dim,
        max(config["splits"] for config in candidates),
    )
    workspace = 0 if shapes is None else sum(math.prod(shape) * 4 for shape in shapes)
    return cache + tensors + workspace + 384 * 1024**2


def _workloads(
    max_tokens,
    max_len,
    max_seqs,
    *,
    memory_budget_bytes=None,
    dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    heads=8,
    kv_heads=1,
    dim=128,
    page=16,
    cache_layouts=(),
    cache_budget_bytes=None,
    max_query_len=MAX_QUERY_LEN,
    physical_max_len=None,
):
    query_limit = min(max_tokens, max_len, max_query_len, MAX_QUERY_LEN)
    queries = {q for q in _QUERY_BUCKETS if q <= query_limit}
    if query_limit:
        queries.add(query_limit)
    for query_len in sorted(queries):
        batch_limit = min(max_seqs, max_tokens - query_len + 1)
        if batch_limit < 1:
            continue
        if query_len <= 2:
            batches = {min(batch, batch_limit) for batch in _BATCH_BUCKETS}
        else:
            batches = {1, min(4, batch_limit), batch_limit}
        if query_len <= 2:
            sequences = {
                seq for seq in _SEQUENCE_BUCKETS if query_len <= seq <= max_len
            }
            sequences.update((query_len, max_len))
        else:
            sequences = {
                query_len,
                max(query_len, min(max_len, 2048)),
                max(query_len, min(max_len, 8192)),
                max_len,
            }
            # Keep long-context verification and prefill buckets even when
            # the model's maximum length is pruned by the KV memory budget.
            sequences.update(
                seq for seq in (32768, 131072) if query_len <= seq <= max_len
            )
        for batch, seq_len in itertools.product(sorted(batches), sorted(sequences)):
            workload = (batch, query_len, seq_len)
            if (
                memory_budget_bytes is not None
                and _scratch_bytes(
                    dtype,
                    kv_dtype,
                    heads,
                    kv_heads,
                    dim,
                    page,
                    max_tokens,
                    workload,
                    physical_max_len,
                )
                > memory_budget_bytes
            ):
                continue
            if cache_budget_bytes is not None:
                cache_bytes = batch * sum(
                    math.ceil(seq_len / block) * size if block else size
                    for block, size in cache_layouts
                )
                if cache_bytes > cache_budget_bytes:
                    continue
            yield workload


def _make_inputs(
    device,
    dtype,
    kv_dtype,
    heads,
    kv_heads,
    dim,
    page,
    max_tokens,
    workload,
    physical_seq_len=None,
):
    batch, query_len, seq_len = workload
    physical_seq_len = max(seq_len, physical_seq_len or seq_len)
    generator = torch.Generator(device=device).manual_seed(1234)

    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

    queries = _query_lengths(batch, query_len, max_tokens)
    tokens = sum(queries)
    q = randn(tokens, heads, dim)
    k = randn(tokens, kv_heads, dim)
    v = randn(tokens, kv_heads, dim)
    blocks_per_seq = math.ceil(seq_len / page)
    physical_blocks_per_seq = math.ceil(physical_seq_len / page)
    cache_blocks_per_seq = (
        physical_blocks_per_seq if batch == 1 and query_len <= 8 else blocks_per_seq
    )
    cache_shape = (batch * cache_blocks_per_seq, 2, page, kv_heads, dim)
    scale_value = 0.125 if kv_dtype.itemsize == 1 else 1.0
    cache = (randn(*cache_shape).float() / scale_value).to(kv_dtype)
    kc, vc = cache[:, 0], cache[:, 1]
    table = (
        torch.arange(physical_blocks_per_seq, device=device, dtype=torch.int32)[None, :]
        .remainder(cache_blocks_per_seq)
        .expand(batch, -1)
        .contiguous()
    )
    table += (
        torch.arange(batch, device=device, dtype=torch.int32)[:, None]
        * cache_blocks_per_seq
    )
    starts = torch.tensor(
        [0, *itertools.accumulate(queries)], device=device, dtype=torch.int32
    )
    lengths = torch.full((batch,), physical_seq_len, device=device, dtype=torch.int32)
    scale = torch.full((), scale_value, device=device, dtype=torch.float32)
    return q, k, v, kc, vc, table, starts, lengths, scale


def _config_key(config: Mapping[str, object]) -> ConfigKey:
    return tuple(sorted(config.items()))


def _rotated(configs, offset):
    offset %= len(configs)
    return configs[offset:] + configs[:offset]


def _measure_configs(
    configs: list[SegmentedAttentionConfig],
    run: Callable[[SegmentedAttentionConfig], object],
    device: torch.device | str,
    eviction: torch.Tensor,
    rounds: int,
    graph_calls: int = 0,
) -> dict[ConfigKey, list[float]]:
    """Interleave candidates so clock and thermal drift affect them evenly."""
    samples: dict[tuple, list[float]] = {_config_key(config): [] for config in configs}
    graphs = {}
    calls = max(1, graph_calls)
    for config in configs:
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if graph_calls <= 1:
                eviction.zero_()
            start.record()
            for _ in range(calls):
                run(config)
            end.record()
        graphs[_config_key(config)] = graph, start, end
    for graph, _, _ in graphs.values():
        graph.replay()
    torch.accelerator.synchronize(device)
    for round_index in range(rounds):
        events = []
        order = _rotated(configs, round_index)
        if round_index % 2:
            order = list(reversed(order))
        for config in order:
            graph, start, end = graphs[_config_key(config)]
            graph.replay()
            end.synchronize()
            events.append((_config_key(config), start, end))
        for key, start, end in events:
            samples[key].append(start.elapsed_time(end) * 1000 / calls)
    return samples


def _condition_configs(
    configs: list[SegmentedAttentionConfig],
    run: Callable[[SegmentedAttentionConfig], object],
    device: torch.device | str,
) -> None:
    for round_index in range(_CONDITION_ROUNDS):
        for config in _rotated(configs, round_index):
            run(config)
    torch.accelerator.synchronize(device)


def _select_tuned_winner(
    default: SegmentedAttentionConfig,
    finalists: list[SegmentedAttentionConfig],
    samples: dict[ConfigKey, list[float]],
    min_promotion_speedup: float = _MIN_PROMOTION_SPEEDUP,
    *,
    warm_samples: dict[ConfigKey, list[float]] | None = None,
) -> tuple[_ConfigComparison, list[_ConfigComparison]]:
    """Return a finalist only when it reliably beats the static incumbent."""
    default_key = _config_key(default)
    default_samples = samples[default_key]
    comparisons: list[_ConfigComparison] = []
    for config in finalists:
        key = _config_key(config)
        config_samples = samples[key]
        paired_speedup = statistics.median(
            baseline / candidate
            for baseline, candidate in zip(default_samples, config_samples, strict=True)
        )
        comparisons.append(
            {
                "config": config,
                "us": statistics.median(config_samples),
                "paired_speedup_vs_default": paired_speedup,
            }
        )
        if warm_samples is not None:
            warm_speedup = statistics.median(
                baseline / candidate
                for baseline, candidate in zip(
                    warm_samples[default_key], warm_samples[key], strict=True
                )
            )
            comparisons[-1].update(
                {
                    "paired_warm_speedup_vs_default": warm_speedup,
                    "rejected_warm_regression": warm_speedup < 1 / _MAX_WARM_SLOWDOWN,
                    "rejected_cold_regression": paired_speedup < 1 / _MAX_WARM_SLOWDOWN,
                    "paired_balanced_speedup_vs_default": math.sqrt(
                        paired_speedup * warm_speedup
                    ),
                }
            )
    challenger = max(
        (
            row
            for row in comparisons
            if not row.get("rejected_warm_regression", False)
            and not row.get("rejected_cold_regression", False)
        ),
        key=lambda result: result.get(
            "paired_balanced_speedup_vs_default", result["paired_speedup_vs_default"]
        ),
    )
    if (
        challenger.get(
            "paired_balanced_speedup_vs_default",
            challenger["paired_speedup_vs_default"],
        )
        >= min_promotion_speedup
    ):
        return challenger, comparisons
    return (
        next(result for result in comparisons if result["config"] == default),
        comparisons,
    )


def _precompile_configs(
    configs,
    q,
    output,
    kc,
    vc,
    table,
    starts,
    lengths,
    query_len,
    kv_scale,
    scale,
    workspace,
    sliding_window,
    causal,
    sinks=None,
):
    workers = min(_COMPILE_WORKERS, len(configs))
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        triton.AsyncCompileMode(executor, ignore_errors=True),
    ):
        for config in configs:
            compile_segmented_attention(
                q,
                output,
                kc,
                vc,
                table,
                starts,
                lengths,
                query_len,
                kv_scale,
                kv_scale,
                scale,
                config,
                workspace,
                sliding_window=sliding_window,
                causal=causal,
                sinks=sinks,
            )


def _tune_workload(
    device: torch.device | str,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    heads: int,
    kv_heads: int,
    dim: int,
    page: int,
    scale: float,
    max_tokens: int,
    workload: tuple[int, int, int],
    *,
    sliding_window: int = -1,
    causal: bool = True,
    has_sinks: bool = False,
    physical_seq_len: int | None = None,
) -> dict[str, Any]:
    batch, query_len, seq_len = workload
    q, _, _, kc, vc, table, starts, lengths, kv_scale = _make_inputs(
        device,
        dtype,
        kv_dtype,
        heads,
        kv_heads,
        dim,
        page,
        max_tokens,
        workload,
        physical_seq_len,
    )
    default = select_segmented_config(
        batch,
        query_len,
        seq_len,
        heads,
        kv_heads,
        dim,
        kv_dtype.itemsize == 1,
        sliding_window=sliding_window,
        native_fp8=supports_segmented_native_fp8(dtype, kv_dtype),
    )
    configs = _candidate_configs(
        default,
        batch,
        query_len,
        heads,
        dim,
        seq_len=seq_len,
        native_fp8=supports_segmented_native_fp8(dtype, kv_dtype),
        legacy_default=segmented_attention._select_segmented_launch_config(
            batch, query_len, seq_len, heads, kv_heads, dim, kv_dtype.itemsize == 1
        ),
    )
    sinks = (
        torch.linspace(-1, 1, heads, dtype=torch.float32, device=device)
        if has_sinks
        else None
    )
    qcap = segmented_query_capacity(query_len)
    shapes = segmented_workspace_shapes(
        batch, qcap, heads, kv_heads, dim, max(c["splits"] for c in configs)
    )
    workspace = (
        None
        if shapes is None
        else tuple(
            torch.empty(shape, dtype=torch.float32, device=device) for shape in shapes
        )
    )
    reference = torch.empty_like(q)
    output = torch.empty_like(q)

    def run(config, out=output):
        run_segmented_attention(
            q,
            out,
            kc,
            vc,
            table,
            starts,
            lengths,
            query_len,
            seq_len,
            kv_scale,
            kv_scale,
            scale,
            sliding_window=sliding_window,
            causal=causal,
            config=config,
            workspace=workspace,
            sinks=sinks,
        )

    def precompile(candidates: list[SegmentedAttentionConfig]) -> None:
        _precompile_configs(
            candidates,
            q,
            output,
            kc,
            vc,
            table,
            starts,
            lengths,
            query_len,
            kv_scale,
            scale,
            workspace,
            sliding_window,
            causal,
            sinks,
        )

    initial_configs = _initial_search_configs(configs)
    precompile(initial_configs)
    incumbent = configs[0]
    reference.fill_(float("nan"))
    run(incumbent, reference)
    torch.accelerator.synchronize(device)
    if not torch.isfinite(reference).all():
        raise AssertionError("Default segmented attention output is not finite")
    reference_norm = reference.float().norm().clamp_min(1e-6)
    eviction = torch.empty(256 * 1024**2, dtype=torch.int8, device=device)
    results: list[dict[str, Any]] = []

    def validate(
        candidates: list[SegmentedAttentionConfig],
    ) -> list[SegmentedAttentionConfig]:
        valid = []
        for config in candidates:
            try:
                output.fill_(float("nan"))
                run(config)
                torch.accelerator.synchronize(device)
                torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
                relative_l2 = (
                    (output.float() - reference.float()).norm() / reference_norm
                ).item()
                if relative_l2 > 0.005:
                    raise AssertionError(
                        f"Relative L2 error {relative_l2} exceeds 0.5%"
                    )
                results.append({"config": config, "relative_l2": relative_l2})
                valid.append(config)
            except (
                torch.OutOfMemoryError,
                triton.OutOfResources,
                triton.CompilationError,
                AssertionError,
                RuntimeError,
            ) as error:
                logger.warning(
                    "Rejected segmented attention config %s: %s", config, error
                )
        return valid

    valid_configs = validate(initial_configs)
    if incumbent not in valid_configs:
        raise RuntimeError("Static segmented attention configuration was rejected")

    graph_calls = 1
    screen_baselines = []

    def screen(
        candidates: list[SegmentedAttentionConfig], stage: int, warm_calls: int = 0
    ) -> tuple[list[_ConfigComparison], int]:
        _condition_configs(candidates, run, device)
        cold = _measure_configs(
            candidates, run, device, eviction, _SCREEN_ROUNDS, graph_calls
        )
        if not warm_calls:
            warm_calls = min(
                _MAX_WARM_GRAPH_CALLS,
                max(
                    _WARM_GRAPH_CALLS,
                    math.ceil(
                        _WARM_TARGET_US
                        / statistics.median(cold[_config_key(incumbent)])
                    ),
                ),
            )
        warm = _measure_configs(
            candidates, run, device, eviction, _SCREEN_ROUNDS, warm_calls
        )
        _, comparisons = _select_tuned_winner(
            incumbent, candidates, cold, warm_samples=warm
        )
        screen_baselines.append(
            dict(
                stage=stage,
                cold_us=cold[_config_key(incumbent)],
                warm_us=warm[_config_key(incumbent)],
            )
        )
        by_key = {_config_key(result["config"]): result for result in results}
        for row in comparisons:
            key = _config_key(row["config"])
            if row["config"] == incumbent and stage != 0:
                continue
            by_key[key].update(
                search_stage=stage,
                screen_us=statistics.median(cold[key]),
                screen_samples_us=cold[key],
                warm_screen_us=statistics.median(warm[key]),
                warm_screen_samples_us=warm[key],
                screen_paired_cold_speedup=row["paired_speedup_vs_default"],
                screen_paired_warm_speedup=row["paired_warm_speedup_vs_default"],
                screen_paired_balanced_speedup=row[
                    "paired_balanced_speedup_vs_default"
                ],
            )
        return comparisons, warm_calls

    screen_comparisons, warm_calls = screen(valid_configs, 0)
    expanded_configs = _upcast_expansion_configs(configs, screen_comparisons)
    if expanded_configs:
        precompile(expanded_configs)
        expanded_valid = validate(expanded_configs)
        if expanded_valid:
            expanded_comparisons, _ = screen(
                [incumbent, *expanded_valid], 1, warm_calls
            )
            screen_comparisons.extend(
                row for row in expanded_comparisons if row["config"] != incumbent
            )
            valid_configs.extend(expanded_valid)

    by_key = {_config_key(result["config"]): result for result in results}
    warm_eligible = {
        _config_key(row["config"])
        for row in screen_comparisons
        if not row.get("rejected_warm_regression", False)
        and not row.get("rejected_cold_regression", False)
    }
    challengers = sorted(
        (
            config
            for config in valid_configs
            if config != incumbent and _config_key(config) in warm_eligible
        ),
        key=lambda config: next(
            row["paired_balanced_speedup_vs_default"]
            for row in screen_comparisons
            if row["config"] == config
        ),
        reverse=True,
    )
    finalists = [incumbent, *challengers[: _FINALIST_COUNT - 1]]
    _condition_configs(finalists, run, device)
    final_samples = _measure_configs(
        finalists, run, device, eviction, _FINAL_ROUNDS, graph_calls
    )
    warm_final = _measure_configs(
        finalists, run, device, eviction, _FINAL_ROUNDS, warm_calls
    )
    min_promotion_speedup = (
        1.10
        if dim == 64 and has_sinks and sliding_window >= 0
        else _MIN_PROMOTION_SPEEDUP
    )
    winner, comparisons = _select_tuned_winner(
        incumbent,
        finalists,
        final_samples,
        min_promotion_speedup,
        warm_samples=warm_final,
    )
    for comparison in comparisons:
        result = by_key[_config_key(comparison["config"])]
        result["final_us"] = comparison["us"]
        result["final_samples_us"] = final_samples[_config_key(comparison["config"])]
        result["paired_speedup_vs_default"] = comparison["paired_speedup_vs_default"]
        if warm_final is not None:
            key = _config_key(comparison["config"])
            result.update(
                warm_final_us=statistics.median(warm_final[key]),
                warm_final_samples_us=warm_final[key],
                paired_warm_speedup_vs_default=comparison[
                    "paired_warm_speedup_vs_default"
                ],
                rejected_warm_regression=comparison["rejected_warm_regression"],
                rejected_cold_regression=comparison["rejected_cold_regression"],
                paired_balanced_speedup_vs_default=comparison[
                    "paired_balanced_speedup_vs_default"
                ],
            )
    winner_result = by_key[_config_key(winner["config"])]
    logger.info(
        "ROCm segmented attention autotune: B=%d Q=%d seq=%d best=%s "
        "%.2f us speedup=%.3fx%s",
        batch,
        query_len,
        seq_len,
        winner["config"],
        winner["us"],
        winner["paired_speedup_vs_default"],
        " (static retained)" if winner["config"] == incumbent else "",
    )
    return {
        "workload": list(workload),
        "query_lengths": _query_lengths(batch, query_len, max_tokens),
        "default": default,
        "best": winner["config"],
        "search": {
            "available_candidates": len(configs),
            "initial_candidates": len(initial_configs),
            "expanded_candidates": len(expanded_configs),
            "valid_candidates": len(results),
            "screen_baselines": screen_baselines,
        },
        "selection": {
            "incumbent": incumbent,
            "finalists": finalists,
            "min_promotion_speedup": min_promotion_speedup,
            "winner_us": winner_result["final_us"],
            "warm_graph_calls": warm_calls,
            "paired_speedup_vs_default": winner["paired_speedup_vs_default"],
            **(
                {
                    "paired_warm_speedup_vs_default": winner[
                        "paired_warm_speedup_vs_default"
                    ],
                    "max_warm_slowdown": _MAX_WARM_SLOWDOWN,
                    "paired_balanced_speedup_vs_default": winner[
                        "paired_balanced_speedup_vs_default"
                    ],
                }
                if warm_final is not None
                else {}
            ),
        },
        "results": results,
    }


def _save(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_records(path, identity, heads, kv_heads, dim, fp8):
    """Load a canonical table and any crash-recovery TP shards."""
    records: dict[tuple, dict] = {}
    merged_shards = []
    sources = [path, *sorted(path.parent.glob(f"{path.name}.tp*.part"))]
    for source in sources:
        if not source.exists():
            continue
        try:
            saved = json.loads(source.read_text())
            if saved["identity"] != identity or not all(
                _valid_record(
                    record,
                    heads,
                    kv_heads,
                    dim,
                    fp8,
                    native_fp8=len(identity["kv_upcast_modes"]) > 1,
                    sliding_window=identity["sliding_window"],
                )
                for record in saved["records"]
            ):
                raise ValueError("identity or record validation failed")
            records.update(
                (tuple(record["workload"]), record) for record in saved["records"]
            )
            if source != path:
                merged_shards.append(source)
        except (ValueError, KeyError, TypeError, AssertionError):
            logger.warning(
                "Ignoring invalid segmented attention tuning cache %s", source
            )
    return {"identity": identity, "records": list(records.values())}, merged_shards


def _tp_context():
    """Return the initialized TP coordinator, or local-only tuning state."""
    from vllm.distributed.parallel_state import (
        get_tp_group,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return None, 0, 1
    group = get_tp_group()
    return group, group.rank_in_group, group.world_size


def _gather_tp(group, value):
    if group is None or group.world_size == 1:
        return [value]
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, value, group=group.cpu_group)
    return gathered


def _workload_weight(workload, max_tokens, heads, kv_heads, dim, fp8):
    """Estimate compile plus execution cost for TP workload balancing."""
    batch, query_len, seq_len = workload
    default = select_segmented_config(
        batch, query_len, seq_len, heads, kv_heads, dim, fp8
    )
    candidates = len(
        _candidate_configs(
            default,
            batch,
            query_len,
            heads,
            dim,
            seq_len=seq_len,
            native_fp8=fp8,
            legacy_default=segmented_attention._select_segmented_launch_config(
                batch, query_len, seq_len, heads, kv_heads, dim, fp8
            ),
        )
    )
    query_tokens = sum(_query_lengths(batch, query_len, max_tokens))
    attention_work = query_tokens * seq_len * heads * dim
    return candidates * (1 << 40) + attention_work


def _shard_workloads(workloads, world_size, max_tokens, heads, kv_heads, dim, fp8):
    """Balance compile-affine ``(batch, query)`` groups across TP ranks."""
    assignments: list[list[tuple[int, tuple]]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    groups: dict[tuple, dict] = {}
    for index, workload in enumerate(workloads):
        compile_group = groups.setdefault(workload[:2], {"weight": 0, "items": []})
        compile_group["weight"] += _workload_weight(
            workload, max_tokens, heads, kv_heads, dim, fp8
        )
        compile_group["items"].append((index, workload))
    for compile_group in sorted(
        groups.values(), key=lambda item: item["weight"], reverse=True
    ):
        rank = min(
            range(world_size), key=lambda candidate: (loads[candidate], candidate)
        )
        assignments[rank].extend(compile_group["items"])
        loads[rank] += compile_group["weight"]
    return [
        [workload for _, workload in sorted(assignment)] for assignment in assignments
    ]


def _valid_record(
    record, heads, kv_heads, dim, fp8, *, native_fp8=False, sliding_window=-1
):
    workload = record["workload"]
    if len(workload) != 3 or any(
        type(value) is not int or value < 1 for value in workload
    ):
        return False
    batch, query_len, seq_len = workload
    default = select_segmented_config(
        batch,
        query_len,
        seq_len,
        heads,
        kv_heads,
        dim,
        fp8,
        sliding_window=sliding_window,
        native_fp8=native_fp8,
    )
    return record["best"] in _candidate_configs(
        default,
        batch,
        query_len,
        heads,
        dim,
        seq_len=seq_len,
        native_fp8=native_fp8,
        legacy_default=segmented_attention._select_segmented_launch_config(
            batch, query_len, seq_len, heads, kv_heads, dim, fp8
        ),
    )


@contextmanager
def _temporary_tuning_kernels(device):
    """Release tuning-only HIP modules without invalidating existing graphs."""
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    caches = []
    for kernel in (_segmented_attention_stage, _segmented_attention_reduce):
        cache = kernel.device_caches[device_index][0]
        caches.append((cache, set(cache)))
    try:
        yield
    finally:
        torch.accelerator.synchronize(device)
        for cache, existing_keys in caches:
            for key in cache.keys() - existing_keys:
                del cache[key]
        # CompiledKernel destruction unloads the HIP module. Keep pre-existing
        # kernels alive, since an earlier graph capture may reference them.
        gc.collect()
        torch.accelerator.empty_cache()


@torch.inference_mode()
def warmup_segmented_attention(
    device,
    dtype,
    heads,
    kv_heads,
    dim,
    page,
    scale,
    max_tokens,
    max_len,
    max_seqs,
    *,
    memory_budget_bytes=None,
    cache_layouts=(),
    cache_budget_bytes=None,
    kv_dtype=torch.bfloat16,
    sliding_window=-1,
    causal=True,
    max_query_len=MAX_QUERY_LEN,
    has_sinks=False,
    physical_max_len=None,
):
    """Tune reachable segmented buckets before graph capture."""
    if (
        dtype not in (torch.bfloat16, torch.float16)
        or kv_dtype
        not in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        )
        or dim not in (64, 128, 256, 512)
        or kv_heads < 1
        or heads % kv_heads
        or not 1 <= heads // kv_heads <= 16
    ):
        return
    key = _key(
        device,
        dtype,
        kv_dtype,
        heads,
        kv_heads,
        dim,
        page,
        scale,
        sliding_window,
        causal,
        has_sinks,
    )
    identity = _identity(
        device,
        dtype,
        kv_dtype,
        heads,
        kv_heads,
        dim,
        page,
        scale,
        max_tokens,
        max_len,
        max_seqs,
        sliding_window,
        causal,
        max_query_len,
        has_sinks,
        physical_max_len,
    )
    if memory_budget_bytes is None:
        memory_budget_bytes = _memory_budget(device)
    workload_args = dict(
        memory_budget_bytes=memory_budget_bytes,
        dtype=dtype,
        kv_dtype=kv_dtype,
        heads=heads,
        kv_heads=kv_heads,
        dim=dim,
        page=page,
        cache_layouts=cache_layouts,
        cache_budget_bytes=cache_budget_bytes,
        max_query_len=max_query_len,
        physical_max_len=(
            physical_max_len if has_sinks and sliding_window >= 0 else None
        ),
    )
    workloads = list(_workloads(max_tokens, max_len, max_seqs, **workload_args))
    active = _TABLES.get(key)
    warmed = (
        {tuple(record["workload"]) for record in active.get("records", [])}
        if active is not None and active.get("identity") == identity
        else set()
    )
    if all(workload in warmed for workload in workloads):
        return
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(envs.VLLM_CACHE_ROOT) / "rocm_segmented_attention" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    group, tp_rank, tp_size = _tp_context()
    if group is not None:
        digests = _gather_tp(group, digest)
        if any(peer != digest for peer in digests):
            logger.warning(
                "ROCm segmented attention TP ranks have different tuning "
                "identities; tuning this rank independently"
            )
            group, tp_rank, tp_size = None, 0, 1

    unpruned = len(
        list(_workloads(max_tokens, max_len, max_seqs, max_query_len=max_query_len))
    )
    start = time.monotonic()
    lock = FileLock(str(path) + ".lock") if tp_rank == 0 else None
    if lock is not None:
        lock.acquire()
    try:
        if tp_rank == 0:
            data, merged_shards = _load_records(
                path, identity, heads, kv_heads, dim, kv_dtype.itemsize == 1
            )
        else:
            data, merged_shards = None, []
        if group is not None:
            data = group.broadcast_object(data, src=0)
        assert data is not None
        records = {tuple(record["workload"]): record for record in data["records"]}
        missing = [workload for workload in workloads if workload not in records]
        loaded = len(workloads) - len(missing)
        assignments = _shard_workloads(
            missing,
            tp_size,
            max_tokens,
            heads,
            kv_heads,
            dim,
            kv_dtype.itemsize == 1,
        )
        local_workloads = assignments[tp_rank]
        logger.info(
            "ROCm segmented attention tuning plan: workloads=%d missing=%d "
            "pruned=%d tp_rank=%d/%d assigned=%d scratch_budget=%d MiB "
            "cache_budget=%s",
            len(workloads),
            len(missing),
            unpruned - len(workloads),
            tp_rank,
            tp_size,
            len(local_workloads),
            memory_budget_bytes // 2**20,
            cache_budget_bytes,
        )

        if not missing:
            if tp_rank == 0 and merged_shards:
                _save(path, data)
                for shard in merged_shards:
                    shard.unlink(missing_ok=True)
            _TABLES[key] = data
            logger.info(
                "ROCm segmented attention autotune ready: tp_rank=%d/%d "
                "local_tuned=0 tuned=0 loaded=%d failed=0 elapsed=%.2fs cache=%s",
                tp_rank,
                tp_size,
                loaded,
                time.monotonic() - start,
                path,
            )
            return

        shard_path = path.with_suffix(path.suffix + f".tp{tp_rank}.part")
        local_records = {}
        local_failed = 0
        fatal = None
        try:
            with _temporary_tuning_kernels(device):
                for workload in local_workloads:
                    try:
                        record = _tune_workload(
                            device,
                            dtype,
                            kv_dtype,
                            heads,
                            kv_heads,
                            dim,
                            page,
                            scale,
                            max_tokens,
                            workload,
                            sliding_window=sliding_window,
                            causal=causal,
                            has_sinks=has_sinks,
                            physical_seq_len=(
                                physical_max_len
                                if has_sinks and sliding_window >= 0
                                else None
                            ),
                        )
                    except (
                        torch.OutOfMemoryError,
                        RuntimeError,
                        AssertionError,
                    ) as error:
                        local_failed += 1
                        logger.warning(
                            "Skipping segmented attention tuning workload %s: %s",
                            workload,
                            error,
                        )
                    else:
                        local_records[workload] = record
                        _save(
                            shard_path,
                            {
                                "identity": identity,
                                "records": list(local_records.values()),
                            },
                        )
                    finally:
                        gc.collect()
                        torch.accelerator.empty_cache()
        except BaseException as error:
            fatal = f"{type(error).__name__}: {error}"

        gathered = _gather_tp(
            group,
            {
                "records": list(local_records.values()),
                "failed": local_failed,
                "fatal": fatal,
            },
        )
        if tp_rank == 0:
            fatal_errors = []
            for rank, contribution in enumerate(gathered):
                records.update(
                    (tuple(record["workload"]), record)
                    for record in contribution["records"]
                )
                if contribution["fatal"] is not None:
                    fatal_errors.append(f"TP rank {rank}: {contribution['fatal']}")
            data = {"identity": identity, "records": list(records.values())}
            _save(path, data)
            for shard in path.parent.glob(f"{path.name}.tp*.part"):
                shard.unlink(missing_ok=True)
            result = {
                "data": data,
                "tuned": sum(len(item["records"]) for item in gathered),
                "failed": sum(item["failed"] for item in gathered),
                "fatal": fatal_errors,
            }
        else:
            result = None
        if group is not None:
            result = group.broadcast_object(result, src=0)
        assert result is not None
        data = result["data"]
    finally:
        if lock is not None:
            lock.release()

    _TABLES[key] = data
    logger.info(
        "ROCm segmented attention autotune ready: tp_rank=%d/%d "
        "local_tuned=%d tuned=%d loaded=%d failed=%d elapsed=%.2fs cache=%s",
        tp_rank,
        tp_size,
        len(local_records),
        result["tuned"],
        loaded,
        result["failed"],
        time.monotonic() - start,
        path,
    )
    if result["fatal"]:
        raise RuntimeError(
            "ROCm segmented attention distributed tuning failed: "
            + "; ".join(result["fatal"])
        )


def _runtime_config(
    record: _ConfigRecord,
    batch: int,
    query_len: int,
    seq_len: int,
    heads: int,
    kv_heads: int,
    dim: int,
    fp8: bool,
    *,
    sliding_window: int = -1,
) -> SegmentedAttentionConfig:
    """Preserve tuned parallelism below a warmed batch/query ceiling."""
    config = record["best"].copy()
    if config["splits"] == 1 and 0 <= sliding_window < 255:
        return config
    warm_batch, warm_query, _ = record["workload"]
    gqa = heads // kv_heads
    warm_groups = warm_batch * triton.cdiv(warm_query * gqa, config["bm"])
    groups = batch * triton.cdiv(query_len * gqa, config["bm"])
    if groups < warm_groups:
        default = select_segmented_config(
            batch, query_len, seq_len, heads, kv_heads, dim, fp8, native_fp8=False
        )
        cap = max(config["splits"], default["splits"])
        factor = triton.next_power_of_2(triton.cdiv(warm_groups, groups))
        config["splits"] = min(cap, config["splits"] * factor)
    return config


def get_segmented_config(
    device: torch.device | str,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    heads: int,
    kv_heads: int,
    dim: int,
    page: int,
    scale: float,
    batch: int,
    query_len: int,
    seq_len: int,
    sliding_window: int = -1,
    causal: bool = True,
    has_sinks: bool = False,
) -> SegmentedAttentionConfig | None:
    """Return a warmed ceiling bucket, or None for the static fallback."""
    data = _TABLES.get(
        _key(
            device,
            dtype,
            kv_dtype,
            heads,
            kv_heads,
            dim,
            page,
            scale,
            sliding_window,
            causal,
            has_sinks,
        )
    )
    if data is None:
        return None
    records = data["records"]
    eligible = [
        record
        for record in records
        if record["workload"][0] >= batch
        and record["workload"][1] >= query_len
        and record["workload"][2] >= seq_len
    ]
    if not eligible:
        return None
    winner = min(
        eligible,
        key=lambda record: (
            record["workload"][1],
            record["workload"][0],
            record["workload"][2],
        ),
    )
    return _runtime_config(
        winner,
        batch,
        query_len,
        seq_len,
        heads,
        kv_heads,
        dim,
        kv_dtype.itemsize == 1,
        sliding_window=sliding_window,
    )


def _warmup_segmented_attention(layer, device, **limits) -> None:
    impl = layer.impl
    config = impl._segmented_attention_config
    assert config is not None
    spec = layer.get_kv_cache_spec(config)
    assert spec is not None
    sliding_window = impl.sliding_window[0]
    query_limit = getattr(layer, "segmented_query_limit", None)
    if not isinstance(query_limit, int):
        query_limit = MAX_QUERY_LEN
    causal = getattr(layer, "segmented_causal", True)
    if not isinstance(causal, bool):
        causal = True
    tuning_max_len = config.model_config.max_model_len
    if sliding_window >= 0:
        tuning_max_len = min(tuning_max_len, sliding_window + query_limit)
    warmup_segmented_attention(
        device,
        config.model_config.dtype,
        impl.num_heads,
        impl.num_kv_heads,
        impl.head_size,
        spec.block_size,
        impl.scale,
        config.scheduler_config.max_num_batched_tokens,
        tuning_max_len,
        config.scheduler_config.max_num_seqs,
        kv_dtype=spec.dtype if spec.dtype != torch.uint8 else impl.fp8_dtype,
        sliding_window=sliding_window,
        causal=causal,
        has_sinks=impl.sinks is not None,
        physical_max_len=config.model_config.max_model_len,
        max_query_len=query_limit,
        **limits,
    )
    impl._segmented_attention_warmed_up = True


def rocm_segmented_attn_autotune_warmup(worker: Worker) -> None:
    """Autotune RDNA segmented attention after KV allocation and before capture."""
    if (
        not current_platform.is_rocm()
        or not worker.vllm_config.kernel_config.enable_rocm_segmented_attn_autotune
    ):
        return

    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.backends.rocm_segmented_attn import (
        RocmSegmentedAttentionImpl,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    if not on_gfx1x():
        return
    layers = []
    for layer in worker.vllm_config.compilation_config.static_forward_context.values():
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, RocmSegmentedAttentionImpl):
            continue
        config = impl._segmented_attention_config
        if (
            config is not None
            and config.kernel_config.enable_rocm_segmented_attn_autotune
            and not impl._segmented_attention_warmed_up
            and impl.alibi_slopes is None
            and not impl.logits_soft_cap
        ):
            layers.append(layer)
    if not layers:
        return

    specs = worker.get_kv_cache_spec()
    layouts = tuple(
        (
            (spec.block_size, spec.page_size_bytes)
            if isinstance(spec, FullAttentionSpec)
            else (0, spec.max_memory_usage_bytes(worker.vllm_config))
        )
        for spec in specs.values()
    )
    cache_budget = sum(
        tensor.size
        for tensor in worker.model_runner.kv_cache_config.kv_cache_tensors
        if not tensor.host_resident
    )
    budget = _memory_budget(worker.device)
    for layer in layers:
        _warmup_segmented_attention(
            layer,
            worker.device,
            memory_budget_bytes=budget,
            cache_layouts=layouts,
            cache_budget_bytes=cache_budget,
        )
