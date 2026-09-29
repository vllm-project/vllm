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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from filelock import FileLock

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import triton

from .segmented_attention import (
    MAX_QUERY_LEN,
    compile_segmented_attention,
    run_segmented_attention,
    segmented_query_capacity,
    segmented_workspace_shapes,
    select_segmented_config,
)

logger = init_logger(__name__)

_TABLES: dict[tuple, dict] = {}
_QUERY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)
_SEQUENCE_BUCKETS = (128, 512, 2048, 8192, 32768, 131072, 262144)
_COMPILE_WORKERS = 2
_CONDITION_ROUNDS = 2
_SCREEN_ROUNDS = 5
_FINALIST_COUNT = 3
_FINAL_ROUNDS = 7
_MIN_PROMOTION_SPEEDUP = 1.02


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


def _normalized_config(config, batch, query_len, heads, dim):
    qcap = segmented_query_capacity(query_len)
    result = dict(config)
    result.setdefault("waves_per_eu", 2)
    result.setdefault("reduce_d", 64 if batch * qcap * heads < 64 else dim)
    result.setdefault("reduce_warps", 4)
    return result


def _candidate_configs(default, batch, query_len, heads, dim, seq_len=None):
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
    candidates: list[dict] = []
    seen = set()

    def add(**updates):
        config = {**base, **updates}
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
    kernel = source.with_name("segmented_attention.py")
    return {
        "schema": 5,
        "gpu": properties.name,
        "arch": properties.gcnArchName,
        "compute_units": properties.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "compiler_env": get_cache_invalidating_env_vars(),
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
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
        default, batch, query_len, heads, dim, seq_len=seq_len
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


def _config_key(config):
    return tuple(sorted(config.items()))


def _rotated(configs, offset):
    offset %= len(configs)
    return configs[offset:] + configs[:offset]


def _measure_configs(configs, run, device, eviction, rounds, graph_calls=0):
    """Interleave candidates so clock and thermal drift affect them evenly."""
    stream = torch.cuda.current_stream(device)
    samples: dict[tuple, list[float]] = {_config_key(config): [] for config in configs}
    graphs = {}
    if graph_calls:
        for config in configs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_calls):
                    run(config)
            graphs[_config_key(config)] = graph
        for graph in graphs.values():
            graph.replay()
        torch.accelerator.synchronize(device)
    for round_index in range(rounds):
        events = []
        order = _rotated(configs, round_index)
        if round_index % 2:
            order = list(reversed(order))
        for config in order:
            if not graph_calls:
                eviction.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            if graph_calls:
                graphs[_config_key(config)].replay()
            else:
                run(config)
            end.record(stream)
            events.append((_config_key(config), start, end))
        events[-1][2].synchronize()
        for key, start, end in events:
            samples[key].append(start.elapsed_time(end) * 1000 / max(1, graph_calls))
    return samples


def _condition_configs(configs, run, device):
    for round_index in range(_CONDITION_ROUNDS):
        for config in _rotated(configs, round_index):
            run(config)
    torch.accelerator.synchronize(device)


def _select_tuned_winner(
    default, finalists, samples, min_promotion_speedup=_MIN_PROMOTION_SPEEDUP
):
    """Return a finalist only when it reliably beats the static incumbent."""
    default_key = _config_key(default)
    default_samples = samples[default_key]
    comparisons = []
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
    challenger = max(
        comparisons, key=lambda result: result["paired_speedup_vs_default"]
    )
    if challenger["paired_speedup_vs_default"] >= min_promotion_speedup:
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
    *,
    sliding_window=-1,
    causal=True,
    has_sinks=False,
    physical_seq_len=None,
):
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
        batch, query_len, seq_len, heads, kv_heads, dim, kv_dtype.itemsize == 1
    )
    configs = _candidate_configs(default, batch, query_len, heads, dim, seq_len=seq_len)
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

    _precompile_configs(
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
        sinks,
    )
    incumbent = configs[0]
    reference.fill_(float("nan"))
    run(incumbent, reference)
    torch.accelerator.synchronize(device)
    if not torch.isfinite(reference).all():
        raise AssertionError("Default segmented attention output is not finite")
    reference_norm = reference.float().norm().clamp_min(1e-6)
    eviction = torch.empty(256 * 1024**2, dtype=torch.int8, device=device)
    results = []
    valid_configs = []
    for config in configs:
        try:
            output.fill_(float("nan"))
            run(config)
            torch.accelerator.synchronize(device)
            torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
            relative_l2 = (
                (output.float() - reference.float()).norm() / reference_norm
            ).item()
            if relative_l2 > 0.005:
                raise AssertionError(f"Relative L2 error {relative_l2} exceeds 0.5%")
            results.append({"config": config, "relative_l2": relative_l2})
            valid_configs.append(config)
        except (
            torch.OutOfMemoryError,
            triton.OutOfResources,
            triton.CompilationError,
            AssertionError,
            RuntimeError,
        ) as error:
            logger.warning("Rejected segmented attention config %s: %s", config, error)
    if not results:
        raise RuntimeError("No valid segmented attention launch configurations")
    if incumbent not in valid_configs:
        raise RuntimeError("Static segmented attention configuration was rejected")

    _condition_configs(valid_configs, run, device)
    graph_calls = 5 if dim == 64 and has_sinks else 0
    screen_samples = _measure_configs(
        valid_configs, run, device, eviction, _SCREEN_ROUNDS, graph_calls
    )
    by_key = {_config_key(result["config"]): result for result in results}
    for key, samples in screen_samples.items():
        by_key[key]["screen_us"] = statistics.median(samples)
        by_key[key]["screen_samples_us"] = samples

    challengers = sorted(
        (config for config in valid_configs if config != incumbent),
        key=lambda config: statistics.median(screen_samples[_config_key(config)]),
    )
    finalists = [incumbent, *challengers[: _FINALIST_COUNT - 1]]
    _condition_configs(finalists, run, device)
    final_samples = _measure_configs(
        finalists, run, device, eviction, _FINAL_ROUNDS, graph_calls
    )
    min_promotion_speedup = (
        1.10
        if dim == 64 and has_sinks and sliding_window >= 0
        else _MIN_PROMOTION_SPEEDUP
    )
    winner, comparisons = _select_tuned_winner(
        incumbent, finalists, final_samples, min_promotion_speedup
    )
    for comparison in comparisons:
        result = by_key[_config_key(comparison["config"])]
        result["final_us"] = comparison["us"]
        result["final_samples_us"] = final_samples[_config_key(comparison["config"])]
        result["paired_speedup_vs_default"] = comparison["paired_speedup_vs_default"]
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
        "selection": {
            "incumbent": incumbent,
            "finalists": finalists,
            "min_promotion_speedup": min_promotion_speedup,
            "winner_us": winner_result["final_us"],
            "paired_speedup_vs_default": winner["paired_speedup_vs_default"],
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
                _valid_record(record, heads, kv_heads, dim, fp8)
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
        _candidate_configs(default, batch, query_len, heads, dim, seq_len=seq_len)
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


def _valid_record(record, heads, kv_heads, dim, fp8):
    workload = record["workload"]
    if len(workload) != 3 or any(
        type(value) is not int or value < 1 for value in workload
    ):
        return False
    batch, query_len, seq_len = workload
    default = select_segmented_config(
        batch, query_len, seq_len, heads, kv_heads, dim, fp8
    )
    return record["best"] in _candidate_configs(
        default, batch, query_len, heads, dim, seq_len=seq_len
    )


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
    """Tune reachable segmented buckets before KV-cache allocation."""
    if (
        dtype not in (torch.bfloat16, torch.float16)
        or kv_dtype
        not in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        )
        or dim not in (64, 128, 256)
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
    active = _TABLES.get(key, {})
    warmed = (
        {tuple(record["workload"]) for record in active.get("records", [])}
        if active.get("identity") == identity
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
                except (torch.OutOfMemoryError, RuntimeError, AssertionError) as error:
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


def get_segmented_config(
    device,
    dtype,
    kv_dtype,
    heads,
    kv_heads,
    dim,
    page,
    scale,
    batch,
    query_len,
    seq_len,
    sliding_window=-1,
    causal=True,
    has_sinks=False,
):
    """Return a warmed ceiling bucket, or None for the static fallback."""
    if not envs.VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE:
        return None
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
    return dict(winner["best"])


def warmup_rocm_segmented_attention(config, device):
    """Tune the selected backend before KV-cache memory profiling."""
    if not envs.VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE:
        return

    from vllm.v1.attention.backends.rocm_segmented_attn import (
        RocmSegmentedAttentionImpl,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    layers = [
        layer
        for layer in config.compilation_config.static_forward_context.values()
        if isinstance(getattr(layer, "impl", None), RocmSegmentedAttentionImpl)
    ]
    if not layers:
        return

    specs = get_kv_cache_spec(config)
    layouts = tuple(
        (
            (spec.block_size, spec.page_size_bytes)
            if isinstance(spec, FullAttentionSpec)
            else (0, spec.max_memory_usage_bytes(config))
        )
        for spec in specs.values()
    )
    budget = _memory_budget(device)
    for layer in layers:
        impl = layer.impl
        impl._segmented_attention_config = config
        impl._warmup_segmented_attention(
            layer,
            device,
            config.model_config.dtype,
            memory_budget_bytes=budget,
            cache_layouts=layouts,
            cache_budget_bytes=budget,
        )
