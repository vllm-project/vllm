# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for segmented conversion selection and startup tuning."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup import (
    rocm_segmented_attn_autotune_warmup as tuning,
)
from vllm.v1.attention.ops import segmented_attention as segmented


@pytest.mark.parametrize("dim", (64, 128, 256, 512))
@pytest.mark.parametrize("gqa", range(1, 17))
@pytest.mark.parametrize("query_len,seq_len", [(1, 32768), (17, 1024), (257, 32768)])
def test_every_eligible_launch_searches_all_conversion_modes(
    dim, gqa, query_len, seq_len
):
    """Changing geometry must not silently drop normal or either native mode."""
    heads, kv_heads = 2 * gqa, 2
    default = segmented.select_segmented_config(
        1, query_len, seq_len, heads, kv_heads, dim, True
    )
    legacy = segmented._select_segmented_launch_config(
        1, query_len, seq_len, heads, kv_heads, dim, True
    )
    configs = tuning._candidate_configs(
        default,
        1,
        query_len,
        heads,
        dim,
        seq_len,
        native_fp8=True,
        legacy_default=legacy,
    )
    assert configs[0] == tuning._normalized_config(default, 1, query_len, heads, dim)
    assert len(configs) == len({tuning._config_key(c) for c in configs})
    launches: dict[tuple, set[str]] = {}
    for config in configs:
        geometry = tuple(sorted((k, v) for k, v in config.items() if k != "kv_upcast"))
        launches.setdefault(geometry, set()).add(config["kv_upcast"])
    assert all(modes == set(segmented.KV_UPCAST_MODES) for modes in launches.values())
    initial = tuning._initial_search_configs(configs)
    assert initial[0] == configs[0]
    assert {tuning._geometry_key(c) for c in initial} == set(launches)
    assert all(c == configs[0] or c["kv_upcast"] == "native4" for c in initial)
    assert len(initial) == len({tuning._config_key(c) for c in initial})
    incumbent_launches = tuning._launch_candidate_configs(
        legacy, 1, query_len, heads, dim, seq_len
    )
    assert all(
        tuple(sorted((k, v) for k, v in c.items() if k != "kv_upcast")) in launches
        for c in incumbent_launches
    )
    limit = min(64, 2 * default["splits"]) if dim == 64 else default["splits"]
    assert all(c["splits"] <= limit for c in configs)


@pytest.mark.parametrize("dim", (64, 128, 256, 512))
def test_ineligible_search_deduplicates_inactive_native_modes(dim):
    """BF16 caches, FP16 queries, and unsupported hardware need one conversion."""
    default = segmented.select_segmented_config(1, 1, 8192, 16, 2, dim, True)
    configs = tuning._candidate_configs(default, 1, 1, 16, dim, native_fp8=False)
    assert all(c["kv_upcast"] == "normal" for c in configs)
    assert len(configs) == len({tuning._config_key(c) for c in configs})
    assert tuning._initial_search_configs(configs) == configs
    assert tuning._upcast_expansion_configs(configs, []) == []


def _shortlist_configs():
    return [
        dict(
            bm=bm,
            bn=bn,
            bk=64,
            splits=1,
            warps=warps,
            stages=1,
            waves_per_eu=2,
            reduce_d=64,
            reduce_warps=4,
            kv_upcast=mode,
        )
        for bm in (16, 32, 64)
        for bn in (32, 64)
        for warps in (4, 8)
        for mode in segmented.KV_UPCAST_MODES
    ]


@pytest.mark.parametrize("failed_anchor", (False, True))
def test_conversion_shortlist_keeps_default_and_recovers_failed_anchors(failed_anchor):
    """An unmeasurable native4 tile must not hide its other conversions."""
    configs = _shortlist_configs()
    anchors = [c for c in configs if c["kv_upcast"] == "native4"]
    comparisons = [
        dict(config=c, paired_balanced_speedup_vs_default=1 + index / 100)
        for index, c in enumerate(anchors)
        if not (failed_anchor and index == 1)
    ]
    expanded = tuning._upcast_expansion_configs(configs, comparisons)
    expected = {tuning._geometry_key(c) for c in anchors[-8:]}
    expected.add(tuning._geometry_key(configs[0]))
    if failed_anchor:
        expected.add(tuning._geometry_key(anchors[1]))
    assert {tuning._geometry_key(c) for c in expanded} == expected
    initial_keys = {
        tuning._config_key(c) for c in tuning._initial_search_configs(configs)
    }
    assert not initial_keys.intersection(tuning._config_key(c) for c in expanded)
    combined = tuning._initial_search_configs(configs) + expanded
    for geometry in expected:
        assert {
            c["kv_upcast"] for c in combined if tuning._geometry_key(c) == geometry
        } == set(segmented.KV_UPCAST_MODES)


def test_staged_tuning_uses_contemporaneous_baselines_and_compiles_only_shortlist(
    monkeypatch,
):
    """Clock drift between stages must not suppress a better native1 candidate."""
    configs = _shortlist_configs()
    geometries = list(dict.fromkeys(tuning._geometry_key(c) for c in configs))
    compiled = []
    measure_calls = 0

    def measure(candidates, run, device, eviction, rounds, graph_calls):
        nonlocal measure_calls
        measure_calls += 1
        baseline = 200 if measure_calls in (3, 4) else 100
        samples = {}
        for config in candidates:
            index = geometries.index(tuning._geometry_key(config))
            mode = config["kv_upcast"]
            factor = 1.0
            if mode == "native4":
                factor = 0.95 - index * 0.005
            elif mode == "native1":
                factor = 0.8 if index == 8 else 1.08
            samples[tuning._config_key(config)] = [baseline * factor] * rounds
        return samples

    monkeypatch.setattr(tuning, "select_segmented_config", lambda *a, **kw: configs[0])
    monkeypatch.setattr(tuning, "_candidate_configs", lambda *a, **kw: configs)
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *a: True)
    monkeypatch.setattr(
        tuning,
        "_precompile_configs",
        lambda candidates, *a: compiled.append(candidates),
    )
    monkeypatch.setattr(
        tuning, "run_segmented_attention", lambda q, out, *a, **kw: out.copy_(q)
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a: None)
    monkeypatch.setattr(tuning, "_condition_configs", lambda *a: None)
    monkeypatch.setattr(tuning, "_measure_configs", measure)
    record = tuning._tune_workload(
        "cpu",
        torch.bfloat16,
        torch.float8_e4m3fn,
        1,
        1,
        64,
        16,
        0.125,
        16,
        (1, 1, 128),
    )
    assert record["best"]["kv_upcast"] == "native1"
    assert tuning._geometry_key(record["best"]) == geometries[8]
    assert len(compiled) == 2
    assert sum(map(len, compiled)) == len(record["results"]) < len(configs)
    assert len({tuning._config_key(c) for batch in compiled for c in batch}) == sum(
        map(len, compiled)
    )
    assert record["search"]["screen_baselines"][0]["cold_us"] == [100] * 5
    assert record["search"]["screen_baselines"][1]["cold_us"] == [200] * 5


@pytest.mark.parametrize("gfx12", (False, True))
@pytest.mark.parametrize(
    "dtype,kv_dtype,eligible",
    [
        (torch.bfloat16, torch.float8_e4m3fn, True),
        (torch.bfloat16, torch.float8_e4m3fnuz, False),
        (torch.float16, torch.float8_e4m3fn, False),
        (torch.bfloat16, torch.bfloat16, False),
    ],
)
def test_native_gate_checks_query_format_cache_format_and_architecture(
    monkeypatch, gfx12, dtype, kv_dtype, eligible
):
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(on_gfx12x=lambda: gfx12),
    )
    assert segmented.supports_segmented_native_fp8(dtype, kv_dtype) == (
        gfx12 and eligible
    )


def test_normal_mode_overrides_legacy_native_flags():
    """An explicit normal choice must survive configs built from old templates."""
    assert (
        segmented.segmented_kv_upcast_mode(
            {"kv_upcast": "normal", "native_fp8_upcast": True, "native_fp8_pack": 4}
        )
        == "normal"
    )
    assert (
        segmented.segmented_kv_upcast_mode(
            {"native_fp8_upcast": True, "native_fp8_pack": 4}
        )
        == "native4"
    )


@pytest.mark.parametrize(
    "warm,cold,promote", [(80, 100, True), (60, 104, False), (99, 100, False)]
)
def test_balanced_promotion_realizes_warm_gains_without_cold_regressions(
    warm, cold, promote
):
    incumbent = {"kv_upcast": "normal"}
    challenger = {"kv_upcast": "native4"}
    keys = list(map(tuning._config_key, (incumbent, challenger)))
    winner, _ = tuning._select_tuned_winner(
        incumbent,
        [incumbent, challenger],
        {keys[0]: [100.0] * 7, keys[1]: [float(cold)] * 7},
        warm_samples={keys[0]: [100.0] * 7, keys[1]: [float(warm)] * 7},
    )
    assert winner["config"] == (challenger if promote else incumbent)


def test_cache_record_validation_respects_native_eligibility():
    default = segmented.select_segmented_config(1, 1, 128, 8, 2, 128, True)
    config = dict(tuning._normalized_config(default, 1, 1, 8, 128), kv_upcast="native4")
    record = {"workload": [1, 1, 128], "best": config}
    assert tuning._valid_record(record, 8, 2, 128, True, native_fp8=True)
    assert not tuning._valid_record(record, 8, 2, 128, True, native_fp8=False)


def test_cache_identity_invalidates_records_after_selector_changes(
    monkeypatch, tmp_path
):
    """The extracted selector must remain part of persistent cache identity."""
    selector = tmp_path / "selector.py"
    selector.write_text("initial selection")
    monkeypatch.setattr(tuning.segmented_attention_config, "__file__", str(selector))
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *a: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="test GPU", gcnArchName="gfx1201", multi_processor_count=64
        ),
    )
    args = (
        "cuda:0",
        torch.bfloat16,
        torch.float8_e4m3fn,
        8,
        2,
        128,
        32,
        0.125,
        8192,
        131072,
        32,
        -1,
        True,
        8192,
    )
    first = tuning._identity(*args)
    selector.write_text("updated selection")
    second = tuning._identity(*args)
    assert first["selector_sha256"] != second["selector_sha256"]
    assert {k: v for k, v in first.items() if k != "selector_sha256"} == {
        k: v for k, v in second.items() if k != "selector_sha256"
    }


@pytest.mark.parametrize("dim", (64, 128, 256, 512))
@pytest.mark.parametrize("mode", segmented.KV_UPCAST_MODES)
@pytest.mark.parametrize("supported", (False, True))
@pytest.mark.parametrize("compile_only", (False, True))
def test_explicit_conversion_reaches_runtime_and_compile_specializations(
    monkeypatch, dim, mode, supported, compile_only
):
    """Both launch paths honor the selected mode and the hardware gate."""
    calls = []

    class Stage:
        def __getitem__(self, grid):
            return self.warmup

        def warmup(self, *args, **kwargs):
            calls.append(args)

    monkeypatch.setattr(segmented, "_segmented_attention_stage", Stage())
    monkeypatch.setattr(
        segmented, "supports_segmented_native_fp8", lambda *_: supported
    )
    q = torch.empty((1, 4, dim), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    cache = torch.empty((1, 16, 2, dim), dtype=torch.float8_e4m3fn)
    table = torch.zeros((1, 1), dtype=torch.int32)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    scale = torch.tensor(1.0)
    config = dict(bm=16, bn=32, bk=64, splits=1, warps=4, stages=1, kv_upcast=mode)
    args = (q, output, cache, cache, table, starts, lengths, 1)
    if compile_only:
        segmented.compile_segmented_attention(*args, scale, scale, 1.0, config, None)
    else:
        assert (
            segmented.run_segmented_attention(
                *args, 1, scale, scale, 1.0, config=config
            )
            is output
        )
    assert len(calls) == 1
    native = supported and mode != "normal"
    assert calls[0][-2:] == (native, 4 if native and mode == "native4" else 1)


@pytest.mark.parametrize("graph_calls", (0, 1, 32))
def test_graph_timing_excludes_eviction_and_normalizes_call_count(
    monkeypatch, graph_calls
):
    """The startup objective must measure kernels independently of eviction cost."""
    clock = SimpleNamespace(us=0.0, capture=None, evictions=0)

    def emit(operation):
        if clock.capture is None:
            operation()
        else:
            clock.capture.append(operation)

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True, "external": True}
            self.time = 0.0

        def record(self):
            emit(lambda: setattr(self, "time", clock.us))

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return (end.time - self.time) / 1000

    class Graph:
        def __init__(self):
            self.operations: list = []

        def replay(self):
            for operation in self.operations:
                operation()

    @contextmanager
    def capture(graph):
        clock.capture = graph.operations
        yield
        clock.capture = None

    def evict():
        def operation():
            clock.us += 1000
            clock.evictions += 1

        emit(operation)

    def run(config):
        emit(lambda: setattr(clock, "us", clock.us + config["cost_us"]))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda _: None)
    configs = [{"cost_us": 10.0}, {"cost_us": 20.0}]
    samples = tuning._measure_configs(
        configs, run, "cuda:0", SimpleNamespace(zero_=evict), 5, graph_calls
    )
    assert samples == {tuning._config_key(c): [c["cost_us"]] * 5 for c in configs}
    assert clock.evictions == (12 if graph_calls <= 1 else 0)


@pytest.mark.parametrize("dim", (128, 256, 512))
def test_ceiling_lookup_preserves_parallelism_within_reserved_workspace(dim):
    """Q259 needs more splits than the Q512 ceiling that supplied its tile."""
    best = tuning._normalized_config(
        segmented.select_segmented_config(1, 512, 32768, 1, 1, dim, True),
        1,
        512,
        1,
        dim,
    )
    best["splits"] = 16
    record = {"workload": [1, 512, 32768], "best": best}
    adapted = tuning._runtime_config(record, 1, 259, 16411, 1, 1, dim, True)
    assert best["splits"] == 16 and adapted["splits"] == 32
    assert record["best"] == best
    assert tuning._runtime_config(record, 1, 512, 32768, 1, 1, dim, True) == best
    sizes = segmented.segmented_attention_workspace_size(
        1, 1, 1, dim, 32768, max_tokens=512, fp8=True, autotune=True
    )
    shapes = segmented.segmented_workspace_shapes(1, 512, 1, 1, dim, adapted["splits"])
    assert all(torch.Size(s).numel() <= limit for s, limit in zip(shapes, sizes))


def test_lookup_keeps_verified_long_decode_splits_across_sequence_boundary():
    """A one-token sequence boundary must not discard the tuned 512-way decode."""
    best = dict(bm=16, splits=512, kv_upcast="native4")
    record = {"workload": [1, 1, 131072], "best": best}
    assert tuning._runtime_config(record, 1, 1, 131071, 1, 1, 64, True) == best


@pytest.mark.parametrize("dim", (128, 256, 512))
@pytest.mark.parametrize("gqa", (1, 3, 8, 16))
def test_narrow_window_lookup_keeps_a_single_pass_winner(monkeypatch, dim, gqa):
    """Short windows must not pay for a reduction absent from startup tuning."""
    best = tuning._normalized_config(
        segmented.select_segmented_config(1, 512, 512, gqa, 1, dim, True),
        1,
        512,
        gqa,
        dim,
    )
    best["splits"] = 1
    record = {"workload": [1, 512, 512], "best": best}
    key = tuning._key(
        "cuda:0", torch.bfloat16, torch.float8_e4m3fn, gqa, 1, dim, 32, 1.0, 127
    )
    monkeypatch.setattr(tuning, "_TABLES", {key: {"records": [record]}})
    actual = tuning.get_segmented_config(
        "cuda:0",
        torch.bfloat16,
        torch.float8_e4m3fn,
        gqa,
        1,
        dim,
        32,
        1.0,
        1,
        259,
        386,
        sliding_window=127,
    )
    assert actual == best
    assert record["best"] == best


def test_wider_window_lookup_can_add_useful_partition_parallelism():
    """A ragged batch below its warm ceiling still benefits from more CTAs."""
    best = tuning._normalized_config(
        segmented.select_segmented_config(4, 32, 2048, 14, 1, 128, True),
        4,
        32,
        14,
        128,
    )
    best["splits"] = 1
    record = {"workload": [4, 32, 2048], "best": best}
    actual = tuning._runtime_config(
        record, 3, 19, 1042, 14, 1, 128, True, sliding_window=1023
    )
    assert actual["splits"] > 1 and record["best"]["splits"] == 1
