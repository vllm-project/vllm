# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal segmented attention configuration and tuning contracts."""

import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup import (
    rocm_segmented_attn_autotune_warmup as tuning,
)
from vllm.v1.attention.ops import segmented_attention as segmented
from vllm.v1.attention.ops import segmented_attention_config as selection


def test_tuning_table_stride_is_fixed_without_extending_the_workload():
    """Reuse compiled launches across sequence buckets while timing actual lengths."""
    for seq_len in (128, 512):
        inputs = tuning._make_inputs(
            "cpu",
            torch.bfloat16,
            torch.bfloat16,
            2,
            1,
            64,
            32,
            64,
            (2, 16, seq_len),
            table_max_len=1024,
        )
        assert inputs[5].shape == (2, 32)
        assert inputs[5].stride(0) == 32
        assert inputs[7].tolist() == [seq_len, seq_len]
        assert inputs[3].shape[0] == 2 * (seq_len // 32)


@pytest.mark.parametrize(
    "mode,image_span,expected",
    [
        ("none", 1089, 1279),
        ("clamped", 1089, 2367),
        ("unclamped", 2048, 4350),
        ("clamped", 0, 32768),
    ],
)
def test_mm_span_includes_reopened_image_keys(mode, image_span, expected):
    """Long text prefixes must not inflate a windowed image workload's picker."""
    assert (
        selection.segmented_attention_span(32768, 256, 1023, mode, image_span)
        == expected
    )
    assert selection.segmented_attention_span(32768, 256, -1, mode, image_span) == 32768


def test_mm_tuning_keys_isolate_masks_image_spans_and_range_counts():
    """Neither text nor incompatible image geometry may supply an MM winner."""
    args = ("cuda:0", torch.bfloat16, torch.float8_e4m3fn, 16, 8, 256, 32, 1.0)
    text = tuning._key(*args)
    image = tuning._key(
        *args, mm_prefix_mode="clamped", mm_prefix_max_span=1089, mm_prefix_max_ranges=1
    )
    assert image == tuning._key(
        *args, mm_prefix_mode="clamped", mm_prefix_max_span=1120, mm_prefix_max_ranges=1
    )
    assert (
        len(
            {
                text,
                image,
                tuning._key(
                    *args,
                    mm_prefix_mode="unclamped",
                    mm_prefix_max_span=1089,
                    mm_prefix_max_ranges=1,
                ),
                tuning._key(
                    *args,
                    mm_prefix_mode="clamped",
                    mm_prefix_max_span=560,
                    mm_prefix_max_ranges=1,
                ),
                tuning._key(
                    *args,
                    mm_prefix_mode="clamped",
                    mm_prefix_max_span=1089,
                    mm_prefix_max_ranges=2,
                ),
            }
        )
        == 5
    )


def test_mm_startup_uses_image_sized_query_ceilings_and_skips_clamped_q1():
    """A 1089-token image should not borrow a mostly-text 2048-query profile."""
    workloads = list(
        tuning._workloads(
            4096,
            4096,
            4,
            mm_prefix_mode="clamped",
            mm_prefix_max_span=1152,
            mm_prefix_max_ranges=2,
        )
    )
    queries = {q for _, q, _ in workloads}
    assert 1184 in queries and 2344 in queries
    assert all(q > 1 for q in queries)
    assert not list(
        tuning._workloads(1, 128, 1, max_query_len=1, mm_prefix_mode="clamped")
    )


@pytest.mark.parametrize(
    "dim,logical_page,physical_page", [(512, 16, 64), (256, 64, 16)]
)
def test_layer_warmup_uses_allocated_hybrid_cache_pages(
    monkeypatch, dim, logical_page, physical_page
):
    """A recomputed logical spec must not make warmed profiles miss at runtime."""
    clamps = dim == 256
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4),
    )
    impl = SimpleNamespace(
        _segmented_attention_config=config,
        sliding_window=(1023, 0) if clamps else (-1, -1),
        num_heads=16,
        num_kv_heads=8 if clamps else 1,
        head_size=dim,
        scale=1.0,
        sinks=None,
    )
    impl._split_kv_cache = lambda cache: (cache[:, 0], cache[:, 1])
    layer = SimpleNamespace(
        impl=impl,
        kv_cache=torch.empty((1, 2, physical_page, dim)),
        get_kv_cache_spec=lambda _: SimpleNamespace(
            block_size=logical_page, dtype=torch.bfloat16
        ),
        mm_prefix_clamp_sliding_window=clamps,
    )
    calls = []
    monkeypatch.setattr(
        tuning, "warmup_segmented_attention", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr(tuning, "_mm_warmup_profiles", lambda *_: [(1152, 1)])
    tuning._warmup_segmented_attention(layer, "cpu", model_clamps_mm_prefix=True)
    assert len(calls) == (2 if clamps else 1)
    assert all(args[5] == physical_page for args in calls)
    assert impl._segmented_attention_warmed_up


def test_mm_specific_winner_survives_persistent_cache_reload(tmp_path):
    """A valid image-only geometry must not be rejected by text-only validation."""
    options = dict(
        mm_prefix_mode="clamped", mm_prefix_max_span=1152, mm_prefix_max_ranges=1
    )
    default = selection.select_segmented_config(
        1, 1117, 1117, 16, 8, 256, True, sliding_window=1023, **options
    )
    configs = tuning._candidate_configs(
        default,
        1,
        1117,
        16,
        256,
        1117,
        native_fp8=True,
        sliding_window=1023,
        mm_prefix_mode="clamped",
    )
    winner = next(
        c
        for c in configs
        if c["bm"] == 64
        and c["bn"] == 64
        and c["warps"] == 8
        and c["stages"] == 2
        and c["kv_upcast"] == "native4"
    )
    record = dict(workload=[1, 1117, 1117], best=winner)
    assert not tuning._valid_record(
        record, 16, 8, 256, True, native_fp8=True, sliding_window=1023
    )
    assert tuning._valid_record(
        record, 16, 8, 256, True, native_fp8=True, sliding_window=1023, **options
    )
    identity = dict(
        kv_upcast_modes=list(selection.KV_UPCAST_MODES), sliding_window=1023, **options
    )
    path = tmp_path / "mm.json"
    path.write_text(json.dumps(dict(identity=identity, records=[record])))
    loaded, _ = tuning._load_records(path, identity, 16, 8, 256, True)
    assert loaded["records"] == [record]


@pytest.mark.parametrize("mode", ["clamped", "unclamped"])
@pytest.mark.parametrize("fp8,query,seq", [(False, 70, 80), (True, 33, 48)])
def test_mm_ceiling_does_not_override_a_different_static_tiling_regime(
    monkeypatch, mode, fp8, query, seq
):
    """A larger query ceiling must not replace a validated short-image tile."""
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    args = ("cuda:0", torch.bfloat16, dtype, 16, 8, 256, 32, 1.0)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode=mode,
        mm_prefix_max_span=64,
        mm_prefix_max_ranges=1,
    )
    best = tuning._normalized_config(
        selection.select_segmented_config(
            1, 128, 128, 16, 8, 256, fp8, native_fp8=fp8, **options
        ),
        1,
        128,
        16,
        256,
    )
    key = tuning._key(*args, **options)
    monkeypatch.setattr(
        tuning,
        "_TABLES",
        {key: {"records": [{"workload": [1, 128, 128], "best": best}]}},
    )
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *_: fp8)
    assert tuning.get_segmented_config(*args, 1, query, seq, **options) is None
    assert tuning.get_segmented_config(*args, 1, 128, 128, **options) == best
    assert tuning.get_segmented_config(*args, 1, 119, 128, **options) == best


@pytest.mark.parametrize(
    "batch,query,seq,tokens,warm_queries,expected_hit",
    [
        (1, 257, 300, 257, [512], False),
        (4, 128, 2048, 131, [128, 128, 128, 128], False),
        (4, 128, 2048, 131, [128, 1, 1, 1], True),
        (1, 512, 512, 512, [512], True),
    ],
)
def test_mm_wider_tuned_tile_preserves_useful_parallelism(
    monkeypatch, batch, query, seq, tokens, warm_queries, expected_hit
):
    """Dense ceilings must not underfill a smaller or newly ragged request."""
    warm_query = max(warm_queries)
    workload = [len(warm_queries), warm_query, 512 if batch == 1 else 2048]
    best = tuning._normalized_config(
        selection.SegmentedAttentionConfig(
            bm=128, bn=32, bk=256, splits=1, warps=8, stages=1, kv_upcast="normal"
        ),
        workload[0],
        workload[1],
        16,
        256,
    )
    args = ("cuda:0", torch.bfloat16, torch.bfloat16, 16, 8, 256, 16, 1.0)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=280,
        mm_prefix_max_ranges=1,
    )
    key = tuning._key(*args, **options)
    monkeypatch.setattr(
        tuning,
        "_TABLES",
        {
            key: {
                "records": [
                    {
                        "workload": workload,
                        "query_lengths": warm_queries,
                        "best": best,
                    }
                ]
            }
        },
    )
    config = tuning.get_segmented_config(
        *args, batch, query, seq, num_query_tokens=tokens, **options
    )
    assert (config is not None) == expected_hit


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("dim", [64, 128, 256, 512])
def test_mm_search_keeps_incumbent_and_all_eligible_conversions(native, dim):
    """Image tile pruning retains a measured fallback and conversion choices."""
    default = selection.select_segmented_config(
        1,
        1024,
        2048,
        16,
        8,
        dim,
        native,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=1089,
    )
    configs = tuning._candidate_configs(
        default, 1, 1024, 16, dim, 2048, native_fp8=native, mm_prefix_mode="clamped"
    )
    incumbent = tuning._normalized_config(default, 1, 1024, 16, dim)
    assert incumbent.items() <= configs[0].items()
    assert len(configs) == len({tuning._config_key(c) for c in configs})
    geometries = {tuning._geometry_key(c) for c in configs}
    expected = set(selection.KV_UPCAST_MODES) if native else {"normal"}
    assert all(
        {c["kv_upcast"] for c in configs if tuning._geometry_key(c) == geometry}
        == expected
        for geometry in geometries
    )


@pytest.mark.parametrize("dim", [64, 128, 256, 512])
@pytest.mark.parametrize("gqa", [1, 2, 8, 16])
@pytest.mark.parametrize("kv_heads", [1, 3])
@pytest.mark.parametrize("fp8", [False, True])
def test_mm_workspace_reserves_reachable_static_and_tuned_splits(
    dim, gqa, kv_heads, fp8
):
    """An image-specific tile must not outgrow scratch reserved before capture."""
    heads = gqa * kv_heads
    reserved, _ = segmented.segmented_attention_workspace_size(
        4,
        heads,
        kv_heads,
        dim,
        32768,
        max_tokens=2048,
        fp8=fp8,
        autotune=True,
        mm_prefix=True,
        sliding_window=1023,
    )
    for batch, query, image_span, mode in itertools.product(
        (1, 3), (17, 129, 257, 1025), (70, 1089), ("clamped", "unclamped")
    ):
        config = selection.select_segmented_config(
            batch,
            query,
            32768,
            heads,
            kv_heads,
            dim,
            fp8,
            sliding_window=1023,
            mm_prefix_mode=mode,
            mm_prefix_max_span=image_span,
            mm_prefix_max_ranges=1,
        )
        candidates = tuning._candidate_configs(
            config,
            batch,
            query,
            heads,
            dim,
            32768,
            native_fp8=fp8,
            mm_prefix_mode=mode,
            legacy_default=segmented._select_segmented_launch_config(
                batch, query, 32768, heads, kv_heads, dim, fp8
            ),
        )
        splits = max(c["splits"] for c in candidates)
        if splits > 1:
            needed = (
                batch * selection.segmented_query_capacity(query) * heads * dim * splits
            )
            assert needed <= reserved, (
                batch,
                query,
                image_span,
                mode,
                config,
                needed,
                reserved,
            )


@pytest.mark.parametrize("query_len,splits", [(17, 1), (65, 1), (17, 2), (65, 2)])
def test_single_pass_compile_reuses_capacity_without_shrinking_the_grid(
    monkeypatch, query_len, splits
):
    """Unused scratch indexing must not compile a new single-pass stage per Q."""
    position = segmented._segmented_attention_stage.arg_names.index("QCAP")
    calls = []

    class Kernel:
        def warmup(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(segmented, "_segmented_attention_stage", Kernel())
    monkeypatch.setattr(segmented, "_segmented_attention_reduce", Kernel())
    q = torch.empty((query_len, 2, 64), dtype=torch.bfloat16)
    cache = torch.empty((8, 32, 1, 64), dtype=torch.bfloat16)
    table = torch.zeros((1, 8), dtype=torch.int32)
    starts = torch.tensor([0, query_len], dtype=torch.int32)
    lens = torch.tensor([256], dtype=torch.int32)
    scale = torch.tensor(1.0)
    qcap = selection.segmented_query_capacity(query_len)
    shapes = segmented.segmented_workspace_shapes(1, qcap, 2, 1, 64, splits)
    workspace = None if shapes is None else tuple(torch.empty(s) for s in shapes)
    cfg = dict(bm=16, bn=32, bk=64, splits=splits, warps=4, stages=1)
    segmented.compile_segmented_attention(
        q,
        torch.empty_like(q),
        cache,
        cache,
        table,
        starts,
        lens,
        query_len,
        scale,
        scale,
        1.0,
        cfg,
        workspace,
    )
    args, options = calls[0]
    assert args[position] == (qcap if splits > 1 else 1)
    assert options["grid"] == (1, (qcap * 2 + 15) // 16, splits)


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mm_shuffled_tables_cover_disjoint_cache_pages_and_preserve_padding(kv_dtype):
    inputs = tuning._make_inputs(
        "cpu",
        torch.bfloat16,
        kv_dtype,
        2,
        1,
        64,
        32,
        64,
        (3, 16, 128),
        table_max_len=1024,
        shuffle_pages=True,
    )
    table = inputs[5]
    pages = table[:, :4]
    assert table.shape == (3, 32) and table.dtype == torch.int32
    assert sorted(pages.flatten().tolist()) == list(range(12))
    assert torch.equal(table, pages.repeat(1, 8))
    assert inputs[7].tolist() == [128, 128, 128]


@pytest.mark.parametrize("matched_density", [False, True])
def test_mm_lookup_rejects_only_unmeasured_sparse_query_density(
    monkeypatch, matched_density
):
    query_lengths = [256, 5, 1, 1, 1, 1, 1, 0]
    tokens = sum(query_lengths)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=1089,
        mm_prefix_max_ranges=1,
    )
    args = ("cuda:0", torch.bfloat16, torch.float8_e4m3fn, 16, 8, 256, 16, 1.0)
    best = tuning._normalized_config(
        selection.select_segmented_config(8, 256, 8192, 16, 8, 256, True, **options),
        8,
        256,
        16,
        256,
    )
    best.update({"bn": 32, "warps": 4})
    record = dict(
        workload=[8, 256, 8192],
        best=best,
        query_lengths=query_lengths if matched_density else [256] * 8,
    )
    monkeypatch.setattr(
        tuning, "_TABLES", {tuning._key(*args, **options): {"records": [record]}}
    )
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *_: True)
    result = tuning.get_segmented_config(
        *args, 8, 256, 2367, num_query_tokens=tokens, **options
    )
    assert (result is not None) == matched_density


def test_mm_grid_retains_short_profiles_midpoints_and_two_request_batches():
    workloads = list(
        tuning._workloads(
            4096,
            4096,
            4,
            mm_prefix_mode="clamped",
            mm_prefix_max_span=1152,
            mm_prefix_max_ranges=2,
        )
    )
    assert {4, 16, 32, 48, 64, 96, 256, 384, 1536}.issubset(
        {q for _, q, _ in workloads}
    )
    assert any(b == 2 and q == 384 for b, q, _ in workloads)

    assert (1, 48, 64) in workloads


@pytest.mark.parametrize("fp8,partitions", [(False, {1, 2, 4}), (True, {1, 4, 8})])
def test_mm_search_preserves_incumbent_partition_alternatives(fp8, partitions):
    default = selection.select_segmented_config(
        1,
        64,
        8192,
        16,
        8,
        256,
        fp8,
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=1152,
        mm_prefix_max_ranges=1,
    )
    configs = tuning._candidate_configs(
        default,
        1,
        64,
        16,
        256,
        8192,
        native_fp8=fp8,
        mm_prefix_mode="clamped",
        sliding_window=1023,
    )
    assert partitions.issubset(
        {
            c["splits"]
            for c in configs
            if c["bm"] == default["bm"]
            and c["bn"] == default["bn"]
            and c["bk"] == default["bk"]
            and c["warps"] == default["warps"]
            and c["stages"] == default["stages"]
        }
    )


@pytest.mark.parametrize(
    "seq_len,expected_splits", [(32, 1), (64, 1), (65, 2), (128, 2)]
)
def test_mm_runtime_discards_partitions_without_cache_tiles(seq_len, expected_splits):
    best = tuning._normalized_config(
        selection.select_segmented_config(1, 32, 8192, 16, 8, 256, True),
        1,
        32,
        16,
        256,
    )
    best.update({"bn": 64, "splits": 8})
    record: tuning._ConfigRecord = {"workload": [1, 32, 2048], "best": best}
    result = tuning._runtime_config(
        record,
        1,
        32,
        seq_len,
        16,
        8,
        256,
        True,
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=64,
        mm_prefix_max_ranges=1,
        native_fp8=True,
    )
    assert result["splits"] == expected_splits
    assert record["best"]["splits"] == 8


@pytest.mark.parametrize(
    "query,seq_len,bm,bn,expected_hit",
    [(128, 128, 64, 64, False), (128, 128, 64, 128, True), (64, 64, 32, 64, True)],
)
def test_exact_short_mm_lookup_keeps_kv_reuse_and_measured_wider_tiles(
    monkeypatch, query, seq_len, bm, bn, expected_hit
):
    args = ("cuda:0", torch.bfloat16, torch.bfloat16, 16, 8, 256, 16, 1.0)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=query - 12,
        mm_prefix_max_ranges=1,
    )
    best = tuning._normalized_config(
        selection.select_segmented_config(
            1, query, seq_len, 16, 8, 256, False, **options
        ),
        1,
        query,
        16,
        256,
    )
    best.update({"bm": bm, "bn": bn, "warps": 8, "stages": 1, "splits": 1})
    record: tuning._ConfigRecord = {
        "workload": [1, query, seq_len],
        "best": best,
        "query_lengths": [query],
    }
    monkeypatch.setattr(
        tuning, "_TABLES", {tuning._key(*args, **options): {"records": [record]}}
    )
    result = tuning.get_segmented_config(
        *args, 1, query, seq_len, num_query_tokens=query, **options
    )
    assert (result is not None) == expected_hit
    assert record["best"]["bm"] == bm
