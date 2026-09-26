# SPDX-License-Identifier: Apache-2.0
"""Parsing guards against the quiet failure mode of benchmarking: a run that
produced no usable data being reported as a data point anyway."""

import json
from pathlib import Path

import pytest
from bench.collect import (
    MalformedExport,
    _dedupe_by_concurrency,
    collect_run,
    concurrency_from_path,
    load_export,
    parse_export,
)
from conftest import make_point, write_export


class TestParseExport:
    def test_extracts_both_pareto_axes(self, export_factory):
        point = parse_export(export_factory(), concurrency=64, label="baseline")
        assert point.tokens_per_s_per_user == 75.0
        assert point.tokens_per_s_per_gpu == 4800.0
        assert point.concurrency == 64
        assert point.label == "baseline"

    def test_throughput_axis_is_per_gpu(self, export_factory):
        point = parse_export(
            export_factory(total_tps=9600.0), concurrency=64, label="tp2", num_gpus=2
        )
        assert point.tokens_per_s_per_gpu == 4800.0
        assert point.output_token_throughput == 9600.0, "raw total is also kept"

    def test_rejects_zero_gpus(self, export_factory):
        with pytest.raises(ValueError, match="num_gpus must be >= 1"):
            parse_export(export_factory(), concurrency=1, label="x", num_gpus=0)

    def test_latency_metrics_are_carried_through(self, export_factory):
        point = parse_export(export_factory(), concurrency=8, label="b")
        assert point.ttft_ms == 120.0
        assert point.ttft_p99_ms == 240.0
        assert point.itl_ms == 13.3
        assert point.request_latency_ms == 3500.0
        assert point.request_latency_p99_ms == 7000.0
        assert point.output_sequence_length == 256.0

    def test_percentiles_land_in_metrics_map(self, export_factory):
        point = parse_export(export_factory(), concurrency=8, label="b")
        assert point.metrics["output_token_throughput_per_user.p99"] == 60.0
        assert point.metrics["time_to_first_token.p50"] == 120.0
        assert point.metrics["output_token_throughput"] == 4800.0

    def test_missing_throughput_is_a_hard_error(self, export_factory):
        export = export_factory()
        del export["output_token_throughput"]
        with pytest.raises(MalformedExport, match="no 'output_token_throughput'"):
            parse_export(export, concurrency=1, label="b")

    def test_non_streaming_derives_per_user_from_osl_and_latency(self, export_factory):
        export = export_factory(per_user=None, osl=200.0, latency=4000.0)
        point = parse_export(export, concurrency=1, label="b")
        assert point.tokens_per_s_per_user == pytest.approx(50.0)

    def test_unusable_export_without_either_source(self, export_factory):
        export = export_factory(per_user=None, osl=None, latency=None)
        with pytest.raises(MalformedExport, match="--streaming"):
            parse_export(export, concurrency=1, label="b")

    def test_zero_latency_does_not_divide_by_zero(self, export_factory):
        export = export_factory(per_user=None)
        export["request_latency"]["avg"] = 0.0
        with pytest.raises(MalformedExport):
            parse_export(export, concurrency=1, label="b")

    def test_non_numeric_stat_is_treated_as_absent(self, export_factory):
        export = export_factory()
        export["time_to_first_token"]["avg"] = "n/a"
        point = parse_export(export, concurrency=1, label="b")
        assert point.ttft_ms is None

    def test_scalar_metric_shaped_as_bare_value_is_ignored(self, export_factory):
        export = export_factory()
        export["request_throughput"] = 12.5
        point = parse_export(export, concurrency=1, label="b")
        assert point.request_throughput is None


class TestErrorRate:
    def test_clean_run(self, export_factory):
        point = parse_export(export_factory(), concurrency=1, label="b")
        assert point.error_rate == 0.0

    def test_partial_failures_surface(self, export_factory):
        export = export_factory(request_count=200.0, errors=10.0)
        assert parse_export(export, concurrency=1, label="b").error_rate == 0.05

    def test_no_request_count_means_no_rate(self, export_factory):
        export = export_factory(request_count=None, errors=5.0)
        assert parse_export(export, concurrency=1, label="b").error_rate == 0.0

    def test_as_row_drops_metrics_and_adds_rate(self, export_factory):
        row = parse_export(
            export_factory(request_count=100.0, errors=1.0), concurrency=1, label="b"
        ).as_row()
        assert "metrics" not in row
        assert row["error_rate"] == 0.01


class TestConcurrencyFromPath:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("r/baseline/slug/concurrency0032/x/profile_export_aiperf.json", 32),
            ("concurrency1/profile_export_aiperf.json", 1),
            ("a/concurrency0008/b/concurrency-notes/f.json", 8),
        ],
    )
    def test_recovers_concurrency(self, path, expected):
        assert concurrency_from_path(Path(path)) == expected

    @pytest.mark.parametrize(
        "path",
        ["r/baseline/slug/f.json", "concurrencyABC/f.json", "concurrency/f.json"],
    )
    def test_returns_none_when_absent(self, path):
        assert concurrency_from_path(Path(path)) is None

    def test_innermost_marker_wins(self):
        path = Path("concurrency0002/concurrency0016/f.json")
        assert concurrency_from_path(path) == 16


class TestLoadExport:
    def test_reads_object(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text('{"a": 1}')
        assert load_export(path) == {"a": 1}

    def test_rejects_non_object(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text("[1, 2]")
        with pytest.raises(MalformedExport, match="expected a JSON object"):
            load_export(path)

    def test_propagates_invalid_json(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text("{oops")
        with pytest.raises(json.JSONDecodeError):
            load_export(path)


class TestCollectRun:
    def test_collects_and_orders_points(self, run_tree, export_factory):
        run_dir = run_tree(
            "baseline",
            {
                128: export_factory(total_tps=6000.0, per_user=30.0),
                1: export_factory(total_tps=90.0, per_user=90.0),
                16: export_factory(total_tps=1400.0, per_user=85.0),
            },
        )
        points = collect_run(run_dir, label="baseline")
        assert [p.concurrency for p in points] == [1, 16, 128]
        assert points[0].tokens_per_s_per_user == 90.0

    def test_label_defaults_to_dir_name(self, run_tree, export_factory):
        run_dir = run_tree("baseline", {1: export_factory()})
        assert collect_run(run_dir)[0].label == run_dir.name

    def test_num_gpus_applies_to_all_points(self, run_tree, export_factory):
        run_dir = run_tree("b", {1: export_factory(total_tps=200.0)})
        assert collect_run(run_dir, num_gpus=2)[0].tokens_per_s_per_gpu == 100.0

    def test_empty_tree_yields_no_points(self, tmp_path):
        assert collect_run(tmp_path) == []

    def test_exports_outside_a_concurrency_dir_are_skipped(
        self, tmp_path, export_factory
    ):
        write_export(tmp_path / "warmup", export_factory())
        assert collect_run(tmp_path) == []

    def test_rerun_of_a_point_overwrites_the_earlier_one(self):
        first, second = make_point(8, 50.0, 100.0), make_point(8, 50.0, 200.0)
        deduped = _dedupe_by_concurrency([first, second])
        assert len(deduped) == 1
        assert deduped[0].tokens_per_s_per_gpu == 200.0
