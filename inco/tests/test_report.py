# SPDX-License-Identifier: Apache-2.0
"""The report turns a sweep into the numbers that get quoted in the write-up,
so the headline scalars (throughput at an interactivity SLO, the frontier) are
asserted against hand-computed values."""

import csv

import pytest
from bench.report import (
    CSV_COLUMNS,
    integrity_warnings,
    is_steep,
    interactivity_at_load,
    markdown_table,
    pareto_frontier,
    plot_pareto,
    summarize,
    throughput_at_interactivity,
    write_csv,
)
from conftest import make_point as point


@pytest.fixture
def curve():
    """A plausible 8B-on-one-H100 shape: interactivity falls as load rises."""
    return [
        point(1, 95.0, 95.0, ttft_ms=35.0, itl_ms=10.5, request_count=24.0),
        point(8, 88.0, 700.0, ttft_ms=60.0, itl_ms=11.4, request_count=64.0),
        point(32, 70.0, 2240.0, ttft_ms=180.0, itl_ms=14.3, request_count=256.0),
        point(128, 35.0, 4480.0, ttft_ms=900.0, itl_ms=28.6, request_count=1024.0),
        point(256, 18.0, 4600.0, ttft_ms=2100.0, itl_ms=55.6, request_count=1024.0),
    ]


class TestThroughputAtInteractivity:
    def test_picks_highest_throughput_meeting_the_slo(self, curve):
        best = throughput_at_interactivity(curve, 30.0)
        assert best.concurrency == 128
        assert best.tokens_per_s_per_gpu == 4480.0

    def test_tighter_slo_forces_lower_load(self, curve):
        assert throughput_at_interactivity(curve, 50.0).concurrency == 32

    def test_slo_boundary_is_inclusive(self, curve):
        assert throughput_at_interactivity(curve, 35.0).concurrency == 128

    def test_unreachable_slo_returns_none(self, curve):
        assert throughput_at_interactivity(curve, 200.0) is None

    def test_empty_sweep_returns_none(self):
        assert throughput_at_interactivity([], 10.0) is None


class TestParetoFrontier:
    def test_monotone_tradeoff_keeps_every_point(self, curve):
        assert [p.concurrency for p in pareto_frontier(curve)] == [256, 128, 32, 8, 1]

    def test_dominated_point_is_dropped(self):
        points = [point(1, 50.0, 500.0), point(2, 40.0, 400.0), point(4, 60.0, 600.0)]
        assert [p.concurrency for p in pareto_frontier(points)] == [4]

    def test_duplicate_points_are_both_kept(self):
        """Equal points do not dominate each other, so neither is discarded."""
        points = [point(1, 50.0, 500.0), point(2, 50.0, 500.0)]
        assert len(pareto_frontier(points)) == 2

    def test_frontier_is_sorted_by_interactivity(self, curve):
        xs = [p.tokens_per_s_per_user for p in pareto_frontier(curve)]
        assert xs == sorted(xs)

    def test_empty(self):
        assert pareto_frontier([]) == []


class TestInteractivityAtLoad:
    def test_finds_the_point(self, curve):
        assert interactivity_at_load(curve, 32).tokens_per_s_per_user == 70.0

    def test_missing_concurrency(self, curve):
        assert interactivity_at_load(curve, 17) is None


class TestMarkdownTable:
    def test_renders_one_row_per_point(self, curve):
        lines = markdown_table(curve).splitlines()
        assert len(lines) == len(curve) + 2  # header + divider
        assert lines[0].startswith("| Conc |")

    def test_missing_metrics_render_as_dash(self):
        assert "| - |" in markdown_table([point(1, 50.0, 500.0)])

    def test_error_rate_is_a_percentage(self):
        row = markdown_table(
            [point(1, 50.0, 500.0, request_count=100.0, error_request_count=3.0)]
        )
        assert "3.00%" in row

    def test_empty_sweep(self):
        assert markdown_table([]) == "_no results_"


class TestWriteCsv:
    def test_columns_and_values(self, curve, tmp_path):
        path = write_csv(curve, tmp_path / "out" / "pareto.csv")
        rows = list(csv.DictReader(path.read_text().splitlines()))
        assert list(rows[0]) == list(CSV_COLUMNS)
        assert [int(r["concurrency"]) for r in rows] == [1, 8, 32, 128, 256]
        assert float(rows[3]["tokens_per_s_per_gpu"]) == 4480.0

    def test_creates_parent_directories(self, curve, tmp_path):
        assert write_csv(curve, tmp_path / "a" / "b" / "c.csv").exists()

    def test_empty_sweep_still_writes_a_header(self, tmp_path):
        path = write_csv([], tmp_path / "empty.csv")
        assert path.read_text().strip() == ",".join(CSV_COLUMNS)


class TestSummarize:
    def test_contains_table_and_slo_block(self, curve):
        text = summarize(curve)
        assert "| Conc |" in text
        assert "interactivity SLO" in text
        assert "4480" in text

    def test_unreachable_slo_is_labelled(self, curve):
        assert "not reached" in summarize(curve, slo_tokens_per_s_per_user=(500,))


class TestPlot:
    def test_writes_a_png(self, curve, tmp_path):
        pytest.importorskip("matplotlib")
        path = plot_pareto({"baseline": curve}, tmp_path / "p" / "pareto.png")
        assert path.exists() and path.stat().st_size > 0

    def test_overlays_multiple_runs(self, curve, tmp_path):
        pytest.importorskip("matplotlib")
        faster = [
            point(p.concurrency, p.tokens_per_s_per_user, p.tokens_per_s_per_gpu * 1.2)
            for p in curve
        ]
        path = plot_pareto({"baseline": curve, "opt": faster}, tmp_path / "cmp.png")
        assert path.exists()

    def test_returns_none_without_matplotlib(self, curve, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fail_matplotlib(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_matplotlib)
        assert plot_pareto({"baseline": curve}, tmp_path / "none.png") is None


class TestLabelPlacement:
    """Overlaid sweeps trace nearly the same curve and meet outright at low
    concurrency, so `c=` labels have to be kept off each other and off the
    line -- which side is free depends on the local slope."""

    SPANS = (100.0, 5000.0)

    def test_the_saturated_tail_reads_as_vertical(self):
        """Throughput climbing while interactivity barely moves."""
        xs, ys = [40.0, 38.0, 37.0], [3000.0, 4000.0, 5000.0]
        assert is_steep(xs, ys, 1, self.SPANS)

    def test_the_low_load_end_reads_as_horizontal(self):
        """Interactivity collapsing for little throughput."""
        xs, ys = [95.0, 88.0, 70.0], [95.0, 700.0, 900.0]
        assert not is_steep(xs, ys, 1, self.SPANS)

    def test_slope_is_judged_in_axis_fractions_not_raw_units(self):
        """tok/s/gpu is ~50x tok/s/user, so raw units call everything steep."""
        xs, ys = [95.0, 88.0, 70.0], [95.0, 700.0, 900.0]
        assert is_steep(xs, ys, 1, (100.0, 100.0))

    def test_an_endpoint_uses_the_segment_it_has(self):
        xs, ys = [40.0, 38.0, 37.0], [3000.0, 4000.0, 5000.0]
        assert is_steep(xs, ys, 0, self.SPANS)
        assert is_steep(xs, ys, len(xs) - 1, self.SPANS)

    def test_the_two_sides_never_coincide(self):
        from bench.report import _label_placement

        placements = {
            _label_placement(steep, outward)
            for steep in (True, False)
            for outward in (True, False)
        }
        assert len(placements) == 4


class TestIntegrityWarnings:
    """Two defects that invalidate a curve rather than just adding noise, both
    seen in a real run: tokens/s/user rising with concurrency (impossible), and
    too few waves through the batch to dilute the fill/drain transient."""

    def test_clean_curve_has_no_warnings(self, curve):
        assert integrity_warnings(curve) == []

    def test_rising_interactivity_is_flagged_as_impossible(self):
        points = [
            point(320, 37.6, 12032, request_count=2560.0),
            point(336, 38.5, 12936, request_count=2688.0),
        ]
        problems = integrity_warnings(points)
        assert any("impossible" in p for p in problems)

    def test_too_few_waves_is_flagged(self):
        points = [point(400, 33.0, 13200, request_count=1024.0)]
        problems = integrity_warnings(points)
        assert any("only 2.6 waves" in p for p in problems)

    def test_wave_threshold_is_configurable(self):
        points = [point(100, 50.0, 5000, request_count=500.0)]
        assert integrity_warnings(points, min_waves=4.0) == []
        assert integrity_warnings(points, min_waves=8.0)

    def test_missing_request_count_is_not_flagged(self):
        assert integrity_warnings([point(64, 50.0, 3200)]) == []

    def test_warnings_appear_in_summary(self):
        points = [
            point(320, 37.6, 12032, request_count=1024.0),
            point(336, 38.5, 12936, request_count=1024.0),
        ]
        assert "Measurement integrity warnings" in summarize(points)
