# SPDX-License-Identifier: Apache-2.0
"""Before/after comparison: the speedup numbers that go in the report must be
computed from the SLO-matched points, not from whichever point looks best."""

import pytest
from bench.compare import delta_table, load_runs, main, per_concurrency_table
from conftest import make_point


def point(label, concurrency, per_user, per_gpu, ttft=None):
    """Label-first spelling: these tests are all about comparing runs."""
    return make_point(concurrency, per_user, per_gpu, label=label, ttft_ms=ttft)


@pytest.fixture
def runs():
    base = [
        point("baseline", 1, 95.0, 95.0, 35.0),
        point("baseline", 32, 70.0, 2240.0, 180.0),
        point("baseline", 128, 35.0, 4480.0, 900.0),
    ]
    opt = [
        point("opt", 1, 100.0, 100.0, 30.0),
        point("opt", 32, 80.0, 2560.0, 150.0),
        point("opt", 128, 45.0, 5760.0, 700.0),
    ]
    return {"baseline": base, "opt": opt}


class TestDeltaTable:
    def test_speedup_is_computed_at_the_matched_slo(self, runs):
        table = delta_table(runs, slos=(30.0,))
        assert "4480 (c=128)" in table
        assert "5760 (c=128)" in table
        assert "1.286x (+28.6%)" in table

    def test_tighter_slo_compares_different_concurrencies(self, runs):
        table = delta_table(runs, slos=(75.0,))
        # baseline can only hit 75 tok/s/user at c=1; opt manages it at c=32.
        assert "95 (c=1)" in table
        assert "2560 (c=32)" in table

    def test_unreachable_slo_is_labelled_and_not_ratioed(self, runs):
        table = delta_table(runs, slos=(300.0,))
        assert "not reached" in table
        assert "x (" not in table

    def test_regression_shows_a_negative_delta(self, runs):
        slower = {
            "baseline": runs["baseline"],
            "slow": [
                point(
                    "slow",
                    c.concurrency,
                    c.tokens_per_s_per_user,
                    c.tokens_per_s_per_gpu * 0.5,
                )
                for c in runs["baseline"]
            ],
        }
        assert "0.500x (-50.0%)" in delta_table(slower, slos=(30.0,))

    def test_single_run_has_no_delta_column(self, runs):
        table = delta_table({"baseline": runs["baseline"]}, slos=(30.0,))
        assert "vs baseline" not in table

    def test_three_runs_all_compare_to_the_first(self, runs):
        runs["opt2"] = [
            point(
                "opt2",
                c.concurrency,
                c.tokens_per_s_per_user,
                c.tokens_per_s_per_gpu * 2,
            )
            for c in runs["baseline"]
        ]
        table = delta_table(runs, slos=(30.0,))
        assert "opt vs baseline" in table and "opt2 vs baseline" in table
        assert "2.000x (+100.0%)" in table


class TestPerConcurrencyTable:
    def test_row_per_concurrency_with_ratios(self, runs):
        table = per_concurrency_table(runs)
        lines = table.splitlines()
        assert len(lines) == 2 + 3
        assert "1.286x" in table

    def test_concurrency_only_in_one_run_renders_dashes(self, runs):
        runs["opt"].append(point("opt", 256, 20.0, 6000.0))
        table = per_concurrency_table(runs)
        assert table.splitlines()[-1].startswith("| 256 | - | - | - |")

    def test_missing_ttft_renders_dash(self):
        runs = {"a": [point("a", 1, 50.0, 50.0)]}
        assert per_concurrency_table(runs).splitlines()[-1].endswith("| 50.0 | - |")


class TestLoadRuns:
    def test_reads_both_runs_from_disk(self, tmp_path, run_tree, export_factory):
        for label, tps in (("baseline", 4480.0), ("opt", 5760.0)):
            run_tree(label, {128: export_factory(total_tps=tps, per_user=35.0)})
        runs = load_runs(["baseline", "opt"], tmp_path)
        assert runs["opt"][0].tokens_per_s_per_gpu == 5760.0

    def test_missing_run_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no aiperf exports under"):
            load_runs(["nope"], tmp_path)


class TestMain:
    def test_writes_comparison_artifacts(
        self, tmp_path, run_tree, export_factory, capsys
    ):
        for label, tps in (("baseline", 4480.0), ("opt", 5760.0)):
            run_tree(
                label,
                {c: export_factory(total_tps=tps, per_user=tps / c) for c in (1, 128)},
            )
        rc = main(["baseline", "opt", "--artifact-root", str(tmp_path)])
        out_dir = tmp_path / "comparisons"
        assert rc == 0
        assert (out_dir / "baseline-vs-opt.md").exists()
        assert (out_dir / "baseline-vs-opt.csv").exists()
        assert "Throughput at interactivity SLO" in capsys.readouterr().out

    def test_names_retitle_the_report_without_renaming_the_files(
        self, tmp_path, run_tree, export_factory, capsys
    ):
        """Run labels are directory names; the write-up wants readable ones."""
        for label in ("baseline", "opt"):
            run_tree(label, {c: export_factory() for c in (1, 128)})
        rc = main(
            [
                "baseline", "opt",
                "--artifact-root", str(tmp_path),
                "--names", "unpruned,REAP 50% pruned",
            ]
        )
        assert rc == 0
        assert (tmp_path / "comparisons" / "baseline-vs-opt.md").exists()
        out = capsys.readouterr().out
        assert "# unpruned vs REAP 50% pruned" in out
        assert "REAP 50% pruned tok/s/gpu" in out

    def test_a_name_per_run_is_required(
        self, tmp_path, run_tree, export_factory, capsys
    ):
        for label in ("baseline", "opt"):
            run_tree(label, {c: export_factory() for c in (1, 128)})
        rc = main(
            ["baseline", "opt", "--artifact-root", str(tmp_path), "--names", "only-one"]
        )
        assert rc == 2
