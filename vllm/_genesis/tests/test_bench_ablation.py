# SPDX-License-Identifier: Apache-2.0
"""Unit tests for D3 — bench-suite ablation comparison helpers.

We test `_ablation_compare()` and `_print_ablation_table()` from
`tools/genesis_bench_suite.py` directly. The full bench loop is not
exercised here (it requires a running vllm server); we feed synthetic
result dicts that mirror the real shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# Allow `import genesis_bench_suite` from tools/
TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import genesis_bench_suite as gbs  # noqa: E402


# ─── Synthetic result builders ─────────────────────────────────────────


def _make_result(name: str, wall_tps_samples: list[float],
                 decode_tpot_samples: list[float],
                 ttft_samples: list[float | None]) -> dict:
    flat = []
    for i, (wt, dt, tt) in enumerate(zip(wall_tps_samples,
                                         decode_tpot_samples,
                                         ttft_samples)):
        flat.append({
            "prompt_idx": i % 5, "run_idx": i // 5,
            "wall_tps": wt, "decode_tpot_ms": dt, "ttft_ms": tt,
            "completion_tokens": 200, "elapsed_s": 200 / wt if wt else 0,
            "finish": "stop",
        })
    return {
        "name": name,
        "decode_bench": {
            "wall_TPS": gbs.mean_std_cv(wall_tps_samples),
            "decode_TPOT_ms": gbs.mean_std_cv(decode_tpot_samples),
            "TTFT_ms": gbs.mean_std_cv([t for t in ttft_samples if t]),
            "flat_results": flat,
        },
    }


# ─── Tests for _ablation_compare ───────────────────────────────────────


class TestAblationCompare:
    def test_baseline_load_failure_returns_error(self, tmp_path):
        result = _make_result("cur", [100.0]*5, [10.0]*5, [120.0]*5)
        ab = gbs._ablation_compare(str(tmp_path / "nonexistent.json"),
                                   result, "no-PN14")
        assert "error" in ab
        assert "load failed" in ab["error"].lower()

    def test_missing_decode_bench_returns_error(self, tmp_path):
        # Baseline JSON without decode_bench
        baseline_path = tmp_path / "bad_baseline.json"
        baseline_path.write_text(json.dumps({"name": "broken"}))
        result = _make_result("cur", [100.0]*5, [10.0]*5, [120.0]*5)
        ab = gbs._ablation_compare(str(baseline_path), result, "no-PN14")
        assert "error" in ab
        assert "decode_bench missing" in ab["error"]

    def test_no_regression_yields_NOT_SIGNIFICANT(self, tmp_path):
        """Identical-distribution baseline+current → Welch p high → NOT_SIG."""
        baseline = _make_result("base",
                                [103.0]*10, [9.4]*10, [125.0]*10)
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        # Slight perturbation (still well within noise)
        current = _make_result("cur",
                               [103.1]*10, [9.5]*10, [125.5]*10)
        ab = gbs._ablation_compare(str(baseline_path), current, "no-X")
        assert "error" not in ab
        for m in ("wall_TPS", "decode_TPOT_ms", "TTFT_ms"):
            assert m in ab["metrics"]
            entry = ab["metrics"][m]
            # No regression should yield non-significant verdict
            assert entry["verdict"] in ("NOT_SIGNIFICANT", "IDENTICAL"), (
                f"{m} verdict={entry['verdict']} (p={entry['welch_p']})"
            )

    def test_clear_regression_detected(self, tmp_path):
        """Distinct distributions (15% TPS drop) → Welch SIGNIFICANT."""
        baseline = _make_result("base",
                                [120.0, 121.0, 119.5, 120.5, 120.2, 121.3, 119.8, 120.1, 120.9, 120.4],
                                [8.3]*10, [115.0]*10)
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        # 15% TPS drop, low overlap
        current = _make_result("cur",
                               [102.0, 102.5, 101.8, 102.2, 102.7, 101.9, 102.1, 102.6, 102.3, 102.0],
                               [9.8]*10, [125.0]*10)
        ab = gbs._ablation_compare(str(baseline_path), current, "no-PN14")
        wt = ab["metrics"]["wall_TPS"]
        assert wt["verdict"] == "SIGNIFICANT", wt
        assert wt["pct_change"] is not None
        assert wt["pct_change"] < -10.0  # drop, not gain

    def test_pct_change_signs_correct(self, tmp_path):
        """current > baseline → positive pct_change; current < baseline → negative."""
        baseline = _make_result("base", [100.0]*8, [10.0]*8, [120.0]*8)
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        # Current TPS higher (improvement)
        current_up = _make_result("up", [110.0]*8, [9.0]*8, [110.0]*8)
        ab_up = gbs._ablation_compare(str(baseline_path), current_up, "improvement")
        assert ab_up["metrics"]["wall_TPS"]["pct_change"] > 0
        # decode_TPOT lower in current = improvement; pct_change is negative
        # (because current_mean < baseline_mean) — semantically: TPOT down is good
        assert ab_up["metrics"]["decode_TPOT_ms"]["pct_change"] < 0

    def test_ablate_tag_preserved(self, tmp_path):
        baseline = _make_result("base", [100.0]*5, [10.0]*5, [120.0]*5)
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        current = _make_result("cur", [99.0]*5, [10.1]*5, [121.0]*5)
        ab = gbs._ablation_compare(str(baseline_path), current, "no-PN14")
        assert ab["ablate_tag"] == "no-PN14"
        assert ab["baseline"] == "base"
        assert ab["current"] == "cur"

    def test_sample_counts_reported(self, tmp_path):
        baseline = _make_result("base", [100.0]*7, [10.0]*7, [120.0]*7)
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        current = _make_result("cur", [99.0]*5, [10.1]*5, [121.0]*5)
        ab = gbs._ablation_compare(str(baseline_path), current, "tag")
        assert ab["metrics"]["wall_TPS"]["n_baseline"] == 7
        assert ab["metrics"]["wall_TPS"]["n_current"] == 5

    def test_handles_none_ttft_samples(self, tmp_path):
        """TTFT samples can be None when http stream didn't capture timestamp.
        Compare must filter them out, not crash."""
        baseline = _make_result("base", [100.0]*6, [10.0]*6,
                                [120.0, None, 122.0, None, 121.0, 119.0])
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        current = _make_result("cur", [99.0]*6, [10.1]*6,
                               [None, 121.0, None, 122.5, 121.5, None])
        ab = gbs._ablation_compare(str(baseline_path), current, "tag")
        # No KeyError, no None-arithmetic crash
        assert "TTFT_ms" in ab["metrics"]
        assert ab["metrics"]["TTFT_ms"]["n_baseline"] == 4
        assert ab["metrics"]["TTFT_ms"]["n_current"] == 3


# ─── Tests for _print_ablation_table ───────────────────────────────────


class TestAblationTablePrint:
    def test_print_error_branch(self, capsys):
        gbs._print_ablation_table({"error": "baseline load failed: foo"})
        captured = capsys.readouterr()
        assert "ablation skipped" in captured.out
        assert "baseline load failed" in captured.out

    def test_print_full_table(self, capsys, tmp_path):
        baseline = _make_result("base", [100.0]*5, [10.0]*5, [120.0]*5)
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        current = _make_result("cur", [102.0]*5, [9.8]*5, [119.0]*5)
        ab = gbs._ablation_compare(str(baseline_path), current, "no-PN14")
        gbs._print_ablation_table(ab)
        out = capsys.readouterr().out
        assert "no-PN14" in out  # ablate_tag in header
        assert "wall_TPS" in out
        assert "decode_TPOT_ms" in out
        assert "TTFT_ms" in out
        # Verdict columns rendered (any of the welch_t enum values)
        assert any(v in out for v in ("SIGNIFICANT", "NOT_SIGNIFICANT", "IDENTICAL"))

    def test_print_handles_missing_metric(self, capsys):
        ab = {
            "baseline": "base", "current": "cur", "ablate_tag": "tag",
            "metrics": {
                "wall_TPS": {"error": "summary mean missing"},
                "decode_TPOT_ms": {
                    "baseline_mean": 10.0, "current_mean": 10.5,
                    "delta": 0.5, "pct_change": 5.0,
                    "welch_t": 1.5, "welch_p": 0.15,
                    "verdict": "NOT_SIGNIFICANT",
                    "n_baseline": 5, "n_current": 5,
                },
            },
        }
        gbs._print_ablation_table(ab)
        out = capsys.readouterr().out
        assert "summary mean missing" in out  # error row rendered, not crashed
        assert "decode_TPOT_ms" in out
