# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmarks/issues/69408/analyze.py"
spec = importlib.util.spec_from_file_location("issue_69408", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_regression_metrics_match_report_shape():
    old = [
        mod._get_metric({"completed": 96, "duration": 140, "median_tpot_ms": 140}, 32)
    ]
    new = [
        mod._get_metric({"completed": 96, "duration": 160, "median_tpot_ms": 163}, 32)
    ]
    result = mod.summarize(old, new)
    assert result["candidate_samples_per_s"] < result["baseline_samples_per_s"]
    assert result["throughput_change_pct"] == pytest.approx(-12.5)
    assert result["tpot_change_pct"] == pytest.approx(16.43, abs=0.01)


def test_fallback_to_request_throughput():
    assert mod._get_metric({"request_throughput": 2.5, "median_tpot_ms": 50}, 32) == (
        80.0,
        50.0,
    )


def test_invalid_duration_and_empty_results():
    with pytest.raises(ValueError, match="duration"):
        mod._get_metric({"completed": 4, "duration": 0, "median_tpot_ms": 2}, 2)
    with pytest.raises(ValueError, match="Need baseline"):
        mod.summarize([], [(1.0, 1.0)])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("median_tpot_ms", float("nan")),
        ("median_tpot_ms", float("inf")),
        ("median_tpot_ms", -1),
        ("median_tpot_ms", 0),
        ("duration", float("inf")),
        ("duration", -1),
        ("completed", True),
        ("completed", 1.25),
        ("completed", 0),
        ("failed", 1),
    ],
)
def test_rejects_unusable_run(field, bad_value):
    result = {"completed": 96, "duration": 140, "median_tpot_ms": 140, "failed": 0}
    result[field] = bad_value
    with pytest.raises(ValueError, match=field):
        mod._get_metric(result, 32)


def test_loads_real_bench_serve_result():
    # Fields retained from vllm bench serve against a local SSE fixture server.
    result = Path(__file__).parent / "fixtures/bench_serve_completions_result.json"
    data = json.loads(result.read_text())
    assert mod._load_files([str(result)], 1) == [
        (pytest.approx(data["request_throughput"]), data["median_tpot_ms"])
    ]


@pytest.mark.parametrize("unmatched", ["run_004.json", "typo_*.json"])
def test_cli_rejects_unmatched_input_among_valid_runs(tmp_path, unmatched):
    for i in range(1, 4):
        (tmp_path / f"run_{i:03}.json").write_text(
            '{"request_throughput": 2.5, "median_tpot_ms": 50}', encoding="utf-8"
        )

    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--baseline",
            "run_*.json",
            unmatched,
            "--candidate",
            "run_*.json",
            "--control-baseline",
            "run_*.json",
            "--control-candidate",
            "run_*.json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert unmatched in result.stderr
    assert not result.stdout
