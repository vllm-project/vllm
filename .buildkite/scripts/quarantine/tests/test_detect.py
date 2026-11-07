# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from detect import (  # noqa: E402
    ANALYSIS_WINDOW_DAYS,
    REINSTATEMENT_PASSES,
    QuarantineEntry,
    TestResult,
    check_reinstatement,
    detect_flaky_tests,
    normalize_backend,
    parse_test_result,
    update_quarantine_list,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
RECENT = NOW - timedelta(days=1)


def _result(
    test_id: str = "tests/test_a.py::test_one",
    backend: str = "cuda",
    passed: bool = True,
    build_number: int = 1,
    ts_offset_hours: int = 0,
) -> TestResult:
    return TestResult(
        test_id=test_id,
        backend=backend,
        passed=passed,
        build_number=build_number,
        timestamp=RECENT + timedelta(hours=ts_offset_hours),
    )


def _make_runs(
    n_pass: int,
    n_fail: int,
    test_id: str = "tests/test_a.py::test_one",
    backend: str = "cuda",
) -> list[TestResult]:
    results = []
    for i in range(n_pass):
        results.append(
            _result(
                test_id=test_id,
                backend=backend,
                passed=True,
                build_number=i,
                ts_offset_hours=i,
            )
        )
    for i in range(n_fail):
        results.append(
            _result(
                test_id=test_id,
                backend=backend,
                passed=False,
                build_number=n_pass + i,
                ts_offset_hours=n_pass + i,
            )
        )
    return results


# --- normalize_backend ---


class TestNormalizeBackend:
    @pytest.mark.parametrize(
        "device,expected",
        [
            ("h100", "cuda"),
            ("h200_18gb", "cuda"),
            ("h200_35gb", "cuda"),
            ("b200", "cuda"),
            ("b200-k8s", "cuda"),
            ("a100", "cuda"),
            ("l4", "cuda"),
            ("gh200", "cuda"),
            ("mi250_1", "rocm"),
            ("mi300_4", "rocm"),
            ("mi325_8", "rocm"),
            ("mi355_dpx", "rocm"),
            ("amd_cpu", "rocm"),
            ("zen5", "rocm"),
            ("intel_cpu", "intel"),
            ("intel_hpu", "intel"),
            ("intel_gpu", "intel"),
            ("cpu", "cpu"),
            ("cpu-small", "cpu"),
            ("cpu-medium", "cpu"),
            ("arm_cpu", "cpu"),
            ("ascend_npu", "other"),
            ("dgx-spark", "other"),
            ("unknown_device", "other"),
        ],
    )
    def test_device_mapping(self, device: str, expected: str):
        assert normalize_backend(device) == expected

    def test_empty_device_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            normalize_backend("")

    def test_case_insensitive(self):
        assert normalize_backend("H200_18GB") == "cuda"
        assert normalize_backend("MI300_1") == "rocm"


# --- parse_test_result ---


class TestParseTestResult:
    def test_valid(self):
        raw = {
            "test_id": "tests/test_a.py::test_one",
            "backend": "cuda",
            "passed": True,
            "build_number": 100,
            "timestamp": "2026-10-01T12:00:00+00:00",
        }
        result = parse_test_result(raw)
        assert result.test_id == "tests/test_a.py::test_one"
        assert result.backend == "cuda"
        assert result.passed is True
        assert result.build_number == 100

    def test_missing_field(self):
        raw = {"test_id": "x", "backend": "cuda", "passed": True}
        with pytest.raises(ValueError, match="Missing required"):
            parse_test_result(raw)

    def test_invalid_backend(self):
        raw = {
            "test_id": "x",
            "backend": "tpu",
            "passed": True,
            "build_number": 1,
            "timestamp": "2026-10-01T12:00:00+00:00",
        }
        with pytest.raises(ValueError, match="Invalid backend"):
            parse_test_result(raw)

    def test_empty_dict(self):
        with pytest.raises(ValueError, match="Missing required"):
            parse_test_result({})


# --- detect_flaky_tests ---


class TestDetectFlakyTests:
    def test_flaky_test_detected(self):
        results = _make_runs(n_pass=5, n_fail=5)
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 1
        assert entries[0].test_id == "tests/test_a.py::test_one"
        assert entries[0].backend == "cuda"
        assert entries[0].fail_rate == 0.5

    def test_consistently_passing_not_quarantined(self):
        results = _make_runs(n_pass=10, n_fail=0)
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 0

    def test_consistently_failing_not_quarantined(self):
        results = _make_runs(n_pass=0, n_fail=10)
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 0

    def test_too_few_runs(self):
        results = _make_runs(n_pass=2, n_fail=2)
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 0

    def test_at_low_threshold_boundary_not_quarantined(self):
        # Exactly 10% fail rate (1 fail in 10 runs) → ≤ threshold
        results = _make_runs(n_pass=9, n_fail=1)
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 0

    def test_at_high_threshold_boundary_not_quarantined(self):
        # Exactly 90% fail rate (9 fail in 10 runs) → ≥ threshold
        results = _make_runs(n_pass=1, n_fail=9)
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 0

    def test_just_above_low_threshold_quarantined(self):
        # 2 fail in 10 runs = 20% → above 10%
        results = _make_runs(n_pass=8, n_fail=2)
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 1
        assert entries[0].fail_rate == 0.2

    def test_old_results_excluded(self):
        old = NOW - timedelta(days=ANALYSIS_WINDOW_DAYS + 1)
        results = [
            TestResult(
                "tests/test_a.py::test_one",
                "cuda",
                False,
                i,
                old,
            )
            for i in range(5)
        ] + [
            TestResult(
                "tests/test_a.py::test_one",
                "cuda",
                True,
                10 + i,
                old,
            )
            for i in range(5)
        ]
        entries = detect_flaky_tests(results, now=NOW)
        assert len(entries) == 0

    def test_flaky_on_one_backend_only(self):
        cuda_runs = _make_runs(n_pass=10, n_fail=0, backend="cuda")
        rocm_runs = _make_runs(n_pass=5, n_fail=5, backend="rocm")
        entries = detect_flaky_tests(cuda_runs + rocm_runs, now=NOW)
        assert len(entries) == 1
        assert entries[0].backend == "rocm"

    def test_empty_input(self):
        assert detect_flaky_tests([], now=NOW) == []

    def test_reason_format(self):
        results = _make_runs(n_pass=5, n_fail=5)
        entries = detect_flaky_tests(results, now=NOW)
        assert "50%" in entries[0].reason
        assert "10 runs" in entries[0].reason
        assert "cuda" in entries[0].reason


# --- check_reinstatement ---


class TestCheckReinstatement:
    def _entry(
        self,
        test_id: str = "tests/test_a.py::test_one",
        backend: str = "cuda",
        quarantined_at: str = "2026-09-25T00:00:00+00:00",
        fail_rate: float = 0.5,
        total_runs: int = 10,
        reason: str = "test",
    ) -> QuarantineEntry:
        return QuarantineEntry(
            test_id=test_id,
            backend=backend,
            quarantined_at=quarantined_at,
            fail_rate=fail_rate,
            total_runs=total_runs,
            reason=reason,
        )

    def test_reinstate_after_consecutive_passes(self):
        entry = self._entry()
        results = [
            _result(passed=True, build_number=i, ts_offset_hours=i)
            for i in range(REINSTATEMENT_PASSES)
        ]
        assert check_reinstatement(entry, results, now=NOW) is True

    def test_no_reinstate_with_recent_failure(self):
        entry = self._entry()
        results = [
            _result(passed=True, build_number=i, ts_offset_hours=i)
            for i in range(REINSTATEMENT_PASSES - 1)
        ] + [
            _result(
                passed=False,
                build_number=REINSTATEMENT_PASSES - 1,
                ts_offset_hours=REINSTATEMENT_PASSES - 1,
            )
        ]
        assert check_reinstatement(entry, results, now=NOW) is False

    def test_no_reinstate_too_few_runs(self):
        entry = self._entry()
        results = [
            _result(passed=True, build_number=i, ts_offset_hours=i)
            for i in range(REINSTATEMENT_PASSES - 1)
        ]
        assert check_reinstatement(entry, results, now=NOW) is False

    def test_no_reinstate_no_runs(self):
        entry = self._entry()
        assert check_reinstatement(entry, [], now=NOW) is False

    def test_reinstate_ignores_other_backends(self):
        entry = self._entry(backend="cuda")
        results = [
            _result(
                backend="rocm",
                passed=True,
                build_number=i,
                ts_offset_hours=i,
            )
            for i in range(REINSTATEMENT_PASSES)
        ]
        assert check_reinstatement(entry, results, now=NOW) is False

    def test_reinstate_checks_last_n(self):
        entry = self._entry()
        # Old failure followed by enough passes
        results = [_result(passed=False, build_number=0, ts_offset_hours=0)] + [
            _result(
                passed=True,
                build_number=1 + i,
                ts_offset_hours=1 + i,
            )
            for i in range(REINSTATEMENT_PASSES)
        ]
        assert check_reinstatement(entry, results, now=NOW) is True


# --- update_quarantine_list ---


class TestUpdateQuarantineList:
    def test_add_new_flaky(self):
        results = _make_runs(n_pass=5, n_fail=5)
        updated = update_quarantine_list([], results, now=NOW)
        assert len(updated) == 1
        assert updated[0].test_id == "tests/test_a.py::test_one"

    def test_reinstate_fixed_test(self):
        entry = QuarantineEntry(
            test_id="tests/test_a.py::test_one",
            backend="cuda",
            quarantined_at="2026-09-25T00:00:00+00:00",
            fail_rate=0.5,
            total_runs=10,
            reason="test",
        )
        results = [
            _result(passed=True, build_number=i, ts_offset_hours=i)
            for i in range(REINSTATEMENT_PASSES)
        ]
        updated = update_quarantine_list([entry], results, now=NOW)
        assert len(updated) == 0

    def test_keep_quarantined_if_still_flaky(self):
        entry = QuarantineEntry(
            test_id="tests/test_a.py::test_one",
            backend="cuda",
            quarantined_at="2026-09-25T00:00:00+00:00",
            fail_rate=0.5,
            total_runs=10,
            reason="original",
        )
        results = _make_runs(n_pass=5, n_fail=5)
        updated = update_quarantine_list([entry], results, now=NOW)
        assert len(updated) == 1
        # Preserves original entry's quarantined_at
        assert updated[0].quarantined_at == "2026-09-25T00:00:00+00:00"

    def test_keep_quarantined_if_not_enough_passes(self):
        entry = QuarantineEntry(
            test_id="tests/test_a.py::test_one",
            backend="cuda",
            quarantined_at="2026-09-25T00:00:00+00:00",
            fail_rate=0.5,
            total_runs=10,
            reason="test",
        )
        # Only 3 passes — not enough to reinstate
        results = [
            _result(passed=True, build_number=i, ts_offset_hours=i) for i in range(3)
        ]
        updated = update_quarantine_list([entry], results, now=NOW)
        assert len(updated) == 1

    def test_mixed_backends(self):
        cuda_runs = _make_runs(n_pass=5, n_fail=5, backend="cuda")
        rocm_runs = _make_runs(n_pass=10, n_fail=0, backend="rocm")
        updated = update_quarantine_list([], cuda_runs + rocm_runs, now=NOW)
        assert len(updated) == 1
        assert updated[0].backend == "cuda"

    def test_empty_results_keeps_existing(self):
        entry = QuarantineEntry(
            test_id="tests/test_a.py::test_one",
            backend="cuda",
            quarantined_at="2026-09-25T00:00:00+00:00",
            fail_rate=0.5,
            total_runs=10,
            reason="test",
        )
        updated = update_quarantine_list([entry], [], now=NOW)
        assert len(updated) == 1

    def test_sorted_output(self):
        results_b = _make_runs(
            n_pass=5,
            n_fail=5,
            test_id="tests/test_b.py::test_b",
        )
        results_a = _make_runs(
            n_pass=5,
            n_fail=5,
            test_id="tests/test_a.py::test_a",
        )
        updated = update_quarantine_list([], results_b + results_a, now=NOW)
        assert updated[0].test_id == "tests/test_a.py::test_a"
        assert updated[1].test_id == "tests/test_b.py::test_b"


# --- CLI integration (file I/O) ---


class TestCLI:
    def _write_jsonl(self, tmp_path: Path, results: list[TestResult]) -> Path:
        p = tmp_path / "results.jsonl"
        with open(p, "w") as f:
            for r in results:
                line = {
                    "test_id": r.test_id,
                    "backend": r.backend,
                    "passed": r.passed,
                    "build_number": r.build_number,
                    "timestamp": r.timestamp.isoformat(),
                }
                f.write(json.dumps(line) + "\n")
        return p

    def test_cli_creates_quarantine_file(self, tmp_path: Path):
        from cli import main  # noqa: E402

        results = _make_runs(n_pass=5, n_fail=5)
        results_file = self._write_jsonl(tmp_path, results)
        quarantine_file = tmp_path / "quarantine.json"

        rc = main(
            [
                str(results_file),
                "--quarantine-file",
                str(quarantine_file),
            ]
        )
        assert rc == 0
        assert quarantine_file.exists()

        with open(quarantine_file) as f:
            data = json.load(f)
        assert len(data["entries"]) == 1
        assert data["entries"][0]["backend"] == "cuda"

    def test_cli_empty_results(self, tmp_path: Path):
        from cli import main  # noqa: E402

        results_file = tmp_path / "empty.jsonl"
        results_file.write_text("")

        rc = main(
            [
                str(results_file),
                "--quarantine-file",
                str(tmp_path / "q.json"),
            ]
        )
        assert rc == 0

    def test_cli_malformed_json(self, tmp_path: Path):
        from cli import main  # noqa: E402

        results_file = tmp_path / "bad.jsonl"
        results_file.write_text("not json\n")

        rc = main(
            [
                str(results_file),
                "--quarantine-file",
                str(tmp_path / "q.json"),
            ]
        )
        assert rc == 1

    def test_cli_invalid_record(self, tmp_path: Path):
        from cli import main  # noqa: E402

        results_file = tmp_path / "bad.jsonl"
        results_file.write_text('{"test_id": "x"}\n')

        rc = main(
            [
                str(results_file),
                "--quarantine-file",
                str(tmp_path / "q.json"),
            ]
        )
        assert rc == 1
