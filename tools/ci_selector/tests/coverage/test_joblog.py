# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test outcomes read off a job log, and what counts as having run nothing."""

from __future__ import annotations

import gzip
from pathlib import Path

from ci_selector.coverage.joblog import read_counts


def write(path: Path, body: str, *, gz: bool = False) -> Path:
    if gz:
        path.write_bytes(gzip.compress(body.encode()))
    else:
        path.write_text(body)
    return path


class TestParsing:
    def test_reads_a_summary(self, tmp_path: Path):
        log = write(
            tmp_path / "job.log",
            "collected 10 items\n===== 8 passed, 2 skipped in 3.21s =====\n",
        )
        counts = read_counts(log).counts
        assert (counts.passed, counts.skipped, counts.collected) == (8, 2, 10)
        assert counts.invocations == 1

    def test_reads_gzipped_logs(self, tmp_path: Path):
        log = write(
            tmp_path / "job.log.gz",
            "collected 4 items\n=== 4 passed in 1.0s ===\n",
            gz=True,
        )
        assert read_counts(log).counts.passed == 4

    def test_strips_buildkite_timestamps_inside_a_token(self, tmp_path: Path):
        # Buildkite splices markers mid-word, so "2 passed" arrives as
        # "2 _bk;t=123passed". Without stripping, the counts come out silently
        # low and a healthy job reads as one that ran nothing.
        log = write(
            tmp_path / "job.log",
            "=== 2 _bk;t=1786062793172passed, 1 skipped in 2.0s ===\n",
        )
        counts = read_counts(log).counts
        assert (counts.passed, counts.skipped) == (2, 1)

    def test_sums_every_invocation(self, tmp_path: Path):
        # A step usually runs several pytest commands and only the total
        # describes the run the record came from.
        log = write(
            tmp_path / "job.log",
            "=== 3 passed in 1.0s ===\n=== 5 passed, 1 failed in 2.0s ===\n",
        )
        counts = read_counts(log).counts
        assert (counts.passed, counts.failed, counts.invocations) == (8, 1, 2)

    def test_an_unreadable_log_says_so(self, tmp_path: Path):
        assert read_counts(tmp_path / "absent.log").unreadable


class TestRanNothing:
    def test_all_skipped_counts_as_having_run_nothing(self, tmp_path: Path):
        log = write(tmp_path / "job.log", "=== 6 skipped in 1.0s ===\n")
        assert read_counts(log).counts.ran_nothing

    def test_one_executed_test_is_enough(self, tmp_path: Path):
        # Threshold-free on purpose. "Almost all skipped" needs a fraction
        # nobody can derive; "executed none at all" is unambiguous.
        log = write(tmp_path / "job.log", "=== 1 passed, 99 skipped in 1.0s ===\n")
        assert not read_counts(log).counts.ran_nothing

    def test_a_step_that_is_not_pytest_is_not_thin_for_it(self, tmp_path: Path):
        # Plenty of steps are shell scripts. Silence from pytest says nothing
        # about them, so this signal has to stay quiet rather than accuse.
        log = write(tmp_path / "job.log", "building wheel\ndone\n")
        counts = read_counts(log).counts
        assert counts.invocations == 0 and not counts.ran_nothing

    def test_errors_count_as_executed(self, tmp_path: Path):
        log = write(tmp_path / "job.log", "=== 2 errors, 3 skipped in 1.0s ===\n")
        assert not read_counts(log).counts.ran_nothing

    def test_all_deselected_ran_nothing_too(self, tmp_path: Path):
        # Zero skips, so the old `skipped > 0` clause read this as healthy.
        log = write(tmp_path / "job.log", "=== 12 deselected in 0.4s ===\n")
        assert read_counts(log).counts.ran_nothing

    def test_no_tests_ran_is_not_a_healthy_row(self, tmp_path: Path):
        # pytest's own wording when a filter matches nothing. Also zero skips.
        log = write(tmp_path / "job.log", "=== no tests ran in 0.1s ===\n")
        assert read_counts(log).counts.ran_nothing


class TestSummaryUnparsed:
    """The witness for `_SUMMARY` drift, which `ran_nothing` structurally
    cannot be: if the summary regex matches nothing then every count it feeds
    is zero, which is indistinguishable from a healthy quiet step."""

    def test_a_collection_line_with_no_summary_is_caught(self, tmp_path: Path):
        # What a job killed mid-run leaves behind, and what a pytest release
        # that changes its summary format would leave on EVERY job at once.
        log = write(tmp_path / "job.log", "collected 40 items\ntest_a.py .\n")
        counts = read_counts(log).counts
        assert counts.invocations == 0 and counts.summary_unparsed

    def test_a_step_that_is_not_pytest_stays_silent(self, tmp_path: Path):
        # The floor. Collecting nothing means there was no pytest output to misread.
        log = write(tmp_path / "job.log", "building wheel\ndone\n")
        assert not read_counts(log).counts.summary_unparsed

    def test_a_readable_summary_is_not_drift(self, tmp_path: Path):
        log = write(
            tmp_path / "job.log", "collected 3 items\n=== 3 passed in 1.0s ===\n"
        )
        assert not read_counts(log).counts.summary_unparsed
