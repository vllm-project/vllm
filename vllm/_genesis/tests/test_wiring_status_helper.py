# SPDX-License-Identifier: Apache-2.0
"""Unit tests for B2 — shared `result_to_wiring_status` helper in
`vllm._genesis.wiring.text_patch`.

The helper centralizes (TextPatchResult, TextPatchFailure) → (status,
reason) translation that was previously duplicated across ~25 wiring
modules with subtle drift. Specifically: many copies forgot to
distinguish APPLIED from SKIPPED/IDEMPOTENT and reported "applied"
even when the file was unchanged (silent boot-log lie).
"""
from __future__ import annotations


from vllm._genesis.wiring.text_patch import (
    TextPatchFailure,
    TextPatchResult,
    result_to_wiring_status,
)


class TestResultMapping:
    def test_applied_returns_applied_status(self):
        status, reason = result_to_wiring_status(
            TextPatchResult.APPLIED, None,
            applied_message="PN14 applied: clamp engaged",
            patch_name="PN14",
        )
        assert status == "applied"
        assert reason == "PN14 applied: clamp engaged"

    def test_idempotent_returns_skipped_with_marker_reason(self):
        status, reason = result_to_wiring_status(
            TextPatchResult.IDEMPOTENT, None,
            applied_message="not used",
            patch_name="PN14",
        )
        assert status == "skipped"
        assert "PN14" in reason
        assert "marker present" in reason

    def test_skipped_with_failure_reason_and_detail(self):
        f = TextPatchFailure(reason="upstream_merged",
                              detail="marker 'safe_page_idx' present")
        status, reason = result_to_wiring_status(
            TextPatchResult.SKIPPED, f,
            applied_message="not used",
            patch_name="PN14",
        )
        assert status == "skipped"
        assert "PN14" in reason
        assert "upstream_merged" in reason
        assert "safe_page_idx" in reason

    def test_skipped_without_failure_uses_unknown_marker(self):
        status, reason = result_to_wiring_status(
            TextPatchResult.SKIPPED, None,
            applied_message="not used",
            patch_name="PN14",
        )
        assert status == "skipped"
        assert "unknown_skip" in reason

    def test_skipped_with_failure_no_detail(self):
        f = TextPatchFailure(reason="required_anchor_missing", detail="")
        status, reason = result_to_wiring_status(
            TextPatchResult.SKIPPED, f,
            applied_message="not used",
            patch_name="PN14",
        )
        assert status == "skipped"
        assert "required_anchor_missing" in reason
        # No trailing " — " when detail is empty
        assert not reason.rstrip().endswith("—")

    def test_failed_with_failure_reason_and_detail(self):
        f = TextPatchFailure(reason="ambiguous_anchor",
                              detail="anchor matches 3 times")
        status, reason = result_to_wiring_status(
            TextPatchResult.FAILED, f,
            applied_message="not used",
            patch_name="PN14",
        )
        assert status == "failed"
        assert "PN14" in reason
        assert "ambiguous_anchor" in reason
        assert "3 times" in reason

    def test_failed_without_failure(self):
        status, reason = result_to_wiring_status(
            TextPatchResult.FAILED, None,
            applied_message="not used",
            patch_name="PN14",
        )
        assert status == "failed"
        assert "unknown" in reason

    def test_status_values_match_wiring_contract(self):
        """All possible TextPatchResult values must map to one of the
        three wiring statuses — never raises, never returns None."""
        for r in TextPatchResult:
            status, reason = result_to_wiring_status(
                r, None,
                applied_message="msg", patch_name="X",
            )
            assert status in ("applied", "skipped", "failed")
            assert isinstance(reason, str)
            assert reason  # non-empty


class TestRegressionAgainstOldBug:
    """The original bug this helper fixes: many wiring `apply()` copies
    only checked `if result == FAILED` and fell through with "applied"
    for IDEMPOTENT and SKIPPED. Caught by PN14 TDD 2026-04-29."""

    def test_idempotent_must_NOT_report_applied(self):
        """The pre-helper bug returned 'applied' here. The helper must
        return 'skipped' so boot logs are honest."""
        status, _ = result_to_wiring_status(
            TextPatchResult.IDEMPOTENT, None,
            applied_message="lie that file was modified",
            patch_name="X",
        )
        assert status == "skipped"

    def test_skipped_must_NOT_report_applied(self):
        """Drift detection caused a SKIPPED return in PN14 fixture; the
        old code reported 'applied' anyway. The helper must return
        'skipped'."""
        f = TextPatchFailure(reason="upstream_merged", detail="marker present")
        status, _ = result_to_wiring_status(
            TextPatchResult.SKIPPED, f,
            applied_message="lie that file was modified",
            patch_name="X",
        )
        assert status == "skipped"
