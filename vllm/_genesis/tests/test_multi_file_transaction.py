# SPDX-License-Identifier: Apache-2.0
"""TDD for MultiFilePatchTransaction (audit A-03/A-04/A-05 fix).

Verifies two-phase commit semantics:
  1. dry-run validates all anchors WITHOUT writing
  2. commit applies all-or-(skip with no writes)
  3. mid-commit failure reports partial state with clear message
  4. idempotency: re-running on already-applied set returns applied
  5. file missing = atomic skip (zero writes)
"""
from __future__ import annotations

import pytest


def _make_synthetic_patcher(tmp_path, fname, anchor, replacement, marker, required=True):
    """Helper: create file + return TextPatcher targeting it."""
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher,
    )
    target = tmp_path / fname
    target.write_text("# header\n" + anchor + "\n# footer\n")
    return TextPatcher(
        patch_name=f"test {fname}",
        target_file=str(target),
        marker=marker,
        sub_patches=[TextPatch(name="sp", anchor=anchor,
                                replacement=replacement, required=required)],
    )


def test_all_files_apply_atomically(tmp_path):
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction
    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    p2 = _make_synthetic_patcher(tmp_path, "f2.py", "BBB", "BBB_FIXED", "MARKER2")
    p3 = _make_synthetic_patcher(tmp_path, "f3.py", "CCC", "CCC_FIXED", "MARKER3")
    txn = MultiFilePatchTransaction([p1, p2, p3], name="ATOMIC")
    status, reason = txn.apply_or_skip()
    assert status == "applied", f"unexpected: {status} / {reason}"
    assert "3/3 files committed" in reason
    # Verify each file modified
    for p in (p1, p2, p3):
        assert p.marker in open(p.target_file).read()


def test_dry_run_skip_when_anchor_missing_in_one_file(tmp_path):
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction
    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    p2 = _make_synthetic_patcher(tmp_path, "f2.py", "BBB", "BBB_FIXED", "MARKER2")
    # Sabotage: write file 2 with no anchor
    open(p2.target_file, "w").write("# no anchor here\n")
    txn = MultiFilePatchTransaction([p1, p2], name="ATOMIC_SKIP")
    status, reason = txn.apply_or_skip()
    assert status == "skipped", f"unexpected: {status} / {reason}"
    assert "dry-run failed" in reason
    assert "anchor" in reason.lower()
    # CRITICAL: file 1 must remain UNMODIFIED (atomic skip, no writes)
    assert "MARKER1" not in open(p1.target_file).read(), (
        "atomic violation: file 1 was written despite dry-run failure"
    )


def test_idempotency_already_applied(tmp_path):
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction
    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    p2 = _make_synthetic_patcher(tmp_path, "f2.py", "BBB", "BBB_FIXED", "MARKER2")
    txn1 = MultiFilePatchTransaction([p1, p2], name="IDEM")
    s1, _ = txn1.apply_or_skip()
    assert s1 == "applied"
    # Re-run — markers present → dry-run treats as already-applied
    txn2 = MultiFilePatchTransaction([p1, p2], name="IDEM")
    s2, r2 = txn2.apply_or_skip()
    assert s2 == "applied", f"second apply not idempotent: {s2} / {r2}"


def test_missing_patcher_means_skip(tmp_path):
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction
    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    txn = MultiFilePatchTransaction([p1, None], name="ATOMIC_NONE")
    status, reason = txn.apply_or_skip()
    assert status == "skipped"
    assert "patcher is None" in reason
    # File 1 untouched
    assert "MARKER1" not in open(p1.target_file).read()


def test_missing_target_file_means_skip(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        MultiFilePatchTransaction, TextPatch, TextPatcher,
    )
    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    # Patcher pointing at nonexistent file
    p2 = TextPatcher(
        patch_name="missing",
        target_file=str(tmp_path / "does_not_exist.py"),
        marker="MARKER2",
        sub_patches=[TextPatch(name="sp", anchor="X", replacement="Y", required=True)],
    )
    txn = MultiFilePatchTransaction([p1, p2], name="ATOMIC_FILE_MISSING")
    status, reason = txn.apply_or_skip()
    assert status == "skipped"
    assert "missing" in reason.lower() or "file" in reason.lower()
    # File 1 untouched (atomicity)
    assert "MARKER1" not in open(p1.target_file).read()


def test_optional_subpatch_doesnt_block_dry_run(tmp_path):
    """If a sub-patch is required=False and its anchor missing,
    dry-run still passes (only required sub-patches gate)."""
    from vllm._genesis.wiring.text_patch import (
        MultiFilePatchTransaction, TextPatch, TextPatcher,
    )
    target = tmp_path / "f.py"
    target.write_text("AAA\n# no BBB\n")
    p = TextPatcher(
        patch_name="optional",
        target_file=str(target),
        marker="MARKER",
        sub_patches=[
            TextPatch(name="req", anchor="AAA", replacement="AAA_FIXED",
                       required=True),
            TextPatch(name="opt", anchor="BBB", replacement="BBB_FIXED",
                       required=False),
        ],
    )
    txn = MultiFilePatchTransaction([p], name="OPT")
    status, _ = txn.apply_or_skip()
    assert status == "applied"  # required anchor was found, optional skipped


# ─── G-POST-08: TRUE rollback on phase-2 race ────────────────────────────
#
# Audit `genesis_post_fix_rescan_audit_2026-05-05` G-POST-08 flagged that
# `_failed_with_partial` only LOGGED — partial commits stayed on disk.
# These tests pin the new byte-for-byte rollback contract.


def test_phase2_race_rollback_restores_files(tmp_path, monkeypatch):
    """File 0 commits OK; file 1 raises during apply — file 0 must be
    restored to its pre-transaction byte content."""
    from vllm._genesis.wiring import text_patch as tp
    from vllm._genesis.wiring.text_patch import (
        MultiFilePatchTransaction, TextPatchResult,
    )

    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    p2 = _make_synthetic_patcher(tmp_path, "f2.py", "BBB", "BBB_FIXED", "MARKER2")
    pre_f1 = open(p1.target_file).read()

    # Sabotage: monkey-patch p2.apply() to raise AFTER p1 has committed
    def _boom(*a, **kw):
        raise RuntimeError("simulated phase-2 race")
    monkeypatch.setattr(p2, "apply", _boom)

    txn = MultiFilePatchTransaction([p1, p2], name="RACE_ROLLBACK")
    status, reason = txn.apply_or_skip()
    assert status == "failed"
    assert "ROLLED BACK" in reason
    assert "simulated phase-2 race" in reason
    # CRITICAL: f1 must be restored byte-for-byte
    post_f1 = open(p1.target_file).read()
    assert post_f1 == pre_f1, (
        f"rollback failed: f1 still mutated.\nPRE:  {pre_f1!r}\nPOST: {post_f1!r}"
    )
    assert "MARKER1" not in post_f1


def test_phase2_race_rollback_skipped_does_not_unapply_idempotent(tmp_path):
    """Files that resolve to IDEMPOTENT (already had marker BEFORE this
    transaction) must NOT be restored — restoring would unapply prior
    state belonging to a different transaction."""
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction

    # First transaction applies p_already
    p_already = _make_synthetic_patcher(
        tmp_path, "fA.py", "AAA", "AAA_FIXED", "PRIOR_MARKER",
    )
    MultiFilePatchTransaction([p_already], name="PRIOR").apply_or_skip()
    assert "PRIOR_MARKER" in open(p_already.target_file).read()
    pre_fA = open(p_already.target_file).read()

    # Second transaction reuses p_already (will be IDEMPOTENT) plus p_new
    # plus p_fail (anchor missing — fails phase 2)
    p_new = _make_synthetic_patcher(
        tmp_path, "fN.py", "BBB", "BBB_FIXED", "NEW_MARKER",
    )
    p_fail = _make_synthetic_patcher(
        tmp_path, "fF.py", "ZZZ", "ZZZ_FIXED", "FAIL_MARKER",
    )
    # Sabotage AFTER snapshot but before commit by removing anchor from fF
    # post dry-run. We do this via simulating the race: replace anchor with
    # something else after phase 1 passes.
    pre_fF = open(p_fail.target_file).read()

    # Simulate "phase-2 race" by monkey-patching apply on the SPECIFIC
    # instance (not the class — class-level patch would clobber p_already
    # and p_new too, which use the same TextPatcher class).
    from vllm._genesis.wiring.text_patch import TextPatchResult, TextPatchFailure
    def _race_fail(*a, **kw):
        return TextPatchResult.SKIPPED, TextPatchFailure(
            reason="anchor not found (race)", detail="post-dry-run mutation",
        )
    p_fail.apply = _race_fail  # bind on instance
    txn = MultiFilePatchTransaction(
        [p_already, p_new, p_fail], name="MIXED_RACE",
    )
    status, reason = txn.apply_or_skip()

    assert status == "failed"
    # IDEMPOTENT file unchanged
    assert open(p_already.target_file).read() == pre_fA
    # NEW file (truly applied this txn) MUST be rolled back
    fN_post = open(p_new.target_file).read()
    assert "NEW_MARKER" not in fN_post, (
        "p_new was applied this txn and must be rolled back"
    )
    # Confirm rollback summary mentions the right file
    assert "ROLLED BACK" in reason
    assert p_new.target_file in reason


def test_rollback_aid_written_when_restore_fails(tmp_path, monkeypatch):
    """If restore-write itself fails (filesystem error), a `.genesis_rollback`
    sibling file must be written as manual recovery aid."""
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction

    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    p2 = _make_synthetic_patcher(tmp_path, "f2.py", "BBB", "BBB_FIXED", "MARKER2")

    def _boom(*a, **kw):
        raise RuntimeError("phase-2 race")
    monkeypatch.setattr(p2, "apply", _boom)

    # Make the restore fail by intercepting os.replace for this test.
    import os
    real_replace = os.replace
    def _failing_replace(src, dst):
        if str(dst) == p1.target_file:
            raise PermissionError("simulated restore failure")
        return real_replace(src, dst)
    monkeypatch.setattr(os, "replace", _failing_replace)

    txn = MultiFilePatchTransaction([p1, p2], name="ROLLBACK_AID")
    status, reason = txn.apply_or_skip()
    assert status == "failed"
    assert "MANUAL RECOVERY NEEDED" in reason
    # Aid file must exist next to f1
    assert os.path.isfile(p1.target_file + ".genesis_rollback")


def test_no_rollback_when_first_file_fails(tmp_path, monkeypatch):
    """If the very first commit-phase apply fails, no files were APPLIED
    yet — rollback summary should say no files needed restoring."""
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction

    p1 = _make_synthetic_patcher(tmp_path, "f1.py", "AAA", "AAA_FIXED", "MARKER1")
    p2 = _make_synthetic_patcher(tmp_path, "f2.py", "BBB", "BBB_FIXED", "MARKER2")
    pre_f1 = open(p1.target_file).read()
    pre_f2 = open(p2.target_file).read()

    def _boom(*a, **kw):
        raise RuntimeError("first-file failure")
    monkeypatch.setattr(p1, "apply", _boom)

    txn = MultiFilePatchTransaction([p1, p2], name="FIRST_FAIL")
    status, reason = txn.apply_or_skip()
    assert status == "failed"
    assert "no files needed rollback" in reason
    # Both files genuinely untouched
    assert open(p1.target_file).read() == pre_f1
    assert open(p2.target_file).read() == pre_f2
