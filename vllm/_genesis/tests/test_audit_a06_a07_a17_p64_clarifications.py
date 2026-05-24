# SPDX-License-Identifier: Apache-2.0
"""Audit A-06 / A-07 / A-17 — P64 documentation + intentional-deferral test.

Background (audit 2026-05-05):
  A-06: P64 docstring обещает "3 sub-patches" but Pydantic/null fix
        (`_create_remaining_args_delta` → SERVING_CRD_NEW) is NOT in
        sub_patches list.
  A-07: P64 QWEN3COD_FNEND_NEW removes `self.json_closed = True` line
        from old branch — looks like state-machine bug.
  A-17: SERVING_CRD_OLD/NEW dead code constants.

Findings after deep analysis:
  A-06: docstring count is stale (says "3" but real is "4 wired + 1
        deferred"); fix → update docstring.
  A-07: NOT a runtime bug — `self.json_closed = True` is set upstream
        at parser line 626 inside `if not self.json_closed and ...`
        BEFORE our P64 anchor's branch executes. Fix → add explanatory
        comment in P64 replacement.
  A-17: NOT dead code — intentionally deferred per design comment in
        `_make_serving_patcher`. Fix → add explicit AUDIT A-17 marker
        in constants so future readers don't misread as forgotten code.

These tests gate against silent regression of the documentation/comments
that explain the design intent.
"""
from __future__ import annotations

import re

import pytest


def _wiring():
    from vllm._genesis.wiring.structured_output import (
        patch_64_qwen3coder_mtp_streaming as M,
    )
    return M


def _wiring_source() -> str:
    from vllm._genesis.wiring.structured_output import (
        patch_64_qwen3coder_mtp_streaming,
    )
    return open(patch_64_qwen3coder_mtp_streaming.__file__).read()


# ─── A-06 / A-17 ─────────────────────────────────────────────────────────────


def test_a06_docstring_count_matches_actual_subpatches():
    """A-06: module docstring sub-patch count must reflect reality."""
    src = _wiring_source()
    # Count actual `TextPatch(name=...)` instantiations in sub_patches lists
    actual_subpatches = src.count('TextPatch(name="p64_')
    # Docstring should mention this count (or a range that covers it)
    # Acceptable: "4 sub-patches", "4-5 sub-patches", "4 wired + 1 deferred"
    docstring_match = re.search(
        r'(\d+)\s+(?:wired\s+)?sub-patch', src[:2000], re.IGNORECASE
    )
    if docstring_match:
        claimed = int(docstring_match.group(1))
        assert claimed == actual_subpatches, (
            f"A-06 violation: docstring claims {claimed} sub-patches but "
            f"code has {actual_subpatches} wired. Update docstring."
        )


def test_a17_serving_crd_constants_marked_intentionally_deferred():
    """A-17: SERVING_CRD_OLD/NEW must have explicit deferral comment so
    future readers don't think they're dead code. Window is 1200 chars
    before/after the constant — comments may be a paragraph long."""
    src = _wiring_source()
    crd_idx = src.find("SERVING_CRD_OLD")
    assert crd_idx > 0, "SERVING_CRD_OLD constant must exist"
    # Wider window — comment paragraph may be 800+ chars
    context = src[max(0, crd_idx - 1200):crd_idx + 2000]
    assert "AUDIT A-17" in context or "audit A-17" in context, (
        "A-17 violation: SERVING_CRD_OLD/NEW constants are defined but "
        "not in sub_patches. Add a clear `# AUDIT A-17: intentionally "
        "deferred — see _make_serving_patcher comment` near the constants."
    )


def test_a17_make_serving_patcher_has_deferral_explanation():
    """A-17: _make_serving_patcher must contain comment explaining why
    SERVING_CRD sub-patch is omitted."""
    src = _wiring_source()
    fn_idx = src.find("def _make_serving_patcher")
    assert fn_idx > 0
    # Look at function body (next 2000 chars)
    fn_body = src[fn_idx:fn_idx + 2000]
    assert "_create_remaining_args_delta" in fn_body, (
        "A-17: _make_serving_patcher must reference the deferred function name"
    )
    keywords = ["belt-and-braces", "deferred", "leave", "unchanged", "primary symptom"]
    assert any(kw in fn_body for kw in keywords), (
        "A-17: _make_serving_patcher must explain WHY CRD sub-patch is omitted"
    )


# ─── A-07 ────────────────────────────────────────────────────────────────────


def test_a07_qwen3cod_fnend_new_documents_json_closed_invariant():
    """A-07: QWEN3COD_FNEND_NEW removes `self.json_closed = True` line.
    NOT a bug (set upstream at parser line ~626), but readers need a
    clear comment explaining why the line is missing here."""
    M = _wiring()
    new = M.QWEN3COD_FNEND_NEW
    # Must explicitly mention json_closed for traceability
    assert "json_closed" in new, (
        "A-07 violation: QWEN3COD_FNEND_NEW removes json_closed line from "
        "OLD branch but doesn't reference it in a comment. Add: "
        "`# json_closed=True is set at parser line ~626 (top of branch); "
        "we don't repeat here.`"
    )


def test_a07_qwen3cod_fnend_new_explains_upstream_set():
    """A-07: comment must clarify upstream sets json_closed."""
    M = _wiring()
    new = M.QWEN3COD_FNEND_NEW
    keywords = ["upstream", "above", "already", "top of", "set at"]
    assert any(kw in new.lower() for kw in keywords), (
        "A-07: QWEN3COD_FNEND_NEW comment must clarify json_closed is "
        "set BY UPSTREAM ABOVE this anchor (at `if not self.json_closed:`)"
    )
