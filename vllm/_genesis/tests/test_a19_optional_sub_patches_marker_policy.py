# SPDX-License-Identifier: Apache-2.0
"""TDD for A-19 — forbid `required=False` sub-patches sharing a TextPatcher marker.

Audit P1 finding (genesis_deep_cross_audit_2026-05-05, P1.5):
A `TextPatcher` with one shared `marker` field hosting MULTIPLE
`required=False` sub-patches has a partial-apply trap:
  1. Boot 1: anchor for sub-patch A is found, anchor B is missing → A
     applies, B silently skipped, marker is written.
  2. Boot 2: marker present → TextPatcher returns idempotent → B is
     never re-attempted, even after upstream brought B's anchor back.

Fix landed for PN40 (split into separate markers per subpatch). This
test forbids the pattern repeating elsewhere via static AST scan.

Allowlist: only legitimate cases (e.g. all-required sub-patches sharing
a marker is fine — they apply atomically). Add new patches here ONLY
after a senior review confirms the partial-apply trap is impossible.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

WIRING_DIR = (
    Path(__file__).parent.parent / "wiring"
).resolve()


def _collect_text_patcher_calls() -> list[tuple[Path, ast.Call]]:
    """Walk the wiring tree, parse each .py, return all TextPatcher() calls."""
    calls: list[tuple[Path, ast.Call]] = []
    for py_file in WIRING_DIR.rglob("*.py"):
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            target_name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name)
                else None
            )
            if target_name == "TextPatcher":
                calls.append((py_file, node))
    return calls


def _extract_sub_patches(call: ast.Call) -> list[ast.Call]:
    """Extract the list of TextPatch() inner calls from a TextPatcher() call."""
    for kw in call.keywords:
        if kw.arg != "sub_patches":
            continue
        if not isinstance(kw.value, ast.List):
            return []
        return [
            elt for elt in kw.value.elts
            if isinstance(elt, ast.Call)
        ]
    return []


def _is_required_false(text_patch_call: ast.Call) -> bool:
    """Inspect TextPatch(required=...) to determine optional-vs-required."""
    for kw in text_patch_call.keywords:
        if kw.arg == "required" and isinstance(kw.value, ast.Constant):
            return kw.value.value is False
    return False  # default required=True per TextPatch dataclass


# Patches explicitly cleared via review (graceful-degradation pattern
# where partial apply is INTENTIONAL — opposite of PN40 which promised
# all-or-nothing). Add here ONLY with reason.
_KNOWN_SAFE_OPTIONAL_SHARING = frozenset({
    # Legacy MoE tuning — multi-arch fallback set; any subset of arch-
    # specific TextPatch hits is fine, no all-or-nothing contract.
    "legacy/patch_24_moe_tune.py",
    # Qwen3 BEFORE-THINK fallback (P27) — three independent fallback
    # paths (non-stream capture / non-stream return / stream start).
    # Any subset apply is graceful degradation by design.
    "legacy/patch_27_reasoning_before_think.py",
    # P59 — empirically DISPROVEN backport of vllm#39055; deprecated
    # opt-in research artifact. partial-apply has zero PROD impact.
    "structured_output/patch_59_qwen3_reasoning_tool_call_recovery.py",
    # P83 — empirically DISPROVEN MTP keep-last-cached-block; kept as
    # opt-in research artifact for a workload we don't trigger.
    "spec_decode/patch_83_mtp_keep_last_cached_block.py",
})


def test_no_optional_subpatches_share_marker():
    """Static AST gate: no TextPatcher with marker= AND >=2 required=False subs."""
    violations: list[str] = []
    for path, call in _collect_text_patcher_calls():
        sub_patches = _extract_sub_patches(call)
        if not sub_patches:
            continue
        optional_count = sum(_is_required_false(sp) for sp in sub_patches)
        if optional_count < 2:
            continue  # fewer than 2 optional → no partial-apply trap
        # Check rel-path is not in allowlist
        rel_str = str(path.relative_to(WIRING_DIR))
        if rel_str in _KNOWN_SAFE_OPTIONAL_SHARING:
            continue
        violations.append(
            f"{rel_str}: TextPatcher contains {optional_count} required=False "
            f"sub-patches sharing one marker. Partial-apply trap risk — "
            "split into separate TextPatcher instances with separate markers, "
            "or change required=False → required=True if all-or-nothing is "
            "the desired contract."
        )
    assert not violations, "A-19 violations:\n  " + "\n  ".join(violations)


def test_text_patcher_walk_finds_real_calls():
    """Sanity: the AST walk actually discovers TextPatcher calls."""
    calls = _collect_text_patcher_calls()
    assert len(calls) >= 5, (
        f"Expected ≥5 TextPatcher calls in wiring/, found {len(calls)}. "
        "AST walk may be broken."
    )
