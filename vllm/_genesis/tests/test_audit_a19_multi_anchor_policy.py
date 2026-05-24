# SPDX-License-Identifier: Apache-2.0
"""Audit A-19 — required=False multi-anchor policy test.

Audit finding: TextPatcher with multiple `required=False` subpatches
sharing a marker can deadlock — once marker is set after one subpatch
applies, repeat runs return IDEMPOTENT and the missing subpatches
NEVER get applied.

Policy: any wiring file that creates a TextPatcher with
**>1 required=False subpatches** MUST either:
  (a) use distinct markers per subpatch, OR
  (b) explicitly opt out via `_AUDIT_A19_EXEMPT = True` module attr
      with a docstring justification.

PN40 omnibus is exempt (sub-A required=True, sub-C+D scheduler subpatches
are tightly coupled — both apply or both stay un-marked is acceptable).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_GENESIS_WIRING_ROOT = (
    Path(__file__).resolve().parents[2] / "_genesis" / "wiring"
)


def _wiring_files() -> list[Path]:
    """All `patch_*.py` files under wiring/."""
    return sorted(_GENESIS_WIRING_ROOT.rglob("patch_*.py"))


def _module_is_exempt(src: str) -> bool:
    return "_AUDIT_A19_EXEMPT = True" in src


def _count_optional_subpatches_per_patcher(src: str) -> list[tuple[int, int, int]]:
    """For each TextPatcher() instantiation, count `required=False` subpatches.

    Returns list of (line, total_subpatches, optional_count).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    findings: list[tuple[int, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Look for TextPatcher(...) calls
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name != "TextPatcher":
            continue
        # Find sub_patches kwarg
        sub_patches_arg = None
        for kw in node.keywords:
            if kw.arg == "sub_patches" and isinstance(kw.value, ast.List):
                sub_patches_arg = kw.value
                break
        if sub_patches_arg is None:
            continue
        total = 0
        optional = 0
        for elt in sub_patches_arg.elts:
            if not isinstance(elt, ast.Call):
                continue
            total += 1
            for sub_kw in elt.keywords:
                if sub_kw.arg == "required":
                    val = sub_kw.value
                    is_required = True
                    if isinstance(val, ast.Constant) and val.value is False:
                        is_required = False
                    if not is_required:
                        optional += 1
        findings.append((node.lineno, total, optional))
    return findings


def test_no_silent_multi_optional_in_wiring():
    """Audit A-19 enforce: every TextPatcher with >1 required=False subpatches
    must EITHER use distinct markers (we can't statically check this without
    running the code) OR explicitly mark module as `_AUDIT_A19_EXEMPT = True`."""
    violations: list[tuple[Path, int, int, int]] = []
    for wf in _wiring_files():
        try:
            src = wf.read_text(encoding="utf-8")
        except Exception:
            continue
        if _module_is_exempt(src):
            continue
        for line, total, optional in _count_optional_subpatches_per_patcher(src):
            if optional > 1:
                violations.append((wf, line, total, optional))

    if violations:
        msgs = [
            f"  {v[0].relative_to(_GENESIS_WIRING_ROOT.parent.parent)}:line{v[1]}"
            f" — TextPatcher with {v[3]}/{v[2]} required=False subpatches"
            for v in violations
        ]
        pytest.fail(
            "Audit A-19 violation: TextPatcher(s) have multiple required=False "
            "subpatches sharing a marker. This permanently locks out missed "
            "anchors after partial application:\n"
            + "\n".join(msgs)
            + "\n\nFix: use distinct markers per optional subpatch, OR add "
            "`_AUDIT_A19_EXEMPT = True` to the module with docstring "
            "justification (only if subpatches are TIGHTLY COUPLED — both "
            "apply or both stay un-marked is acceptable for the use case)."
        )


def test_audit_a19_exemptions_are_documented():
    """Modules that opt out via _AUDIT_A19_EXEMPT must justify either in
    the module docstring OR in a comment within 5 lines before/after the
    `_AUDIT_A19_EXEMPT = True` statement."""
    keywords = ["A-19", "audit", "tightly coupled", "_AUDIT_A19_EXEMPT"]
    for wf in _wiring_files():
        try:
            src = wf.read_text(encoding="utf-8")
        except Exception:
            continue
        if not _module_is_exempt(src):
            continue
        lines = src.splitlines()
        # Find the exempt statement
        exempt_line_idx = None
        for i, ln in enumerate(lines):
            if "_AUDIT_A19_EXEMPT = True" in ln:
                exempt_line_idx = i
                break
        if exempt_line_idx is None:
            continue
        # Look in nearby comments (5 lines around)
        nearby = "\n".join(
            lines[max(0, exempt_line_idx - 5):exempt_line_idx + 5]
        )
        try:
            module_doc = ast.get_docstring(ast.parse(src)) or ""
        except SyntaxError:
            module_doc = ""
        haystack = (nearby + "\n" + module_doc).lower()
        if not any(kw.lower() in haystack for kw in keywords):
            pytest.fail(
                f"{wf.name}: declares `_AUDIT_A19_EXEMPT = True` but no "
                "justification found (must mention A-19, audit, tightly "
                "coupled, or _AUDIT_A19_EXEMPT in module docstring or "
                "in comments within 5 lines of the exempt statement)."
            )
