# SPDX-License-Identifier: Apache-2.0
"""Pin the PATCHES.md ↔ PATCH_REGISTRY sync contract.

`PATCHES.md` is advertised as the **single source of truth** for every
Genesis runtime patch. The dispatcher's `PATCH_REGISTRY` is what
actually runs at boot. These two have drifted in the past — at one
point PATCHES.md was missing 9 entries that existed in the registry —
because there was no automated check.

This test catches that class of drift on every push/PR: if a
contributor adds a patch to `PATCH_REGISTRY` but forgets to document
it in `PATCHES.md`, this fails with a clean error message naming the
missing IDs.

The reverse direction (PATCHES.md mentions an ID that's not in the
registry) is intentionally NOT enforced — PATCHES.md still documents
the legacy P1-P55 set that lives in `apply_all.py` as a dry-run
diagnostic, not in the registry.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PATCHES_MD = REPO_ROOT / "docs" / "PATCHES.md"


@pytest.fixture(scope="module")
def patches_md_text() -> str:
    if not PATCHES_MD.is_file():
        pytest.skip(f"PATCHES.md not present at {PATCHES_MD}")
    return PATCHES_MD.read_text(encoding="utf-8")


def test_every_registry_entry_documented(patches_md_text: str):
    """Every PATCH_REGISTRY ID must be referenced somewhere in
    PATCHES.md — title, table row, callout, anything."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    missing: list[str] = []
    for pid in sorted(PATCH_REGISTRY):
        # \b doesn't match across letter↔digit boundaries the way we
        # want for things like "P67" vs "P67b" — use explicit
        # non-alphanumeric / start-of-line / end-of-line boundaries.
        pattern = (
            r"(?:^|[^A-Za-z0-9])"
            + re.escape(pid)
            + r"(?:[^A-Za-z0-9]|$)"
        )
        if not re.search(pattern, patches_md_text):
            missing.append(pid)

    assert not missing, (
        f"PATCHES.md is missing entries for {len(missing)} "
        f"PATCH_REGISTRY ID(s): {missing}\n"
        "Add a row in the appropriate category table at PATCHES.md, or "
        "if the registry entry is intentionally undocumented, explain "
        "why in this test's docstring."
    )


def test_total_count_in_header_matches_registry(patches_md_text: str):
    """The 'Total PATCH_REGISTRY entries: N' line at the top of
    PATCHES.md must match `len(PATCH_REGISTRY)`. This catches the
    drift class where someone adds a patch + a row but forgets to
    bump the headline number."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    # Match e.g. "**Total PATCH_REGISTRY entries:** 48"
    m = re.search(
        r"\*\*Total PATCH_REGISTRY entries:\*\*\s*(\d+)",
        patches_md_text,
    )
    assert m is not None, (
        "PATCHES.md header must contain a "
        "'**Total PATCH_REGISTRY entries:** <N>' line so operators can "
        "see at a glance how many patches the dispatcher carries"
    )
    declared = int(m.group(1))
    actual = len(PATCH_REGISTRY)
    assert declared == actual, (
        f"PATCHES.md header declares {declared} registry entries but "
        f"len(PATCH_REGISTRY) == {actual}. Update the headline number."
    )
