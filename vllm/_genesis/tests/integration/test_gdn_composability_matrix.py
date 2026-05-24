# SPDX-License-Identifier: Apache-2.0
"""Variant D Phase 3 — GDN composability matrix tests.

Static composability validation: for every pair of active GDN patches
in a typical Genesis configuration, verify either:
  (a) explicit conflict declared (apply check enforces mutex), OR
  (b) different code sites (truly orthogonal), OR
  (c) shared site WITH explicit composes_with declaration

This catches silent regressions when a new GDN patch is added without
updating the composability registry.

PN59 (Variant D Phase 2) is the focus: must compose cleanly with PN50
(proj input fusion), PN54 (ssm_state dedup), PN29 (chunk_o scale-fold),
PN30 (DS conv), PN11 (interleaved branch).
"""
from __future__ import annotations

import pytest

from vllm._genesis.utils.gdn_composability import (
    GdnSite,
    all_gdn_patches,
    find_composability_warnings,
    get_profile,
    has_site_overlap,
    is_explicit_conflict,
)


# ─── Profile sanity ─────────────────────────────────────────────────────────


def test_pn59_profile_complete():
    """PN59 must be registered with chunk_orchestrator + scratch_pool sites."""
    p = get_profile("PN59")
    assert p is not None
    assert GdnSite.CHUNK_ORCHESTRATOR in p.sites
    assert GdnSite.SCRATCH_POOL in p.sites


def test_p103_explicitly_conflicts_with_pn59():
    """P103 outer wrap is superseded by PN59. Mutex must be declared."""
    assert is_explicit_conflict("P103", "PN59")


def test_pn50_pn59_orthogonal():
    """PN50 (proj input) + PN59 (chunk orchestrator) — different sites."""
    assert not has_site_overlap("PN50", "PN59")
    assert not is_explicit_conflict("PN50", "PN59")


def test_pn54_pn59_orthogonal():
    """PN54 (ssm_state dedup) + PN59 (chunk orchestrator) — different sites."""
    assert not has_site_overlap("PN54", "PN59")
    assert not is_explicit_conflict("PN54", "PN59")


def test_pn29_pn59_orthogonal():
    """PN29 (chunk_o kernel arithmetic) + PN59 (orchestrator) — different sites."""
    assert not has_site_overlap("PN29", "PN59")
    assert not is_explicit_conflict("PN29", "PN59")


def test_pn30_pn59_orthogonal():
    """PN30 (DS conv) + PN59 (chunk orchestrator) — different files."""
    assert not has_site_overlap("PN30", "PN59")


def test_pn11_pn59_orthogonal():
    """PN11 (interleaved branch only) + PN59 (non-interleaved) — disjoint."""
    assert not has_site_overlap("PN11", "PN59")


def test_pn26b_p67_not_gdn_at_all():
    """PN26b sparse-V + P67 multi-query — non-GDN paths, empty sites."""
    pn26b = get_profile("PN26b")
    p67 = get_profile("P67")
    assert pn26b is not None and len(pn26b.sites) == 0
    assert p67 is not None and len(p67.sites) == 0


# ─── Genesis 27B Lorbus typical-configuration matrix ───────────────────────


GENESIS_27B_TYPICAL_GDN_STACK = {
    "PN50",   # GDN proj fusion
    "PN54",   # contiguous dedup
    "PN59",   # streaming GDN (Variant D)
    "PN29",   # chunk_o scale-fold
    "PN30",   # DS conv
    "PN11",   # AB contiguous
    "PN32",   # GDN chunked-prefill
}


def test_typical_stack_no_explicit_conflicts():
    """27B Lorbus typical GDN stack — no patch pair declares conflict."""
    patches = list(GENESIS_27B_TYPICAL_GDN_STACK)
    for i, a in enumerate(patches):
        for b in patches[i + 1:]:
            assert not is_explicit_conflict(a, b), (
                f"Conflict in typical stack: {a} <-> {b} — "
                "must be resolved before deployment"
            )


def test_p103_excluded_from_typical_stack():
    """If user mistakenly enables P103 alongside PN59, find_warnings catches it."""
    stack = GENESIS_27B_TYPICAL_GDN_STACK | {"P103"}
    warnings = find_composability_warnings(stack)
    assert any("P103" in w and "PN59" in w for w in warnings), (
        "find_composability_warnings must catch P103+PN59 conflict"
    )


def test_typical_stack_no_unexpected_warnings():
    """Typical 27B stack should produce 0 explicit-conflict warnings.
    Site-overlap warnings (info level) are acceptable."""
    warnings = find_composability_warnings(GENESIS_27B_TYPICAL_GDN_STACK)
    explicit_conflicts = [w for w in warnings if "explicit conflict" in w]
    assert explicit_conflicts == [], (
        f"Unexpected conflicts in typical stack: {explicit_conflicts}"
    )


# ─── Registry completeness ────────────────────────────────────────────────


def test_every_gdn_patch_has_description():
    """Each profile must have a non-empty description."""
    for pid in all_gdn_patches():
        p = get_profile(pid)
        assert p is not None
        assert p.description, f"{pid} missing description"
        assert len(p.description) > 30, f"{pid} description too terse"


def test_no_self_conflict():
    """Sanity: no profile declares itself as conflict."""
    for pid in all_gdn_patches():
        p = get_profile(pid)
        assert p is not None
        assert pid not in p.conflicts_with, f"{pid} self-conflict"


# ─── Phase 3 readiness gate ────────────────────────────────────────────────


def test_phase3_pn59_ready_for_composability_validation():
    """Final gate: PN59 declared, all 27B stack patches profiled, no
    explicit conflicts in typical stack. PROD-A/B can proceed."""
    pn59 = get_profile("PN59")
    assert pn59 is not None
    for pid in GENESIS_27B_TYPICAL_GDN_STACK:
        p = get_profile(pid)
        assert p is not None, f"Patch {pid} missing from composability registry"
    # No conflicts in typical stack
    warnings = find_composability_warnings(GENESIS_27B_TYPICAL_GDN_STACK)
    explicit = [w for w in warnings if "explicit conflict" in w]
    assert not explicit, f"Phase 3 NOT ready: {explicit}"
