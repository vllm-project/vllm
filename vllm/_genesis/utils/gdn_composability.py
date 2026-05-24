# SPDX-License-Identifier: Apache-2.0
"""GDN-stack composability registry — Variant D Phase 3.

Tracks which Genesis patches touch the GDN forward path, what code site
each modifies, and which combinations are known-safe vs known-incompatible.

Used by `genesis doctor` to warn operators before enabling combinations
that haven't been validated.

Design principle: SINGLE SOURCE OF TRUTH for "does PN59 conflict with X?"
Replaces ad-hoc `if env_X and env_Y: ...` checks scattered across patches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GdnSite(Enum):
    """Code site within GDN forward path that a patch modifies."""
    PROJ_INPUT = "proj_input"           # in_proj_qkvz / in_proj_ba (BEFORE chunk_*)
    GDN_BODY = "gdn_body"                # gdn_linear_attn.py forward body
    CHUNK_ORCHESTRATOR = "chunk_orch"   # chunk_gated_delta_rule_fwd
    CHUNK_FWD_H = "chunk_fwd_h"          # state production
    CHUNK_FWD_O = "chunk_fwd_o"          # output computation
    SSM_STATE = "ssm_state"              # post-FLA state writeback
    OUTER_WRAP = "outer_wrap"            # wrapper above orchestrator
    SCRATCH_POOL = "scratch_pool"        # Genesis-managed buffers


@dataclass(frozen=True)
class GdnPatchProfile:
    """Profile of one Genesis patch's interaction with GDN forward path."""
    patch_id: str
    sites: frozenset[GdnSite]
    description: str
    composes_with: frozenset[str] = field(default_factory=frozenset)
    conflicts_with: frozenset[str] = field(default_factory=frozenset)


# Static registry — updated when new patches land in GDN area
_GDN_PATCH_PROFILES: dict[str, GdnPatchProfile] = {
    "PN50": GdnPatchProfile(
        patch_id="PN50",
        sites=frozenset({GdnSite.PROJ_INPUT}),
        description="Fused split/reshape/cat Triton kernel for Qwen3.5/3.6 "
                    "contiguous projection. Operates BEFORE chunk_*.",
    ),
    "PN54": GdnPatchProfile(
        patch_id="PN54",
        sites=frozenset({GdnSite.SSM_STATE, GdnSite.GDN_BODY}),
        description="GDN .contiguous() dedup. ssm_state advanced-index "
                    "removal + LoRA branch chunk halves cleanup.",
    ),
    "PN59": GdnPatchProfile(
        patch_id="PN59",
        sites=frozenset({GdnSite.CHUNK_ORCHESTRATOR, GdnSite.SCRATCH_POOL}),
        description="Streaming-GDN window-iterative driver. Replaces "
                    "fwd_h+fwd_o pair with windowed loop using "
                    "GdnScratchPool. Variant D Phase 2.",
    ),
    "P103": GdnPatchProfile(
        patch_id="P103",
        sites=frozenset({GdnSite.OUTER_WRAP}),
        description="FLA Cliff2 chunked-prefill OUTER orchestrator wrap "
                    "(splits T into sub-T blocks before calling FLA). "
                    "Superseded by PN59 streaming when PN59 enabled.",
        conflicts_with=frozenset({"PN59"}),  # PN59 supersedes
    ),
    "PN32": GdnPatchProfile(
        patch_id="PN32",
        sites=frozenset({GdnSite.GDN_BODY}),
        description="GDN chunked-prefill inside gdn_linear_attn (output "
                    "buffer reduction). Complementary to PN59 (different "
                    "code path).",
    ),
    "PN29": GdnPatchProfile(
        patch_id="PN29",
        sites=frozenset({GdnSite.CHUNK_FWD_O}),
        description="GDN chunk_o scale-fold optimization. Triton kernel "
                    "arithmetic — composable with PN59 (PN59 calls "
                    "chunk_fwd_o without modifying its kernel).",
    ),
    "PN30": GdnPatchProfile(
        patch_id="PN30",
        sites=frozenset({GdnSite.GDN_BODY}),
        description="DS conv state layout fix (causal_conv1d). Different "
                    "kernel from FLA chunk_*; no overlap with PN59.",
    ),
    "PN11": GdnPatchProfile(
        patch_id="PN11",
        sites=frozenset({GdnSite.GDN_BODY}),
        description="GDN AB tensor contiguous fix in gqa_interleaved branch "
                    "(Qwen3-Next path). PN59 operates only in non-interleaved "
                    "Qwen3.5/3.6 branch — disjoint code paths.",
    ),
    "PN26b": GdnPatchProfile(
        patch_id="PN26b",
        sites=frozenset({}),  # Non-GDN attention layers
        description="Sparse-V kernel for non-GDN attention path "
                    "(35B Qwen3MoE). Doesn't touch GDN.",
    ),
    "P67": GdnPatchProfile(
        patch_id="P67",
        sites=frozenset({}),  # Non-GDN attention (TQ multi-query)
        description="TQ multi-query verify kernel for spec-decode. "
                    "Non-GDN attention path. Doesn't touch GDN.",
    ),
}


def get_profile(patch_id: str) -> GdnPatchProfile | None:
    """Return profile for patch, or None if not GDN-related."""
    return _GDN_PATCH_PROFILES.get(patch_id)


def has_site_overlap(patch_a: str, patch_b: str) -> bool:
    """True if two patches modify the same GDN code site."""
    pa, pb = get_profile(patch_a), get_profile(patch_b)
    if pa is None or pb is None:
        return False
    return bool(pa.sites & pb.sites)


def is_explicit_conflict(patch_a: str, patch_b: str) -> bool:
    """True if a patch explicitly declares conflict with another."""
    pa = get_profile(patch_a)
    pb = get_profile(patch_b)
    if pa is None and pb is None:
        return False
    a_conflicts_b = pa is not None and patch_b in pa.conflicts_with
    b_conflicts_a = pb is not None and patch_a in pb.conflicts_with
    return a_conflicts_b or b_conflicts_a


def _is_explicitly_compatible(patch_a: str, patch_b: str) -> bool:
    """Audit P3 fix 2026-05-05: honour `composes_with` declaration.

    If pa.composes_with includes pb (or vice versa), the pair is declared
    safe — skip the site-overlap warning.
    """
    pa = get_profile(patch_a)
    pb = get_profile(patch_b)
    if pa is not None and patch_b in pa.composes_with:
        return True
    if pb is not None and patch_a in pb.composes_with:
        return True
    return False


def find_composability_warnings(active_patches: set[str]) -> list[str]:
    """Scan active patch set for composability warnings.

    Returns human-readable warning lines. Empty list = clean.
    """
    warnings: list[str] = []
    patches = sorted(active_patches)
    for i, a in enumerate(patches):
        for b in patches[i + 1:]:
            if is_explicit_conflict(a, b):
                # Audit P3 fix 2026-05-05: prefer the side that DECLARED
                # the conflict for the description (was always pa, even
                # when conflict was on pb's side).
                pa = get_profile(a)
                pb = get_profile(b)
                source = (
                    pa if (pa is not None and b in pa.conflicts_with) else pb
                )
                msg = (
                    f"⚠ {a} + {b} explicit conflict — "
                    + (source.description if source else "see registry")
                )
                warnings.append(msg)
            elif _is_explicitly_compatible(a, b):
                # composes_with explicitly allows this pair; no warning
                continue
            elif has_site_overlap(a, b):
                pa = get_profile(a)
                pb = get_profile(b)
                shared_sites = pa.sites & pb.sites if pa and pb else set()
                site_names = ", ".join(s.value for s in shared_sites)
                msg = (
                    f"ℹ {a} + {b} share GDN site(s) [{site_names}] — "
                    "verify combined behavior; both apply means careful "
                    "ordering required (see PATCH_REGISTRY)."
                )
                warnings.append(msg)
    return warnings


def all_gdn_patches() -> list[str]:
    """List of all known GDN-area patch IDs."""
    return sorted(_GDN_PATCH_PROFILES.keys())
