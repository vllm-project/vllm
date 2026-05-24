# SPDX-License-Identifier: Apache-2.0
"""Genesis compat — patch lifecycle state machine.

Each entry in `PATCH_REGISTRY` can declare a `lifecycle` field
describing where the patch is in its useful-life curve. This serves
two main purposes:

  1. **Operator UX** — `genesis doctor` can warn on `experimental`,
     surface `superseded_by` for `deprecated`, hide `retired`.
  2. **Safety against accidental deletion** — code removal requires
     prior `lifecycle: retired` for at least one release. The
     `lifecycle-audit` tool enforces this.

States
------

  experimental — newly added, may change/break across releases. Doctor
                 emits warning if engaged.
  stable       — proven on PROD hardware + tests pass + has been
                 default-on for at least one release.
  deprecated   — superseded by another patch (see `superseded_by`).
                 Still works, but doctor recommends migration. Has a
                 `removal_planned` field with target version.
  research    — kept as reference for future hardware / config classes
                 (e.g. P57 +850 MiB capture buffers — useless on 24 GB
                 cards but valuable on H100). Doctor lists but does
                 not engage.
  community   — contributed via plugin entry-point, NOT in core repo.
                 Same engagement gate as `experimental`.
  retired     — removed from active use. Code may still exist for
                 git-blame / archeology, but `apply_all` no longer
                 attempts to apply it. Should be removed from the
                 repository in the next release after retirement.

State transitions (forward-only on the public timeline):

  experimental → stable    (after empirical validation + 1 release)
  stable       → deprecated (when superseded; doctor warns)
  deprecated   → retired   (after `removal_planned` version released)
  research     → stable    (rare — when the future scenario arrives)
  research     → retired   (when it stops being a useful reference)
  community    → stable    (if accepted upstream into core)
  community    → retired   (if abandoned)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Recognised lifecycle states — a patch with any other value triggers a
# validator warning.
KNOWN_STATES = frozenset({
    "experimental",
    "stable",
    "deprecated",
    "research",
    "community",
    "retired",
    "legacy",
})

# States that should NOT be auto-engaged even when env_flag is set
# — doctor will refuse to apply unless operator passes an extra
# `--allow-<state>` flag.
SAFETY_GATED_STATES = frozenset({"retired"})


@dataclass(frozen=True)
class LifecycleAuditEntry:
    """One row in the lifecycle audit table."""
    patch_id: str
    state: str
    note: str
    severity: str  # "ok" | "warn" | "error"


def get_state(meta: dict[str, Any]) -> str:
    """Extract the lifecycle state from a PATCH_REGISTRY entry, with
    backwards-compat fallback to the legacy `deprecated: True` flag."""
    explicit = meta.get("lifecycle")
    if explicit:
        return explicit
    if meta.get("deprecated"):
        return "deprecated"
    # Default: assume `stable` unless declared otherwise. New patches
    # SHOULD declare `experimental` until they've been observed in PROD.
    return "stable"


def audit_registry(registry: dict[str, dict[str, Any]]) -> list[LifecycleAuditEntry]:
    """Walk the registry and produce a human-readable audit for each
    patch's lifecycle state. Used by `genesis lifecycle-audit` and
    by `genesis doctor` for the lifecycle section."""
    entries: list[LifecycleAuditEntry] = []

    for pid, meta in registry.items():
        state = get_state(meta)
        if state not in KNOWN_STATES:
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state,
                note=f"unknown lifecycle state — must be one of {sorted(KNOWN_STATES)}",
                severity="error",
            ))
            continue

        if state == "experimental":
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state,
                note=meta.get("experimental_note", "experimental — may change"),
                severity="warn",
            ))
        elif state == "deprecated":
            superseded = meta.get("superseded_by") or []
            if isinstance(superseded, str):
                superseded = [superseded]
            removal = meta.get("removal_planned", "<unscheduled>")
            sup = f" — superseded by {', '.join(superseded)}" if superseded else ""
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state,
                note=f"removal planned in {removal}{sup}",
                severity="warn",
            ))
        elif state == "research":
            note = meta.get("research_note", "kept as reference / future hardware")
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state, note=note, severity="ok",
            ))
        elif state == "community":
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state,
                note=meta.get("community_credit", "community plugin"),
                severity="warn",
            ))
        elif state == "retired":
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state,
                note="retired — apply_all will not attempt this patch",
                severity="warn",
            ))
        elif state == "legacy":
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state,
                note=meta.get("credit", "pre-dispatcher patch — minimal metadata"),
                severity="ok",
            ))
        else:  # stable
            entries.append(LifecycleAuditEntry(
                patch_id=pid, state=state,
                note=meta.get("stable_since", "stable"),
                severity="ok",
            ))

    return entries


def is_engageable(meta: dict[str, Any], allow_gated: bool = False) -> tuple[bool, str]:
    """Return (engageable, reason). False for retired patches unless
    `allow_gated=True`. All other states pass."""
    state = get_state(meta)
    if state == "retired" and not allow_gated:
        return False, (
            f"lifecycle=retired — patch removed from active use. "
            f"Pass --allow-retired to engage anyway."
        )
    return True, f"lifecycle={state}"


def format_audit_table(entries: list[LifecycleAuditEntry]) -> list[str]:
    """Pretty-print the audit table as text lines (for doctor / CLI)."""
    if not entries:
        return ["(empty registry)"]
    by_state: dict[str, list[LifecycleAuditEntry]] = {}
    for e in entries:
        by_state.setdefault(e.state, []).append(e)

    lines = []
    for state in ("experimental", "stable", "deprecated", "research",
                  "community", "retired", "legacy"):
        if state not in by_state:
            continue
        ents = by_state[state]
        lines.append(f"  [{state}] ({len(ents)} patches)")
        for e in ents:
            mark = "  •" if e.severity == "ok" else "  ⚠" if e.severity == "warn" else "  ✗"
            lines.append(f"  {mark} {e.patch_id:<8} — {e.note}")

    # Unknown states (errors)
    unknown = [e for e in entries if e.state not in KNOWN_STATES]
    if unknown:
        lines.append(f"  [unknown] ({len(unknown)} patches — registry errors)")
        for e in unknown:
            lines.append(f"  ✗ {e.patch_id:<8} — {e.note}")
    return lines
