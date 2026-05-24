# SPDX-License-Identifier: Apache-2.0
"""Genesis schema validator — `python3 -m vllm._genesis.compat.schema_validator`.

Validates each `PATCH_REGISTRY` entry against
`schemas/patch_entry.schema.json`. Catches typos, missing required
fields, malformed shapes, and lifecycle-conditional rules (e.g. a
`deprecated` patch must declare `deprecation_note` or `superseded_by`).

Implementation choice: hand-rolled validator instead of the
`jsonschema` library. Genesis aims to stay dependency-free at runtime;
adding `jsonschema` would force every operator to install it. The
hand-rolled validator covers the specific JSON Schema features our
schema uses (required, type, enum, pattern, additionalProperties,
allOf+if+then). If the operator has `jsonschema` installed, we can
optionally upgrade to it for richer messages, but pure-stdlib path
must always work.

Exit code from the CLI:
  0 — registry is schema-clean
  1 — at least one entry failed validation

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Schema path (relative to repo root). The CLI/validator reads this lazily.
_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "patch_entry.schema.json"


@dataclass(frozen=True)
class SchemaIssue:
    """One validation failure."""
    patch_id: str
    field: str        # field path (e.g. "applies_to.is_turboquant")
    severity: str     # "ERROR" / "WARNING"
    message: str


# ─── Hand-rolled validator (covers the subset we need) ──────────────────


_KNOWN_LIFECYCLE = {"experimental", "stable", "deprecated", "research",
                    "community", "retired", "legacy"}

# Top-level fields permitted in a PATCH_REGISTRY entry.
# Mirrors the `properties` block of patch_entry.schema.json.
_KNOWN_FIELDS = {
    "title", "env_flag", "default_on", "lifecycle", "category",
    "credit", "deprecation_note", "experimental_note", "research_note",
    "community_credit", "stable_since", "since_version", "deprecated_since",
    "removal_planned", "deprecated", "superseded_by", "upstream_pr",
    "applies_to", "requires_patches", "conflicts_with", "composes_with",
    # Pin-bump retirement metadata (added 2026-05-05 alongside vllm#39931
    # P4 supersede). When `superseded_by` is set, `retire_after_pin` says
    # which vllm pin contains the upstream merge — operator can drop the
    # patch from PATCH_REGISTRY once that pin is on the allowlist.
    "retire_after_pin",
    # Some legacy entries also have these informational-only fields:
    "notes", "cross_rig_validation", "files", "files_added", "files_removed",
    "verified_in_main_2026_04_24", "verified_in_main_2026_04_29",
    "verified_in_main_2026_04_30",
    # Free-form metadata that shouldn't fail validation:
    "applies_to_combinations",
    # Plugin-extension protocol fields (Phase 5b community plugins):
    # `patch_id` is the plugin's chosen registry key; `_plugin_origin` is
    # provenance metadata stamped during entry-point discovery. Both are
    # informational and allowed but not required.
    "patch_id", "_plugin_origin",
    # Phase 5c — `apply_callable` lets a plugin RUN code, not just
    # declare metadata. Value is either a callable or a "module:func"
    # string resolved at boot via importlib.
    "apply_callable",
}


def validate_entry(patch_id: str, meta: dict[str, Any]) -> list[SchemaIssue]:
    """Validate one PATCH_REGISTRY entry. Returns list of issues
    (empty list = clean)."""
    issues: list[SchemaIssue] = []

    if not isinstance(meta, dict):
        issues.append(SchemaIssue(
            patch_id=patch_id, field=".", severity="ERROR",
            message=f"entry must be dict, got {type(meta).__name__}",
        ))
        return issues

    # Required fields (top-level)
    for req in ("title", "env_flag", "default_on"):
        if req not in meta:
            issues.append(SchemaIssue(
                patch_id=patch_id, field=req, severity="ERROR",
                message=f"missing required field {req!r}",
            ))

    # Unknown / typo'd top-level fields
    for key in meta:
        if key not in _KNOWN_FIELDS:
            issues.append(SchemaIssue(
                patch_id=patch_id, field=key, severity="ERROR",
                message=f"unknown field {key!r} (typo? "
                        f"valid: {sorted(_KNOWN_FIELDS)})",
            ))

    # Field-level type checks
    if "title" in meta and not isinstance(meta["title"], str):
        issues.append(SchemaIssue(
            patch_id=patch_id, field="title", severity="ERROR",
            message=f"title must be str, got {type(meta['title']).__name__}",
        ))
    if "default_on" in meta and not isinstance(meta["default_on"], bool):
        issues.append(SchemaIssue(
            patch_id=patch_id, field="default_on", severity="ERROR",
            message=f"default_on must be bool, got "
                    f"{type(meta['default_on']).__name__}",
        ))

    # env_flag pattern
    if "env_flag" in meta:
        if not isinstance(meta["env_flag"], str):
            issues.append(SchemaIssue(
                patch_id=patch_id, field="env_flag", severity="ERROR",
                message="env_flag must be string",
            ))
        elif not re.match(r"^GENESIS_[A-Z][A-Z0-9_]*$", meta["env_flag"]):
            issues.append(SchemaIssue(
                patch_id=patch_id, field="env_flag", severity="ERROR",
                message=f"env_flag {meta['env_flag']!r} doesn't match "
                        f"^GENESIS_[A-Z][A-Z0-9_]*$",
            ))

    # Lifecycle enum
    lc = meta.get("lifecycle")
    if lc is not None:
        if lc not in _KNOWN_LIFECYCLE:
            issues.append(SchemaIssue(
                patch_id=patch_id, field="lifecycle", severity="ERROR",
                message=f"unknown lifecycle {lc!r} "
                        f"(must be one of {sorted(_KNOWN_LIFECYCLE)})",
            ))

    # Conditional requirements per lifecycle state
    if lc == "deprecated":
        if "deprecation_note" not in meta and "superseded_by" not in meta:
            issues.append(SchemaIssue(
                patch_id=patch_id, field="lifecycle", severity="ERROR",
                message="deprecated lifecycle requires either "
                        "'deprecation_note' or 'superseded_by'",
            ))
    if lc == "research" and "research_note" not in meta:
        issues.append(SchemaIssue(
            patch_id=patch_id, field="lifecycle", severity="WARNING",
            message="research lifecycle should declare 'research_note'",
        ))
    if lc == "community" and "community_credit" not in meta:
        issues.append(SchemaIssue(
            patch_id=patch_id, field="lifecycle", severity="WARNING",
            message="community lifecycle should declare 'community_credit'",
        ))

    # upstream_pr type
    if "upstream_pr" in meta:
        v = meta["upstream_pr"]
        if v is not None and (not isinstance(v, int) or v < 1):
            issues.append(SchemaIssue(
                patch_id=patch_id, field="upstream_pr", severity="ERROR",
                message=f"upstream_pr must be positive int or null, "
                        f"got {v!r}",
            ))

    # requires_patches / conflicts_with must be list[str]
    for key in ("requires_patches", "conflicts_with"):
        if key in meta:
            v = meta[key]
            if not isinstance(v, list):
                issues.append(SchemaIssue(
                    patch_id=patch_id, field=key, severity="ERROR",
                    message=f"{key} must be list, got "
                            f"{type(v).__name__}",
                ))
            else:
                for i, item in enumerate(v):
                    if not isinstance(item, str):
                        issues.append(SchemaIssue(
                            patch_id=patch_id, field=f"{key}[{i}]",
                            severity="ERROR",
                            message=f"{key}[{i}] must be str, got "
                                    f"{type(item).__name__}",
                        ))

    # superseded_by — accept str OR list[str]
    if "superseded_by" in meta:
        v = meta["superseded_by"]
        if not isinstance(v, str) and not (
            isinstance(v, list) and all(isinstance(x, str) for x in v)
        ):
            issues.append(SchemaIssue(
                patch_id=patch_id, field="superseded_by", severity="ERROR",
                message="superseded_by must be str or list[str]",
            ))

    # applies_to must be dict
    if "applies_to" in meta and not isinstance(meta["applies_to"], dict):
        issues.append(SchemaIssue(
            patch_id=patch_id, field="applies_to", severity="ERROR",
            message="applies_to must be dict (legacy flat or compound)",
        ))

    return issues


def validate_registry(
    registry: dict[str, dict[str, Any]],
) -> list[SchemaIssue]:
    """Validate every entry in a PATCH_REGISTRY-shaped dict."""
    issues: list[SchemaIssue] = []
    for pid, meta in registry.items():
        issues.extend(validate_entry(pid, meta))
    return issues


def load_schema() -> dict[str, Any]:
    """Read schemas/patch_entry.schema.json. Useful for IDE / external tooling."""
    if not _SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"schema not found at {_SCHEMA_PATH} — "
            "did you move it without updating the path?"
        )
    return json.loads(_SCHEMA_PATH.read_text())


# ─── CLI ─────────────────────────────────────────────────────────────────


def _format_issues(issues: list[SchemaIssue]) -> list[str]:
    if not issues:
        return ["✓ PATCH_REGISTRY schema clean (no issues)"]
    lines = []
    by_patch: dict[str, list[SchemaIssue]] = {}
    for i in issues:
        by_patch.setdefault(i.patch_id, []).append(i)
    for pid, ents in sorted(by_patch.items()):
        lines.append(f"  {pid}:")
        for i in ents:
            mark = "✗" if i.severity == "ERROR" else "⚠"
            lines.append(f"    {mark} [{i.severity}] {i.field}: {i.message}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.schema_validator",
        description="Validate PATCH_REGISTRY against "
                    "schemas/patch_entry.schema.json",
    )
    parser.add_argument("--json", action="store_true",
                        help="JSON output for CI / dashboards")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only issue lines (skip header / summary)")
    args = parser.parse_args(argv)

    from vllm._genesis.dispatcher import PATCH_REGISTRY

    issues = validate_registry(PATCH_REGISTRY)

    if args.json:
        print(json.dumps({
            "total_entries": len(PATCH_REGISTRY),
            "issue_count": len(issues),
            "issues": [
                {"patch_id": i.patch_id, "field": i.field,
                 "severity": i.severity, "message": i.message}
                for i in issues
            ],
        }, indent=2))
    else:
        if not args.quiet:
            print("=" * 72)
            print(f"Genesis schema validator — {len(PATCH_REGISTRY)} "
                  f"PATCH_REGISTRY entries, {len(issues)} issues")
            print("=" * 72)
        for line in _format_issues(issues):
            print(line)
        if not args.quiet:
            print("=" * 72)

    has_error = any(i.severity == "ERROR" for i in issues)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
