# SPDX-License-Identifier: Apache-2.0
"""Genesis migrate-vllm — pin-bump migration runbook generator.

Usage:
  python3 -m vllm._genesis.compat.migrate /path/to/upstream-vllm-clone
  python3 -m vllm._genesis.compat.migrate /tmp/vllm --patches PN14,PN13,P67
  python3 -m vllm._genesis.compat.migrate /tmp/vllm --json --out runbook.json

Reads each Genesis text-patch's anchor + upstream marker, checks
against the target upstream-vllm tree, and produces a per-patch
verdict with actionable migration guidance:

  ✓ clean / would_apply  — anchor still matches; patch ports cleanly
  ⚠ anchor_drift         — upstream refactored; re-derive anchor
  ✓ upstream_merged      — fix already there; patch self-retires
  ✗ file_missing         — target file moved / renamed; update wiring path
  ⚠ ambiguous_anchor     — anchor appears N>1 times in new source

Read-only against upstream. Output: structured dict (JSON) + markdown
runbook file (default `genesis_migration_<sha>.md`).

This is **D1's offline-aware sibling**: D1 (`tools/check_upstream_drift.py`)
runs on a CI cron and reports drift; `migrate.py` is operator-driven and
produces actionable runbook before a planned pin bump.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("genesis.compat.migrate")


# ─── Helpers ─────────────────────────────────────────────────────────────


def _resolve_for_upstream(rel: str, upstream_root: Path) -> str | None:
    """Mimic resolve_vllm_file pointed at the upstream clone."""
    candidate = upstream_root / "vllm" / rel
    if candidate.is_file():
        return str(candidate)
    return None


def _make_patcher_for_patch(patch_id: str, upstream_root: Path):
    """Locate the wiring module for `patch_id`, redirect its
    resolve_vllm_file to the upstream tree, and return the patcher.
    Returns (patcher, error_str | None)."""
    from vllm._genesis.compat.categories import module_for
    mod_path = module_for(patch_id)
    if mod_path is None:
        return None, f"no wiring module found for patch {patch_id!r}"

    try:
        mod = importlib.import_module(mod_path)
    except Exception as e:
        return None, f"import failed: {e}"

    make_patcher = getattr(mod, "_make_patcher", None)
    if make_patcher is None:
        return None, f"module {mod_path} has no _make_patcher() — non-text-patch"

    # Redirect resolve_vllm_file
    orig_resolve = getattr(mod, "resolve_vllm_file", None)
    orig_install_root = getattr(mod, "vllm_install_root", None)

    if orig_resolve:
        mod.resolve_vllm_file = lambda rel: _resolve_for_upstream(rel, upstream_root)
    if orig_install_root:
        mod.vllm_install_root = lambda: str(upstream_root)

    # Compute kwargs for parameterized _make_patcher (PN9, P77, etc.)
    sig = inspect.signature(make_patcher)
    kwargs: dict[str, Any] = {}
    for pname, p in sig.parameters.items():
        if p.default is not inspect._empty:
            continue
        ann = p.annotation
        if ann is inspect._empty:
            kwargs[pname] = None
        elif "int" in str(ann):
            kwargs[pname] = 0
        elif "bool" in str(ann):
            kwargs[pname] = False
        elif "float" in str(ann):
            kwargs[pname] = 0.0
        else:
            kwargs[pname] = None

    try:
        patcher = make_patcher(**kwargs) if kwargs else make_patcher()
    except Exception as e:
        # Restore
        if orig_resolve:
            mod.resolve_vllm_file = orig_resolve
        if orig_install_root:
            mod.vllm_install_root = orig_install_root
        return None, f"_make_patcher raised: {e}"

    # Don't restore originals here — caller may need patcher.target_file etc.
    # The redirected resolve sticks for the duration of the call. This is
    # safe because we never write to the upstream tree.
    return patcher, None


# ─── Per-patch check ────────────────────────────────────────────────────


def check_patch_against_upstream(
    patch_id: str,
    upstream_root: str | Path,
) -> dict[str, Any]:
    """Verify one Genesis patch against an upstream-vllm checkout.

    Returns a verdict dict with shape:
      {
        "patch_id": "PN14",
        "status": "clean" | "anchor_drift" | "upstream_merged" |
                  "file_missing" | "ambiguous_anchor" | "unknown_patch" |
                  "non_text_patch" | "error",
        "message": "human-readable explanation",
        "target_file": "vllm/v1/.../triton_turboquant_decode.py",
        "anchor_count": int,  # how many times the anchor matched
        "drift_marker_hit": str | None,
        "action": "human-readable next step",
      }
    """
    upstream_root = Path(upstream_root).resolve()
    if not upstream_root.is_dir():
        return {
            "patch_id": patch_id, "status": "error",
            "message": f"upstream path does not exist: {upstream_root}",
            "action": "verify upstream-vllm clone path",
        }

    patcher, err = _make_patcher_for_patch(patch_id, upstream_root)
    if err:
        if "no wiring module" in err:
            return {
                "patch_id": patch_id, "status": "unknown_patch",
                "message": err, "action": "check Genesis registry",
            }
        if "non-text-patch" in err:
            return {
                "patch_id": patch_id, "status": "non_text_patch",
                "message": err,
                "action": "non-text wiring; manually verify behavior",
            }
        return {
            "patch_id": patch_id, "status": "error",
            "message": err, "action": "investigate wiring module",
        }

    if patcher is None:
        return {
            "patch_id": patch_id, "status": "file_missing",
            "message": "patcher could not resolve target file in upstream",
            "action": "check if upstream renamed/moved the target file",
        }

    target = Path(patcher.target_file)
    if not target.is_file():
        return {
            "patch_id": patch_id, "status": "file_missing",
            "message": f"target file does not exist in upstream: {target}",
            "target_file": str(target),
            "action": "check if upstream renamed/moved the target file; "
                      "update wiring/<patch>.py resolve_vllm_file path",
        }

    try:
        content = target.read_text()
    except Exception as e:
        return {
            "patch_id": patch_id, "status": "error",
            "message": f"target read failed: {e}",
            "action": "check file permissions / encoding",
        }

    # Layer 1: drift markers (= upstream merged equivalent fix)
    drift_markers = list(getattr(patcher, "upstream_drift_markers", []) or [])
    for m in drift_markers:
        # Skip Genesis's own marker — we want UPSTREAM markers, not our own
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return {
                "patch_id": patch_id, "status": "upstream_merged",
                "message": f"upstream marker {m!r} present in target — "
                           f"upstream PR has merged equivalent fix",
                "target_file": str(target.relative_to(upstream_root)),
                "drift_marker_hit": m,
                "anchor_count": 0,
                "action": "patch will self-retire automatically; no action "
                          "needed except verify after pin bump",
            }

    # Layer 2: count anchor matches in upstream
    sub_patches = list(getattr(patcher, "sub_patches", []) or [])
    if not sub_patches:
        return {
            "patch_id": patch_id, "status": "error",
            "message": "patcher declared no sub_patches",
            "action": "investigate wiring module",
        }

    total_count = 0
    missing_count = 0
    ambiguous_count = 0
    for sp in sub_patches:
        anchor = getattr(sp, "anchor", None)
        if not anchor:
            continue
        c = content.count(anchor)
        total_count += c
        if c == 0 and getattr(sp, "required", True):
            missing_count += 1
        elif c > 1:
            ambiguous_count += 1

    target_rel = str(target.relative_to(upstream_root))

    if missing_count > 0:
        return {
            "patch_id": patch_id, "status": "anchor_drift",
            "message": f"{missing_count} required anchor(s) NOT FOUND in "
                       f"upstream — upstream refactored this region",
            "target_file": target_rel,
            "anchor_count": total_count,
            "drift_marker_hit": None,
            "action": f"open {target_rel} in upstream, find the new shape "
                      f"of the region, re-derive the anchor in "
                      f"wiring/<patch>.py and re-run this tool",
        }

    if ambiguous_count > 0:
        return {
            "patch_id": patch_id, "status": "ambiguous_anchor",
            "message": f"{ambiguous_count} anchor(s) match >1 times — "
                       f"upstream may have duplicated the region",
            "target_file": target_rel,
            "anchor_count": total_count,
            "action": "narrow the anchor to disambiguate (add more context "
                      "lines around the unique segment)",
        }

    return {
        "patch_id": patch_id, "status": "clean",
        "message": "all anchors match exactly once in upstream — patch "
                   "ports cleanly",
        "target_file": target_rel,
        "anchor_count": total_count,
        "drift_marker_hit": None,
        "action": "no action — patch will apply identically",
    }


# ─── Aggregate runbook ──────────────────────────────────────────────────


def generate_runbook(
    upstream_root: str | Path,
    patch_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run check_patch_against_upstream over a set of patches and
    aggregate into a runbook structure.

    If `patch_ids` is None, walks every text-patch wiring module.
    """
    upstream_root = Path(upstream_root).resolve()

    if patch_ids is None:
        # Walk all patches in PATCH_REGISTRY
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        patch_ids = sorted(PATCH_REGISTRY.keys())

    results = []
    for pid in patch_ids:
        results.append(check_patch_against_upstream(pid, upstream_root))

    # Aggregate counts
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream_root": str(upstream_root),
        "summary": {
            "total_checked": len(results),
            "by_status": by_status,
            "needs_operator_action": sum(
                v for k, v in by_status.items()
                if k in ("anchor_drift", "ambiguous_anchor", "file_missing")
            ),
            "auto_self_retire": by_status.get("upstream_merged", 0),
            "ports_cleanly": by_status.get("clean", 0),
        },
        "patches": results,
    }


# ─── Markdown formatter ─────────────────────────────────────────────────


_STATUS_BADGES = {
    "clean":            "✅ ports cleanly",
    "would_apply":      "✅ would apply",
    "anchor_present":   "✅ anchor present",
    "upstream_merged":  "🎉 upstream merged",
    "anchor_drift":     "⚠️ anchor drift",
    "ambiguous_anchor": "⚠️ ambiguous anchor",
    "file_missing":     "❌ file moved",
    "non_text_patch":   "ℹ️ non-text-patch",
    "unknown_patch":    "❓ unknown patch",
    "error":            "❌ error",
}


def format_runbook_md(runbook: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# Genesis migration runbook")
    L.append("")
    L.append(f"Generated: {runbook['generated_at']}")
    L.append(f"Upstream:  `{runbook['upstream_root']}`")
    L.append("")
    L.append("## Summary")
    L.append("")
    s = runbook["summary"]
    L.append(f"- Total patches checked: **{s['total_checked']}**")
    L.append(f"- Ports cleanly:         **{s['ports_cleanly']}**")
    L.append(f"- Auto-self-retire:      **{s['auto_self_retire']}**")
    L.append(f"- Needs operator action: **{s['needs_operator_action']}**")
    L.append("")
    L.append("### Status breakdown")
    L.append("")
    for status, count in sorted(s["by_status"].items()):
        L.append(f"- {_STATUS_BADGES.get(status, status)}: {count}")
    L.append("")

    # Group by status for the per-patch sections
    groups: dict[str, list] = {}
    for p in runbook["patches"]:
        groups.setdefault(p["status"], []).append(p)

    # Operator actions first (drift / ambiguous / missing)
    priority = ["anchor_drift", "ambiguous_anchor", "file_missing", "error",
                "unknown_patch", "non_text_patch", "upstream_merged",
                "clean", "would_apply", "anchor_present"]

    for status in priority:
        if status not in groups:
            continue
        L.append(f"## {_STATUS_BADGES.get(status, status)} ({len(groups[status])})")
        L.append("")
        for p in groups[status]:
            L.append(f"### {p['patch_id']}")
            L.append("")
            L.append(f"- **Status:** {p['status']}")
            L.append(f"- **Message:** {p['message']}")
            if p.get("target_file"):
                L.append(f"- **File:** `{p['target_file']}`")
            if p.get("drift_marker_hit"):
                L.append(f"- **Marker hit:** `{p['drift_marker_hit']}`")
            L.append(f"- **Action:** {p.get('action', '(no action)')}")
            L.append("")

    return "\n".join(L)


# ─── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.migrate",
        description="Generate a pin-bump migration runbook for a target "
                    "upstream-vllm checkout.",
    )
    parser.add_argument("upstream_path",
                        help="Path to upstream-vllm clone (read-only)")
    parser.add_argument("--patches", default=None,
                        help="Comma-separated patch IDs to check "
                             "(default: all)")
    parser.add_argument("--json", action="store_true",
                        help="JSON output to stdout")
    parser.add_argument("--out", default=None,
                        help="Path to write the markdown runbook "
                             "(default: don't write)")
    args = parser.parse_args(argv)

    upstream = Path(args.upstream_path)
    if not upstream.exists():
        print(f"upstream path not found: {upstream}", file=sys.stderr)
        return 2
    if not upstream.is_dir():
        print(f"upstream path is not a directory: {upstream}", file=sys.stderr)
        return 2

    patch_list = None
    if args.patches:
        patch_list = [p.strip() for p in args.patches.split(",") if p.strip()]

    runbook = generate_runbook(upstream, patch_list)

    if args.json:
        print(json.dumps(runbook, indent=2, default=str))
    else:
        md = format_runbook_md(runbook)
        print(md)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(format_runbook_md(runbook))
        print(f"\n(Markdown runbook written to: {out_path})", file=sys.stderr)

    # Exit non-zero if any patch needs operator action
    needs = runbook["summary"]["needs_operator_action"]
    return 1 if needs > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
