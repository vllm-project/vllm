# SPDX-License-Identifier: Apache-2.0
"""Genesis explain — `python3 -m vllm._genesis.compat.explain <patch_id>`.

Produces a structured per-patch report — what the patch is, what it does,
when it engages, what it depends on / conflicts with, what the upstream
status is, and whether it would APPLY on the current system right now.

Usage:
  python3 -m vllm._genesis.compat.explain PN14
  python3 -m vllm._genesis.compat.explain P67 --json
  python3 -m vllm._genesis.compat.explain --list

This is the per-patch counterpart of `genesis doctor` — doctor shows
the whole forest, explain zooms into one tree.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

log = logging.getLogger("genesis.compat.explain")


# ─── Core data extraction ────────────────────────────────────────────────


def _normalize_str_or_list(value: Any) -> list[str]:
    """`'X'` or `['X', 'Y']` or None → `[]` / `['X']` / `['X', 'Y']`."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


def explain_patch(patch_id: str) -> dict[str, Any]:
    """Return a structured per-patch explanation dict.

    Returns a dict with keys: patch_id, title, env_flag, default_on,
    category, lifecycle, dependencies, applies_to, upstream, decision,
    credit, notes. On unknown patch_id, returns {"error": "..."}.
    """
    from vllm._genesis.dispatcher import PATCH_REGISTRY, should_apply

    if not patch_id:
        return {"error": "empty patch_id"}

    meta = PATCH_REGISTRY.get(patch_id)
    if meta is None:
        return {
            "error": f"unknown patch_id {patch_id!r}",
            "available_count": len(PATCH_REGISTRY),
        }

    # ─── Lifecycle section ──────────────────────────────────────────────
    from vllm._genesis.compat.lifecycle import get_state
    lc_state = get_state(meta)
    lifecycle: dict[str, Any] = {"state": lc_state}
    for field in ("stable_since", "since_version", "deprecated_since",
                  "removal_planned", "experimental_note", "research_note",
                  "community_credit"):
        if field in meta:
            lifecycle[field] = meta[field]
    lifecycle["superseded_by"] = _normalize_str_or_list(meta.get("superseded_by"))

    # ─── Dependencies section ───────────────────────────────────────────
    dependencies = {
        "requires": _normalize_str_or_list(meta.get("requires_patches")),
        "conflicts_with": _normalize_str_or_list(meta.get("conflicts_with")),
    }

    # ─── applies_to section + live evaluation ───────────────────────────
    applies_to_block: dict[str, Any] = {"rule": meta.get("applies_to") or {}}
    if applies_to_block["rule"]:
        # Build a flat profile dict (model_detect aliases + version stuff)
        try:
            from vllm._genesis.model_detect import get_model_profile
            profile = get_model_profile() or {}
        except Exception:
            profile = {}
        # Mirror the boolean aliases the dispatcher uses
        flat_profile = dict(profile)
        for applies_key, profile_key in (("is_moe", "moe"),
                                         ("is_hybrid", "hybrid"),
                                         ("is_turboquant", "turboquant")):
            if profile_key in profile and applies_key not in flat_profile:
                flat_profile[applies_key] = profile[profile_key]

        try:
            from vllm._genesis.compat.predicates import explain as predicate_explain
            applies_to_block["explanation"] = predicate_explain(
                applies_to_block["rule"], flat_profile,
            )
        except Exception as e:
            applies_to_block["explanation"] = [f"(predicate explain failed: {e})"]
        applies_to_block["resolved_profile"] = flat_profile
    else:
        applies_to_block["explanation"] = ["(no applies_to declared — applies on every model)"]
        applies_to_block["resolved_profile"] = {}

    # ─── Upstream tracking section ──────────────────────────────────────
    upstream: dict[str, Any] = {
        "pr_number": meta.get("upstream_pr"),
        "marker": None,
        "marker_file": None,
        "merged_status": "unknown",
    }
    try:
        from vllm._genesis.patches.upstream_compat import UPSTREAM_MARKERS

        # Match priority:
        #   1. EXACT key match — `PR_<num>_*` where num == meta.upstream_pr.
        #      This is the canonical link from a patch to its tracker.
        #   2. `affects_patch` field where patch_id appears as a standalone
        #      token (not just substring) — e.g. "PN14 TQ decode" matches
        #      patch_id PN14, but "PN14, P40 — re-derivation" does NOT win
        #      over an exact PR-number match for PN14.
        chosen_key, chosen_info = None, None
        pr_num = meta.get("upstream_pr")

        # Priority 1: exact PR-number key prefix
        if pr_num:
            for key, info in UPSTREAM_MARKERS.items():
                if key.startswith(f"PR_{pr_num}_"):
                    chosen_key, chosen_info = key, info
                    break

        # Priority 2: standalone token in affects_patch (only if no PR match)
        if chosen_info is None:
            import re as _re
            tok = _re.compile(rf"\b{_re.escape(patch_id)}\b")
            for key, info in UPSTREAM_MARKERS.items():
                affects = info.get("affects_patch", "")
                if isinstance(affects, str) and tok.search(affects):
                    chosen_key, chosen_info = key, info
                    break

        if chosen_info is not None:
            upstream["marker"] = chosen_info.get("marker")
            upstream["marker_file"] = (
                chosen_info.get("file")
                or (chosen_info.get("files") or [None])[0]
            )
            upstream["merged_status"] = chosen_info.get("merged_date", "unknown")
            upstream["compat_key"] = chosen_key
    except Exception:
        # compat-rules JSON optional — explain proceeds without merged_status
        pass

    # ─── Decision today (live should_apply) ─────────────────────────────
    try:
        decision_bool, decision_reason = should_apply(patch_id)
    except Exception as e:
        decision_bool, decision_reason = False, f"should_apply raised: {e}"
    decision = {
        "applied": bool(decision_bool),
        "reason": decision_reason,
    }

    # ─── Assemble ──────────────────────────────────────────────────────
    return {
        "patch_id": patch_id,
        "title": meta.get("title", patch_id),
        "env_flag": meta.get("env_flag", ""),
        "default_on": bool(meta.get("default_on", False)),
        "category": meta.get("category", "uncategorized"),
        "lifecycle": lifecycle,
        "dependencies": dependencies,
        "applies_to": applies_to_block,
        "upstream": upstream,
        "decision": decision,
        "credit": meta.get("credit", ""),
        "notes": meta.get("deprecation_note") or meta.get("notes", ""),
    }


# ─── Text formatter ──────────────────────────────────────────────────────


def format_explain_text(report: dict[str, Any]) -> list[str]:
    """Render the explain report as human-readable text lines."""
    L: list[str] = []

    if "error" in report:
        L.append("=" * 72)
        L.append(f"✗ {report['error']}")
        if "available_count" in report:
            L.append(f"  ({report['available_count']} patches in registry — "
                     f"run `python3 -m vllm._genesis.compat.doctor` for the list)")
        L.append("=" * 72)
        return L

    patch_id = report["patch_id"]
    L.append("=" * 72)
    L.append(f"{patch_id} — {report['title']}")
    L.append("=" * 72)

    # Identity
    L.append("")
    L.append(f"  Env flag:     {report['env_flag']}")
    L.append(f"  Default:      {'ON' if report['default_on'] else 'OFF'}")
    L.append(f"  Category:     {report['category']}")

    # Lifecycle
    lc = report["lifecycle"]
    L.append("")
    L.append(f"  Lifecycle:    {lc['state']}")
    if lc["state"] == "deprecated":
        sup = ", ".join(lc.get("superseded_by", [])) or "(none specified)"
        L.append(f"    superseded by:    {sup}")
        if "removal_planned" in lc:
            L.append(f"    removal planned:  {lc['removal_planned']}")
    elif lc["state"] == "experimental":
        if "experimental_note" in lc:
            L.append(f"    note:             {lc['experimental_note']}")
    elif lc["state"] == "research":
        if "research_note" in lc:
            L.append(f"    research note:    {lc['research_note']}")
    elif lc["state"] == "stable":
        if "stable_since" in lc:
            L.append(f"    stable since:     {lc['stable_since']}")

    # Dependencies
    deps = report["dependencies"]
    L.append("")
    L.append(f"  Dependencies:")
    if deps["requires"]:
        L.append(f"    requires:         {', '.join(deps['requires'])}")
    else:
        L.append(f"    requires:         (none)")
    if deps["conflicts_with"]:
        L.append(f"    conflicts with:   {', '.join(deps['conflicts_with'])}")
    else:
        L.append(f"    conflicts with:   (none)")

    # applies_to
    L.append("")
    L.append(f"  applies_to:")
    for line in report["applies_to"].get("explanation", []):
        L.append(f"    {line}")

    # Upstream
    up = report["upstream"]
    L.append("")
    L.append(f"  Upstream tracking:")
    if up.get("pr_number"):
        L.append(f"    PR:               vllm#{up['pr_number']}")
    else:
        L.append(f"    PR:               (Genesis-original or no upstream tracker)")
    if up.get("marker"):
        L.append(f"    marker symbol:    {up['marker']}")
    if up.get("marker_file"):
        L.append(f"    marker file:      {up['marker_file']}")
    if up.get("merged_status") and up["merged_status"] != "unknown":
        L.append(f"    merged status:    {up['merged_status']}")

    # Decision
    dec = report["decision"]
    L.append("")
    mark = "✓ APPLY" if dec["applied"] else "✗ SKIP"
    L.append(f"  Decision today:  {mark}")
    L.append(f"    reason: {dec['reason']}")

    # Credit / notes
    if report.get("credit"):
        L.append("")
        L.append(f"  Credit / what + why:")
        # Wrap long credit lines for readability
        for line in _wrap(report["credit"], width=70, indent=4):
            L.append(line)

    if report.get("notes"):
        L.append("")
        L.append(f"  Notes:")
        for line in _wrap(str(report["notes"]), width=70, indent=4):
            L.append(line)

    L.append("=" * 72)
    return L


def _wrap(text: str, *, width: int = 70, indent: int = 4) -> list[str]:
    """Simple word-wrap with leading-indent. Multiline-aware."""
    pad = " " * indent
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append(pad)
            continue
        words = paragraph.split()
        cur = pad
        for w in words:
            if len(cur) + 1 + len(w) > width + indent:
                lines.append(cur)
                cur = pad + w
            else:
                cur = (cur + " " + w) if cur != pad else (cur + w)
        if cur.strip():
            lines.append(cur)
    return lines


# ─── CLI ─────────────────────────────────────────────────────────────────


def _list_patches() -> int:
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    print(f"Genesis patches — {len(PATCH_REGISTRY)} entries")
    print("─" * 72)
    for pid, meta in sorted(PATCH_REGISTRY.items()):
        title = meta.get("title", pid)
        cat = meta.get("category", "—")
        print(f"  {pid:<8} [{cat:<22}] {title[:35]}")
    print("─" * 72)
    print("Detail: python3 -m vllm._genesis.compat.explain <patch_id>")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.explain",
        description="Per-patch detailed explanation — what does this Genesis "
                    "patch do, when does it engage, what does it depend on?",
    )
    parser.add_argument("patch_id", nargs="?",
                        help="Patch ID (e.g. PN14, P67, PN16)")
    parser.add_argument("--list", action="store_true",
                        help="List all patches and exit")
    parser.add_argument("--json", action="store_true",
                        help="Output the report as JSON")
    args = parser.parse_args(argv)

    if args.list:
        return _list_patches()

    if not args.patch_id:
        parser.print_usage()
        sys.exit(2)

    report = explain_patch(args.patch_id)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for line in format_explain_text(report):
            print(line)

    return 1 if "error" in report else 0


if __name__ == "__main__":
    sys.exit(main())
