# SPDX-License-Identifier: Apache-2.0
"""Genesis lifecycle-audit — `python3 -m vllm._genesis.compat.lifecycle_audit_cli`.

Walks PATCH_REGISTRY and produces a per-state breakdown:
  - experimental — patches that may break (warn the operator)
  - stable       — proven, the default UX
  - deprecated   — superseded; surface migration path
  - research     — kept as reference; doctor lists but doesn't engage
  - community    — plugin entry-points (future Phase 5)
  - retired      — code may still exist; not applied
  - <unknown>    — registry error (fails CI)

Exit code:
  0 — clean (no error-severity issues)
  1 — at least one patch has an unknown / malformed lifecycle state

Usage:
  python3 -m vllm._genesis.compat.lifecycle_audit_cli
  python3 -m vllm._genesis.compat.lifecycle_audit_cli --state deprecated
  python3 -m vllm._genesis.compat.lifecycle_audit_cli --json --quiet

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.lifecycle_audit_cli",
        description="Audit lifecycle states across PATCH_REGISTRY.",
    )
    parser.add_argument("--state", choices=[
        "experimental", "stable", "deprecated", "research",
        "community", "retired",
    ], default=None,
        help="Filter to one lifecycle state",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (for CI / dashboards)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print rows with severity != 'ok'")
    args = parser.parse_args(argv)

    from vllm._genesis.dispatcher import PATCH_REGISTRY
    from vllm._genesis.compat.lifecycle import (
        audit_registry, format_audit_table,
    )

    entries = audit_registry(PATCH_REGISTRY)

    # Filter by state if requested
    if args.state:
        entries = [e for e in entries if e.state == args.state]

    # Filter by quiet (skip "ok" severity)
    if args.quiet:
        entries = [e for e in entries if e.severity != "ok"]

    if args.json:
        # Group by state (parallel to lifecycle.audit_registry style)
        by_state: dict[str, list] = {}
        for e in entries:
            by_state.setdefault(e.state, []).append({
                "patch_id": e.patch_id,
                "note": e.note,
                "severity": e.severity,
            })
        print(json.dumps({
            "by_state": by_state,
            "entries": [
                {"patch_id": e.patch_id, "state": e.state,
                 "note": e.note, "severity": e.severity}
                for e in entries
            ],
            "total": len(entries),
        }, indent=2, default=str))
    else:
        print("=" * 72)
        print(f"Genesis lifecycle audit — {len(entries)} entries"
              + (f" (filter: state={args.state})" if args.state else "")
              + (" (quiet: errors+warnings only)" if args.quiet else ""))
        print("=" * 72)
        if entries:
            for line in format_audit_table(entries):
                print(line)
        else:
            print("  (no entries match)")
        print("=" * 72)

    # Exit non-zero ONLY on error severity (unknown state, malformed entries)
    has_error = any(e.severity == "error" for e in entries)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
