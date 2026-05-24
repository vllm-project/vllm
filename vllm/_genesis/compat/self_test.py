# SPDX-License-Identifier: Apache-2.0
"""Genesis self-test — operator sanity check.

Quick CI-style verification that runs after a git pull or pin bump:

  1. VERSION constant readable + sane
  2. All compat modules import cleanly
  3. All wiring modules import cleanly
  4. PATCH_REGISTRY validates against schema
  5. Lifecycle audit clean (no unknown states)
  6. Categories index builds without errors
  7. Predicates evaluator works on the real registry
  8. JSON schema file present + parseable

Exit code:
  0 = all critical checks passed
  1 = at least one failure (operator action required)

Usage:
  python3 -m vllm._genesis.compat.self_test
  python3 -m vllm._genesis.compat.self_test --json
  python3 -m vllm._genesis.compat.self_test --quiet

This is the "is Genesis itself working?" tool. Different from doctor,
which is "is my SYSTEM healthy?". A doctor failure can be hardware /
config; a self-test failure is a Genesis bug.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("genesis.compat.self_test")


# ─── Check dataclass shape ───────────────────────────────────────────────


def _check(name: str, fn: Callable[[], tuple[str, str]]) -> dict[str, str]:
    """Run a check function. Returns {name, status, message}.

    Check function returns (status, message) where status in
    {pass, fail, warn, skip}.
    """
    try:
        status, message = fn()
    except Exception as e:
        # Self-test must NEVER crash; failed import / unexpected error
        # = "fail" status with traceback summary
        status = "fail"
        message = f"{type(e).__name__}: {e}"
    return {"name": name, "status": status, "message": message}


# ─── Individual checks ───────────────────────────────────────────────────


def _check_version() -> tuple[str, str]:
    """VERSION constant present + readable."""
    from vllm._genesis import __version__
    if not isinstance(__version__, str):
        return "fail", f"__version__ is {type(__version__).__name__}, want str"
    if not __version__:
        return "fail", "__version__ is empty"
    return "pass", f"version: {__version__}"


def _check_compat_imports() -> tuple[str, str]:
    """Every compat/* module imports without error."""
    modules = [
        "vllm._genesis.compat.predicates",
        "vllm._genesis.compat.version_check",
        "vllm._genesis.compat.lifecycle",
        "vllm._genesis.compat.schema_validator",
        "vllm._genesis.compat.lifecycle_audit_cli",
        "vllm._genesis.compat.categories",
        "vllm._genesis.compat.explain",
        "vllm._genesis.compat.recipes",
        "vllm._genesis.compat.plugins",
        "vllm._genesis.compat.telemetry",
        "vllm._genesis.compat.update_channel",
        "vllm._genesis.compat.cli",
        "vllm._genesis.compat.bench",
        "vllm._genesis.compat.doctor",
        "vllm._genesis.compat.init_wizard",
        "vllm._genesis.compat.migrate",
        "vllm._genesis.compat.models.registry",
        "vllm._genesis.compat.models.list_cli",
        "vllm._genesis.compat.models.pull",
    ]
    failed = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {type(e).__name__}: {e}")
    if failed:
        return "fail", f"{len(failed)} compat module(s) failed to import:\n  " + \
                       "\n  ".join(failed)
    return "pass", f"{len(modules)} compat modules import cleanly"


def _check_wiring_imports() -> tuple[str, str]:
    """Every wiring/patch_*.py module imports without error.

    Some require resolve_vllm_file (i.e. a real vllm install). Those
    are SKIPPED rather than failed when vllm isn't importable here.
    """
    wiring_dir = Path(__file__).resolve().parent.parent / "wiring"
    if not wiring_dir.is_dir():
        return "skip", f"wiring/ dir missing at {wiring_dir}"

    # rglob to handle both flat layout and Phase-2.1 category subdirs.
    files = sorted(wiring_dir.rglob("patch_*.py"))
    failed = []
    skipped_count = 0
    for f in files:
        # Compute dotted module path from the file's location under
        # wiring/ — works for flat AND categorical layouts.
        rel_parts = f.relative_to(wiring_dir.parent.parent.parent).parts
        mod_name = ".".join(list(rel_parts[:-1]) + [f.stem])
        try:
            importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            # vllm not installed → skip rather than fail
            if "vllm" in str(e) and "_genesis" not in str(e):
                skipped_count += 1
                continue
            failed.append(f"{mod_name}: {type(e).__name__}: {e}")
        except Exception as e:
            failed.append(f"{mod_name}: {type(e).__name__}: {e}")

    if failed:
        return "fail", f"{len(failed)}/{len(files)} wiring modules broken:\n  " + \
                       "\n  ".join(failed[:5])  # cap at 5 for readability
    if skipped_count > 0:
        return "warn", (
            f"{len(files) - skipped_count}/{len(files)} wiring modules "
            f"imported; {skipped_count} skipped (vllm not installed in this env)"
        )
    return "pass", f"{len(files)} wiring modules import cleanly"


def _check_schema_validator() -> tuple[str, str]:
    """PATCH_REGISTRY schema-validates."""
    from vllm._genesis.compat.schema_validator import validate_registry
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    issues = validate_registry(PATCH_REGISTRY)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        return "fail", (
            f"{len(errors)} schema error(s) in PATCH_REGISTRY: "
            + "; ".join(f"{i.patch_id}.{i.field}" for i in errors[:5])
        )
    if issues:
        return "warn", f"{len(issues)} non-error schema issue(s)"
    return "pass", f"all {len(PATCH_REGISTRY)} entries schema-clean"


def _check_lifecycle_audit() -> tuple[str, str]:
    """Lifecycle audit clean (no unknown states)."""
    from vllm._genesis.compat.lifecycle import audit_registry
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    entries = audit_registry(PATCH_REGISTRY)
    errors = [e for e in entries if e.severity == "error"]
    if errors:
        return "fail", (
            f"{len(errors)} lifecycle error(s): "
            + "; ".join(f"{e.patch_id}: {e.note}" for e in errors[:3])
        )
    return "pass", f"{len(entries)} entries — no unknown lifecycle states"


def _check_categories_build() -> tuple[str, str]:
    """Categories index builds without errors + every patch placed."""
    from vllm._genesis.compat.categories import _build_categories
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    cats = _build_categories()
    placed = sum(len(v) for v in cats.values())
    if placed != len(PATCH_REGISTRY):
        return "fail", (
            f"category placement mismatch: registry={len(PATCH_REGISTRY)}, "
            f"categorized={placed}"
        )
    return "pass", f"{len(PATCH_REGISTRY)} patches → {len(cats)} categories"


def _check_predicates_evaluate() -> tuple[str, str]:
    """Predicates evaluator works on real entries."""
    from vllm._genesis.compat.predicates import evaluate
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    failures = []
    for pid, meta in PATCH_REGISTRY.items():
        applies_to = meta.get("applies_to")
        if not applies_to:
            continue
        try:
            ok, _reason = evaluate(applies_to, {})
            assert isinstance(ok, bool)
        except Exception as e:
            failures.append(f"{pid}: {type(e).__name__}: {e}")
    if failures:
        return "fail", (
            f"{len(failures)} predicate eval failures:\n  "
            + "\n  ".join(failures[:5])
        )
    return "pass", "predicates evaluator works on every applies_to in registry"


def _check_schema_file() -> tuple[str, str]:
    """schemas/patch_entry.schema.json file present + parseable.

    The schema file is a repo-only artifact — it ships with the source
    tree but is NOT installed alongside the `vllm._genesis` package.
    Operators running self-test from inside a container that only has
    the package mounted (no repo) will not have the schema file
    available; that's a "skip", not a "fail".
    """
    import os

    candidates: list[Path] = []
    # 1. Explicit override via env (lets sysadmins point at any location)
    env_root = os.environ.get("GENESIS_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "schemas" / "patch_entry.schema.json")
    # 2. Repo-root relative to this file (works in a git checkout)
    candidates.append(
        Path(__file__).resolve().parents[3] / "schemas" / "patch_entry.schema.json"
    )
    # 3. Cwd-relative (works when invoked from the repo root)
    candidates.append(Path.cwd() / "schemas" / "patch_entry.schema.json")

    schema_path = next((p for p in candidates if p.is_file()), None)
    if schema_path is None:
        # Not finding the schema file is expected in a slim container
        # deployment — flag as skip so it doesn't break exit code.
        return "skip", (
            "schema file not present in this deployment "
            "(repo-only artifact; set GENESIS_REPO_ROOT to override)"
        )
    try:
        with open(schema_path) as f:
            data = json.load(f)
    except Exception as e:
        return "fail", f"schema file not valid JSON: {e}"
    required_keys = ("$schema", "title", "type", "properties")
    missing = [k for k in required_keys if k not in data]
    if missing:
        return "warn", f"schema missing keys: {missing}"
    return "pass", "schema file parseable + has required keys"


_CHECKS: list[tuple[str, Callable[[], tuple[str, str]]]] = [
    ("version constant",       _check_version),
    ("compat imports",          _check_compat_imports),
    ("wiring imports",          _check_wiring_imports),
    ("schema validator",        _check_schema_validator),
    ("lifecycle audit",         _check_lifecycle_audit),
    ("categories build",        _check_categories_build),
    ("predicates evaluator",    _check_predicates_evaluate),
    ("schema file",             _check_schema_file),
]


# ─── Driver ──────────────────────────────────────────────────────────────


def run_self_test() -> dict[str, Any]:
    """Run all checks. Returns a dict with `checks` (list) and
    `summary` (counts)."""
    results = []
    for name, fn in _CHECKS:
        results.append(_check(name, fn))

    summary = {"passed": 0, "failed": 0, "warned": 0,
               "skipped": 0, "total": len(results)}
    for r in results:
        if r["status"] == "pass":
            summary["passed"] += 1
        elif r["status"] == "fail":
            summary["failed"] += 1
        elif r["status"] == "warn":
            summary["warned"] += 1
        elif r["status"] == "skip":
            summary["skipped"] += 1

    return {"checks": results, "summary": summary}


def _format_check(c: dict[str, str]) -> str:
    icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "•"}.get(c["status"], "?")
    return f"  {icon} [{c['status'].upper():<4}] {c['name']:<25} {c['message']}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.self_test",
        description="Run Genesis structural self-tests.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Show only fail / warn rows")
    args = parser.parse_args(argv)

    result = run_self_test()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        s = result["summary"]
        if not args.quiet:
            print("=" * 72)
            print("Genesis self-test")
            print("=" * 72)
        for c in result["checks"]:
            if args.quiet and c["status"] == "pass":
                continue
            print(_format_check(c))
        if not args.quiet:
            print("=" * 72)
            print(f"Summary: {s['passed']} pass, {s['failed']} fail, "
                  f"{s['warned']} warn, {s['skipped']} skip")
            print("=" * 72)

    return 1 if result["summary"]["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
