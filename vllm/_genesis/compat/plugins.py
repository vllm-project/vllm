# SPDX-License-Identifier: Apache-2.0
"""Genesis plugins — community-shipped patches via setuptools entry-points.

Third-party packages can extend Genesis without forking the core repo
by declaring entry-points in the `vllm_genesis_patches` group:

    # In a third-party package's pyproject.toml:
    [project.entry-points."vllm_genesis_patches"]
    my_patch = "my_pkg.genesis_plugin:get_patch_metadata"

The callable returns either a dict (single patch) or list of dicts
(multiple patches per package). Each dict is validated against
schemas/patch_entry.schema.json, auto-tagged with `lifecycle: community`,
auto-stamped with `_plugin_origin`, and registered into the
PATCH_REGISTRY at boot.

OPT-IN security gate
────────────────────
Plugin discovery is OFF by default. Set `GENESIS_ALLOW_PLUGINS=1` to
enable. Genesis must boot identically with or without plugins
installed — discovery is a no-op when the gate is closed.

Once enabled, discovery walks `importlib.metadata.entry_points` for
the `vllm_genesis_patches` group, loads each callable in a try/except
(one bad plugin doesn't break others), validates the returned shape,
checks for collisions with core PATCH_REGISTRY ids, and registers
survivors.

Provenance tracking
───────────────────
Every plugin entry gets `_plugin_origin: "<entry_point_name>:<value>"`
attached. Doctor + explain CLIs surface this so operators always know
which patches came from third-party packages.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

log = logging.getLogger("genesis.compat.plugins")

# setuptools entry-points group name
ENTRY_POINT_GROUP = "vllm_genesis_patches"


# ─── Opt-in gate ─────────────────────────────────────────────────────────


def _plugins_enabled() -> bool:
    """True when GENESIS_ALLOW_PLUGINS is set to a truthy value."""
    return os.environ.get("GENESIS_ALLOW_PLUGINS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ─── Entry-point discovery ───────────────────────────────────────────────


def _discover_entry_points() -> list[Any]:
    """Walk importlib.metadata for our entry-point group.

    Returns a list of EntryPoint-like objects (each has .name, .value,
    .load()). Tests monkey-patch this function to inject synthetic
    entry points without needing a real installed package.
    """
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:
        return []
    try:
        # Python 3.10+ select() API
        eps = importlib_metadata.entry_points(group=ENTRY_POINT_GROUP)
        return list(eps)
    except Exception as e:
        log.debug("entry-points discovery failed: %s", e)
        return []


# ─── Plugin loading + validation ─────────────────────────────────────────


def _load_one_plugin(ep) -> list[dict[str, Any]]:
    """Load a single entry-point, normalize to list[dict]. On any
    failure, log + return empty list (one bad plugin can't break others)."""
    try:
        callable_obj = ep.load()
    except Exception as e:
        log.warning("plugin %r load failed: %s", ep.name, e)
        return []

    try:
        result = callable_obj()
    except Exception as e:
        log.warning("plugin %r callable raised: %s", ep.name, e)
        return []

    # Normalize to list
    if isinstance(result, dict):
        items = [result]
    elif isinstance(result, list):
        items = [it for it in result if isinstance(it, dict)]
        if len(items) != len(result):
            log.warning(
                "plugin %r returned a list with %d non-dict items "
                "— ignoring those", ep.name, len(result) - len(items),
            )
    else:
        log.warning(
            "plugin %r callable must return dict or list[dict], got %s — "
            "ignoring", ep.name, type(result).__name__,
        )
        return []

    # Stamp origin + force community lifecycle
    enriched = []
    for it in items:
        out = dict(it)  # copy so we don't mutate plugin's data
        out["lifecycle"] = "community"  # always — no plugin claims stable
        ep_value = getattr(ep, "value", "<unknown>")
        out["_plugin_origin"] = f"{ep.name}:{ep_value}"
        # Ensure `community_credit` is present (schema requires it for
        # community lifecycle)
        if "community_credit" not in out:
            out["community_credit"] = (
                f"plugin entry-point: {ep.name}"
            )
        enriched.append(out)
    return enriched


def _validate_plugin(plugin: dict[str, Any]) -> tuple[bool, list[str]]:
    """Run the schema validator on a plugin dict. Returns (ok, reasons).

    `patch_id` is plugin-API metadata (becomes the key in PATCH_REGISTRY,
    not a stored field). `_plugin_origin` is provenance, also plugin-only.
    Strip both before passing to the schema validator, which validates
    the SHAPE of one PATCH_REGISTRY VALUE.
    """
    try:
        from vllm._genesis.compat.schema_validator import validate_entry
    except Exception as e:
        return False, [f"schema_validator import failed: {e}"]

    pid = plugin.get("patch_id") or plugin.get("name") or "<unknown>"

    # Plugin-API metadata that doesn't belong in the PATCH_REGISTRY value
    plugin_only_fields = ("patch_id", "_plugin_origin")
    schema_dict = {k: v for k, v in plugin.items() if k not in plugin_only_fields}

    issues = validate_entry(pid, schema_dict)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        return False, [f"{i.field}: {i.message}" for i in errors]
    return True, []


def _check_collision(plugin: dict[str, Any]) -> tuple[bool, str | None]:
    """A plugin can't claim a patch_id already in core PATCH_REGISTRY."""
    pid = plugin.get("patch_id")
    if not pid:
        return False, "plugin has no patch_id"
    try:
        from vllm._genesis.dispatcher import PATCH_REGISTRY
    except Exception:
        return True, None  # conservative pass if dispatcher unavailable
    if pid in PATCH_REGISTRY:
        existing = PATCH_REGISTRY[pid]
        # Check if the existing entry is itself a plugin (re-registration is OK)
        if existing.get("lifecycle") == "community" and \
                existing.get("_plugin_origin"):
            return True, None
        return False, (
            f"patch_id {pid!r} collides with core PATCH_REGISTRY entry "
            f"(title: {existing.get('title', '?')!r})"
        )
    return True, None


# ─── Public API ──────────────────────────────────────────────────────────


def discover_plugins() -> list[dict[str, Any]]:
    """Walk entry-points, load + validate + dedupe. Returns the list of
    plugin patches that pass all checks. Honors GENESIS_ALLOW_PLUGINS
    gate: returns [] when env unset.
    """
    if not _plugins_enabled():
        return []

    survivors: list[dict[str, Any]] = []
    for ep in _discover_entry_points():
        loaded = _load_one_plugin(ep)
        for plugin in loaded:
            # Validate shape against schema
            ok, reasons = _validate_plugin(plugin)
            if not ok:
                log.warning(
                    "plugin %r SKIPPED — schema violation(s): %s",
                    plugin.get("patch_id", ep.name), "; ".join(reasons),
                )
                continue

            # Check for collision with core registry
            ok, reason = _check_collision(plugin)
            if not ok:
                log.warning(
                    "plugin %r SKIPPED — %s",
                    plugin.get("patch_id", ep.name), reason,
                )
                continue

            survivors.append(plugin)

    return survivors


# Track plugins we registered so unregister_plugins() is idempotent
_REGISTERED_PLUGIN_IDS: set[str] = set()


def register_plugins() -> int:
    """Discover + register plugins into PATCH_REGISTRY. Returns the
    count of newly registered plugins. Idempotent — calling twice
    doesn't double-register.

    Each registered plugin is added to PATCH_REGISTRY with its
    auto-tagged lifecycle (community) and provenance metadata. The
    A3/D2 validator + lifecycle audit + doctor will all see them like
    any other patch but with the community-state badge.
    """
    plugins = discover_plugins()
    if not plugins:
        return 0

    try:
        from vllm._genesis.dispatcher import PATCH_REGISTRY
    except Exception as e:
        log.warning("PATCH_REGISTRY unavailable, skipping plugin register: %s", e)
        return 0

    n = 0
    for plugin in plugins:
        pid = plugin["patch_id"]
        if pid in PATCH_REGISTRY:
            # Already there (idempotent reregister). Skip.
            continue
        # Strip non-PATCH_REGISTRY fields before registration so schema
        # validator on the live registry stays clean (`_plugin_origin`
        # is informational metadata, not a registry field)
        registry_entry = {k: v for k, v in plugin.items()
                          if k not in ("_plugin_origin",)}
        # But preserve provenance via 'community_credit' string
        if "_plugin_origin" in plugin:
            origin = plugin["_plugin_origin"]
            existing_credit = registry_entry.get("community_credit", "")
            if origin not in existing_credit:
                registry_entry["community_credit"] = (
                    f"{existing_credit} [via plugin: {origin}]"
                ).strip()
        PATCH_REGISTRY[pid] = registry_entry
        _REGISTERED_PLUGIN_IDS.add(pid)
        n += 1
        log.info(
            "[Genesis plugins] registered community patch %r "
            "(via %s)", pid, plugin.get("_plugin_origin", "?"),
        )
    return n


def unregister_plugins() -> int:
    """Remove all plugin-registered patches from PATCH_REGISTRY.
    Returns the count removed. Useful for tests + dynamic plugin
    refresh."""
    try:
        from vllm._genesis.dispatcher import PATCH_REGISTRY
    except Exception:
        return 0
    n = 0
    for pid in list(_REGISTERED_PLUGIN_IDS):
        if pid in PATCH_REGISTRY:
            del PATCH_REGISTRY[pid]
            n += 1
        _REGISTERED_PLUGIN_IDS.discard(pid)
    return n


# ─── Phase 5c — apply_callable resolution + apply lifecycle ──────────────


def _resolve_apply_callable(spec: Any):
    """Resolve a plugin's apply_callable to an actual callable.

    `spec` may be:
      - already a callable → returned as-is
      - a string of the form "module.path:attr" → imported via
        importlib, then attribute-resolved
      - None / empty / unparseable → returned as None

    Errors are caught (returns None) so plugin failures never propagate
    out of discovery.
    """
    if spec is None or spec == "":
        return None
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        return None

    # Standard "module.path:attr" entry-point format
    if ":" not in spec:
        return None
    module_path, _, attr_path = spec.partition(":")
    if not module_path or not attr_path:
        return None

    try:
        import importlib
        mod = importlib.import_module(module_path)
    except Exception as e:
        log.warning(
            "[plugin] could not import module %r for apply_callable %r: %s",
            module_path, spec, e,
        )
        return None

    # Attribute path may itself have dots (rare but supported)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            log.warning(
                "[plugin] apply_callable %r: attribute %r not found on %s",
                spec, part, getattr(obj, "__name__", module_path),
            )
            return None
    if not callable(obj):
        log.warning(
            "[plugin] apply_callable %r resolved to non-callable %s",
            spec, type(obj).__name__,
        )
        return None
    return obj


def apply_plugin_patch(plugin: dict[str, Any]) -> tuple[str, str]:
    """Apply one community plugin's patch. Returns (status, reason).

    Status:
      - 'applied'  → apply_callable ran and returned ('applied', ...)
      - 'skipped'  → env flag unset OR plugin has no apply_callable
      - 'failed'   → apply_callable raised OR resolution failed

    Mirrors the (status, reason) contract used by core wiring modules
    so apply_all can render plugin patches in the same matrix.
    """
    pid = plugin.get("patch_id", "<unknown_plugin>")
    env_flag = plugin.get("env_flag", "")
    apply_spec = plugin.get("apply_callable")

    # No apply_callable → metadata-only plugin (Phase 5b style). Skip
    # cleanly with a clear reason.
    if not apply_spec:
        return "skipped", (
            f"{pid}: metadata-only plugin (no apply_callable declared)"
        )

    # Env-flag gate (operator opt-in to engage)
    env_value = os.environ.get(env_flag, "") if env_flag else ""
    if env_value.strip().lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            f"{pid}: opt-in only — set {env_flag}=1 to engage"
        )

    # Resolve the callable
    fn = _resolve_apply_callable(apply_spec)
    if fn is None:
        return "failed", (
            f"{pid}: apply_callable {apply_spec!r} could not be resolved"
        )

    # Invoke with full error isolation
    try:
        result = fn()
    except Exception as e:
        log.exception("[plugin %s] apply_callable raised", pid)
        return "failed", f"{pid}: {type(e).__name__}: {e}"

    # Normalize return value. Plugin may return:
    #   - tuple(status, reason) — preferred contract
    #   - any other type → treat as opaque success (give plugin benefit
    #     of doubt) and report what they returned
    if isinstance(result, tuple) and len(result) == 2:
        status, reason = result
        if status not in ("applied", "skipped", "failed"):
            return "failed", (
                f"{pid}: invalid status {status!r} returned by plugin "
                f"(expected applied/skipped/failed)"
            )
        return str(status), str(reason)
    if isinstance(result, str):
        return "applied", f"{pid}: {result}"
    if result is None:
        return "applied", f"{pid}: plugin executed (no message)"
    return "applied", (
        f"{pid}: plugin executed with non-tuple return type "
        f"({type(result).__name__})"
    )


def apply_all_plugins() -> dict[str, int]:
    """Walk discovered plugins and call apply_plugin_patch on each.

    Returns a stats dict: {total, applied, skipped, failed}. Honors
    the GENESIS_ALLOW_PLUGINS gate (returns zeroes when closed).
    """
    plugins = discover_plugins()
    stats = {"total": 0, "applied": 0, "skipped": 0, "failed": 0}
    for plugin in plugins:
        stats["total"] += 1
        status, reason = apply_plugin_patch(plugin)
        if status not in stats:
            # Defensive: unexpected status. Count as failed.
            stats["failed"] += 1
            log.warning(
                "[plugin %s] unexpected status %r — %s",
                plugin.get("patch_id"), status, reason,
            )
            continue
        stats[status] += 1
        log.info(
            "[Genesis plugin] %s %s — %s",
            status.upper(), plugin.get("patch_id", "?"), reason[:120],
        )
    return stats


# ─── CLI ─────────────────────────────────────────────────────────────────


def _format_plugin(plugin: dict[str, Any]) -> list[str]:
    L = []
    L.append(f"  Plugin: {plugin.get('patch_id', '?')}")
    L.append(f"    title:           {plugin.get('title', '?')}")
    L.append(f"    env_flag:        {plugin.get('env_flag', '?')}")
    L.append(f"    category:        {plugin.get('category', 'uncategorized')}")
    L.append(f"    lifecycle:       {plugin.get('lifecycle', '?')}")
    if plugin.get("community_credit"):
        L.append(f"    community credit: {plugin['community_credit']}")
    if plugin.get("_plugin_origin"):
        L.append(f"    origin:          {plugin['_plugin_origin']}")
    if plugin.get("credit"):
        L.append(f"    credit:          {plugin['credit'][:80]}")
    return L


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.plugins",
        description="Manage Genesis community plugin entry-points.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List discovered plugins")
    sp_list.add_argument("--json", action="store_true")

    sp_show = sub.add_parser("show", help="Display a plugin")
    sp_show.add_argument("patch_id")

    sub.add_parser("validate",
                   help="Walk plugins, exit 1 on schema violations")

    args = parser.parse_args(argv)

    if not _plugins_enabled() and args.cmd != "validate":
        # Be informative: tell operator why no plugins show
        if args.cmd == "list":
            print("Plugin discovery is OFF — set GENESIS_ALLOW_PLUGINS=1 "
                  "to enable.")
            print("(0 plugins discovered)")
            return 0

    plugins = discover_plugins()

    if args.cmd == "list":
        if args.json:
            print(json.dumps(plugins, indent=2, default=str))
            return 0
        print("=" * 72)
        print(f"Genesis community plugins — {len(plugins)} discovered")
        print("=" * 72)
        if not plugins:
            print("  (no plugins installed; see "
                  "https://github.com/Sandermage/genesis-vllm-patches#plugins)")
            return 0
        for p in plugins:
            for line in _format_plugin(p):
                print(line)
            print()
        return 0

    if args.cmd == "show":
        match = next(
            (p for p in plugins if p.get("patch_id") == args.patch_id), None,
        )
        if match is None:
            print(f"plugin {args.patch_id!r} not found", file=sys.stderr)
            return 2
        print("=" * 72)
        print(f"Genesis plugin — {args.patch_id}")
        print("=" * 72)
        for line in _format_plugin(match):
            print(line)
        return 0

    if args.cmd == "validate":
        # Walk entry points, validate each, report
        issues = []
        eps = _discover_entry_points()

        # If gate is closed, validation still works (operator can check
        # plugins before turning them on)
        for ep in eps:
            try:
                loaded = ep.load()
                result = loaded()
            except Exception as e:
                issues.append((ep.name, f"load/call raised: {e}"))
                continue

            # Normalize
            items = [result] if isinstance(result, dict) else (
                result if isinstance(result, list) else []
            )
            for it in items:
                if not isinstance(it, dict):
                    issues.append((ep.name, "non-dict in returned list"))
                    continue
                # Forced community lifecycle for validation purposes
                test_dict = {**it, "lifecycle": "community"}
                if "community_credit" not in test_dict:
                    test_dict["community_credit"] = f"plugin: {ep.name}"
                if "patch_id" not in test_dict:
                    test_dict["patch_id"] = f"PLUGIN_{ep.name.upper()}"
                ok, reasons = _validate_plugin(test_dict)
                if not ok:
                    issues.append((ep.name, "; ".join(reasons)))

        if not issues:
            print(f"✓ {len(eps)} plugin entry-point(s) pass schema validation")
            return 0

        print(f"✗ {len(issues)} plugin issue(s) found:", file=sys.stderr)
        for name, reason in issues:
            print(f"  - {name}: {reason}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
