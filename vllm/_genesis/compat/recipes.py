# SPDX-License-Identifier: Apache-2.0
"""Genesis recipes — `python3 -m vllm._genesis.compat.recipes`.

A recipe captures everything needed to reproduce a Genesis launch:
  - hardware target (model_key, vllm pin, tensor parallel)
  - container settings (image, mounts, ports, resource limits)
  - env variables (Genesis env flags + system env)
  - vllm serve command-line args
  - expected metrics (for regression detection)
  - human-readable notes / quirks

Recipes are stored at $GENESIS_RECIPES_DIR (default ~/.genesis/recipes/)
as JSON. JSON, not YAML, because Genesis ships zero runtime
dependencies — we don't pull in PyYAML for this. The structure is
shallow enough that JSON is readable.

Workflow:
  # Save current PROD container as a recipe
  python3 -m vllm._genesis.compat.recipes save prod-v794 \\
      --from-container vllm-server-mtp-test \\
      --description "27B Lorbus INT4 + TQ k8v4 + 5 PN-family patches"

  # List local recipes
  python3 -m vllm._genesis.compat.recipes list

  # Display one recipe
  python3 -m vllm._genesis.compat.recipes show prod-v794

  # Generate launch script from a recipe
  python3 -m vllm._genesis.compat.recipes load prod-v794 --out start.sh

  # Delete
  python3 -m vllm._genesis.compat.recipes delete old-recipe

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("genesis.compat.recipes")


# Current recipe schema version. Bump on incompatible changes.
RECIPE_VERSION = "1.0"

# Recipe name validation — must be filesystem-safe (no path traversal,
# no separators, no hidden files).
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


# ─── Storage ─────────────────────────────────────────────────────────────


def _resolve_recipes_dir() -> Path:
    """Resolve recipes directory.
    Override via GENESIS_RECIPES_DIR; default ~/.genesis/recipes/."""
    override = os.environ.get("GENESIS_RECIPES_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path("~/.genesis/recipes").expanduser()


def _validate_name(name: str) -> str:
    """Reject path-traversal / unsafe characters in recipe names."""
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise ValueError(
            f"recipe name {name!r} invalid — must match "
            f"^[a-zA-Z0-9][a-zA-Z0-9._-]{{0,63}}$ "
            f"(no path separators, no '..', no leading dot)"
        )
    if name in (".", ".."):
        raise ValueError(f"recipe name {name!r} reserved")
    return name


def _path_for(name: str) -> Path:
    name = _validate_name(name)
    base = _resolve_recipes_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.json"


def save(name: str, recipe: dict[str, Any]) -> Path:
    """Persist `recipe` as JSON under recipes-dir. Returns the path."""
    p = _path_for(name)
    # Ensure schema version + name are set on the recipe
    recipe = dict(recipe)
    recipe.setdefault("genesis_recipe_version", RECIPE_VERSION)
    recipe["name"] = name
    if "created_at" not in recipe:
        recipe["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(recipe, indent=2, default=str))
    return p


def load(name: str) -> dict[str, Any] | None:
    """Read a recipe by name. Returns None if not found."""
    p = _path_for(name)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.warning("recipe %r is corrupt (%s) — returning None", name, e)
        return None


def list_names() -> list[str]:
    """List recipe names found in recipes-dir."""
    base = _resolve_recipes_dir()
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.json"))


def delete(name: str) -> bool:
    """Remove a recipe. Returns True if deleted, False if not found."""
    p = _path_for(name)
    if not p.is_file():
        return False
    p.unlink()
    return True


# ─── Validation ──────────────────────────────────────────────────────────


def validate_recipe(recipe: dict[str, Any]) -> list[str]:
    """Return list of human-readable issues. Empty list = clean.

    Errors (always reported):
      - missing required field (name, target, container, vllm_serve)
      - schema_version mismatch

    Warnings (informational):
      - unknown GENESIS_ENABLE_* env that doesn't match a registered patch
    """
    issues: list[str] = []
    if not isinstance(recipe, dict):
        return [f"recipe must be dict, got {type(recipe).__name__}"]

    # Required top-level fields
    for key in ("name",):
        if key not in recipe or not recipe[key]:
            issues.append(f"missing required field {key!r}")

    # Schema version
    sv = recipe.get("genesis_recipe_version")
    if sv and sv != RECIPE_VERSION:
        issues.append(
            f"schema_version mismatch: recipe has {sv!r}, "
            f"current is {RECIPE_VERSION!r}"
        )

    # Validate envs against PATCH_REGISTRY
    envs = recipe.get("envs") or {}
    if envs:
        try:
            from vllm._genesis.dispatcher import PATCH_REGISTRY
            known_flags = {
                meta.get("env_flag") for meta in PATCH_REGISTRY.values()
                if meta.get("env_flag")
            }
            for env_key in envs:
                if env_key.startswith("GENESIS_ENABLE_") and env_key not in known_flags:
                    issues.append(
                        f"unknown GENESIS_ENABLE_* env {env_key!r} — "
                        f"no patch registered with this flag"
                    )
        except Exception as e:
            log.debug("env validation skipped (%s)", e)

    return issues


# ─── URL adoption (community recipe sharing) ────────────────────────────


_DEFAULT_MAX_BODY_BYTES = 100 * 1024  # 100 KB; recipes are tiny by design


def _fetch_url_body(url: str, max_bytes: int = _DEFAULT_MAX_BODY_BYTES) -> str:
    """Fetch the body of `url` as text. Caps total bytes at `max_bytes`
    to refuse a malicious giant payload. Designed to be monkey-patched
    in tests."""
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "Genesis-recipe-adopt/1.0",
        "Accept": "application/json, text/plain;q=0.9, */*;q=0.5",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        # Read up to max_bytes + 1 — if we hit max_bytes+1 the body
        # exceeds the cap and should be refused.
        body = resp.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise RuntimeError(
                f"recipe at {url} exceeded {max_bytes} byte limit "
                "(refusing to load oversized recipe)"
            )
    return body.decode("utf-8", errors="strict")


def adopt_recipe(
    url: str,
    name: str,
    *,
    allow_http: bool = False,
    max_bytes: int = _DEFAULT_MAX_BODY_BYTES,
) -> dict[str, Any] | None:
    """Pull a recipe from `url`, validate, save under `name`. Returns
    the saved recipe dict on success.

    Security model:
      - HTTPS-only by default; HTTP refused unless allow_http=True
      - Body capped to max_bytes (default 100 KB)
      - Schema-validated before saving (no garbage to disk)
      - Name-validated by `_validate_name()` (no path traversal)

    Raises ValueError on bad URL / invalid recipe shape.
    Raises RuntimeError on network / size errors.
    """
    # URL shape check
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")
    lower = url.lower()
    if lower.startswith("https://"):
        pass
    elif lower.startswith("http://"):
        if not allow_http:
            raise ValueError(
                f"refusing to fetch over plain HTTP: {url!r}. "
                "Pass allow_http=True (CLI: --allow-http) to override "
                "for testing, but production should use HTTPS."
            )
    else:
        raise ValueError(
            f"unsupported URL scheme in {url!r} — must be https:// "
            "(or http:// with allow_http)"
        )

    # Validate target name BEFORE fetching (cheap fast-fail)
    _validate_name(name)

    # Fetch
    body = _fetch_url_body(url, max_bytes)

    # Parse + schema-validate
    try:
        recipe = json.loads(body)
    except Exception as e:
        raise ValueError(f"recipe at {url!r} is not valid JSON: {e}")
    if not isinstance(recipe, dict):
        raise ValueError(
            f"recipe at {url!r} is not a JSON object (got "
            f"{type(recipe).__name__})"
        )

    # Name in the dict is replaced with the operator's chosen name
    recipe = dict(recipe)
    recipe["name"] = name

    issues = validate_recipe(recipe)
    errors = [i for i in issues if "missing" in i.lower()
                                  or "schema_version" in i.lower()]
    if errors:
        raise ValueError(
            f"recipe at {url!r} failed validation: " + "; ".join(errors)
        )

    # Adoption-specific stricter check: a remote recipe must declare
    # AT LEAST one substantive field (target / container / vllm_serve /
    # envs / mounts). A near-empty {"name": "..."} would technically
    # validate clean but produces no useful launch — refuse here so we
    # don't quietly save garbage from an untrusted URL.
    substantive_fields = (
        "target", "container", "vllm_serve", "envs", "mounts",
        "expected_metrics", "notes",
    )
    declared = [f for f in substantive_fields if recipe.get(f)]
    if not declared:
        raise ValueError(
            f"recipe at {url!r} failed adoption validation: "
            f"no substantive fields (expected at least one of "
            f"{substantive_fields}). Refusing to save a near-empty recipe."
        )

    # Provenance: tag with origin URL + adoption timestamp
    recipe["_adopted_from"] = url
    recipe["_adopted_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
    )

    # Save
    save(name, recipe)
    return recipe


# ─── docker inspect → recipe ────────────────────────────────────────────


def _docker_inspect(container_name: str) -> str:
    """Wrap `docker inspect <name>` so it can be monkey-patched in tests."""
    out = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout


def _parse_vllm_command(cmd_list: list[str]) -> dict[str, Any]:
    """Parse a vllm serve command-line into a structured dict.

    The command typically arrives as ['-c', 'set -e; ... exec vllm serve --model X --tp 2 ...'].
    We extract the serve args by splitting on whitespace and walking flags.
    """
    if not cmd_list:
        return {}
    # Concatenate to one string for simpler parsing
    cmd_str = " ".join(cmd_list)
    # Locate `vllm serve ...` segment
    serve_idx = cmd_str.find("vllm serve")
    if serve_idx < 0:
        return {"raw_command": cmd_str}
    serve_segment = cmd_str[serve_idx:]
    tokens = serve_segment.split()
    result: dict[str, Any] = {"raw_command": cmd_str}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        flag = t[2:].replace("-", "_")
        # Look ahead for value
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            value = tokens[i + 1]
            # Type coercion
            try:
                value = int(value)
            except (ValueError, TypeError):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    # Not numeric — keep raw string value (e.g. enum-like flag)
                    pass
            result[flag] = value
            i += 2
        else:
            result[flag] = True
            i += 1
    return result


def from_container(container_name: str) -> dict[str, Any] | None:
    """Build a recipe dict from a running docker container."""
    try:
        raw = _docker_inspect(container_name)
    except Exception as e:
        log.warning("docker inspect %r failed: %s", container_name, e)
        return None

    try:
        data = json.loads(raw)
    except Exception as e:
        log.warning("docker inspect output not JSON: %s", e)
        return None

    if not data or not isinstance(data, list):
        return None
    info = data[0]
    cfg = info.get("Config", {})
    host_cfg = info.get("HostConfig", {})

    # Envs as dict
    envs: dict[str, str] = {}
    for kv in cfg.get("Env", []) or []:
        if "=" in kv:
            k, _, v = kv.partition("=")
            envs[k] = v

    # Mounts
    mounts: list[dict[str, Any]] = []
    for bind in host_cfg.get("Binds") or []:
        # Format: "host:container[:flags]"
        parts = bind.split(":")
        m = {"host": parts[0], "container": parts[1] if len(parts) > 1 else ""}
        if len(parts) > 2:
            m["readonly"] = "ro" in parts[2:]
        mounts.append(m)

    # Ports
    ports = []
    for p in (host_cfg.get("PortBindings") or {}):
        ports.append(p.split("/")[0])

    return {
        "genesis_recipe_version": RECIPE_VERSION,
        "name": container_name.lstrip("/"),
        "description": f"Captured from running container {container_name!r}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_by": os.environ.get("USER", "unknown"),
        "container": {
            "image": cfg.get("Image", ""),
            "name": (info.get("Name") or "").lstrip("/"),
            "ports": ports,
            "shm_size": _format_size(host_cfg.get("ShmSize", 0)),
            "memory": _format_size(host_cfg.get("Memory", 0)),
        },
        "mounts": mounts,
        "envs": envs,
        "vllm_serve": _parse_vllm_command(cfg.get("Cmd") or []),
        "target": {
            "vllm_pin": "(detected at boot — see envs)",
        },
    }


def _format_size(bytes_val: int) -> str:
    """Convert byte count to human-readable docker-style size string."""
    if bytes_val <= 0:
        return "0"
    for suffix, factor in [("g", 2**30), ("m", 2**20), ("k", 2**10)]:
        if bytes_val >= factor:
            return f"{bytes_val // factor}{suffix}"
    return str(bytes_val)


# ─── Recipe → launch script ──────────────────────────────────────────────


# ─── Diff (community A/B compare) ───────────────────────────────────────


# Keys that vary by who-saved-when and don't reflect launch behavior.
# Excluded from `diff_recipes` so two operators with the same effective
# config see "no differences" rather than meaningless metadata noise.
_DIFF_PROVENANCE_KEYS = frozenset({
    "created_at", "created_by",
    "_adopted_from", "_adopted_at",
    "name", "description",
    "genesis_recipe_version",
})


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys for stable diffing.

    Lists are kept as-is (compared opaquely — order matters), since
    most recipe lists (ports, mounts) are short and reordering changes
    behavior.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def diff_recipes(
    a: dict[str, Any], b: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Structured diff between two recipes.

    Returns ``{"added": {key: value}, "removed": {key: value},
    "changed": {key: (old, new)}}`` keyed by dotted paths into the
    recipe (e.g. ``vllm_serve.max_model_len``).

    Provenance fields (``created_at``, ``created_by``, ``_adopted_*``,
    ``name``, ``description``, ``genesis_recipe_version``) are
    excluded — two operators saving the same effective config at
    different times should diff clean.
    """
    fa = _flatten(a)
    fb = _flatten(b)

    def _is_provenance(key: str) -> bool:
        head = key.split(".", 1)[0]
        return head in _DIFF_PROVENANCE_KEYS

    fa = {k: v for k, v in fa.items() if not _is_provenance(k)}
    fb = {k: v for k, v in fb.items() if not _is_provenance(k)}

    added = {k: fb[k] for k in fb if k not in fa}
    removed = {k: fa[k] for k in fa if k not in fb}
    changed = {
        k: (fa[k], fb[k])
        for k in fa
        if k in fb and fa[k] != fb[k]
    }
    return {"added": added, "removed": removed, "changed": changed}


def _format_diff(
    diff: dict[str, dict[str, Any]],
    name_a: str,
    name_b: str,
) -> list[str]:
    """Format a structured diff for human consumption."""
    lines: list[str] = []
    if not diff["added"] and not diff["removed"] and not diff["changed"]:
        lines.append(f"recipes {name_a!r} and {name_b!r} are identical "
                     "(no differences ignoring provenance fields)")
        return lines

    lines.append(f"diff: {name_a}  →  {name_b}")
    if diff["changed"]:
        lines.append("")
        lines.append("CHANGED:")
        for k in sorted(diff["changed"]):
            old, new = diff["changed"][k]
            lines.append(f"  {k}")
            lines.append(f"    - {old!r}")
            lines.append(f"    + {new!r}")
    if diff["removed"]:
        lines.append("")
        lines.append(f"REMOVED (only in {name_a!r}):")
        for k in sorted(diff["removed"]):
            lines.append(f"  - {k} = {diff['removed'][k]!r}")
    if diff["added"]:
        lines.append("")
        lines.append(f"ADDED (only in {name_b!r}):")
        for k in sorted(diff["added"]):
            lines.append(f"  + {k} = {diff['added'][k]!r}")
    return lines


# ────────────────────────────────────────────────────────────────────────


def to_launch_script(recipe: dict[str, Any]) -> str:
    """Generate a bash launch script from a recipe."""
    name = recipe.get("name", "unnamed")
    desc = recipe.get("description", "")
    created_at = recipe.get("created_at", "")
    container = recipe.get("container") or {}
    container_name = container.get("name") or name
    image = container.get("image") or "vllm/vllm-openai:nightly"
    ports = container.get("ports") or [8000]
    shm = container.get("shm_size") or "8g"
    mem = container.get("memory") or "64g"

    envs = recipe.get("envs") or {}
    mounts = recipe.get("mounts") or []
    vllm_serve = recipe.get("vllm_serve") or {}

    L = []
    L.append("#!/usr/bin/env bash")
    L.append("# ════════════════════════════════════════════════════════════════════")
    L.append(f"# Genesis launch script generated from recipe '{name}'")
    if desc:
        L.append(f"# Description: {desc}")
    if created_at:
        L.append(f"# Recipe created: {created_at}")
    L.append(f"# Generated by: python3 -m vllm._genesis.compat.recipes load {name}")
    L.append("# ════════════════════════════════════════════════════════════════════")
    L.append("set -euo pipefail")
    L.append("")
    L.append(f"docker stop {container_name} 2>/dev/null || true")
    L.append(f"docker rm   {container_name} 2>/dev/null || true")
    L.append("")
    L.append("docker run -d \\")
    L.append(f"  --name {container_name} \\")
    L.append(f"  --shm-size={shm} --memory={mem} \\")
    for port in ports:
        L.append(f"  -p {port}:{port} \\")
    L.append("  --gpus all \\")
    L.append("  --security-opt label=disable --entrypoint /bin/bash \\")

    for m in mounts:
        host = m.get("host", "")
        ctr = m.get("container", "")
        ro = ":ro" if m.get("readonly") else ""
        if host and ctr:
            L.append(f"  -v {host}:{ctr}{ro} \\")

    # Envs in deterministic order
    for key in sorted(envs):
        val = envs[key]
        # Quote values with spaces
        if any(c in str(val) for c in " \t\""):
            val = f'"{val}"'
        L.append(f"  -e {key}={val} \\")

    L.append(f"  {image} -c \\")

    # Build the inner command
    inner = ["set -e"]
    inner.append("python3 -m vllm._genesis.patches.apply_all")

    serve_args = []
    raw = vllm_serve.get("raw_command")
    if raw and "vllm serve" in raw:
        # Use raw command if available — preserves exact tokenization
        idx = raw.find("vllm serve")
        serve_args.append("exec " + raw[idx:].strip())
    else:
        # Build from structured fields
        parts = ["exec vllm serve"]
        for k in sorted(vllm_serve):
            if k == "raw_command":
                continue
            v = vllm_serve[k]
            flag = "--" + k.replace("_", "-")
            if v is True:
                parts.append(flag)
            elif v is False:
                continue
            else:
                parts.append(f"{flag} {v}")
        serve_args.append(" ".join(parts))

    inner_str = " ; ".join(inner) + " ; " + " ; ".join(serve_args)
    L.append(f'  "{inner_str}"')
    L.append("")
    L.append("sleep 3")
    L.append(f'docker logs --tail 5 {container_name} 2>&1 | sed "s/^/  /"')
    L.append(f'echo "[recipe:{name}] container started; tail with: docker logs -f {container_name}"')
    return "\n".join(L) + "\n"


# ─── CLI ─────────────────────────────────────────────────────────────────


def _format_show(recipe: dict[str, Any]) -> list[str]:
    L = []
    L.append("=" * 72)
    L.append(f"Recipe: {recipe.get('name', '?')}")
    L.append("=" * 72)
    L.append(f"  Description: {recipe.get('description', '(none)')}")
    L.append(f"  Created at:  {recipe.get('created_at', '(unknown)')}")
    L.append(f"  Created by:  {recipe.get('created_by', '(unknown)')}")
    L.append(f"  Schema:      v{recipe.get('genesis_recipe_version', '?')}")
    L.append("")

    target = recipe.get("target") or {}
    if target:
        L.append("  Target:")
        for k, v in sorted(target.items()):
            L.append(f"    {k}: {v}")
        L.append("")

    container = recipe.get("container") or {}
    if container:
        L.append("  Container:")
        for k, v in sorted(container.items()):
            L.append(f"    {k}: {v}")
        L.append("")

    envs = recipe.get("envs") or {}
    if envs:
        L.append(f"  Envs ({len(envs)} variables):")
        genesis_envs = {k: v for k, v in envs.items() if k.startswith("GENESIS_")}
        for k in sorted(genesis_envs):
            L.append(f"    {k}={genesis_envs[k]}")
        other = {k: v for k, v in envs.items() if not k.startswith("GENESIS_")}
        if other:
            L.append(f"    + {len(other)} non-GENESIS env(s) (omitted)")
        L.append("")

    vllm = recipe.get("vllm_serve") or {}
    if vllm:
        L.append("  vllm serve:")
        for k in sorted(vllm):
            if k == "raw_command":
                continue
            L.append(f"    --{k.replace('_', '-')} {vllm[k]}")
        L.append("")

    expected = recipe.get("expected_metrics") or {}
    if expected:
        L.append("  Expected metrics:")
        for k, v in sorted(expected.items()):
            L.append(f"    {k}: {v}")
        L.append("")

    if recipe.get("notes"):
        L.append("  Notes:")
        for line in str(recipe["notes"]).split("\n"):
            L.append(f"    {line}")
        L.append("")

    L.append("=" * 72)
    return L


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.recipes",
        description="Manage Genesis launch recipes",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    sub.add_parser("list", help="List local recipes")

    # show
    sp_show = sub.add_parser("show", help="Display one recipe")
    sp_show.add_argument("name")
    sp_show.add_argument("--json", action="store_true")

    # save
    sp_save = sub.add_parser("save", help="Save a recipe")
    sp_save.add_argument("name")
    sp_save.add_argument("--from-container",
                         help="Capture from a running docker container")
    sp_save.add_argument("--description", default="")

    # load (= generate launch script)
    sp_load = sub.add_parser("load",
                              help="Generate a bash launch script from a recipe")
    sp_load.add_argument("name")
    sp_load.add_argument("--out", default=None,
                         help="Output path (default: stdout)")

    # delete
    sp_del = sub.add_parser("delete", help="Delete a recipe")
    sp_del.add_argument("name")

    # adopt — pull from URL
    sp_adopt = sub.add_parser(
        "adopt",
        help="Pull a recipe from a URL (community recipe sharing)",
    )
    sp_adopt.add_argument("url",
                          help="HTTPS URL to a JSON recipe")
    sp_adopt.add_argument("name",
                          help="Local name to save the adopted recipe as")
    sp_adopt.add_argument("--allow-http", action="store_true",
                          help="Allow plain HTTP (default: HTTPS-only)")
    sp_adopt.add_argument("--max-bytes", type=int,
                          default=_DEFAULT_MAX_BODY_BYTES,
                          help=f"Refuse bodies > N bytes "
                               f"(default {_DEFAULT_MAX_BODY_BYTES})")

    # validate
    sp_val = sub.add_parser("validate", help="Validate a recipe's shape")
    sp_val.add_argument("name")

    # diff — A/B compare two recipes
    sp_diff = sub.add_parser(
        "diff",
        help="Compare two recipes (community A/B)",
    )
    sp_diff.add_argument("name_a", help="First recipe name (or path)")
    sp_diff.add_argument("name_b", help="Second recipe name (or path)")
    sp_diff.add_argument("--json", action="store_true",
                         help="Output structured diff as JSON")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        names = list_names()
        if not names:
            print("(no recipes — see `recipes save --help` to create one)")
            return 0
        print(f"=== {len(names)} recipe(s) at {_resolve_recipes_dir()} ===")
        for n in names:
            r = load(n) or {}
            desc = r.get("description") or ""
            ts = r.get("created_at") or ""
            print(f"  {n:<32} {ts}  {desc[:40]}")
        return 0

    if args.cmd == "show":
        rec = load(args.name)
        if rec is None:
            print(f"unknown recipe: {args.name!r}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(rec, indent=2, default=str))
        else:
            for line in _format_show(rec):
                print(line)
        return 0

    if args.cmd == "save":
        if args.from_container:
            rec = from_container(args.from_container)
            if rec is None:
                print(f"failed to inspect container {args.from_container!r}",
                      file=sys.stderr)
                return 2
        else:
            print("save without --from-container not yet supported "
                  "— provide a container name", file=sys.stderr)
            return 2
        if args.description:
            rec["description"] = args.description
        path = save(args.name, rec)
        print(f"✓ saved recipe {args.name!r} → {path}")
        return 0

    if args.cmd == "load":
        rec = load(args.name)
        if rec is None:
            print(f"unknown recipe: {args.name!r}", file=sys.stderr)
            return 2
        script = to_launch_script(rec)
        if args.out:
            out = Path(args.out)
            out.write_text(script)
            out.chmod(0o755)
            print(f"✓ launch script written to {out}")
        else:
            print(script)
        return 0

    if args.cmd == "delete":
        ok = delete(args.name)
        if ok:
            print(f"✓ deleted recipe {args.name!r}")
            return 0
        print(f"recipe {args.name!r} not found", file=sys.stderr)
        return 2

    if args.cmd == "adopt":
        try:
            adopt_recipe(
                args.url, args.name,
                allow_http=args.allow_http,
                max_bytes=args.max_bytes,
            )
        except ValueError as e:
            print(f"✗ adopt rejected: {e}", file=sys.stderr)
            return 2
        except RuntimeError as e:
            print(f"✗ adopt failed: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            print(f"✗ adopt failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        print(f"✓ adopted recipe {args.name!r} from {args.url}")
        print(f"  Stored at: {_path_for(args.name)}")
        print(f"  Inspect: python3 -m vllm._genesis.compat.recipes "
              f"show {args.name}")
        return 0

    if args.cmd == "validate":
        rec = load(args.name)
        if rec is None:
            print(f"unknown recipe: {args.name!r}", file=sys.stderr)
            return 2
        issues = validate_recipe(rec)
        if not issues:
            print(f"✓ recipe {args.name!r} validates clean")
            return 0
        print(f"recipe {args.name!r} has {len(issues)} issue(s):", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        return 1

    if args.cmd == "diff":
        rec_a = load(args.name_a)
        if rec_a is None:
            print(f"unknown recipe: {args.name_a!r}", file=sys.stderr)
            return 2
        rec_b = load(args.name_b)
        if rec_b is None:
            print(f"unknown recipe: {args.name_b!r}", file=sys.stderr)
            return 2
        d = diff_recipes(rec_a, rec_b)
        if args.json:
            # Tuples → lists for JSON serializability
            d_json = {
                "added": d["added"],
                "removed": d["removed"],
                "changed": {k: list(v) for k, v in d["changed"].items()},
            }
            print(json.dumps(d_json, indent=2, default=str))
        else:
            for line in _format_diff(d, args.name_a, args.name_b):
                print(line)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
