# SPDX-License-Identifier: Apache-2.0
"""Genesis telemetry — opt-in anonymized stats reporting.

Helps the Genesis community see "what configs work in the wild"
without collecting any personally-identifiable information. Strict
two-gate opt-in: discovery + upload are independent envs.

Gates
─────
  GENESIS_ENABLE_TELEMETRY=1      master gate; default OFF.
  GENESIS_TELEMETRY_UPLOAD=1      upload gate; default OFF.
  GENESIS_TELEMETRY_DIR=<path>    storage override; default ~/.genesis/telemetry/
  GENESIS_TELEMETRY_INCLUDE_PLUGIN_NAMES=1
                                  include community-plugin patch_ids in
                                  reports (default OFF — names could
                                  fingerprint a small group of operators).

Anonymized data the report DOES include
────────────────────────────────────────
  - Stable random instance_id (UUID-shaped, persisted locally only)
  - Hardware class (e.g. "rtx_a5000", "rtx_4090") — categorical
  - Compute capability (sm_86 etc.)
  - vllm / torch / triton / cuda version strings
  - Genesis version + commit (from version_check + git rev)
  - Detected model class (qwen3_5 / qwen3_next / etc.) + flags
    (is_hybrid / is_moe / is_turboquant / quant_format)
  - List of applied core patch IDs (P67, PN14, ...)
  - Skip count, lifecycle distribution (counts, not patch_ids)
  - Plugin count (NAMES default off; opt-in env)
  - Run timestamp

What the report does NOT include
────────────────────────────────
  - Hostname / IP / MAC
  - Username / home dir / paths
  - Container names / launch script paths
  - Env-variable VALUES (only env-FLAG presence)
  - Specific tokens, model paths, config secrets
  - Plugin-author git handles unless plugin opted in

Workflow
────────
  # See your status
  python3 -m vllm._genesis.compat.telemetry status

  # Inspect what would be reported (works only when master gate open)
  python3 -m vllm._genesis.compat.telemetry show

  # Manually collect + save a report
  python3 -m vllm._genesis.compat.telemetry collect

  # Clear local stash
  python3 -m vllm._genesis.compat.telemetry clear

Note: network upload is deferred for now. Reports are written locally
to GENESIS_TELEMETRY_DIR. A later release will add the actual upload
endpoint after the community dashboard is live.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import string
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("genesis.compat.telemetry")


SCHEMA_VERSION = "1.0"


# ─── Gate detection ──────────────────────────────────────────────────────


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """True when GENESIS_ENABLE_TELEMETRY is set truthy."""
    return _truthy(os.environ.get("GENESIS_ENABLE_TELEMETRY", ""))


def is_upload_enabled() -> bool:
    """Upload requires BOTH master gate AND upload gate."""
    return is_enabled() and _truthy(os.environ.get("GENESIS_TELEMETRY_UPLOAD", ""))


def _include_plugin_names() -> bool:
    return _truthy(os.environ.get("GENESIS_TELEMETRY_INCLUDE_PLUGIN_NAMES", ""))


# ─── Storage ─────────────────────────────────────────────────────────────


def _resolve_telemetry_dir() -> Path:
    override = os.environ.get("GENESIS_TELEMETRY_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path("~/.genesis/telemetry").expanduser()


def _ensure_dir() -> Path:
    base = _resolve_telemetry_dir()
    base.mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)
    return base


# ─── Stable anonymous instance ID ────────────────────────────────────────


_ID_ALPHABET = string.ascii_letters + string.digits


def get_or_create_instance_id() -> str:
    """Return the persistent anonymous instance ID, creating one on
    first call. Uses cryptographic randomness — NOT derived from any
    PII source (hostname / mac / username are NEVER consulted)."""
    base = _ensure_dir()
    f = base / "instance_id"
    if f.is_file():
        existing = f.read_text().strip()
        if existing:
            return existing
    # Fresh random — secrets.token_urlsafe → ~22-char URL-safe string
    new = "anon-" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(20))
    f.write_text(new + "\n")
    try:
        f.chmod(0o600)  # private to user
    except OSError:
        # chmod unsupported on Windows / network FS — anon-id still readable
        pass
    return new


# ─── Report assembly ─────────────────────────────────────────────────────


def _detect_hardware() -> dict[str, Any]:
    """Classify GPU hardware — categorical only, no serial / PCIe ID."""
    out: dict[str, Any] = {
        "gpu_class": "unknown", "gpu_count": 0,
        "compute_capability": None,
    }
    try:
        import torch
        if not torch.cuda.is_available():
            return out
        n = torch.cuda.device_count()
        out["gpu_count"] = n
        if n == 0:
            return out
        first = torch.cuda.get_device_name(0).lower()
        # Simple normalization
        for keyword in ("a5000", "a4000", "a6000",
                         "3090", "4090", "5090",
                         "h100", "h200", "rtx pro 6000", "blackwell"):
            if keyword in first:
                out["gpu_class"] = keyword.replace(" ", "_")
                break
        out["compute_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as e:
        log.debug("hardware detect failed: %s", e)
    return out


def _detect_software() -> dict[str, Any]:
    """Capture version strings — no env values, no paths."""
    try:
        from vllm._genesis.compat.version_check import detect_versions
        p = detect_versions()
        return {
            "vllm": p.vllm, "torch": p.torch, "triton": p.triton,
            "cuda_runtime": p.cuda_runtime,
            # Driver version is OK (kernel module, not user identity)
            "nvidia_driver": p.nvidia_driver,
            "python": p.python,
        }
    except Exception as e:
        log.debug("software detect failed: %s", e)
        return {}


def _detect_model() -> dict[str, Any]:
    """Categorical model info — no model name / path."""
    try:
        from vllm._genesis.compat.model_detect import get_model_profile
        profile = get_model_profile() or {}
    except Exception:
        return {"resolved": False}
    return {
        "resolved": bool(profile.get("resolved", False)),
        "model_class": profile.get("model_class"),
        "is_hybrid": profile.get("hybrid", profile.get("is_hybrid")),
        "is_moe": profile.get("moe", profile.get("is_moe")),
        "is_turboquant": profile.get("turboquant",
                                      profile.get("is_turboquant")),
        "quant_format": profile.get("quant_format"),
    }


def _summarize_patches() -> dict[str, Any]:
    """List applied patch IDs + lifecycle stats. No env values."""
    out: dict[str, Any] = {
        "applied": [], "skip_count": 0, "lifecycle_stats": {},
    }
    try:
        from vllm._genesis.dispatcher import (
            PATCH_REGISTRY, get_apply_matrix,
        )
        from vllm._genesis.compat.lifecycle import get_state
    except Exception:
        return out

    matrix = get_apply_matrix()
    out["applied"] = sorted({d["patch_id"] for d in matrix if d["applied"]})
    out["skip_count"] = sum(1 for d in matrix if not d["applied"])

    # Lifecycle distribution
    counts: dict[str, int] = {}
    for pid, meta in PATCH_REGISTRY.items():
        state = get_state(meta)
        counts[state] = counts.get(state, 0) + 1
    out["lifecycle_stats"] = counts

    return out


def _summarize_plugins() -> dict[str, Any]:
    """Plugin count by default; names only with extra opt-in."""
    out: dict[str, Any] = {"count": 0}
    try:
        from vllm._genesis.compat.plugins import discover_plugins
        plugins = discover_plugins()
    except Exception:
        return out
    out["count"] = len(plugins)
    if _include_plugin_names() and plugins:
        out["names"] = sorted(p.get("patch_id", "?") for p in plugins)
    return out


def _detect_genesis_version() -> dict[str, str]:
    """Best-effort Genesis version + commit from local git.

    Version is the single source of truth from `vllm/_genesis/__version__.py`
    (so we don't drift across modules). Commit is short SHA from local git
    if available.
    """
    out: dict[str, str] = {"version": "unknown", "commit": "unknown"}

    # Pull canonical version
    try:
        from vllm._genesis import __version__ as canonical_version
        out["version"] = canonical_version
    except Exception as e:
        log.debug("__version__ import failed: %s", e)

    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / ".git").is_dir():
        try:
            import subprocess
            r = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                out["commit"] = r.stdout.strip()
        except Exception as e:
            log.debug("git rev-parse failed: %s", e)
    return out


def collect_report() -> dict[str, Any]:
    """Assemble a fresh anonymized telemetry report."""
    genesis = _detect_genesis_version()
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instance_id": get_or_create_instance_id(),
        "genesis_version": genesis["version"],
        "genesis_commit": genesis["commit"],
        "hardware": _detect_hardware(),
        "software": _detect_software(),
        "model": _detect_model(),
        "patches": _summarize_patches(),
        "plugins": _summarize_plugins(),
    }


def save_report(report: dict[str, Any]) -> Path | None:
    """Persist a report to `<dir>/reports/<timestamp>.json`. Returns
    the path, or None if the master gate is closed (telemetry off →
    nothing written)."""
    if not is_enabled():
        return None

    base = _ensure_dir()
    ts = report.get("timestamp") or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
    )
    # File-safe timestamp; if a same-second collision exists, suffix
    # with a counter so all reports survive.
    base_fname = ts.replace(":", "-")
    out = base / "reports" / f"{base_fname}.json"
    counter = 1
    while out.exists():
        out = base / "reports" / f"{base_fname}_{counter:03d}.json"
        counter += 1
        if counter > 999:
            log.warning("[telemetry] too many same-second saves, overwriting")
            out = base / "reports" / f"{base_fname}_overflow.json"
            break
    out.write_text(json.dumps(report, indent=2, default=str))
    return out


def clear() -> int:
    """Delete all stored reports (instance_id preserved). Returns the
    count of deleted report files."""
    base = _resolve_telemetry_dir()
    reports = base / "reports"
    if not reports.is_dir():
        return 0
    n = 0
    for f in reports.iterdir():
        if f.is_file():
            f.unlink()
            n += 1
    return n


def upload_report(report: dict[str, Any]) -> dict[str, Any] | None:
    """Network upload — deferred until community dashboard exists.
    Returns None when upload gate is closed (always, for now).

    When the upload endpoint is live, this function will:
      1. Verify is_upload_enabled() is True
      2. POST the report to a public endpoint
      3. Return the server's response (status, accepted, etc.)

    For Phase 5d alpha, upload is permanently a no-op."""
    if not is_upload_enabled():
        return None
    log.info(
        "[telemetry] upload requested but Phase 5d ships local-first only; "
        "see CHANGELOG for community dashboard rollout"
    )
    return None


# ─── CLI ─────────────────────────────────────────────────────────────────


def _format_status() -> list[str]:
    L = []
    L.append("=" * 72)
    L.append("Genesis telemetry status")
    L.append("=" * 72)
    enabled = is_enabled()
    upload = is_upload_enabled()
    L.append(f"  Master gate (GENESIS_ENABLE_TELEMETRY): "
             f"{'ON' if enabled else 'OFF'}")
    L.append(f"  Upload gate (GENESIS_TELEMETRY_UPLOAD):  "
             f"{'ON' if upload else 'OFF'}")
    L.append(f"  Storage dir: {_resolve_telemetry_dir()}")
    if enabled:
        try:
            iid = get_or_create_instance_id()
            L.append(f"  Instance ID: {iid}")
        except Exception:
            # ID file unwritable (read-only FS) — skip line, status report continues
            pass
        # Count local reports
        reports = _resolve_telemetry_dir() / "reports"
        if reports.is_dir():
            n = sum(1 for _ in reports.iterdir() if _.is_file())
            L.append(f"  Local reports: {n}")
        else:
            L.append("  Local reports: 0")
    else:
        L.append("")
        L.append("  Telemetry is DISABLED. To enable:")
        L.append("    export GENESIS_ENABLE_TELEMETRY=1")
        L.append("  No data is collected when disabled.")
    L.append("=" * 72)
    return L


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.telemetry",
        description="Manage Genesis opt-in anonymized telemetry.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="Show telemetry on/off status")
    sub.add_parser("show", help="Print what a report would contain")
    sub.add_parser("collect", help="Collect + save a report")
    sub.add_parser("clear", help="Delete local stash of reports")
    args = parser.parse_args(argv)

    if args.cmd == "status":
        for line in _format_status():
            print(line)
        return 0

    if args.cmd == "show":
        if not is_enabled():
            print("Telemetry is OFF — set GENESIS_ENABLE_TELEMETRY=1 to "
                  "see what would be reported.", file=sys.stderr)
            return 1
        report = collect_report()
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.cmd == "collect":
        if not is_enabled():
            print("Telemetry is OFF — set GENESIS_ENABLE_TELEMETRY=1 first.",
                  file=sys.stderr)
            return 1
        report = collect_report()
        path = save_report(report)
        if path is None:
            print("Save failed (gate closed mid-call?)", file=sys.stderr)
            return 1
        print(f"✓ Report saved to {path}")
        return 0

    if args.cmd == "clear":
        n = clear()
        print(f"✓ Cleared {n} report(s)")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
