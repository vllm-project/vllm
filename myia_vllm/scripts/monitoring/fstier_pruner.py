#!/usr/bin/env python3
"""External pruner for the KV offload fs tier (TieringOffloadingSpec).

Why this exists
---------------
The fs tier (see `_iteration_log/v0320_fstier/NOTES.md`) has **no eviction and
no cap in code** (verified in-code 2026-10-20). Left alone it fills the disk:
24 GB in a 30-minute synthetic window. This script is the mandatory external
pruner that keeps it bounded.

Layout (content-addressed, one file per KV block):
    <root>/<model>_<hash>_r<TP rank>/<3 hex>/<block>
    blocks are 26_247_168 B (~25 MB) on the Swift-1.5 nvfp4 config.

Cap policy (arbitrated with Maintenance 2026-10-20, corrected 2026-10-21):
    cap = min(MAX_GIB, guest_free - 12% of guest total,
              for each guard mount: free - 12% of its total)

The tier physically lives in the Ubuntu WSL ext4.vhdx
(`C:\\Users\\MYIA\\AppData\\Local\\wsl\\{...}\\ext4.vhdx`, 850 GB, on C:),
NOT on D: — measured 2026-10-21 via the Lxss registry key. So the volume to
guard is C: (where the vhdx grows) plus the guest filesystem itself. D: at
90% is a separate host concern and must not cap this tier.

Safety rules
------------
- DRY-RUN by default; `--apply` is required to delete anything.
- Only files under `<root>/*_r<digits>/*/<file>` are candidates — the root and
  the model namespace directories are never removed.
- Files modified less than `--min-age-min` ago are left alone: a fresh block
  may be mid-promotion (lookup = file existence; a delete between lookup and
  load is the documented `kv_load_failure_policy='fail'` risk).
- No symlink following, no deletion outside the resolved root.
- Errors are collected per file and reported, never fatal for the whole run.

State file (readable from Windows and WSL; gitignored under myia_vllm/logs/):
    myia_vllm/logs/fstier_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone

GIB = 1024 ** 3
BLOCK_RE = re.compile(r"_r\d+$")  # namespace dir name suffix: <model>_<hash>_r<rank>
HEX_RE = re.compile(r"^[0-9a-fA-F]{1,3}$")  # shard dir (first hash hex chars)
MID_RE = re.compile(r"^[0-9a-fA-F]{1,2}_g\d+$")  # v0.32.0 mid dir: <2hex>_g<generation>


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_cap(root: str, guard_mounts: list[str], max_gib: float,
                reserve_pct: float, guest_floor_gib: float) -> dict:
    """cap = min(max_gib, guest_free - guest_floor, for each mount: free - reserve%)."""
    checks = {}
    availables = []
    # Guest root filesystem: absolute floor (it hosts the HF cache + WSL system,
    # a percentage of its total would be both huge and meaningless).
    try:
        usage = shutil.disk_usage(root)
        floor = int(guest_floor_gib * GIB)
        available = max(0, usage.free - floor)
        checks["guest_root"] = {
            "path": root, "total_bytes": usage.total, "free_bytes": usage.free,
            "floor_bytes": floor, "available_bytes": available,
        }
        availables.append(available)
    except OSError as exc:
        checks["guest_root"] = {"path": root, "error": str(exc)}
    # Guard mounts (host volumes backing the vhdx): percentage reserve.
    for m in guard_mounts:
        try:
            usage = shutil.disk_usage(m)
        except OSError as exc:
            checks[m] = {"path": m, "error": str(exc)}
            continue
        reserve = int(usage.total * reserve_pct / 100.0)
        available = max(0, usage.free - reserve)
        checks[m] = {
            "path": m, "total_bytes": usage.total, "free_bytes": usage.free,
            "reserve_bytes": reserve, "available_bytes": available,
        }
        availables.append(available)
    cap = min([int(max_gib * GIB)] + availables) if availables else 0
    return {
        "cap_bytes": cap,
        "max_gib": max_gib,
        "reserve_pct": reserve_pct,
        "guest_floor_gib": guest_floor_gib,
        "space_checks": checks,
    }


def collect(root: str, min_age_min: int) -> tuple[list[dict], list[str], int, int]:
    """Return (candidates sorted oldest-first, errors, files_total, bytes_total)."""
    now = time.time()
    min_age_s = min_age_min * 60
    candidates: list[dict] = []
    errors: list[str] = []
    files_total = 0
    bytes_total = 0

    if not os.path.isdir(root):
        return [], [f"root not found: {root}"], 0, 0

    for ns in os.scandir(root):
        if not ns.is_dir(follow_symlinks=False) or not BLOCK_RE.search(ns.name):
            continue  # namespace must be <model>_<hash>_r<rank>
        for shard in os.scandir(ns.path):
            if not shard.is_dir(follow_symlinks=False) or not HEX_RE.match(shard.name):
                continue
            for mid in os.scandir(shard.path):
                # Layouts seen on v0.32.0: <ns>/<3hex>/<2hex>_g<gen>/<hash>.bin
                # (mid = dir); keep support for a block file directly under
                # <3hex> in case the mid level disappears on a future release.
                if mid.is_dir(follow_symlinks=False):
                    if not MID_RE.match(mid.name):
                        continue
                    entries = os.scandir(mid.path)
                elif mid.is_file(follow_symlinks=False):
                    entries = [mid]
                else:
                    continue
                for entry in entries:
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        st = entry.stat(follow_symlinks=False)
                    except OSError as exc:  # vanished between scandir and stat
                        errors.append(f"{entry.path}: {exc}")
                        continue
                    files_total += 1
                    bytes_total += st.st_size
                    if now - st.st_mtime <= min_age_s:
                        continue  # too fresh: possibly in-flight
                    candidates.append({"path": entry.path, "size": st.st_size, "mtime": st.st_mtime})

    candidates.sort(key=lambda c: c["mtime"])
    return candidates, errors, files_total, bytes_total


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune the vLLM KV fs offload tier (dry-run by default).")
    ap.add_argument("--root", default="/home/user/vllm/.kv-tier-fs", help="tier root inside WSL")
    ap.add_argument("--guard-mounts", default="/mnt/c",
                    help="comma-separated volumes whose free space must be preserved "
                         "(default /mnt/c: host of the Ubuntu ext4.vhdx that backs the tier)")
    ap.add_argument("--max-gib", type=float, default=48.0, help="hard cap regardless of free space")
    ap.add_argument("--reserve-pct", type=float, default=12.0, help="percent of a guard mount's total kept free")
    ap.add_argument("--guest-floor-gib", type=float, default=40.0,
                    help="absolute GB kept free on the guest FS (hosts HF cache + WSL system)")
    ap.add_argument("--min-age-min", type=int, default=30, help="never delete blocks younger than this")
    ap.add_argument("--state-file", default="/mnt/d/vllm/myia_vllm/logs/fstier_state.json")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    started = time.time()
    guard = [m.strip() for m in args.guard_mounts.split(",") if m.strip()]
    cap = compute_cap(args.root, guard, args.max_gib, args.reserve_pct, args.guest_floor_gib)
    candidates, errors, files_total, bytes_total = collect(args.root, args.min_age_min)

    over = bytes_total - cap["cap_bytes"]
    deleted_n = 0
    deleted_bytes = 0
    freed = 0
    for c in candidates:
        if freed >= over:
            break
        if args.apply:
            try:
                os.remove(c["path"])
                deleted_n += 1
                deleted_bytes += c["size"]
                freed += c["size"]
            except OSError as exc:
                errors.append(f"delete {c['path']}: {exc}")
        else:
            deleted_n += 1  # would-delete count
            deleted_bytes += c["size"]
            freed += c["size"]

    state = {
        "ts_utc": utcnow(),
        "root": args.root,
        "mode": "apply" if args.apply else "dry-run",
        "min_age_min": args.min_age_min,
        **cap,
        "files_total": files_total,
        "bytes_total": bytes_total,
        "bytes_over_cap": max(0, over),
        "candidates": len(candidates),
        "deleted_files": deleted_n,
        "deleted_bytes": deleted_bytes,
        "bytes_after_expected": bytes_total - freed,
        "errors": errors[:50],
        "errors_count": len(errors),
        "elapsed_s": round(time.time() - started, 2),
    }

    try:
        os.makedirs(os.path.dirname(args.state_file), exist_ok=True)
        tmp = args.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, args.state_file)  # atomic
    except OSError as exc:
        state["state_write_error"] = str(exc)

    print(json.dumps(state, indent=2))
    # Exit 0 even with per-file errors (they are reported); exit 2 only if root missing.
    return 2 if any(e.startswith("root not found") for e in errors) else 0


if __name__ == "__main__":
    sys.exit(main())