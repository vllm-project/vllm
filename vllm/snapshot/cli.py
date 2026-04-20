# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`vllm snapshot ...` subcommand.

Subcommands:
  vllm snapshot create        Create a snapshot for the current version key
  vllm snapshot list          List existing snapshots
  vllm snapshot drop [--all]  Delete snapshot(s)
  vllm snapshot verify <key>  Try a dry-run restore as a sanity check

And the internal entry point:
  vllm.snapshot.cli.try_restore_and_dispatch()   # called from CLI main()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from vllm.snapshot.criu_wrapper import (
    criu_dump,
    criu_installed,
    criu_restore,
    cuda_checkpoint_installed,
    cuda_checkpoint_toggle,
)
from vllm.snapshot.keying import (
    SnapshotKey,
    compute_snapshot_key,
    snapshot_dir,
    snapshot_root,
)
from vllm.snapshot.resume_protocol import write_payload

logger = logging.getLogger(__name__)


def _manifest_path(d: Path) -> Path:
    return d / "MANIFEST"


def _imgs_path(d: Path) -> Path:
    return d / "imgs"


def _write_manifest(d: Path, key: SnapshotKey, bytes_on_disk: int) -> None:
    import dataclasses

    manifest = {
        **dataclasses.asdict(key),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes_on_disk": bytes_on_disk,
    }
    _manifest_path(d).write_text(json.dumps(manifest, indent=2))


def _read_manifest(d: Path) -> dict | None:
    p = _manifest_path(d)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# Internal API called from CLI main() on `vllm serve`
# ---------------------------------------------------------------------------


def try_restore_and_dispatch() -> bool:
    """Attempt to restore from a compatible snapshot. Returns True if the
    snapshot was restored and ownership of the process was transferred;
    False to indicate the caller should fall through to normal startup.

    In the prototype: performs all path/manifest checks, prepares the
    payload, and logs what would happen. The actual criu_restore is
    gated by `VLLM_SNAPSHOT_ENABLED=1` and `VLLM_SNAPSHOT_DRY_RUN=0`.
    """
    if os.environ.get("VLLM_SNAPSHOT_ENABLED") != "1":
        return False
    if not criu_installed():
        logger.debug("snapshot restore: criu not installed; skipping")
        return False

    key = compute_snapshot_key()
    d = snapshot_dir(key)
    manifest = _read_manifest(d)
    if manifest is None:
        logger.debug("snapshot restore: no snapshot for key %s", key.digest())
        return False

    # Verify key match (paranoid — manifest should already match dir)
    if manifest.get("vllm_version") != key.vllm_version:
        logger.info("snapshot restore: MANIFEST version mismatch; skipping")
        return False

    # Prepare payload; restored helper reads it via VLLM_RESUME_PAYLOAD env
    payload_path = write_payload()
    os.environ["VLLM_RESUME_PAYLOAD"] = str(payload_path)

    logger.info(
        "snapshot restore: using %s (key=%s, %.1f MB)",
        d,
        key.digest(),
        manifest.get("bytes_on_disk", 0) / 1e6,
    )

    # Kick the restore. In v1 this is:
    #   1. criu_restore(d/imgs) -> spawned process suspended in signal.pause()
    #   2. cuda_checkpoint_toggle(restored_pid) to push CUDA state onto GPU
    #   3. os.kill(restored_pid, SIGUSR2) wakes its _resume() handler
    #   4. wait on restored_pid; exit with its return code
    #
    # For the prototype, dry-run logs the sequence.
    proc = criu_restore(_imgs_path(d), background=True)
    if proc is None:
        # Dry run or criu absent post-check — fall back.
        os.environ.pop("VLLM_RESUME_PAYLOAD", None)
        try:
            os.unlink(payload_path)
        except OSError:
            pass
        return False

    try:
        restored_pid = proc.pid  # in practice, read from criu --pidfile
        if cuda_checkpoint_installed():
            cuda_checkpoint_toggle(restored_pid)
        os.kill(restored_pid, signal.SIGUSR2)
        proc.wait()
        sys.exit(proc.returncode)
    except Exception:
        logger.exception("snapshot restore failed; falling back")
        return False


# ---------------------------------------------------------------------------
# User-facing `vllm snapshot` CLI subcommand
# ---------------------------------------------------------------------------


def cmd_create(args: argparse.Namespace) -> int:
    key = compute_snapshot_key()
    d = snapshot_dir(key)
    if _manifest_path(d).exists() and not args.force:
        print(f"Snapshot already exists at {d}")
        print("  use --force to rebuild")
        return 0

    d.mkdir(parents=True, exist_ok=True)
    imgs = _imgs_path(d)
    imgs.mkdir(exist_ok=True)
    ready_file = d / ".ready"
    if ready_file.exists():
        ready_file.unlink()

    helper_env = dict(os.environ)
    helper_env["VLLM_SNAPSHOT_READY_FILE"] = str(ready_file)
    if args.dry_run:
        helper_env["VLLM_SNAPSHOT_DRY_RUN"] = "1"

    # Launch helper; it will pause once ready
    print(f"Starting snapshot helper for key {key.digest()}")
    print(f"  {key.describe()}")
    helper = subprocess.Popen(
        [sys.executable, "-m", "vllm.snapshot.helper"],
        env=helper_env,
    )

    # Wait up to N seconds for the ready file
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if ready_file.exists():
            break
        if helper.poll() is not None:
            print(f"Helper exited early (rc={helper.returncode})")
            return 1
        time.sleep(0.5)
    else:
        print("Helper did not reach ready state within 120s")
        helper.kill()
        return 1

    print(f"Helper ready (pid={helper.pid}); initiating snapshot")

    try:
        # GPU → CPU
        if cuda_checkpoint_installed():
            cuda_checkpoint_toggle(helper.pid)
        else:
            print("  cuda-checkpoint not installed; snapshot will NOT include "
                  "CUDA state (saves ~10s per restore instead of ~15-20s)")

        # Snapshot
        criu_dump(helper.pid, imgs, log_file=d / "dump.log")

        # After dump, helper is frozen. kill it to clean up.
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=5)

        bytes_on_disk = _dir_size(imgs) if imgs.exists() else 0
        _write_manifest(d, key, bytes_on_disk)
        print(f"Snapshot created at {d} ({bytes_on_disk / 1e6:.1f} MB)")
        return 0
    except Exception as exc:
        print(f"Snapshot creation failed: {exc!r}")
        try:
            helper.kill()
        except Exception:
            pass
        # Clean up a partial snapshot
        shutil.rmtree(d, ignore_errors=True)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    root = snapshot_root()
    if not root.exists():
        print(f"No snapshots (root {root} does not exist)")
        return 0

    found = False
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = _read_manifest(d)
        if m is None:
            continue
        found = True
        print(f"{d.name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")
        print()
    if not found:
        print("No valid snapshots found")
    return 0


def cmd_drop(args: argparse.Namespace) -> int:
    root = snapshot_root()
    if args.all:
        if root.exists():
            shutil.rmtree(root)
            print(f"Removed {root}")
        return 0

    if args.key:
        target = root / args.key
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed {target}")
        else:
            print(f"No snapshot {target}")
        return 0

    # Default: drop the one matching current key
    key = compute_snapshot_key()
    target = snapshot_dir(key)
    if target.exists():
        shutil.rmtree(target)
        print(f"Removed {target}")
    else:
        print(f"No snapshot for current key {key.digest()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vllm snapshot",
        description="CRIU + cuda-checkpoint based startup snapshot",
    )
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_create = sub.add_parser("create", help="Create snapshot")
    p_create.add_argument("--force", action="store_true")
    p_create.add_argument("--dry-run", action="store_true",
                          help="Skip CUDA init + binary calls; log only")
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List snapshots")
    p_list.set_defaults(func=cmd_list)

    p_drop = sub.add_parser("drop", help="Remove snapshot(s)")
    p_drop.add_argument("--all", action="store_true")
    p_drop.add_argument("--key", help="Specific key digest to drop")
    p_drop.set_defaults(func=cmd_drop)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
