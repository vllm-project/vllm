# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrappers around the `criu` and `cuda-checkpoint` binaries.

In the prototype, the binaries may not be installed. All calls check
for binary availability and log-rather-than-execute in dry-run mode.
Controlled by:
    VLLM_SNAPSHOT_ENABLED=1   # enable the feature
    VLLM_SNAPSHOT_DRY_RUN=1   # log actions but don't invoke binaries
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _is_enabled() -> bool:
    return os.environ.get("VLLM_SNAPSHOT_ENABLED") == "1"


def _is_dry_run() -> bool:
    return os.environ.get("VLLM_SNAPSHOT_DRY_RUN") == "1"


def _find(binary: str) -> str | None:
    return shutil.which(binary)


def criu_installed() -> bool:
    return _find("criu") is not None


def cuda_checkpoint_installed() -> bool:
    return _find("cuda-checkpoint") is not None


def cuda_checkpoint_toggle(pid: int) -> None:
    """Toggle CUDA state between GPU-resident and CPU-pinned.

    Used bracketed around criu dump/restore:
        cuda_checkpoint_toggle(helper_pid)  # park to CPU
        criu_dump(helper_pid, ...)
    And on restore:
        criu_restore(...)
        cuda_checkpoint_toggle(restored_pid)  # push back to GPU
    """
    if not _is_enabled() or _is_dry_run():
        print(f"[criu_wrapper] DRYRUN cuda-checkpoint --toggle --pid {pid}")
        return
    if not cuda_checkpoint_installed():
        raise RuntimeError(
            "cuda-checkpoint not installed; required for CUDA-aware snapshot"
        )
    subprocess.run(
        ["cuda-checkpoint", "--toggle", "--pid", str(pid)],
        check=True,
        timeout=60,
    )


def criu_dump(
    pid: int, images_dir: Path, log_file: Path | None = None
) -> None:
    """Dump the process tree to `images_dir`.

    Caller is responsible for having called cuda_checkpoint_toggle first
    if the process has CUDA state.
    """
    if not _is_enabled() or _is_dry_run():
        print(f"[criu_wrapper] DRYRUN criu dump -t {pid} -D {images_dir}")
        return
    if not criu_installed():
        raise RuntimeError("criu not installed")
    images_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "criu",
        "dump",
        "--tree",
        str(pid),
        "--images-dir",
        str(images_dir),
        "--tcp-established",
        "--ext-unix-sk",
        "--file-locks",
        "--shell-job",
        "--external",
        "file[/dev/nvidiactl]:ignore",
        "--external",
        "file[/dev/nvidia-uvm]:ignore",
        # Could add per-GPU device nodes discovered via /dev scan
    ]
    if log_file is not None:
        cmd += ["--log-file", str(log_file)]
    subprocess.run(cmd, check=True)


def criu_restore(
    images_dir: Path, log_file: Path | None = None, background: bool = True
) -> subprocess.Popen | None:
    """Restore the snapshot. Returns the Popen handle if backgrounded.

    In the prototype, the restored process still needs to have its
    CUDA state re-attached via cuda_checkpoint_toggle(<restored_pid>).
    The restored process pid can be discovered from criu's
    --pidfile option or by inspecting /proc.
    """
    if not _is_enabled() or _is_dry_run():
        print(f"[criu_wrapper] DRYRUN criu restore -D {images_dir}")
        return None
    if not criu_installed():
        raise RuntimeError("criu not installed")
    cmd = [
        "criu",
        "restore",
        "--images-dir",
        str(images_dir),
        "--shell-job",
        "--ext-unix-sk",
    ]
    if log_file is not None:
        cmd += ["--log-file", str(log_file)]
    if background:
        return subprocess.Popen(cmd)
    subprocess.run(cmd, check=True)
    return None
