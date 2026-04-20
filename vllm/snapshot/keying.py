# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot-key computation.

The snapshot key identifies the tuple of versions that a snapshot is
compatible with. If ANY of these change, the snapshot is invalid and
must be rebuilt. We keep the key deterministic and stable so that the
same host can re-find its snapshot after a reboot or shell session.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class SnapshotKey:
    """The tuple that uniquely determines snapshot compatibility."""

    vllm_version: str
    python_version: str
    torch_version: str
    cuda_runtime_version: str
    cuda_driver_version: str
    gpu_arch: str
    platform_machine: str = field(default_factory=platform.machine)

    def digest(self) -> str:
        """A short filesystem-safe digest for use as a directory name."""
        raw = "|".join(
            f"{k}={v}" for k, v in sorted(asdict(self).items())
        ).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def describe(self) -> str:
        return (
            f"vllm={self.vllm_version} py={self.python_version} "
            f"torch={self.torch_version} cuda-rt={self.cuda_runtime_version} "
            f"driver={self.cuda_driver_version} arch={self.gpu_arch}"
        )


def _try_cuda_driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=2,
        )
        # first GPU's driver; all GPUs share it on a host
        return out.splitlines()[0].strip()
    except Exception:
        return "unknown"


def _try_gpu_arch() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=2,
        )
        cc = out.splitlines()[0].strip().replace(".", "_")
        # compute_cap like "9.0" -> sm_90
        return f"sm_{cc}"
    except Exception:
        return "unknown"


def _try_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unknown"


def _try_cuda_runtime_version() -> str:
    try:
        import torch

        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def _try_vllm_version() -> str:
    try:
        from vllm.version import __version__

        return __version__
    except Exception:
        return "unknown"


def compute_snapshot_key() -> SnapshotKey:
    return SnapshotKey(
        vllm_version=_try_vllm_version(),
        python_version=platform.python_version(),
        torch_version=_try_torch_version(),
        cuda_runtime_version=_try_cuda_runtime_version(),
        cuda_driver_version=_try_cuda_driver_version(),
        gpu_arch=_try_gpu_arch(),
    )


def snapshot_root() -> Path:
    """Directory that holds per-key snapshot subdirs."""
    root = os.environ.get("VLLM_SNAPSHOT_ROOT")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "vllm" / "snapshots"


def snapshot_dir(key: SnapshotKey) -> Path:
    return snapshot_root() / key.digest()
