# ABOUTME: Modal helpers for running EPS tests remotely.
# ABOUTME: Provides CPU/GPU functions to execute pytest or custom commands.

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import modal
from modal.mount import Mount
from modal.secret import Secret

app = modal.App("eps-smoke")

SKIP_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
}


def _include(path: str) -> bool:
    parts = path.split("/")
    return not any(part in SKIP_DIRS for part in parts)


repo_mount = Mount._from_local_dir(".", remote_path="/workspace", condition=_include)

common_kwargs: dict[str, object] = {
    "mounts": [repo_mount],
    "timeout": 30 * 60,  # 30 minutes default
}

if os.environ.get("HF_TOKEN"):
    common_kwargs["secrets"] = [
        Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
    ]


cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "numpy>=1.26",
        "pytest>=8.0",
        "Pillow>=10.0",
    )
)

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "curl")
    .pip_install(
        "numpy>=1.26",
        "pytest>=8.0",
        "Pillow>=10.0",
    )
)

GPU_TORCH_SPEC = (
    "torch==2.4.1+cu121",
    "https://download.pytorch.org/whl/cu121",
)


def _ensure_editable_install() -> None:
    marker = Path("/tmp/.vllm_editable_installed")
    if marker.exists():
        return
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd="/workspace",
    )
    marker.touch()


def _ensure_gpu_torch() -> None:
    marker = Path("/tmp/.torch_cuda_installed")
    if marker.exists():
        return
    import subprocess
    import sys

    pkg, index_url = GPU_TORCH_SPEC
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "--index-url", index_url],
        cwd="/workspace",
    )
    marker.touch()


def _prep_env() -> None:
    os.chdir("/workspace")
    os.environ.setdefault("PYTHONPATH", "/workspace")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    if "HF_TOKEN" in os.environ:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


def _run(cmd: Iterable[str]) -> int:
    import subprocess

    print("[modal] running:", " ".join(cmd), flush=True)
    return subprocess.call(list(cmd))


@app.function(image=cpu_image, **common_kwargs)
def run_pytest(test_path: str = "tests/v1/eps/test_smoke_cpu.py") -> int:
    _prep_env()
    _ensure_editable_install()
    return _run(["pytest", "-q", test_path])


@app.function(image=cpu_image, **common_kwargs)
def run_cmd(*cmd: str) -> int:
    if not cmd:
        raise ValueError("run_cmd requires at least one argument")
    _prep_env()
    _ensure_editable_install()
    return _run(cmd)


@app.function(image=gpu_image, gpu="A10G", **common_kwargs)
def run_gpu_cmd(*cmd: str) -> int:
    if not cmd:
        raise ValueError("run_gpu_cmd requires at least one argument")
    _prep_env()
    _ensure_editable_install()
    _ensure_gpu_torch()
    return _run(cmd)
