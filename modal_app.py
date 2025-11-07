# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ABOUTME: Modal helpers for running EPS tests remotely.
# ABOUTME: Provides CPU/GPU functions to execute pytest or custom commands.

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import modal
from modal.mount import Mount
from modal.secret import Secret

app = modal.App("eps-smoke")

SKIP_DIRS: set[str] = {
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
    "timeout": 30 * 60,  # 30 minutes default
}

if os.environ.get("HF_TOKEN"):
    common_kwargs["secrets"] = [Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]


cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "cmake",
        "ninja-build",
        "libnuma-dev",
    )
    .pip_install(
        "numpy>=1.26",
        "pytest>=8.0",
        "Pillow>=10.0",
    )
)
cpu_image = cpu_image._add_mount_layer_or_copy(repo_mount)

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "curl", "cmake", "ninja-build", "libnuma-dev")
    .pip_install(
        "numpy>=1.26",
        "pytest>=8.0",
        "Pillow>=10.0",
    )
)
gpu_image = gpu_image._add_mount_layer_or_copy(repo_mount)

CPU_TORCH_SPEC: tuple[str, str | None] = (
    "torch==2.9.0+cpu",
    "https://download.pytorch.org/whl/cpu",
)

GPU_TORCH_SPEC: tuple[str, str | None] = (
    "torch==2.9.0+cu121",
    "https://download.pytorch.org/whl/cu121",
)


def _ensure_project_install(torch_spec: tuple[str, str | None], *, tag: str) -> None:
    marker = Path(f"/tmp/.vllm_install_{tag}")
    if marker.exists():
        return
    import subprocess
    import sys

    repo_root = Path("/workspace")
    repo_dir = repo_root / "vllm"
    pip_env = os.environ.copy()
    pip_env.setdefault("PIP_PROGRESS_BAR", "off")
    pip_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    pkg, index_url = torch_spec
    cmd = [sys.executable, "-m", "pip", "install", "-q", pkg]
    if index_url:
        cmd.extend(["--index-url", index_url])
    subprocess.check_call(cmd, cwd=repo_root, env=pip_env)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "setuptools>=77.0.3,<80.0.0",
            "setuptools-scm>=8.0",
            "wheel",
            "packaging>=24.2",
            "jinja2",
            "cmake>=3.26.1",
        ],
        cwd=repo_root,
        env=pip_env,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str((repo_root / "requirements" / "common.txt").resolve()),
        ],
        cwd=repo_root,
        env=pip_env,
    )
    if tag == "cpu":
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "numba==0.61.2",
            ],
            cwd=repo_root,
            env=pip_env,
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            str(repo_root),
            "--no-deps",
            "--no-build-isolation",
        ],
        cwd=repo_root,
        env=pip_env,
    )
    marker.touch()


def _prep_env() -> None:
    os.chdir("/workspace")
    os.environ.setdefault("PYTHONPATH", "/workspace")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    os.environ.setdefault("VLLM_CPU_OMP_THREADS_BIND", "all")
    os.environ.setdefault("CPU_VISIBLE_MEMORY_NODES", "0")
    if "HF_TOKEN" in os.environ:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


def _run(cmd: Iterable[str]) -> int:
    import subprocess

    print("[modal] running:", " ".join(cmd), flush=True)
    return subprocess.call(list(cmd))


@app.function(image=cpu_image, **common_kwargs)
def run_pytest(test_path: str = "tests/v1/eps/test_smoke_cpu.py") -> int:
    _prep_env()
    _ensure_project_install(CPU_TORCH_SPEC, tag="cpu")
    try:
        from vllm import platforms  # type: ignore
        from vllm.platforms.cpu import CpuPlatform  # type: ignore

        platforms.current_platform = CpuPlatform()
    except Exception:
        pass
    return _run(["pytest", "-q", test_path])


@app.function(image=cpu_image, **common_kwargs)
def run_cmd(*cmd: str) -> int:
    if not cmd:
        raise ValueError("run_cmd requires at least one argument")
    _prep_env()
    _ensure_project_install(CPU_TORCH_SPEC, tag="cpu")
    try:
        from vllm import platforms  # type: ignore
        from vllm.platforms.cpu import CpuPlatform  # type: ignore

        platforms.current_platform = CpuPlatform()
    except Exception:
        pass
    return _run(cmd)


@app.function(image=gpu_image, gpu="A10G", **common_kwargs)
def run_gpu_cmd(*cmd: str) -> int:
    if not cmd:
        raise ValueError("run_gpu_cmd requires at least one argument")
    _prep_env()
    _ensure_project_install(GPU_TORCH_SPEC, tag="gpu")
    return _run(cmd)
