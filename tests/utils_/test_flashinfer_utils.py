# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import importlib.util
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm.utils.flashinfer as fi


def _make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o755)


@pytest.fixture
def default_cuda_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Hide any real toolkit, keep ninja on PATH, redirect /usr/local/cuda."""
    for var in ("CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC"):
        monkeypatch.delenv(var, raising=False)
    _make_exe(tmp_path / "venv" / "bin" / "ninja")
    monkeypatch.setenv("PATH", str(tmp_path / "venv" / "bin"))
    home = tmp_path / "usr_local_cuda"
    monkeypatch.setattr(fi, "_DEFAULT_CUDA_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def flashinfer_without_cubin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend flashinfer-python is installed without flashinfer-cubin."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            object() if name == "flashinfer" else real_find_spec(name, *args)
        ),
    )
    monkeypatch.setattr(fi, "has_flashinfer_cubin", lambda: False)
    fi.has_flashinfer.cache_clear()
    yield
    fi.has_flashinfer.cache_clear()


def test_has_flashinfer_finds_toolkit_off_path(default_cuda_home: Path):
    """Bare-metal installs keep nvcc in /usr/local/cuda, off PATH, which is
    where FlashInfer's JIT finds it."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    assert fi.has_flashinfer()


def test_has_flashinfer_requires_ninja(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """FlashInfer runs `ninja` from PATH, which lacks the venv's bin directory
    when vLLM is launched without activating the venv."""
    _make_exe(tmp_path / "cuda" / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "cuda" / "bin"))
    assert not fi.has_flashinfer()


_FLASHINFER_VERSION = "0.7.0.post1"


@pytest.fixture
def kernel_warnings(
    monkeypatch: pytest.MonkeyPatch, caplog_vllm, disable_log_dedup
) -> Callable[[dict[str, str], int], list[str]]:
    """Return the kernel warnings for the given installed kernel packages and
    GPU compute capability."""
    monkeypatch.delenv("VLLM_HAS_FLASHINFER_CUBIN", raising=False)
    monkeypatch.setattr(fi, "has_flashinfer_cubin", lambda: True)

    def warnings(kernels: dict[str, str], capability: int) -> list[str]:
        installed = {"flashinfer-python": _FLASHINFER_VERSION, **kernels}

        def version(name: str) -> str:
            if name not in installed:
                raise importlib.metadata.PackageNotFoundError(name)
            return installed[name]

        monkeypatch.setattr(importlib.metadata, "version", version)
        monkeypatch.setattr(
            fi,
            "current_platform",
            SimpleNamespace(
                is_cuda=lambda: True,
                has_device_capability=lambda required: capability >= required,
            ),
        )
        fi.warn_if_flashinfer_kernels_missing()
        return [r.getMessage() for r in caplog_vllm.records if r.levelname == "WARNING"]

    return warnings


@pytest.mark.parametrize(
    ("kernels", "fix"),
    [
        ({}, "Run `flashinfer download-kernels`"),
        (
            {"flashinfer-cubin": "0.7.0", "flashinfer-jit-cache": "0.7.0+cu130"},
            "Run `FLASHINFER_DISABLE_VERSION_CHECK=1 flashinfer download-kernels`",
        ),
    ],
    ids=["missing", "stale"],
)
def test_warns_when_kernels_not_preinstalled(
    kernel_warnings, kernels: dict[str, str], fix: str
):
    """Hopper and newer GPUs pay for missing or stale precompiled kernels with
    downloads and compilation at startup, so the fix must be spelled out."""
    (warning,) = kernel_warnings(kernels, 90)
    assert fix in warning


@pytest.mark.parametrize(
    ("kernels", "capability"),
    [
        (
            {
                "flashinfer-cubin": _FLASHINFER_VERSION,
                "flashinfer-jit-cache": f"{_FLASHINFER_VERSION}+cu130",
            },
            90,
        ),
        ({}, 89),
    ],
    ids=["installed", "pre-hopper"],
)
def test_no_kernel_warning(kernel_warnings, kernels: dict[str, str], capability: int):
    """Matching kernels (jit-cache carries a CUDA local version) and GPUs older
    than Hopper must stay quiet."""
    assert not kernel_warnings(kernels, capability)
