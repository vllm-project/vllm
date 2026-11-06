# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from pathlib import Path

import pytest

import vllm.utils.deep_gemm as dg


def _make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o755)


@pytest.fixture
def default_cuda_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Hide any real toolkit and redirect /usr/local/cuda."""
    for var in (
        "CUDA_HOME",
        "CUDA_PATH",
        "DG_JIT_NVCC_COMPILER",
        "DJ_JIT_NVCC_COMPILER",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    home = tmp_path / "usr_local_cuda"
    monkeypatch.setattr(dg, "_DEFAULT_CUDA_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def deep_gemm_installed(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend DeepGEMM is importable on a GPU it supports."""
    monkeypatch.setattr(dg.current_platform, "support_deep_gemm", lambda: True)
    monkeypatch.setattr(dg, "has_deep_gemm", lambda: True)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    dg.is_deep_gemm_supported.cache_clear()
    yield
    dg.is_deep_gemm_supported.cache_clear()


def test_deep_gemm_unsupported_without_nvcc(default_cuda_home: Path):
    """Wheel-only installs have no CUDA toolkit, so DeepGEMM's JIT cannot run
    and block FP8 must fall back to another kernel (#60261)."""
    assert not dg.is_deep_gemm_supported()


def test_deep_gemm_finds_toolkit_off_path(default_cuda_home: Path):
    _make_exe(default_cuda_home / "bin" / "nvcc")
    assert dg.is_deep_gemm_supported()


def test_deep_gemm_trusts_cuda_home(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """DeepGEMM's JIT does not look past CUDA_HOME, even when it is wrong."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing"))
    assert not dg.is_deep_gemm_supported()
