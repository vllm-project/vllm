# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from pathlib import Path

import pytest

import vllm.utils.deep_gemm as dg
import vllm.utils.platform_utils as pu


def _make_nvcc(path: Path, release: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {release}'\n")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def default_cuda_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Path]:
    """Hide any real toolkit and pretend DeepGEMM is usable."""
    for var in (
        "CUDA_HOME",
        "CUDA_PATH",
        "DG_JIT_NVCC_COMPILER",
        "DJ_JIT_NVCC_COMPILER",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    home = tmp_path / "usr_local_cuda"
    monkeypatch.setattr(pu, "_DEFAULT_CUDA_HOME", str(home))
    monkeypatch.setattr(dg.current_platform, "support_deep_gemm", lambda: True)
    monkeypatch.setattr(dg, "has_deep_gemm", lambda: True)
    dg.is_deep_gemm_supported.cache_clear()
    yield home
    dg.is_deep_gemm_supported.cache_clear()


@pytest.mark.parametrize(
    ("release", "supported"), [("12.8", False), ("12.9", True), ("13.0", True)]
)
def test_nvcc_version_gates_deep_gemm(
    default_cuda_home: Path, release: str, supported: bool
):
    """DeepGEMM's JIT aborts the engine below nvcc 12.9."""
    _make_nvcc(default_cuda_home / "bin" / "nvcc", release)
    assert dg.is_deep_gemm_supported() is supported


def test_missing_nvcc_disables_deep_gemm():
    assert not dg.is_deep_gemm_supported()


def test_jit_nvcc_override_wins(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _make_nvcc(default_cuda_home / "bin" / "nvcc", "12.8")
    nvcc = _make_nvcc(tmp_path / "new" / "nvcc", "13.0")
    monkeypatch.setenv("DG_JIT_NVCC_COMPILER", str(nvcc))
    assert dg.is_deep_gemm_supported()
