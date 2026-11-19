# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

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


@pytest.fixture
def jit_environment() -> Iterator[None]:
    fi._limit_flashinfer_jit_parallelism.cache_clear()
    with patch.dict(os.environ):
        os.environ.pop("MAX_JOBS", None)
        yield
    fi._limit_flashinfer_jit_parallelism.cache_clear()


@pytest.mark.parametrize(
    "available_gib,cpus,expected", [(64, 32, 4), (2, 32, 1), (1024, 4, 4)]
)
def test_limit_jit_parallelism(
    jit_environment: None, monkeypatch, available_gib, cpus, expected
):
    monkeypatch.setattr(fi, "get_available_memory", lambda: available_gib * 1024**3)
    monkeypatch.setattr(fi.os, "cpu_count", lambda: cpus)
    fi._limit_flashinfer_jit_parallelism()
    assert os.environ["MAX_JOBS"] == str(expected)


def test_limit_jit_parallelism_keeps_explicit_max_jobs(
    jit_environment: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(fi, "get_available_memory", Mock(side_effect=AssertionError))
    fi._limit_flashinfer_jit_parallelism()
    assert os.environ["MAX_JOBS"] == "8"


def test_first_dense_cutlass_nvfp4_gemm_limits_jobs(
    jit_environment: None, monkeypatch: pytest.MonkeyPatch
):
    flashinfer = pytest.importorskip("flashinfer")
    if not hasattr(fi, "flashinfer_mm_fp4"):
        pytest.skip("FlashInfer custom ops are unavailable")
    x = torch.empty(2, 2)
    seen = []

    def mm_fp4(*args, **kwargs):
        seen.append(os.getenv("MAX_JOBS"))
        return x

    memory = Mock(return_value=64 * 1024**3)
    monkeypatch.setattr(fi, "get_available_memory", memory)
    monkeypatch.setattr(fi.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(flashinfer, "mm_fp4", mm_fp4)
    args = (x, x, x, x, x, x.dtype, False)
    fi.flashinfer_mm_fp4_fake(*args, backend="cutlass")
    fi.flashinfer_mm_fp4(*args, backend="cudnn")
    fi.flashinfer_mm_fp4(*args, backend="cutlass", use_nvfp4=False)
    memory.assert_not_called()

    memory.return_value = 32 * 1024**3
    fi.flashinfer_mm_fp4(*args, backend="cutlass")
    fi.flashinfer_mm_fp4(*args, backend="cutlass")
    memory.assert_called_once_with()
    assert seen == [None, None, "2", "2"]
