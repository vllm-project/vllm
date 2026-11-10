# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture
def no_max_jobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("MAX_JOBS", raising=False)
    yield


def test_limit_jit_parallelism_keeps_explicit_max_jobs(monkeypatch: pytest.MonkeyPatch):
    """A user-provided MAX_JOBS is never overridden."""
    monkeypatch.setenv("MAX_JOBS", "8")
    assert fi.limit_flashinfer_jit_parallelism() is None
    assert os.environ["MAX_JOBS"] == "8"


def test_limit_jit_parallelism_budgets_available_memory(
    no_max_jobs: None, monkeypatch: pytest.MonkeyPatch
):
    """Half of 64 GiB, at 8 GiB per compiler process, is four jobs."""
    monkeypatch.setattr(fi, "get_available_memory", lambda: 64 * 1024**3)
    monkeypatch.setattr(fi.os, "cpu_count", lambda: 32)
    assert fi.limit_flashinfer_jit_parallelism() == 4
    assert os.environ["MAX_JOBS"] == "4"


def test_limit_jit_parallelism_keeps_at_least_one_job(
    no_max_jobs: None, monkeypatch: pytest.MonkeyPatch
):
    """A nearly-full host still gets one job instead of zero."""
    monkeypatch.setattr(fi, "get_available_memory", lambda: 2 * 1024**3)
    monkeypatch.setattr(fi.os, "cpu_count", lambda: 32)
    assert fi.limit_flashinfer_jit_parallelism() == 1
    assert os.environ["MAX_JOBS"] == "1"


def test_limit_jit_parallelism_never_exceeds_cpu_count(
    no_max_jobs: None, monkeypatch: pytest.MonkeyPatch
):
    """Plenty of memory does not mean more jobs than there are CPUs."""
    monkeypatch.setattr(fi, "get_available_memory", lambda: 1024 * 1024**3)
    monkeypatch.setattr(fi.os, "cpu_count", lambda: 4)
    assert fi.limit_flashinfer_jit_parallelism() == 4
