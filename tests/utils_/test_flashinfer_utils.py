# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
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


def test_has_flashinfer_returns_false_without_toolkit(default_cuda_home: Path):
    assert not fi.has_flashinfer()


def test_has_flashinfer_requires_ninja(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """FlashInfer runs `ninja` from PATH, which lacks the venv's bin directory
    when vLLM is launched without activating the venv."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert not fi.has_flashinfer()
