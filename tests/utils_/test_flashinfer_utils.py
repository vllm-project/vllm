# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import vllm.utils.flashinfer as fi


def _make_exe(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o755)
    return str(path)


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


@pytest.fixture
def warning_once(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Pretend only flashinfer-python (no cubin or jit-cache) is installed on
    SM100."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args):
        if name == "flashinfer":
            return object()
        if name == "flashinfer_jit_cache":
            return None
        return real_find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(fi, "has_flashinfer_cubin", lambda: False)
    monkeypatch.setattr(fi.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        fi.current_platform, "has_device_capability", lambda *args, **kwargs: True
    )
    mock = MagicMock()
    monkeypatch.setattr(fi.logger, "warning_once", mock)
    fi.has_flashinfer.cache_clear()
    yield mock
    fi.has_flashinfer.cache_clear()


def test_has_flashinfer_finds_toolkit_off_path(
    default_cuda_home: Path, warning_once: MagicMock
):
    """Bare-metal installs keep nvcc in /usr/local/cuda, off PATH, which is
    where FlashInfer's JIT finds it."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    assert fi.has_flashinfer()
    warning_once.assert_not_called()


@pytest.mark.parametrize("runtime_only_toolkit", [False, True])
def test_has_flashinfer_warns_without_nvcc(
    default_cuda_home: Path, warning_once: MagicMock, runtime_only_toolkit: bool
):
    """CUDA runtime images have /usr/local/cuda but no bin/nvcc."""
    if runtime_only_toolkit:
        (default_cuda_home / "lib64").mkdir(parents=True)
    assert not fi.has_flashinfer()
    warning_once.assert_called_once()


def test_has_flashinfer_warns_without_ninja(
    default_cuda_home: Path,
    warning_once: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """FlashInfer runs `ninja` from PATH, which lacks the venv's bin directory
    when vLLM is launched without activating the venv."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert not fi.has_flashinfer()
    warning_once.assert_called_once()


def test_jit_cache_does_not_need_ninja(
    default_cuda_home: Path,
    warning_once: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """flashinfer-jit-cache ships prebuilt modules that load without ninja."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: object()
        if name == "flashinfer_jit_cache"
        else real_find_spec(name, *args),
    )
    assert fi.has_flashinfer()
    warning_once.assert_not_called()


def test_nvcc_on_path_resolves_its_toolkit(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The toolkit of nvcc on PATH wins over /usr/local/cuda, as in FlashInfer."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    nvcc = _make_exe(tmp_path / "cuda-13" / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "cuda-13" / "bin"))
    assert fi._flashinfer_nvcc_path() == nvcc


def test_cuda_home_wins_over_default_even_if_broken(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """FlashInfer compiles with $CUDA_HOME/bin/nvcc, so a working default
    toolkit must not mask a CUDA_HOME without nvcc."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing"))
    assert fi._flashinfer_nvcc_path() is None
