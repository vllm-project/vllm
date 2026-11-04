# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The early RTLD_GLOBAL load of libtorch_cpu.so on ROCm (env_override.py) is
gated: VLLM_ROCM_PROMOTE_TORCH_SYMBOLS decides, and unset it follows
VLLM_TORCH_PROFILER_DIR. On stacks whose torch preloads its own ROCm runtime
the load aborts `import vllm`, so it must stay off for plain serving."""

import ctypes
from unittest.mock import patch

import pytest

from vllm import env_override


@pytest.fixture
def rocm_torch(monkeypatch, tmp_path):
    """A fake ROCm torch install: hip version set, libtorch_cpu.so present."""
    (tmp_path / "lib").mkdir()
    lib = tmp_path / "lib" / "libtorch_cpu.so"
    lib.write_bytes(b"")
    monkeypatch.setattr(
        env_override,
        "_get_torch_version_attr",
        lambda attr: "7.1.0" if attr == "hip" else None,
    )
    monkeypatch.setattr(env_override, "_get_torch_root", lambda: str(tmp_path))
    monkeypatch.delenv("VLLM_ROCM_PROMOTE_TORCH_SYMBOLS", raising=False)
    monkeypatch.delenv("VLLM_TORCH_PROFILER_DIR", raising=False)
    return str(lib)


def _promote_calls(lib):
    with patch("ctypes.CDLL") as cdll:
        env_override._maybe_promote_torch_symbols_for_rocm()
    return [c for c in cdll.call_args_list if c.args and c.args[0] == lib]


def test_off_by_default(rocm_torch):
    assert _promote_calls(rocm_torch) == []


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_opt_in(rocm_torch, monkeypatch, value):
    monkeypatch.setenv("VLLM_ROCM_PROMOTE_TORCH_SYMBOLS", value)
    calls = _promote_calls(rocm_torch)
    assert len(calls) == 1
    assert calls[0].kwargs["mode"] == ctypes.RTLD_GLOBAL


def test_profiler_dir_turns_it_on(rocm_torch, monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", str(tmp_path))
    assert len(_promote_calls(rocm_torch)) == 1


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_explicit_off_beats_profiler_dir(rocm_torch, monkeypatch, tmp_path, value):
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ROCM_PROMOTE_TORCH_SYMBOLS", value)
    assert _promote_calls(rocm_torch) == []


def test_not_on_cuda_torch(rocm_torch, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_PROMOTE_TORCH_SYMBOLS", "1")
    monkeypatch.setattr(env_override, "_get_torch_version_attr", lambda attr: None)
    assert _promote_calls(rocm_torch) == []
