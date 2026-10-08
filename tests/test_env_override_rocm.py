# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm libtorch symbol promotion in env_override.py."""

import ctypes
import os
from unittest.mock import patch

import vllm.env_override as env_override

_EVENTS = "_VLLM_TEST_EVENTS"


def _fake_torch_root(tmp_path, with_rocm_init: bool) -> str:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libtorch_cpu.so").write_bytes(b"")
    if with_rocm_init:
        (tmp_path / "_rocm_init.py").write_text(
            "import os\n"
            "def initialize():\n"
            f"    os.environ['{_EVENTS}'] += 'rocm_init,'\n"
        )
    return str(tmp_path)


def _promote(monkeypatch, torch_root: str) -> str:
    monkeypatch.setenv(_EVENTS, "")

    def fake_cdll(name, mode=0):
        os.environ[_EVENTS] += f"cdll_global={mode == ctypes.RTLD_GLOBAL},"

    with (
        patch.object(env_override, "_get_torch_version_attr", return_value="7.13"),
        patch.object(env_override, "_get_torch_root", return_value=torch_root),
        patch.object(ctypes, "CDLL", side_effect=fake_cdll),
    ):
        env_override._maybe_promote_torch_symbols_for_rocm()
    return os.environ[_EVENTS]


def test_rocm_sdk_preload_runs_before_global_promotion(monkeypatch, tmp_path):
    """TheRock's ROCm libs must be preloaded before RTLD_GLOBAL, otherwise HIP
    can resolve to a system ROCm and abort on duplicate LLVM options."""
    events = _promote(monkeypatch, _fake_torch_root(tmp_path, with_rocm_init=True))
    assert events == "rocm_init,cdll_global=True,"


def test_promotion_without_rocm_init(monkeypatch, tmp_path):
    """Non-TheRock ROCm wheels have no _rocm_init.py; promotion is unchanged."""
    events = _promote(monkeypatch, _fake_torch_root(tmp_path, with_rocm_init=False))
    assert events == "cdll_global=True,"
