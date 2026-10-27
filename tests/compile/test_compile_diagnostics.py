# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch

from vllm.compilation.diagnostics import (
    format_compile_diagnostics,
    get_compile_diagnostics,
)

pytestmark = pytest.mark.skip_global_cleanup


class FakeCompilationConfig:
    mode = "VLLM_COMPILE"
    backend = "inductor"
    compile_sizes = [1, 2]
    compile_ranges_split_points = [4, 8]
    debug_dump_path = Path("compile-debug")


class FakeVllmConfig:
    compilation_config = FakeCompilationConfig()
    optimization_level = "O2"


def test_get_compile_diagnostics_returns_json_serializable_dict():
    diagnostics = get_compile_diagnostics(
        FakeVllmConfig(),
        cache_dir="cache",
        local_cache_dir="cache/rank_0_0/backbone",
        cache_factors={
            "env_hash": "env",
            "config_hash": "cfg",
            "compiler_hash": "comp",
            "code_hash": "code",
        },
    )

    assert isinstance(diagnostics, dict)
    json.dumps(diagnostics)
    assert "errors" not in diagnostics


def test_format_compile_diagnostics_returns_readable_string():
    diagnostics = get_compile_diagnostics(FakeVllmConfig())

    formatted = format_compile_diagnostics(diagnostics)

    assert isinstance(formatted, str)
    assert formatted.startswith("{\n")
    assert '"compilation"' in formatted
    assert '"runtime"' in formatted


def test_compile_diagnostics_works_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    diagnostics = get_compile_diagnostics(FakeVllmConfig())

    runtime = diagnostics["runtime"]
    assert runtime["cuda_available"] is False
    assert "cuda_device_capability" not in runtime


def test_compile_diagnostics_handles_missing_config_attributes():
    diagnostics = get_compile_diagnostics(object())

    assert diagnostics["runtime"]["torch_version"] == torch.__version__


def test_compile_diagnostics_handles_raising_config_property():
    class RaisingCompilationConfig:
        mode = "VLLM_COMPILE"

        @property
        def backend(self):
            raise RuntimeError("backend unavailable")

    class RaisingVllmConfig:
        compilation_config = RaisingCompilationConfig()

    diagnostics = get_compile_diagnostics(RaisingVllmConfig())

    assert diagnostics["compilation"]["mode"] == "VLLM_COMPILE"
    assert any("backend unavailable" in error for error in diagnostics["errors"])


def test_compile_diagnostics_includes_provided_cache_factor_hashes():
    diagnostics = get_compile_diagnostics(
        FakeVllmConfig(),
        cache_factors={
            "env_hash": "env",
            "config_hash": "cfg",
            "compiler_hash": "comp",
            "code_hash": "code",
            "raw_env": {"PATH": "not included"},
        },
    )

    cache_factors = diagnostics["cache"]["cache_factors"]
    assert cache_factors == {
        "env_hash": "env",
        "config_hash": "cfg",
        "compiler_hash": "comp",
        "code_hash": "code",
    }
