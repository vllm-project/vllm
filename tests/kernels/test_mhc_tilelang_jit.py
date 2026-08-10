# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

from vllm.platforms import current_platform
from vllm.utils import import_utils


class _PassConfigKey:
    TL_DISABLE_WARP_SPECIALIZED = "disable_warp_specialized"
    TL_DISABLE_TMA_LOWER = "disable_tma_lower"
    TL_PTXAS_REGISTER_USAGE_LEVEL = "ptxas_register_usage_level"


def _install_tilelang_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {"jit_decorate": 0, "compiled_call": 0}

    tilelang: Any = ModuleType("tilelang")

    def jit(**kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            calls["jit_decorate"] += 1

            def compiled(*args: Any, **kw: Any) -> Any:
                calls["compiled_call"] += 1
                return func.__name__

            return compiled

        return decorate

    tilelang.PassConfigKey = _PassConfigKey
    tilelang.jit = jit

    monkeypatch.setattr(import_utils, "has_tilelang", lambda: True)
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    monkeypatch.setitem(
        sys.modules, "tilelang.language", ModuleType("tilelang.language")
    )
    monkeypatch.delitem(sys.modules, "vllm.tilelang_utils", raising=False)

    return calls


def test_tilelang_jit_decorator_is_lazy_only_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        pytest.skip("Test requires CUDA or ROCm")

    calls = _install_tilelang_stub(monkeypatch)
    module_name = "vllm.model_executor.kernels.mhc.tilelang_kernels"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)

    if current_platform.is_rocm():
        assert calls["jit_decorate"] == 0
    else:
        assert calls["jit_decorate"] > 0

    decorated_calls = calls["jit_decorate"]
    assert module.mhc_post_tilelang() == "mhc_post_tilelang"
    if current_platform.is_rocm():
        assert calls["jit_decorate"] == 1
    else:
        assert calls["jit_decorate"] == decorated_calls
    assert calls["compiled_call"] == 1
