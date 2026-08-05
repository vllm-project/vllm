# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


class _Platform:
    def __init__(self, *, is_cuda: bool, is_rocm: bool) -> None:
        self._is_cuda = is_cuda
        self._is_rocm = is_rocm

    def is_cuda(self) -> bool:
        return self._is_cuda

    def is_rocm(self) -> bool:
        return self._is_rocm

    def is_arch_support_pdl(self) -> bool:
        return False


class _PassConfigKey:
    TL_DISABLE_WARP_SPECIALIZED = "disable_warp_specialized"
    TL_DISABLE_TMA_LOWER = "disable_tma_lower"
    TL_PTXAS_REGISTER_USAGE_LEVEL = "ptxas_register_usage_level"


def _install_tilelang_kernel_stubs(
    monkeypatch: pytest.MonkeyPatch, *, is_cuda: bool, is_rocm: bool
) -> dict[str, int]:
    calls = {"jit_decorate": 0, "compiled_call": 0}

    platforms: Any = ModuleType("vllm.platforms")
    platforms.current_platform = _Platform(is_cuda=is_cuda, is_rocm=is_rocm)

    import_utils: Any = ModuleType("vllm.utils.import_utils")
    import_utils.has_tilelang = lambda: True

    math_utils: Any = ModuleType("vllm.utils.math_utils")
    math_utils.cdiv = lambda a, b: (a + b - 1) // b

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

    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.utils.import_utils", import_utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.math_utils", math_utils)
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    monkeypatch.setitem(
        sys.modules, "tilelang.language", ModuleType("tilelang.language")
    )

    return calls


def _load_tilelang_kernels(module_name: str) -> Any:
    repo_root = Path(__file__).parents[2]
    module_path = repo_root / "vllm/model_executor/kernels/mhc/tilelang_kernels.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_tilelang_jit_decorator_is_lazy_only_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_calls = _install_tilelang_kernel_stubs(
        monkeypatch, is_cuda=True, is_rocm=False
    )
    cuda_module = _load_tilelang_kernels("tilelang_kernels_cuda_test")

    assert cuda_calls["jit_decorate"] > 0
    assert cuda_module.mhc_post_tilelang() == "mhc_post_tilelang"
    assert cuda_calls["compiled_call"] == 1

    monkeypatch.delitem(sys.modules, "tilelang_kernels_cuda_test")
    rocm_calls = _install_tilelang_kernel_stubs(
        monkeypatch, is_cuda=False, is_rocm=True
    )
    rocm_module = _load_tilelang_kernels("tilelang_kernels_rocm_test")

    assert rocm_calls["jit_decorate"] == 0
    assert rocm_module.mhc_post_tilelang() == "mhc_post_tilelang"
    assert rocm_calls["jit_decorate"] == 1
    assert rocm_calls["compiled_call"] == 1

    assert rocm_module.mhc_post_tilelang() == "mhc_post_tilelang"
    assert rocm_calls["jit_decorate"] == 1
    assert rocm_calls["compiled_call"] == 2
