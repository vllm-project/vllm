# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import importlib
import importlib.util
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


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Test requires ROCm")
def test_deepseek_v4_import_and_jit_monitor_do_not_hijack_hip_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("tilelang") is None:
        pytest.skip("Test requires TileLang to be installed")

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]

    libdl = ctypes.CDLL("libdl.so.2")
    dlsym = libdl.dlsym
    dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    dlsym.restype = ctypes.c_void_p
    dladdr = libdl.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int

    from vllm.model_executor.layers import mhc  # noqa: F401
    from vllm.models import deepseek_v4  # noqa: F401
    from vllm.utils import jit_monitor

    monkeypatch.setattr(jit_monitor, "_active", False)
    monkeypatch.setattr(jit_monitor, "_setup_triton_autotuning_print", lambda: None)
    monkeypatch.setattr(jit_monitor, "_setup_triton_jit_hook", lambda: None)
    monkeypatch.setattr(jit_monitor, "_setup_cutedsl_jit_hook", lambda: None)
    monkeypatch.setattr(
        jit_monitor,
        "_setup_tilelang_jit_hook",
        lambda: pytest.fail("TileLang JIT monitor must not run on ROCm"),
    )
    jit_monitor.activate()

    assert not any(
        name == "tilelang" or name.startswith("tilelang.") for name in sys.modules
    )

    address = dlsym(None, b"hipFree")
    assert address is not None, "hipFree is not available in the global symbol table"
    info = DlInfo()
    assert dladdr(address, ctypes.byref(info))
    source = info.dli_fname.decode() if info.dli_fname else "<unknown>"
    assert "libhip_stub.so" not in source, source
    assert "libamdhip64.so" in source, source
