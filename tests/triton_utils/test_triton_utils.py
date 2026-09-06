# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from unittest import mock

from vllm.triton_utils import importing as triton_importing
from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def _has_triton_for_backends(**drivers: bool) -> bool:
    """Re-evaluate ``HAS_TRITON`` against a synthetic ``triton.backends`` map.

    ``drivers`` maps backend name to whether its driver reports itself active.
    """
    backends = {}
    for name, is_active in drivers.items():
        driver = mock.Mock()
        driver.is_active.return_value = is_active
        backend = mock.Mock()
        backend.driver = driver
        backends[name] = backend

    triton_mod = types.ModuleType("triton")
    triton_mod.__spec__ = importlib.machinery.ModuleSpec("triton", None)
    backends_mod = types.ModuleType("triton.backends")
    backends_mod.__dict__["backends"] = backends
    triton_mod.__dict__["backends"] = backends_mod

    patched_modules = {"triton": triton_mod, "triton.backends": backends_mod}
    try:
        with (
            mock.patch.dict(sys.modules, patched_modules),
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            return importlib.reload(triton_importing).HAS_TRITON
    finally:
        # Restore the module state derived from the real environment.
        importlib.reload(triton_importing)


def test_triton_placeholder_is_module():
    triton = TritonPlaceholder()
    assert isinstance(triton, types.ModuleType)
    assert triton.__name__ == "triton"


def test_triton_language_placeholder_is_module():
    triton_language = TritonLanguagePlaceholder()
    assert isinstance(triton_language, types.ModuleType)
    assert triton_language.__name__ == "triton.language"


def test_triton_placeholder_decorators():
    triton = TritonPlaceholder()

    @triton.jit
    def foo(x):
        return x

    @triton.autotune
    def bar(x):
        return x

    @triton.heuristics
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_decorators_with_args():
    triton = TritonPlaceholder()

    @triton.jit(debug=True)
    def foo(x):
        return x

    @triton.autotune(configs=[], key="x")
    def bar(x):
        return x

    @triton.heuristics({"BLOCK_SIZE": lambda args: 128 if args["x"] > 1024 else 64})
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_language():
    lang = TritonLanguagePlaceholder()
    assert isinstance(lang, types.ModuleType)
    assert lang.__name__ == "triton.language"
    assert lang.constexpr is None
    assert lang.dtype is None
    assert lang.int64 is None
    assert lang.int32 is None
    assert lang.tensor is None


def test_triton_placeholder_language_from_parent():
    triton = TritonPlaceholder()
    lang = triton.language
    assert isinstance(lang, TritonLanguagePlaceholder)


def test_cpu_backend_does_not_disable_triton():
    # The cpu backend's driver is always active, so counting it alongside a GPU
    # backend used to yield 2 active drivers and disable Triton entirely.
    assert _has_triton_for_backends(amd=True, cpu=True) is True


def test_single_gpu_backend_keeps_triton():
    assert _has_triton_for_backends(amd=True) is True


def test_multiple_active_gpu_backends_disable_triton():
    assert _has_triton_for_backends(amd=True, nvidia=True) is False


def test_cpu_backend_alone_disables_triton():
    assert _has_triton_for_backends(cpu=True) is False


def test_no_triton_fallback():
    # clear existing triton modules
    sys.modules.pop("triton", None)
    sys.modules.pop("triton.language", None)
    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)

    # mock triton not being installed
    with mock.patch.dict(sys.modules, {"triton": None}):
        from vllm.triton_utils import HAS_TRITON, tl, triton

        assert HAS_TRITON is False
        assert triton.__class__.__name__ == "TritonPlaceholder"
        assert triton.language.__class__.__name__ == "TritonLanguagePlaceholder"
        assert tl.__class__.__name__ == "TritonLanguagePlaceholder"
