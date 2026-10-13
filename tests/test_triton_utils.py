# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import subprocess
import sys
import types
from unittest import mock

from vllm.triton_utils import importing as triton_importing
from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def _has_triton_for_backends(*, vllm_version: str = "0.0.0", **drivers: bool) -> bool:
    """Re-evaluate ``HAS_TRITON`` against a synthetic ``triton.backends`` map.

    ``drivers`` maps backend name to whether its driver reports itself active.
    ``vllm_version`` selects the build type: importing.py takes its Triton-CPU
    branch only when "cpu" is in the installed vLLM version, so pin it rather
    than depending on the real install (the CPU CI image is a "+cpu" build).
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
            mock.patch("importlib.metadata.version", return_value=vllm_version),
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
    assert lang.constexpr(2**31 - 1) == 2**31 - 1
    assert lang.constexpr(1.5) == 1.5
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


def test_cpu_build_with_cpu_backend_keeps_triton():
    assert _has_triton_for_backends(vllm_version="0.0.0+cpu", cpu=True) is True


def test_cpu_build_without_cpu_backend_disables_triton():
    assert _has_triton_for_backends(vllm_version="0.0.0+cpu", amd=True) is False


def test_no_triton_fallback():
    # Save the real modules so they can be put back afterward - popping them
    # (rather than mock.patch.dict alone) is what forces vllm.triton_utils to
    # be freshly, unmemoized re-imported under the "triton is absent"
    # condition below.
    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "triton",
            "triton.language",
            "vllm.triton_utils",
            "vllm.triton_utils.importing",
        )
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    try:
        # mock triton not being installed
        with mock.patch.dict(sys.modules, {"triton": None}):
            from vllm.triton_utils import HAS_TRITON, tl, triton

            assert HAS_TRITON is False
            assert triton.__class__.__name__ == "TritonPlaceholder"
            assert triton.language.__class__.__name__ == "TritonLanguagePlaceholder"
            assert tl.__class__.__name__ == "TritonLanguagePlaceholder"
            assert tl.constexpr(2**31 - 1) == 2**31 - 1
    finally:
        # The pops above are outside mock.patch.dict's scope, so exiting the
        # `with` restores "triton" to absent, not to the real module - unlike
        # _has_triton_for_backends above, nothing here puts it back. Left
        # alone, every subsequent import of vllm.triton_utils in this process
        # (including other test files, and other tests' cleanup fixtures that
        # need real triton) would get this fake, HAS_TRITON=False module.
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


def test_kernel_package_imports_without_triton():
    # Exercise module-level Triton references in a fresh interpreter, where
    # package imports cannot reuse modules loaded by other tests.
    script = """
import importlib.util

real_find_spec = importlib.util.find_spec
def find_spec_without_triton(name, *args, **kwargs):
    if name in {"triton", "pytorch-triton-xpu"}:
        return None
    return real_find_spec(name, *args, **kwargs)

importlib.util.find_spec = find_spec_without_triton
from vllm.triton_utils import HAS_TRITON
assert not HAS_TRITON
import vllm.kernels
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=os.environ.copy(),
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
