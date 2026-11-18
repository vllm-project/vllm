# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from unittest import mock

import torch

from vllm.triton_utils import importing as triton_importing
from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def _has_triton_for_backends(*, vllm_version: str = "0.0.0", **drivers: bool) -> bool:
    """Re-evaluate `HAS_TRITON` against a synthetic `triton.backends` map.

    `drivers` maps backend name to whether its driver reports itself active.
    `vllm_version` selects the build type: importing.py takes its Triton-CPU
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


def test_kda_prefill_conv_imports_without_triton():
    # CPU builds import the GLM-5.3-Flash KDA layer, and with it the planar
    # stencil module, without Triton (non-x86 CPUs have no triton); it must
    # import under the placeholder, where tl.constexpr(x) is just x and kernels
    # are plain functions, as the stock causal_conv1d module does. Same module
    # dance as test_no_triton_fallback.
    names = (
        "triton",
        "triton.language",
        "vllm.triton_utils",
        "vllm.triton_utils.importing",
        "vllm.model_executor.warmup.jit_warmup_triton_helper",
        "vllm.models.glm5next.nvidia.ops.kda_prefill_conv",
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
    )
    saved_modules = {name: sys.modules.get(name) for name in names}
    for name in saved_modules:
        sys.modules.pop(name, None)
    try:
        with mock.patch.dict(sys.modules, {"triton": None}):
            from vllm.triton_utils import HAS_TRITON

            assert HAS_TRITON is False
            planar = importlib.import_module(
                "vllm.models.glm5next.nvidia.ops.kda_prefill_conv"
            )
            importlib.import_module(
                "vllm.model_executor.layers.mamba.ops.causal_conv1d"
            )
            assert planar.CONV_WIDTH == 4 and planar.TILE_TOKENS == 16
            # Host-side code works on the plain ints as well.
            plan = planar.planar_conv_warmup_plan(
                dtype=torch.bfloat16,
                weight_dtype=torch.float32,
                bias_dtype=None,
                dim=768,
                num_planes=3,
                row_strides=(1026,),
                conv_state_dim_first=True,
            )
            # One stencil launch for the single row stride, the fixup and the
            # state update.
            assert len(plan) == 3
    finally:
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
