# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest

from vllm.utils import jit_monitor


@pytest.fixture(autouse=True)
def _reset_monitor():
    """Reset global monitor state between tests."""
    jit_monitor._active = False
    jit_monitor._mode = "warn"
    jit_monitor._verbose = False
    jit_monitor._cutedsl_hook_installed = False
    jit_monitor._tilelang_hook_installed = False
    jit_monitor._tilelang_jitimpl_compile_depth = 0
    yield
    jit_monitor._active = False
    jit_monitor._mode = "warn"
    jit_monitor._verbose = False
    jit_monitor._cutedsl_hook_installed = False
    jit_monitor._tilelang_hook_installed = False
    jit_monitor._tilelang_jitimpl_compile_depth = 0


# ------------------------------------------------------------------
# Helpers — lightweight stand-ins for the modules ``activate()`` patches
# ------------------------------------------------------------------


def _make_fake_knobs(*, autotuning_print=False, jit_hook=None):
    """Build a minimal fake ``triton.knobs`` namespace."""
    autotuning = SimpleNamespace(print=autotuning_print)
    runtime = SimpleNamespace(jit_post_compile_hook=jit_hook)
    return SimpleNamespace(autotuning=autotuning, runtime=runtime)


def _fake_cute_import_modules(compile_fn):
    """Fake Python's parent package + submodule for ``import cutlass.cute``."""
    fake_cute = cast(Any, ModuleType("cutlass.cute"))
    fake_cute.compile = compile_fn
    fake_parent_package = cast(Any, ModuleType("cutlass"))
    fake_parent_package.__path__ = []
    fake_parent_package.cute = fake_cute
    return {
        "cutlass": fake_parent_package,
        "cutlass.cute": fake_cute,
    }


def _fake_cute_compile(*args, **kwargs):
    return "compiled"


def _fake_tilelang_import_modules():
    """Fake Python's TileLang modules touched by ``jit_monitor.activate``."""

    class FakeJITKernel:
        def __init__(self, *args, **kwargs):
            pass

    class FakeJITImpl:
        def __init__(self, func, signature):
            self.func = func
            self.signature = signature
            self.mode = "lazy"
            self._kernel_cache = {}

        def __call__(self, *args, **kwargs):
            key, _ = self.func.parse_args(*args, **kwargs)
            kernel = self._kernel_cache.get(key)
            if kernel is None:
                kernel = "compiled"
                self._kernel_cache[key] = kernel
            return kernel

    fake_kernel = cast(Any, ModuleType("tilelang.jit.kernel"))
    fake_kernel.JITKernel = FakeJITKernel

    fake_jit = cast(Any, ModuleType("tilelang.jit"))
    fake_jit.JITImpl = FakeJITImpl
    fake_jit.kernel = fake_kernel

    fake_tilelang = cast(Any, ModuleType("tilelang"))
    fake_tilelang.jit = fake_jit

    return {
        "tilelang": fake_tilelang,
        "tilelang.jit": fake_jit,
        "tilelang.jit.kernel": fake_kernel,
    }


@contextmanager
def _patch_jit_modules(fake_knobs, *, cute_compile=_fake_cute_compile):
    """Patch the Triton and CuTeDSL imports touched by ``jit_monitor.activate``."""
    fake_triton = cast(Any, ModuleType("triton"))
    fake_triton.knobs = fake_knobs
    with (
        mock.patch.dict(
            sys.modules,
            {
                "triton": fake_triton,
                **_fake_cute_import_modules(cute_compile),
                **_fake_tilelang_import_modules(),
            },
        ),
        mock.patch.object(jit_monitor, "HAS_TRITON", True),
    ):
        yield


# ------------------------------------------------------------------
# Unit tests (no GPU required, triton is mocked)
# ------------------------------------------------------------------


class TestActivateBasic:
    def test_sets_active(self):
        assert not jit_monitor.is_active()
        with _patch_jit_modules(_make_fake_knobs()):
            jit_monitor.activate()
        assert jit_monitor.is_active()

    def test_idempotent(self):
        fake = _make_fake_knobs()
        with _patch_jit_modules(fake):
            jit_monitor.activate()
            first_hook = fake.runtime.jit_post_compile_hook
            jit_monitor.activate()
            assert fake.runtime.jit_post_compile_hook is first_hook

    def test_logs_info_on_activation(self):
        with (
            mock.patch.object(jit_monitor.logger, "info") as m,
            _patch_jit_modules(_make_fake_knobs()),
        ):
            jit_monitor.activate()
        m.assert_called_once()
        assert "Kernel JIT monitor activated" in m.call_args[0][0]

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="Unsupported JIT monitor mode"):
            jit_monitor.activate(mode="panic")  # type: ignore[arg-type]


class TestAutotuningPrint:
    def test_enables_autotuning_print(self):
        fake = _make_fake_knobs(autotuning_print=False)
        with _patch_jit_modules(fake):
            jit_monitor.activate()
        assert fake.autotuning.print is True

    def test_respects_user_opt_out(self):
        fake = _make_fake_knobs(autotuning_print=False)
        with (
            mock.patch.dict(os.environ, {"TRITON_PRINT_AUTOTUNING": "0"}),
            _patch_jit_modules(fake),
        ):
            jit_monitor.activate()
        assert fake.autotuning.print is False

    def test_noop_when_user_already_enabled(self):
        fake = _make_fake_knobs(autotuning_print=True)
        with (
            mock.patch.dict(os.environ, {"TRITON_PRINT_AUTOTUNING": "1"}),
            _patch_jit_modules(fake),
        ):
            jit_monitor.activate()
        assert fake.autotuning.print is True


class TestTritonJitHook:
    def test_hook_registered(self):
        fake = _make_fake_knobs()
        assert fake.runtime.jit_post_compile_hook is None
        with _patch_jit_modules(fake):
            jit_monitor.activate()
        assert fake.runtime.jit_post_compile_hook is not None

    def test_hook_logs_warning(self):
        fake = _make_fake_knobs()
        with _patch_jit_modules(fake):
            jit_monitor.activate()

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="test_kernel")

        with (
            mock.patch.object(jit_monitor.logger, "warning_once") as m,
            mock.patch.object(jit_monitor.logger, "warning") as warning,
        ):
            hook(
                key="some_key",
                repr="some_repr",
                fn=mock_fn,
                compile=lambda: None,
                is_manual_warmup=False,
                already_compiled=False,
            )

        m.assert_called_once()
        warning.assert_not_called()
        msg = m.call_args[0][0] % m.call_args[0][1:]
        assert "Triton kernel JIT compilation during inference" in msg
        assert "test_kernel" in msg

    def test_hook_chains_existing_hook(self):
        existing = mock.MagicMock(return_value="existing_result")
        fake = _make_fake_knobs(jit_hook=existing)
        with _patch_jit_modules(fake):
            jit_monitor.activate()

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="chained_kernel")
        kwargs = dict(
            key="k",
            repr="r",
            fn=mock_fn,
            compile=lambda: None,
            is_manual_warmup=False,
            already_compiled=False,
        )
        result = hook(**kwargs)

        existing.assert_called_once()
        assert result == "existing_result"

    def test_hook_works_without_existing_hook(self):
        fake = _make_fake_knobs(jit_hook=None)
        with _patch_jit_modules(fake):
            jit_monitor.activate()

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="solo_kernel")
        result = hook(
            key="k",
            repr="r",
            fn=mock_fn,
            compile=lambda: None,
            is_manual_warmup=False,
            already_compiled=False,
        )
        assert result is None

    def test_error_mode_raises(self):
        fake = _make_fake_knobs()
        with _patch_jit_modules(fake):
            jit_monitor.activate(mode="error")

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="error_kernel")
        with pytest.raises(RuntimeError, match="Triton kernel JIT compilation"):
            hook(
                key="k",
                repr="r",
                fn=mock_fn,
                compile=lambda: None,
                is_manual_warmup=False,
                already_compiled=False,
            )


class TestNoTritonFallback:
    def test_activate_without_triton(self):
        with mock.patch.object(jit_monitor, "HAS_TRITON", False):
            jit_monitor.activate()
        assert jit_monitor.is_active()


class TestCuTeDSLHook:
    def test_compile_logs_warning(self):
        def compile_fn(*args, **kwargs):
            return "compiled"

        with _patch_jit_modules(_make_fake_knobs(), cute_compile=compile_fn):
            import cutlass.cute as cute

            jit_monitor.activate()
            with mock.patch.object(jit_monitor.logger, "warning_once") as warning_once:
                result = cute.compile(lambda: None, "arg", option=True)

        assert result == "compiled"
        warning_once.assert_called_once()
        msg = warning_once.call_args[0][0] % warning_once.call_args[0][1:]
        assert "CuTeDSL JIT compilation during inference" in msg

    def test_compile_logs_verbose_warning(self):
        def compile_fn(*args, **kwargs):
            return "compiled"

        with _patch_jit_modules(_make_fake_knobs(), cute_compile=compile_fn):
            import cutlass.cute as cute

            jit_monitor.activate(verbose=True)
            with mock.patch.object(jit_monitor.logger, "warning") as warning:
                result = cute.compile(lambda: None, "arg", option=True)

        assert result == "compiled"
        warning.assert_called_once()
        msg = warning.call_args[0][0] % warning.call_args[0][1:]
        assert "CuTeDSL JIT compilation during inference" in msg

    def test_error_mode_raises(self):
        def compile_fn(*args, **kwargs):
            return "compiled"

        with _patch_jit_modules(_make_fake_knobs(), cute_compile=compile_fn):
            import cutlass.cute as cute

            jit_monitor.activate(mode="error")
            with pytest.raises(RuntimeError, match="CuTeDSL JIT compilation"):
                cute.compile(lambda: None, "arg", option=True)

    def test_subscripted_compile_is_monitored(self):
        """``cute.compile[options](...)`` (flashinfer >= 0.6.14) must work."""

        class FakeCompileCallable:
            def __getitem__(self, options):
                return self

            def __call__(self, *args, **kwargs):
                return "compiled"

        with _patch_jit_modules(_make_fake_knobs(), cute_compile=FakeCompileCallable()):
            import cutlass.cute as cute

            jit_monitor.activate()
            with mock.patch.object(jit_monitor.logger, "warning_once") as warning_once:
                result = cute.compile[("opt_level", 3)](lambda: None, "arg")

        assert result == "compiled"
        warning_once.assert_called_once()


class TestTileLangHook:
    def test_jit_kernel_logs_warning(self):
        with _patch_jit_modules(_make_fake_knobs()):
            from tilelang.jit.kernel import JITKernel

            func = SimpleNamespace(attrs={"global_symbol": "tl_kernel"})
            jit_monitor.activate()
            with mock.patch.object(jit_monitor.logger, "warning_once") as warning_once:
                JITKernel(func=func, out_idx=None, execution_backend="tvm_ffi")

        warning_once.assert_called_once()
        msg = warning_once.call_args[0][0] % warning_once.call_args[0][1:]
        assert "TileLang JIT compilation during inference" in msg
        assert "tl_kernel" in msg

    def test_jit_impl_logs_warning(self):
        with _patch_jit_modules(_make_fake_knobs()):
            from tilelang.jit import JITImpl

            def tilelang_fn(
                gemm_out_mul,
                hidden_size: int,
                n_splits: int = 1,
                hc_mult: int = 4,
            ):
                return None

            class FakeFunc:
                orig_func = tilelang_fn

                def parse_args(self, *args, **kwargs):
                    return (
                        (
                            "tilelang_key",
                            kwargs["hidden_size"],
                            kwargs.get("n_splits", 1),
                        ),
                        {},
                    )

                def set_mode(self, mode):
                    self.mode = mode

            tensor = SimpleNamespace(
                shape=(2, 16, 24),
                dtype="float32",
                device="cuda:0",
            )
            impl = JITImpl(FakeFunc(), inspect.signature(tilelang_fn))

            jit_monitor.activate()
            with (
                mock.patch.object(jit_monitor.logger, "warning_once") as warning_once,
                mock.patch.object(jit_monitor.logger, "warning") as warning,
            ):
                impl(tensor, hidden_size=7168, n_splits=2)

        warning_once.assert_called_once()
        warning.assert_not_called()
        msg = warning_once.call_args[0][0] % warning_once.call_args[0][1:]
        assert "TileLang JIT compilation during inference" in msg
        assert "tilelang_fn" in msg

    def test_jit_impl_does_not_log_on_cache_hit(self):
        with _patch_jit_modules(_make_fake_knobs()):
            from tilelang.jit import JITImpl

            def tilelang_fn(gemm_out_mul, n_splits: int = 1):
                return None

            class FakeFunc:
                orig_func = tilelang_fn

                def parse_args(self, *args, **kwargs):
                    return (("tilelang_key", kwargs.get("n_splits", 1)), {})

                def set_mode(self, mode):
                    self.mode = mode

            tensor = SimpleNamespace(shape=(2, 16, 24), dtype="float32")
            impl = JITImpl(FakeFunc(), inspect.signature(tilelang_fn))

            jit_monitor.activate()
            with mock.patch.object(jit_monitor.logger, "warning_once") as warning_once:
                impl(tensor, n_splits=2)
                impl(tensor, n_splits=2)

        warning_once.assert_called_once()

    def test_from_database_does_not_log(self):
        with _patch_jit_modules(_make_fake_knobs()):
            from tilelang.jit.kernel import JITKernel

            func = SimpleNamespace(attrs={"global_symbol": "cached_tl_kernel"})
            jit_monitor.activate()
            with mock.patch.object(jit_monitor.logger, "warning_once") as warning_once:
                JITKernel(func=func, from_database=True)

        warning_once.assert_not_called()

    def test_error_mode_raises(self):
        with _patch_jit_modules(_make_fake_knobs()):
            from tilelang.jit.kernel import JITKernel

            func = SimpleNamespace(attrs={"global_symbol": "error_tl_kernel"})
            jit_monitor.activate(mode="error")
            with pytest.raises(RuntimeError, match="TileLang JIT compilation"):
                JITKernel(func=func)


# ------------------------------------------------------------------
# Integration tests (real Triton + GPU)
# ------------------------------------------------------------------

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_skip_no_gpu = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_TRITON),
    reason="Requires CUDA GPU and Triton",
)


if _HAS_TRITON:

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


def _run_add_kernel(n: int, block: int = 256, offset: int = 0) -> None:
    """Launch ``_add_kernel`` with vectors of length *n*."""
    x = torch.randn(n + offset, device="cuda")[offset:]  # affect alignment
    y = torch.randn(n, device="cuda")
    out = torch.empty(n, device="cuda")
    grid = ((n + block - 1) // block,)
    _add_kernel[grid](x, y, out, n, BLOCK=block)
    torch.accelerator.synchronize()


@_skip_no_gpu
class TestTritonJitHookIntegration:
    """End-to-end: real Triton kernel, real GPU, real hook."""

    def test_no_warning_on_cached_shape(self):
        _run_add_kernel(1024)

        jit_monitor.activate()
        with mock.patch.object(jit_monitor.logger, "warning_once") as w:
            _run_add_kernel(1024)
        w.assert_not_called()

    def test_warning_on_new_constexpr(self):
        _run_add_kernel(1024, block=256)

        jit_monitor.activate()
        with mock.patch.object(jit_monitor.logger, "warning_once") as w:
            # Different BLOCK (a tl.constexpr) forces recompilation.
            _run_add_kernel(1024, block=512)
        w.assert_called()
        msg = w.call_args[0][0] % w.call_args[0][1:]
        assert "_add_kernel" in msg

    def test_verbose_warning_on_each_new_pointer_alignment(self):
        _run_add_kernel(1024)

        jit_monitor.activate(verbose=True)
        with (
            mock.patch.object(jit_monitor.logger, "warning") as w,
            mock.patch.object(jit_monitor.logger, "warning_once") as w_once,
        ):
            _run_add_kernel(1024, offset=1)
        assert w.called
        w_once.assert_not_called()
