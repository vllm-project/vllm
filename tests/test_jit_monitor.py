# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm.triton_utils import jit_monitor


@pytest.fixture(autouse=True)
def _reset_monitor():
    """Reset global monitor state between tests."""
    jit_monitor._active = False
    yield
    jit_monitor._active = False


# ------------------------------------------------------------------
# Helpers — lightweight stand-ins for triton.knobs
# ------------------------------------------------------------------


def _make_fake_knobs(*, autotuning_print=False, jit_hook=None):
    """Build a minimal fake ``triton.knobs`` namespace."""
    autotuning = SimpleNamespace(print=autotuning_print)
    runtime = SimpleNamespace(jit_post_compile_hook=jit_hook)
    return SimpleNamespace(autotuning=autotuning, runtime=runtime)


def _patch_triton_knobs(fake_knobs):
    """Context manager that makes ``from triton import knobs`` return *fake_knobs*."""
    fake_triton = SimpleNamespace(knobs=fake_knobs)
    return mock.patch.dict(sys.modules, {"triton": fake_triton})


# ------------------------------------------------------------------
# Unit tests (no GPU required, triton is mocked)
# ------------------------------------------------------------------


class TestActivateBasic:
    def test_sets_active(self):
        assert not jit_monitor.is_active()
        with _patch_triton_knobs(_make_fake_knobs()):
            jit_monitor.activate()
        assert jit_monitor.is_active()

    def test_idempotent(self):
        fake = _make_fake_knobs()
        with _patch_triton_knobs(fake):
            jit_monitor.activate()
            first_hook = fake.runtime.jit_post_compile_hook
            jit_monitor.activate()
            assert fake.runtime.jit_post_compile_hook is first_hook

    def test_logs_info_on_activation(self):
        with (
            mock.patch.object(jit_monitor.logger, "info") as m,
            _patch_triton_knobs(_make_fake_knobs()),
        ):
            jit_monitor.activate()
        m.assert_called_once()
        assert "Kernel JIT monitor activated" in m.call_args[0][0]


class TestAutotuningPrint:
    def test_enables_autotuning_print(self):
        fake = _make_fake_knobs(autotuning_print=False)
        with _patch_triton_knobs(fake):
            jit_monitor.activate()
        assert fake.autotuning.print is True

    def test_respects_user_opt_out(self):
        fake = _make_fake_knobs(autotuning_print=False)
        with (
            mock.patch.dict(os.environ, {"TRITON_PRINT_AUTOTUNING": "0"}),
            _patch_triton_knobs(fake),
        ):
            jit_monitor.activate()
        assert fake.autotuning.print is False

    def test_noop_when_user_already_enabled(self):
        fake = _make_fake_knobs(autotuning_print=True)
        with (
            mock.patch.dict(os.environ, {"TRITON_PRINT_AUTOTUNING": "1"}),
            _patch_triton_knobs(fake),
        ):
            jit_monitor.activate()
        assert fake.autotuning.print is True


class TestJitHook:
    def test_hook_registered(self):
        fake = _make_fake_knobs()
        assert fake.runtime.jit_post_compile_hook is None
        with _patch_triton_knobs(fake):
            jit_monitor.activate()
        assert fake.runtime.jit_post_compile_hook is not None

    def test_hook_logs_warning(self):
        fake = _make_fake_knobs()
        with _patch_triton_knobs(fake):
            jit_monitor.activate()

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="test_kernel")

        with mock.patch.object(jit_monitor.logger, "warning") as m:
            hook(
                key="some_key",
                repr="some_repr",
                fn=mock_fn,
                compile=lambda: None,
                is_manual_warmup=False,
                already_compiled=False,
            )

        m.assert_called_once()
        msg = m.call_args[0][0] % m.call_args[0][1:]
        assert "Triton kernel JIT compilation during inference" in msg
        assert "test_kernel" in msg

    def test_hook_chains_existing_hook(self):
        existing = mock.MagicMock(return_value="existing_result")
        fake = _make_fake_knobs(jit_hook=existing)
        with _patch_triton_knobs(fake):
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
        with _patch_triton_knobs(fake):
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


class TestNoTritonFallback:
    def test_activate_without_triton(self):
        with mock.patch.object(jit_monitor, "HAS_TRITON", False):
            jit_monitor.activate()
        assert jit_monitor.is_active()


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


def _run_add_kernel(n: int, block: int = 256) -> None:
    """Launch ``_add_kernel`` with vectors of length *n*."""
    x = torch.randn(n, device="cuda")
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
        with mock.patch.object(jit_monitor.logger, "warning") as w:
            _run_add_kernel(1024)
        w.assert_not_called()

    def test_warning_on_new_constexpr(self):
        _run_add_kernel(1024, block=256)

        jit_monitor.activate()
        with mock.patch.object(jit_monitor.logger, "warning") as w:
            # Different BLOCK (a tl.constexpr) forces recompilation.
            _run_add_kernel(1024, block=512)
        w.assert_called()
        msg = w.call_args[0][0] % w.call_args[0][1:]
        assert "_add_kernel" in msg
