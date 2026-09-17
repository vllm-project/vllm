# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end JIT monitor tests: real Triton kernel, real GPU, real hook."""

from unittest import mock

import pytest

from vllm.utils import jit_monitor

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

pytestmark = pytest.mark.skipif(
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


def test_no_warning_on_cached_shape():
    _run_add_kernel(1024)

    jit_monitor.activate()
    with mock.patch.object(jit_monitor.logger, "warning_once") as w:
        _run_add_kernel(1024)
    w.assert_not_called()


def test_warning_on_new_constexpr():
    _run_add_kernel(1024, block=256)

    jit_monitor.activate()
    with mock.patch.object(jit_monitor.logger, "warning_once") as w:
        # Different BLOCK (a tl.constexpr) forces recompilation.
        _run_add_kernel(1024, block=512)
    w.assert_called()
    msg = w.call_args[0][0] % w.call_args[0][1:]
    assert "_add_kernel" in msg


def test_verbose_warning_on_each_new_pointer_alignment():
    _run_add_kernel(1024)

    jit_monitor.activate(verbose=True)
    with (
        mock.patch.object(jit_monitor.logger, "warning") as w,
        mock.patch.object(jit_monitor.logger, "warning_once") as w_once,
    ):
        _run_add_kernel(1024, offset=1)
    assert w.called
    w_once.assert_not_called()
