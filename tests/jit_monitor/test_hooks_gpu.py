# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end JIT monitor tests: real Triton kernel, real GPU, real hook."""

from unittest import mock

import pytest

from vllm.platforms import current_platform
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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA 512-column path")
def test_kv_only_512_warmup_covers_runtime_strides_and_offsets():
    """Compile-only warmup must cover runtime tiles and input alignment classes."""
    from vllm.models.common.ops.fused_qk_rmsnorm import (
        _kv_rmsnorm_512,
        fused_kv_rmsnorm,
    )

    dtype, eps = torch.bfloat16, 1e-6
    for key in _kv_rmsnorm_512.get_warmup_keys(dtype=dtype, eps=eps):
        _kv_rmsnorm_512.compile(key)
    jit_monitor.activate(mode="error", verbose=True)
    weight = torch.randn(512, device="cuda", dtype=dtype)
    for n in (1, 3, 4, 5, 128, 129, 2048):
        for stride in (1, 512, 1792, 1793):
            storage = torch.randn(n * stride + 513, device="cuda", dtype=dtype)
            for offset in (0, 1):
                kv = storage.as_strided((n, 512), (stride, 1), offset)
                actual = fused_kv_rmsnorm(kv, weight, eps, q_size=1280)
                x = kv.float()
                expected = (
                    x
                    * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                    * weight.float()
                ).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
