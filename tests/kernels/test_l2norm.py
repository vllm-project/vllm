# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's l2norm_fwd Triton operator.

The l2norm_fwd wrapper dispatches based on the USE_DEFAULT_FLA_NORM flag and the
feature dim D. These tests cover the two Triton kernels selected as follows:

    USE_DEFAULT_FLA_NORM == 0            -> l2norm_fwd_kernel2  (any D)
    USE_DEFAULT_FLA_NORM == 1, D >  512  -> l2norm_fwd_kernel1

Both paths are exercised against a float32 PyTorch reference
(y = x / sqrt(sum(x^2) + eps), per row).

Source: vllm/model_executor/layers/fla/ops/l2norm.py
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops import l2norm as l2norm_mod
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def l2norm_ref(x, eps=1e-6):
    """Pure PyTorch L2 norm over the last dim: y = x / sqrt(sum(x^2) + eps)."""
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _set_flag(monkeypatch, use_default):
    # USE_DEFAULT_FLA_NORM is read from the env at import time; patch the module
    # attribute directly so dispatch is deterministic regardless of import order.
    monkeypatch.setattr(l2norm_mod, "USE_DEFAULT_FLA_NORM", use_default)


def _spy_kernel(monkeypatch, kernel):
    """Patch `kernel` with a launch-recording proxy; return the record list.

    The list gets the kernel name appended each time the kernel is launched, so
    a test can assert which kernel the wrapper actually dispatched to.
    """
    launched = []
    real = getattr(l2norm_mod, kernel)

    class _Spy:
        def __getitem__(self, grid):
            launched.append(kernel)
            return real[grid]

    monkeypatch.setattr(l2norm_mod, kernel, _Spy())
    return launched


# (use_default, T, D, kernel) — each case asserts BOTH that the wrapper routes
# to `kernel` and that its output matches the reference, so coverage of a
# specific kernel is guaranteed rather than assumed from the dispatch rule.
CASES = [
    (0, 16, 64, "l2norm_fwd_kernel2"),  # kernel #19 (default, any D)
    (0, 64, 256, "l2norm_fwd_kernel2"),
    (0, 128, 512, "l2norm_fwd_kernel2"),
    (0, 16, 1024, "l2norm_fwd_kernel2"),  # kernel2 ignores D
    (0, 1024, 128, "l2norm_fwd_kernel2"),
    (1, 16, 1024, "l2norm_fwd_kernel1"),  # kernel #17 (default-fla, D > 512)
    (1, 1, 2048, "l2norm_fwd_kernel1"),
    (1, 64, 768, "l2norm_fwd_kernel1"),
    (1, 128, 1024, "l2norm_fwd_kernel1"),
]


@pytest.mark.parametrize(
    "use_default,T,D,kernel",
    CASES,
    ids=[f"fla{ud}_T{t}_D{d}_{k}" for ud, t, d, k in CASES],
)
@torch.inference_mode()
def test_l2norm(monkeypatch, use_default, T, D, kernel):
    """l2norm_fwd routes to the expected kernel and matches the reference."""
    _set_flag(monkeypatch, use_default)
    launched = _spy_kernel(monkeypatch, kernel)
    torch.manual_seed(0)
    x = torch.randn(T, D, device=DEVICE, dtype=torch.float32)

    y = l2norm_fwd(x, eps=1e-6)
    y_ref = l2norm_ref(x, eps=1e-6)

    assert launched == [kernel], f"expected {kernel}, dispatched {launched}"
    assert y.shape == y_ref.shape
    assert not torch.isnan(y).any()
    torch.testing.assert_close(y.float(), y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_default", [0, 1])
@torch.inference_mode()
def test_l2norm_3d(monkeypatch, use_default):
    """3D input is flattened to 2D rows by the wrapper (both flag paths)."""
    _set_flag(monkeypatch, use_default)
    torch.manual_seed(0)
    x = torch.randn(4, 32, 1024, device=DEVICE, dtype=torch.float32)

    y = l2norm_fwd(x, eps=1e-6)
    y_ref = l2norm_ref(x, eps=1e-6)

    assert y.shape == x.shape
    torch.testing.assert_close(y.float(), y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_default", [0, 1])
@torch.inference_mode()
def test_l2norm_bfloat16(monkeypatch, use_default):
    """bfloat16 input matches the fp32 reference within a loose tolerance."""
    _set_flag(monkeypatch, use_default)
    torch.manual_seed(0)
    x = torch.randn(64, 1024, device=DEVICE, dtype=torch.bfloat16)

    y = l2norm_fwd(x, eps=1e-6)
    y_ref = l2norm_ref(x.float(), eps=1e-6)

    torch.testing.assert_close(y.float(), y_ref, rtol=5e-3, atol=5e-3)
