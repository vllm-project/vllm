# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's l2norm_fwd Triton operator.

The l2norm_fwd wrapper dispatches based on the USE_DEFAULT_FLA_NORM flag and the
feature dim D. These tests cover two of the three Triton kernels:

    USE_DEFAULT_FLA_NORM == 0            -> l2norm_fwd_kernel2  (any D)
    USE_DEFAULT_FLA_NORM == 1, D >  512  -> l2norm_fwd_kernel1

Both paths are exercised against a float32 PyTorch reference
(y = x / sqrt(sum(x^2) + eps), per row).

Source: vllm/third_party/flash_linear_attention/ops/l2norm.py
"""

from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import l2norm as l2norm_mod
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="l2norm_fwd dispatches a Triton kernel that requires a "
    "CUDA-alike or XPU device.",
)

DEVICE = current_platform.device_type
EPS = 1e-6


def l2norm_ref(x, eps=EPS):
    """Pure PyTorch L2 norm over the last dim: y = x / sqrt(sum(x^2) + eps)."""
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _set_flag(monkeypatch, use_default):
    # USE_DEFAULT_FLA_NORM is read from the env at import time; patch the module
    # attribute directly so dispatch is deterministic regardless of import order.
    monkeypatch.setattr(l2norm_mod, "USE_DEFAULT_FLA_NORM", use_default)


# (use_default, D, kernel)
ROUTING_CASES = [
    (0, 64, "l2norm_fwd_kernel2"),
    (0, 1024, "l2norm_fwd_kernel2"),
    (1, 768, "l2norm_fwd_kernel1"),
    (1, 2048, "l2norm_fwd_kernel1"),
]


@pytest.mark.parametrize("use_default,D,kernel", ROUTING_CASES)
@torch.inference_mode()
def test_l2norm_dispatch(monkeypatch, use_default, D, kernel):
    """l2norm_fwd routes to the expected kernel for the given flag and D."""
    _set_flag(monkeypatch, use_default)
    x = torch.randn(8, D, device=DEVICE, dtype=torch.float32)

    real = getattr(l2norm_mod, kernel)
    with patch.object(l2norm_mod, kernel, wraps=real) as spy:
        # triton launches as kernel[grid](...), so the launch is observed on
        # __getitem__ rather than on the mock itself.
        spy.__getitem__.side_effect = real.__getitem__
        l2norm_fwd(x, eps=EPS)

    assert spy.__getitem__.called, f"expected {kernel} to be launched"


# (use_default, shape, dtype, tol) - flag=1 is paired only with D > 512 so the
# matrix stays on l2norm_fwd_kernel1.
NUMERIC_CASES = [
    (0, (16, 64), torch.float32, 1e-4),
    (0, (64, 256), torch.float32, 1e-4),
    (0, (128, 512), torch.float32, 1e-4),
    (0, (1024, 128), torch.float32, 1e-4),
    (0, (4, 32, 1024), torch.float32, 1e-4),
    (0, (64, 1024), torch.bfloat16, 5e-3),
    (1, (16, 1024), torch.float32, 1e-4),
    (1, (1, 2048), torch.float32, 1e-4),
    (1, (64, 768), torch.float32, 1e-4),
    (1, (4, 32, 1024), torch.float32, 1e-4),
    (1, (64, 1024), torch.bfloat16, 5e-3),
]


@pytest.mark.parametrize("use_default,shape,dtype,tol", NUMERIC_CASES)
@torch.inference_mode()
def test_l2norm_matches_reference(monkeypatch, use_default, shape, dtype, tol):
    """l2norm_fwd output matches the float32 reference (2D and 3D inputs)."""
    _set_flag(monkeypatch, use_default)
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)

    y = l2norm_fwd(x, eps=EPS)
    y_ref = l2norm_ref(x.float(), eps=EPS)

    assert y.shape == x.shape
    torch.testing.assert_close(y.float(), y_ref, rtol=tol, atol=tol)
