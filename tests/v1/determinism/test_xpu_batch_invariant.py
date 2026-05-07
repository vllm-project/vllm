# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant kernel overrides on Intel XPU.

Verifies correctness (vs torch reference) and the batch-invariance property
(result for one item is bitwise identical regardless of other items in the
batch) for the ops registered with the "XPU" dispatch key:
  - aten::bmm
  - aten::_log_softmax
  - aten::softmax / aten::_softmax
  - aten::mean.dim

Also tests the Triton rms_norm kernel (called directly, not via aten dispatch)
and verifies that registering the XPU overrides correctly routes standard
torch ops through the batch-invariant implementations.
"""

import pytest
import torch
from utils import skip_unsupported_xpu

from vllm.model_executor.layers.batch_invariant import (
    bmm_batch_invariant,
    log_softmax,
    mean_batch_invariant,
    mean_dim,
    rms_norm,
    softmax_batch_invariant,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


# ---------------------------------------------------------------------------
# BMM tests
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize(
    "B,M,K,N",
    [
        (1, 32, 64, 16),
        (4, 64, 128, 64),
        (8, 512, 1024, 512),
        (2, 1, 64, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_correctness(B, M, K, N, dtype):
    """bmm_batch_invariant matches torch.bmm within tolerance."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(B, M, K, dtype=dtype, device=device)
    b = torch.randn(B, K, N, dtype=dtype, device=device)

    expected = torch.bmm(a, b)
    actual = bmm_batch_invariant(a, b)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_batch_invariance(dtype):
    """Same slice gives bitwise-identical result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a_single = torch.randn(1, 64, 32, dtype=dtype, device=device)
    b_single = torch.randn(1, 32, 128, dtype=dtype, device=device)

    out_single = bmm_batch_invariant(a_single, b_single)

    # Embed the same slice in a larger batch with random neighbors
    a_batch = torch.randn(8, 64, 32, dtype=dtype, device=device)
    b_batch = torch.randn(8, 32, 128, dtype=dtype, device=device)
    a_batch[5] = a_single[0]
    b_batch[5] = b_single[0]

    out_batch = bmm_batch_invariant(a_batch, b_batch)

    assert torch.equal(out_single[0], out_batch[5])


# ---------------------------------------------------------------------------
# Log-softmax tests
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize("rows,cols", [(1, 128), (16, 1024), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_softmax_correctness(rows, cols, dtype):
    """Triton log_softmax matches torch.log_softmax."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(rows, cols, dtype=dtype, device=device)

    expected = torch.log_softmax(x, dim=-1)
    actual = log_softmax(x, dim=-1)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_softmax_batch_invariance(dtype):
    """Same row gives identical result regardless of other rows."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 2048, dtype=dtype, device=device)
    out_single = log_softmax(row, dim=-1)

    batch = torch.randn(16, 2048, dtype=dtype, device=device)
    batch[7] = row[0]
    out_batch = log_softmax(batch, dim=-1)

    assert torch.equal(out_single[0], out_batch[7])


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize("rows,cols", [(1, 128), (16, 1024), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_correctness(rows, cols, dtype):
    """Deterministic softmax matches torch.softmax."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(rows, cols, dtype=dtype, device=device)

    expected = torch.softmax(x, dim=-1)
    actual = softmax_batch_invariant(x, dim=-1)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_batch_invariance(dtype):
    """Same row gives identical result regardless of other rows."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 2048, dtype=dtype, device=device)
    out_single = softmax_batch_invariant(row, dim=-1)

    batch = torch.randn(16, 2048, dtype=dtype, device=device)
    batch[7] = row[0]
    out_batch = softmax_batch_invariant(batch, dim=-1)

    assert torch.equal(out_single[0], out_batch[7])


# ---------------------------------------------------------------------------
# Mean reduction tests
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((16, 128), 1),
        ((4, 64, 32), 1),
        ((8, 256), 0),
        ((2, 16, 64), 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_dim_correctness(shape, dim, dtype):
    """Triton mean_dim matches torch.mean."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(shape, dtype=dtype, device=device)

    expected = torch.mean(x.float(), dim=dim).to(dtype)
    actual = mean_dim(x, dim=dim)
    if actual.dtype != dtype:
        actual = actual.to(dtype)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.float32 else (1e-1, 1e-1)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_batch_invariance(dtype):
    """Same slice gives identical result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 512, dtype=dtype, device=device)
    out_single = mean_dim(row, dim=1)

    batch = torch.randn(8, 512, dtype=dtype, device=device)
    batch[3] = row[0]
    out_batch = mean_dim(batch, dim=1)

    assert torch.equal(out_single[0], out_batch[3])


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_multi_dim_batch_invariance(dtype):
    """mean_batch_invariant over multiple dims is batch-invariant."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 64, 32, dtype=dtype, device=device)
    out_single = mean_batch_invariant(row, dim=[1, 2])

    batch = torch.randn(8, 64, 32, dtype=dtype, device=device)
    batch[5] = row[0]
    out_batch = mean_batch_invariant(batch, dim=[1, 2])

    assert torch.equal(out_single[0], out_batch[5])


# ---------------------------------------------------------------------------
# RMS norm tests (also works on XPU per our testing)
# ---------------------------------------------------------------------------


@skip_unsupported_xpu
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [512, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_correctness_xpu(batch_size, hidden_size, dtype):
    """Triton rms_norm produces results close to a reference implementation."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Reference: manual RMS norm in float32
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = ((x_f32 / rms) * weight.float()).to(dtype)

    actual = rms_norm(x, weight, eps=eps)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_batch_invariance_xpu(dtype):
    """Same row gives identical rms_norm result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)
    hidden_size = 2048
    eps = 1e-6

    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    row = torch.randn(1, hidden_size, dtype=dtype, device=device)

    out_single = rms_norm(row, weight, eps=eps)

    batch = torch.randn(8, hidden_size, dtype=dtype, device=device)
    batch[4] = row[0]
    out_batch = rms_norm(batch, weight, eps=eps)

    assert torch.equal(out_single[0], out_batch[4])


# ---------------------------------------------------------------------------
# Dispatch override tests: verify that registering XPU overrides causes
# standard torch ops to route through batch-invariant implementations.
# ---------------------------------------------------------------------------


@pytest.fixture
def xpu_batch_invariant_lib():
    """Register XPU dispatch overrides and clean up after the test."""
    from vllm.model_executor.layers.batch_invariant import (
        _log_softmax_batch_invariant,
    )

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("aten::_log_softmax", _log_softmax_batch_invariant, "XPU", allow_override=True)
    lib.impl("aten::softmax", softmax_batch_invariant, "XPU", allow_override=True)
    lib.impl("aten::_softmax", softmax_batch_invariant, "XPU", allow_override=True)
    lib.impl("aten::mean.dim", mean_batch_invariant, "XPU", allow_override=True)
    lib.impl("aten::bmm", bmm_batch_invariant, "XPU", allow_override=True)
    yield lib
    del lib


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_bmm(xpu_batch_invariant_lib, dtype):
    """torch.bmm routes to bmm_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(4, 64, 32, dtype=dtype, device=device)
    b = torch.randn(4, 32, 128, dtype=dtype, device=device)

    via_torch = torch.bmm(a, b)
    via_direct = bmm_batch_invariant(a, b)

    assert torch.equal(via_torch, via_direct)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_log_softmax(xpu_batch_invariant_lib, dtype):
    """torch.log_softmax routes to batch-invariant impl after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(16, 1024, dtype=dtype, device=device)

    via_torch = torch.log_softmax(x, dim=-1)
    via_direct = log_softmax(x, dim=-1)

    assert torch.equal(via_torch, via_direct)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_softmax(xpu_batch_invariant_lib, dtype):
    """torch.softmax routes to softmax_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(16, 1024, dtype=dtype, device=device)

    via_torch = torch.softmax(x, dim=-1)
    via_direct = softmax_batch_invariant(x, dim=-1)

    assert torch.equal(via_torch, via_direct)


@skip_unsupported_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_override_mean(xpu_batch_invariant_lib, dtype):
    """torch.mean routes to mean_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(8, 512, dtype=dtype, device=device)

    via_torch = torch.mean(x, dim=1)
    via_direct = mean_batch_invariant(x, dim=[1])

    assert torch.equal(via_torch, via_direct)
