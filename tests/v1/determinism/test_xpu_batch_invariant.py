# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test XPU dispatch overrides for batch-invariant kernels.

Verifies that registering the XPU overrides correctly routes standard torch ops
through the batch-invariant implementations.

Note: Kernel correctness and batch-invariance property tests are in shared
test files (test_matmul_batch_invariant.py, test_rms_norm_batch_invariant.py,
test_common_ops_batch_invariant.py, test_batch_invariance.py) that run on both
CUDA and XPU platforms.
"""

import pytest
import torch
from utils import skip_if_not_xpu

from vllm.model_executor.layers.batch_invariant import (
    _register_common_overrides,
    _register_matmul_overrides,
    bmm_batch_invariant,
    log_softmax,
    mean_batch_invariant,
    mm_batch_invariant,
    softmax_batch_invariant,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


# ---------------------------------------------------------------------------
# Dispatch override tests: verify that registering XPU overrides causes
# standard torch ops to route through batch-invariant implementations.
# ---------------------------------------------------------------------------


@pytest.fixture
def xpu_batch_invariant_lib():
    """Register XPU dispatch overrides and clean up after the test."""
    lib = torch.library.Library("aten", "IMPL")
    _register_matmul_overrides(lib, "XPU")
    _register_common_overrides(lib, "XPU")
    yield lib
    del lib


@skip_if_not_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_mm(xpu_batch_invariant_lib, dtype):
    """torch.mm routes to mm_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(64, 1024, dtype=dtype, device=device)
    b = torch.randn(1024, 512, dtype=dtype, device=device)

    via_torch = torch.mm(a, b)
    via_direct = mm_batch_invariant(a, b)

    assert torch.equal(via_torch, via_direct)


@skip_if_not_xpu
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


@skip_if_not_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_log_softmax(xpu_batch_invariant_lib, dtype):
    """torch.log_softmax routes to batch-invariant impl after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(16, 1024, dtype=dtype, device=device)

    via_torch = torch.log_softmax(x, dim=-1)
    via_direct = log_softmax(x, dim=-1)

    assert torch.equal(via_torch, via_direct)


@skip_if_not_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_override_softmax(xpu_batch_invariant_lib, dtype):
    """torch.softmax routes to softmax_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(16, 1024, dtype=dtype, device=device)

    via_torch = torch.softmax(x, dim=-1)
    via_direct = softmax_batch_invariant(x, dim=-1)

    assert torch.equal(via_torch, via_direct)


@skip_if_not_xpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_override_mean(xpu_batch_invariant_lib, dtype):
    """torch.mean routes to mean_batch_invariant after XPU override."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(8, 512, dtype=dtype, device=device)

    via_torch = torch.mean(x, dim=1)
    via_direct = mean_batch_invariant(x, dim=[1])

    assert torch.equal(via_torch, via_direct)
