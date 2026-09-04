# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant matmul against torch.matmul for various shape combinations.

Tests correctness (matches torch.matmul) and batch invariance (result for one
item doesn't change based on other items in the batch).
"""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.determinism.batch_invariant import matmul_batch_invariant
from vllm.model_executor.determinism.batch_invariant_configs import (
    _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS,
    _get_tuned_matmul_arch_family,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_unsupported
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        # 2D x 2D
        ((32, 64), (64, 16)),
        # 2D x 3D
        ((64, 16), (4, 16, 32)),
        # 3D x 2D
        ((4, 32, 64), (64, 16)),
        # 4D x 2D
        ((1, 4, 32, 64), (64, 16)),
        # 3D x 3D
        ((4, 32, 64), (4, 64, 16)),
        # 3D x 4D
        ((2, 32, 64), (1, 2, 64, 16)),
        # 4D x 3D (Gemma4 pattern)
        ((1, 2, 32, 64), (2, 64, 16)),
        # 4D x 4D
        ((1, 2, 32, 64), (4, 2, 64, 16)),
        # 2D x 4D
        ((32, 64), (1, 2, 64, 16)),
        # 2D x 5D
        ((32, 64), (1, 2, 2, 64, 16)),
        # 5D x 2D
        ((1, 2, 2, 32, 64), (64, 16)),
        # 5D x 5D
        ((1, 2, 4, 32, 64), (1, 2, 4, 64, 16)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_correctness(a_shape, b_shape, dtype):
    """
    Compare matmul_batch_invariant against torch.matmul for various shapes.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a = torch.rand(a_shape, dtype=dtype, device=device)
    b = torch.rand(b_shape, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    standard_output = torch.matmul(a, b)

    # Batch-invariant implementation (Triton)
    triton_output = matmul_batch_invariant(a, b)

    # Compare outputs
    # Use looser tolerance for bfloat16 due to its lower precision
    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1  # 10% relative tolerance for bfloat16
    else:
        rtol, atol = 1e-2, 1e-2  # 1% for float16/float32

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"matmul mismatch for a ndim={a.ndim}, b ndim={b.ndim},",
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_batch_invariance(dtype):
    """
    Verify that the result for one item is bitwise identical regardless
    of what other items are in the batch.
    """

    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a_single = torch.rand((1, 64, 32), dtype=dtype, device=device)
    b = torch.rand((32, 128), dtype=dtype, device=device)

    standard_output = matmul_batch_invariant(a_single, b)

    a_batch = torch.rand((8, 64, 32), dtype=dtype, device=device)
    a_batch[3] = a_single[0]

    batch_output = matmul_batch_invariant(a_batch, b)
    batch_output_a = batch_output[3]

    assert torch.equal(standard_output[0], batch_output_a)


@skip_unsupported
@pytest.mark.parametrize("m", [8, 32, 256, 2048])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["contiguous", "transposed"])
def test_matmul_batch_invariance_across_tuned_m_buckets(m, transpose_b):
    # Tuned M buckets must preserve each row's K-reduction order.
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    arch_family = _get_tuned_matmul_arch_family(capability)
    if arch_family not in _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS:
        pytest.skip("No tuned persistent matmul config for this architecture")

    device = torch.device(DEVICE_TYPE)
    n = k = 2048
    torch.manual_seed(42)
    a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
    if transpose_b:
        b = torch.rand((n, k), dtype=torch.bfloat16, device=device).t()
    else:
        b = torch.rand((k, n), dtype=torch.bfloat16, device=device)

    single_output = matmul_batch_invariant(a[:1], b)
    batch_output = matmul_batch_invariant(a, b)

    assert torch.equal(single_output[0], batch_output[0])
