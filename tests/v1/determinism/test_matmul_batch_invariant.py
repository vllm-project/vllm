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

from vllm.model_executor.layers import batch_invariant as batch_invariant_mod
from vllm.model_executor.layers.batch_invariant import (
    bmm_batch_invariant,
    matmul_batch_invariant,
    matmul_persistent,
)
from vllm.platforms import current_platform
from vllm.utils.mem_utils import get_max_shared_memory_bytes

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
def test_fp32_n_equals_1(monkeypatch: pytest.MonkeyPatch):
    """fp32 matmul with N=1 must launch on ~100KB smem and stay batch-invariant."""
    low_smem = get_max_shared_memory_bytes() <= 106496
    monkeypatch.setattr(
        batch_invariant_mod, "_fp32_block_size_n", 32 if low_smem else 128
    )
    monkeypatch.setattr(batch_invariant_mod, "_fp32_num_stages", 2 if low_smem else 3)
    device = torch.device(DEVICE_TYPE)
    a = torch.randn(4, 2048, device=device, dtype=torch.float32)
    b = torch.randn(2048, 1, device=device, dtype=torch.float32)
    torch.testing.assert_close(
        matmul_persistent(a, b), torch.mm(a, b), rtol=1e-2, atol=1e-2
    )
    a_single = torch.rand((1, 32, 2048), dtype=torch.float32, device=device)
    a_batch = torch.rand((8, 32, 2048), dtype=torch.float32, device=device)
    a_batch[3] = a_single[0]
    ref = matmul_batch_invariant(a_single, b)
    assert torch.equal(ref[0], matmul_batch_invariant(a_batch, b)[3])
    ba = torch.randn(2, 4, 2048, device=device, dtype=torch.float32)
    bb = torch.randn(2, 2048, 1, device=device, dtype=torch.float32)
    torch.testing.assert_close(
        bmm_batch_invariant(ba, bb), torch.bmm(ba, bb), rtol=1e-2, atol=1e-2
    )


@skip_unsupported
def test_fp32_n_gt_1_keeps_wide_tile(monkeypatch: pytest.MonkeyPatch):
    """N>1 must keep 128/stages=3 even when the N=1 override is armed."""
    monkeypatch.setattr(batch_invariant_mod, "_fp32_block_size_n", 32)
    monkeypatch.setattr(batch_invariant_mod, "_fp32_num_stages", 2)
    device = torch.device(DEVICE_TYPE)
    a = torch.randn(8, 2048, device=device, dtype=torch.float32)
    b = torch.randn(2048, 16, device=device, dtype=torch.float32)
    torch.testing.assert_close(
        matmul_persistent(a, b), torch.mm(a, b), rtol=1e-2, atol=1e-2
    )
