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

from vllm.model_executor.layers.batch_invariant import (
    _matmul_config,
    matmul_batch_invariant,
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matmul_config_block_k_fixed_across_shapes(dtype):
    """BLOCK_SIZE_K must not vary with M/N for a given dtype.

    BLOCK_SIZE_K is the only tile parameter that changes the per-output-element
    K-reduction order, so a shape-adaptive config that let it vary would break
    batch invariance. Every other tile parameter is free to change.
    """
    block_ks = {
        _matmul_config(M, N, dtype)["BLOCK_SIZE_K"]
        for M in (1, 8, 16, 64, 128, 512, 2048)
        for N in (64, 3584, 18944, 152064)
    }
    assert len(block_ks) == 1, f"BLOCK_SIZE_K varied across shapes: {block_ks}"


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# n=3584 exercises the narrow-N (block_n=64) small-M branch; n=8192 exercises the
# wide-N (block_n=256) branch, which selects a different tile from prefill.
@pytest.mark.parametrize("n", [3584, 8192])
def test_matmul_batch_invariance_across_config_buckets(dtype, n):
    """A row's output is bitwise identical no matter what batch size it lands in.

    In serving, M (number of batched tokens) changes every step, so the adaptive
    config picks a different tile shape for decode (small M) than for prefill
    (large M). The same row must still produce identical output across those
    config buckets. This crosses every bucket boundary of ``_matmul_config``,
    comparing against the row processed alone at M=1.
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    k = 3584
    b = torch.rand((k, n), dtype=dtype, device=device)
    ref_row = torch.rand((1, k), dtype=dtype, device=device)
    reference = matmul_batch_invariant(ref_row, b)[0]

    for m in (1, 8, 9, 16, 17, 32, 64, 65, 128, 129, 256, 512):
        a = torch.rand((m, k), dtype=dtype, device=device)
        pos = m // 2
        a[pos] = ref_row[0]
        out = matmul_batch_invariant(a, b)[pos]
        assert torch.equal(out, reference), (
            f"batch invariance broke at M={m}, N={n} for dtype={dtype}"
        )
