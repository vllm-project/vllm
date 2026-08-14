# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant matmul against torch.matmul for various shape combinations.

Tests correctness (matches torch.matmul) and batch invariance (result for one
item doesn't change based on other items in the batch).
"""

import pytest
import torch

from vllm.model_executor.layers.batch_invariant import (
    addmm_batch_invariant,
    addmm_out_batch_invariant,
    linear_batch_invariant,
    matmul_batch_invariant,
    matmul_persistent,
)
from vllm.platforms import current_platform

from .utils import skip_unsupported

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
@pytest.mark.parametrize("beta,alpha", [(1, 1), (2.0, 3.0), (0, 1), (1, 0.5)])
@pytest.mark.parametrize("bias_shape", [(16,), (32, 16), (1, 16), (32, 1), ()])
@pytest.mark.parametrize("use_out", [False, True])
def test_addmm_honors_beta_alpha_and_bias_shape(beta, alpha, bias_shape, use_out):
    """
    ``addmm`` takes scalar ``beta``/``alpha`` and any ``self`` broadcastable to
    the product. The matmul kernel can only fold in a row vector at ``alpha=1``,
    so everything else has to fall back to a broadcast add.
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    a = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    b = torch.randn(64, 16, dtype=torch.bfloat16, device=device)
    bias = torch.randn(bias_shape, dtype=torch.bfloat16, device=device)
    ref = beta * bias.float() + alpha * (a.float() @ b.float())

    if use_out:
        out = torch.empty(32, 16, dtype=torch.bfloat16, device=device)
        result = addmm_out_batch_invariant(bias, a, b, beta=beta, alpha=alpha, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = addmm_batch_invariant(bias, a, b, beta=beta, alpha=alpha)

    assert result.shape == (32, 16)
    torch.testing.assert_close(result.float(), ref, rtol=1e-1, atol=1e-1)


@skip_unsupported
def test_linear_folds_bias_into_the_accumulator():
    """Same property as the addmm case below, for the ``aten::linear`` path."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    x = torch.randn(64, 512, dtype=torch.bfloat16, device=device)
    w = torch.randn(128, 512, dtype=torch.bfloat16, device=device)
    bias = torch.randn(128, dtype=torch.bfloat16, device=device)

    result = linear_batch_invariant(x, w, bias)
    fused = matmul_persistent(x, w.t(), bias=bias)
    unfused = matmul_persistent(x, w.t()) + bias
    ref = x.float() @ w.float().t() + bias.float()

    assert torch.equal(result, fused), "bias was not folded into the accumulator"
    assert not torch.equal(fused, unfused), "the two paths agree; test has no power"
    fused_err = (fused.float() - ref).abs().max()
    unfused_err = (unfused.float() - ref).abs().max()
    assert fused_err < unfused_err, f"{fused_err} vs {unfused_err}"


@skip_unsupported
@pytest.mark.parametrize("beta", [1, 2.0])
def test_addmm_folds_bias_into_the_accumulator(beta):
    """
    Adding bias after the product costs a second rounding to bf16, so the fused
    and unfused results differ and the fused one is nearer the fp32 answer.
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    a = torch.randn(64, 512, dtype=torch.bfloat16, device=device)
    b = torch.randn(512, 128, dtype=torch.bfloat16, device=device)
    bias = torch.randn(128, dtype=torch.bfloat16, device=device)

    result = addmm_batch_invariant(bias, a, b, beta=beta)
    scaled = bias if beta == 1 else bias.float() * beta
    fused = matmul_persistent(a, b, bias=scaled)
    unfused = matmul_persistent(a, b) + beta * bias

    assert torch.equal(result, fused), "bias was not folded into the accumulator"
    assert not torch.equal(fused, unfused), "the two paths agree; test has no power"

    ref = a.float() @ b.float() + beta * bias.float()
    fused_err = (fused.float() - ref).abs().max()
    unfused_err = (unfused.float() - ref).abs().max()
    assert fused_err < unfused_err, f"{fused_err} vs {unfused_err}"


@skip_unsupported
@pytest.mark.parametrize("beta", [1, 2.0])
def test_addmm_fused_bias_is_batch_invariant(beta):
    """
    ``beta != 1`` pre-scales the bias into fp32 before handing it to the kernel;
    that rescaling must not depend on how many rows are in the batch.
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    b = torch.randn(64, 16, dtype=torch.bfloat16, device=device)
    bias = torch.randn(16, dtype=torch.bfloat16, device=device)
    a = torch.randn(129, 64, dtype=torch.bfloat16, device=device)

    full = addmm_batch_invariant(bias, a, b, beta=beta)
    single = addmm_batch_invariant(bias, a[17:18], b, beta=beta)

    assert torch.equal(full[17:18], single)
