# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant RMS normalization against standard implementations.

This test compares the Triton-based batch-invariant RMS norm implementation
with the standard CUDA-based implementation to ensure numerical accuracy.
"""

import pytest
import torch
from utils import skip_if_not_cuda, skip_unsupported

from vllm.model_executor.layers.batch_invariant import (
    rms_norm_batch_invariant,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_if_not_cuda
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("hidden_size", [512, 2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rms_norm_batch_invariant_vs_standard(
    default_vllm_config,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
):
    """
    Compare batch-invariant Triton RMS norm against standard CUDA implementation.

    Tests that the Triton-based batch-invariant RMS norm produces numerically
    equivalent results to the standard CUDA implementation across various
    configurations.
    """
    device = torch.device(DEVICE_TYPE)

    # Create test input and weight
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()

    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation (Triton)
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

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
        msg=f"RMS norm mismatch for batch_size={batch_size}, "
        f"hidden_size={hidden_size}, "
        f"dtype={dtype}, eps={eps}",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6])
def test_fused_add_rms_norm_batch_invariant_residual_path(
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
):
    """
    Test the batch-invariant fused residual-add + RMSNorm helper directly.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    x_single = torch.randn(1, hidden_size, dtype=dtype, device=device)
    residual_single = torch.randn(1, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    x_batch = torch.cat(
        [
            x_single,
            torch.randn(3, hidden_size, dtype=dtype, device=device),
        ],
        dim=0,
    )
    residual_batch = torch.cat(
        [
            residual_single,
            torch.randn(3, hidden_size, dtype=dtype, device=device),
        ],
        dim=0,
    )

    def fused_add_rms_norm(x, residual, w, e) -> tuple[torch.Tensor, torch.Tensor]:
        import vllm._custom_ops as ops

        ops.fused_add_rms_norm(x, residual, w, e)
        return x, residual

    out_single, residual_out_single = fused_add_rms_norm(
        x_single.clone(),
        residual_single.clone(),
        weight,
        eps,
    )
    out_batch, residual_out_batch = fused_add_rms_norm(
        x_batch.clone(),
        residual_batch.clone(),
        weight,
        eps,
    )

    merged_single = x_single + residual_single
    ref_out = rms_norm_batch_invariant(merged_single, weight, eps=eps)

    torch.testing.assert_close(
        residual_out_single,
        merged_single,
        rtol=0.0,
        atol=0.0,
        msg="Residual output should equal x + residual exactly",
    )
    torch.testing.assert_close(
        residual_out_batch[:1],
        merged_single,
        rtol=0.0,
        atol=0.0,
        msg="Residual output should be batch invariant",
    )
    torch.testing.assert_close(
        out_single,
        out_batch[:1],
        rtol=0.0,
        atol=0.0,
        msg="Fused add RMSNorm output should be batch invariant",
    )

    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
    else:
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(
        out_single,
        ref_out,
        rtol=rtol,
        atol=atol,
        msg="Fused add RMSNorm output should stay numerically close to the "
        "batch-invariant RMSNorm reference",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_invariant_across_block_size_boundary(
    hidden_size: int,
    dtype: torch.dtype,
):
    """Regression test for the num_tokens-dependent RMSNorm block size.

    The CUDA launch picks ``max_block_size = (num_tokens < 256) ? 1024 : 256``,
    which sets the per-row reduction width and therefore the float accumulation
    order. Under ``VLLM_BATCH_INVARIANT=1`` (enabled here by the autouse
    ``enable_batch_invariant_mode`` fixture in conftest.py) a row must normalize
    identically no matter how many tokens share the launch, so the same rows fed
    through a small launch (num_tokens < 256, block 1024) and a large launch
    (num_tokens >= 256, block 256) must produce bitwise-identical output.

    Existing coverage only used batch sizes < 256, so both launches took the same
    block size and the divergence was never exercised. This test crosses the 256
    boundary. It fails without pinning the block size under batch invariance and
    passes with it.
    """
    import vllm._custom_ops as ops

    device = torch.device(DEVICE_TYPE)
    eps = 1e-6

    torch.manual_seed(0)
    n_large = 300  # >= 256 -> block 256
    rows = torch.randn(n_large, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(n_large, hidden_size, dtype=dtype, device=device) * 0.1
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Large launch: all rows at once (num_tokens >= 256).
    x_large = rows.clone()
    res_large = residual.clone()
    ops.fused_add_rms_norm(x_large, res_large, weight, eps)

    # Small launches: same rows in chunks of 8 (num_tokens < 256 each).
    x_small = torch.empty_like(rows)
    for i in range(0, n_large, 8):
        xi = rows[i : i + 8].clone()
        ri = residual[i : i + 8].clone()
        ops.fused_add_rms_norm(xi, ri, weight, eps)
        x_small[i : i + 8] = xi

    torch.testing.assert_close(
        x_small,
        x_large,
        rtol=0.0,
        atol=0.0,
        msg="fused_add_rms_norm output must not depend on num_tokens "
        "(block size) under VLLM_BATCH_INVARIANT=1",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("add_residual", [False, True])
def test_static_fp8_quant_invariant_across_block_size_boundary(
    hidden_size: int,
    dtype: torch.dtype,
    add_residual: bool,
):
    """Regression test for the num_tokens-dependent block size in the fused
    RMSNorm + static-fp8-quant kernels.

    ``rms_norm_static_fp8_quant`` and ``fused_add_rms_norm_static_fp8_quant``
    (``csrc/libtorch_stable/layernorm_quant_kernels.cu``) pick the same
    ``max_block_size = (num_tokens < 256) ? 1024 : 256`` heuristic as plain
    RMSNorm, so the fp32 reduction width — and thus the float accumulation
    order feeding the fp8 quantization — depends on how many tokens share the
    launch. Under ``VLLM_BATCH_INVARIANT=1`` (enabled by the autouse fixture in
    conftest.py) the same rows must produce bitwise-identical fp8 output whether
    launched in small chunks (block 1024) or one launch that crosses the 256
    boundary (block 256).

    These fused kernels are the RMSNorm implementation reached under batch
    invariance for fp8-quantized models on the default compiled path, where the
    ``RMSNormQuantFusionPass`` rewrites norm + quant into them. It fails without
    pinning the block size and passes with it.
    """
    import vllm._custom_ops as ops  # noqa: F401

    device = torch.device(DEVICE_TYPE)
    fp8_dtype = current_platform.fp8_dtype()
    eps = 1e-6

    torch.manual_seed(0)
    n_large = 300  # >= 256 -> block 256
    scale = 1.0 / (2 * hidden_size)
    rows = torch.randn(n_large, hidden_size, dtype=dtype, device=device) * scale
    residual = (
        torch.randn(n_large, hidden_size, dtype=dtype, device=device) * scale
        if add_residual
        else None
    )
    weight = torch.empty(hidden_size, dtype=dtype, device=device).normal_(
        mean=1.0, std=0.1
    )
    quant_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    def run(x, res):
        out = torch.empty_like(x, dtype=fp8_dtype)
        if add_residual:
            torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                out, x, res, weight, quant_scale, eps
            )
        else:
            torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, quant_scale, eps)
        return out

    # Large launch: all rows at once (num_tokens >= 256, block 256).
    out_large = run(rows.clone(), residual.clone() if add_residual else None)

    # Small launches: same rows in chunks of 8 (num_tokens < 256, block 1024).
    out_small = torch.empty_like(rows, dtype=fp8_dtype)
    for i in range(0, n_large, 8):
        xi = rows[i : i + 8].clone()
        ri = residual[i : i + 8].clone() if add_residual else None
        out_small[i : i + 8] = run(xi, ri)

    torch.testing.assert_close(
        out_small.to(torch.float32),
        out_large.to(torch.float32),
        rtol=0.0,
        atol=0.0,
        msg="static-fp8-quant RMSNorm output must not depend on num_tokens "
        "(block size) under VLLM_BATCH_INVARIANT=1",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_per_block_quant_invariant_across_block_size_boundary(
    hidden_size: int,
    dtype: torch.dtype,
):
    """Regression test for the num_tokens-dependent block size in the fused
    RMSNorm + per-block-quant kernel.

    ``rms_norm_per_block_quant``
    (``fused_layernorm_dynamic_per_token_quant.cu``) picks
    ``max_block_size = (num_tokens <= 256) ? 512 : 256``, which sets the fp32
    reduction width in ``compute_rms``. Under ``VLLM_BATCH_INVARIANT=1`` the same
    rows must produce bitwise-identical quantized output and scales whether fed
    through small launches (block 512) or one launch crossing 256 (block 256).
    Pinned to the ``num_tokens <= 256`` value (512) under batch invariance.
    """
    import vllm._custom_ops as ops

    device = torch.device(DEVICE_TYPE)
    quant_dtype = current_platform.fp8_dtype()
    eps = 1e-6
    group_size = 128
    assert hidden_size % group_size == 0

    torch.manual_seed(0)
    n_large = 300  # > 256 -> block 256
    rows = torch.randn(n_large, hidden_size, dtype=dtype, device=device)
    weight = torch.empty(hidden_size, dtype=dtype, device=device).normal_(
        mean=1.0, std=0.1
    )

    def run(x):
        return ops.rms_norm_per_block_quant(
            x, weight, eps, quant_dtype, [1, group_size]
        )

    out_large, scale_large = run(rows.clone())

    out_small = torch.empty_like(out_large)
    scale_small = torch.empty_like(scale_large)
    for i in range(0, n_large, 8):
        out_i, scale_i = run(rows[i : i + 8].clone())
        out_small[i : i + 8] = out_i
        scale_small[i : i + 8] = scale_i

    torch.testing.assert_close(
        out_small.to(torch.float32),
        out_large.to(torch.float32),
        rtol=0.0,
        atol=0.0,
        msg="per-block-quant RMSNorm output must not depend on num_tokens "
        "(block size) under VLLM_BATCH_INVARIANT=1",
    )
    torch.testing.assert_close(
        scale_small,
        scale_large,
        rtol=0.0,
        atol=0.0,
        msg="per-block-quant scales must not depend on num_tokens "
        "(block size) under VLLM_BATCH_INVARIANT=1",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("batch_size", [1, 16, 128])
@pytest.mark.parametrize("seq_len", [1, 32, 512])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
def test_rms_norm_3d_input(
    default_vllm_config, batch_size: int, seq_len: int, hidden_size: int
):
    """
    Test RMS norm with 3D input tensors (batch, seq_len, hidden_size).

    Ensures that the batch-invariant RMS norm correctly handles multi-dimensional
    inputs that are common in transformer models.
    """
    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16
    eps = 1e-6

    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=dtype, device=device
    )
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()
    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

    # Use looser tolerance for bfloat16
    rtol, atol = 1e-1, 1e-1  # 10% tolerance for bfloat16

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"RMS norm mismatch for 3D input with batch_size={batch_size}, "
        f"seq_len={seq_len}, hidden_size={hidden_size}",
    )


@skip_if_not_cuda
def test_rms_norm_numerical_stability(default_vllm_config):
    """
    Test RMS norm numerical stability with extreme values.

    Ensures that both implementations handle edge cases like very small or large
    values without producing NaN or Inf.
    """
    device = torch.device(DEVICE_TYPE)
    dtype = torch.float16
    eps = 1e-6
    hidden_size = 2048

    # Test cases with extreme values
    test_cases = [
        # Very small values
        torch.ones(4, hidden_size, dtype=dtype, device=device) * 1e-5,
        # Very large values
        torch.ones(4, hidden_size, dtype=dtype, device=device) * 1e4,
        # Mixed small and large
        torch.randn(4, hidden_size, dtype=dtype, device=device) * 100,
        # Values near zero
        torch.randn(4, hidden_size, dtype=dtype, device=device) * 1e-6,
    ]

    weight = torch.ones(hidden_size, dtype=dtype, device=device)

    for idx, input_tensor in enumerate(test_cases):
        # Standard implementation
        rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
        rms_norm_layer.weight.data = weight.clone()
        standard_output = rms_norm_layer.forward_cuda(input_tensor)

        # Batch-invariant implementation
        triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

        # Check for NaN or Inf
        assert not torch.isnan(standard_output).any(), (
            f"Standard RMS norm produced NaN for test case {idx}"
        )
        assert not torch.isinf(standard_output).any(), (
            f"Standard RMS norm produced Inf for test case {idx}"
        )
        assert not torch.isnan(triton_output).any(), (
            f"Triton RMS norm produced NaN for test case {idx}"
        )
        assert not torch.isinf(triton_output).any(), (
            f"Triton RMS norm produced Inf for test case {idx}"
        )

        # Compare outputs - very lenient for extreme values with float16
        torch.testing.assert_close(
            triton_output,
            standard_output,
            rtol=2e-1,  # 20% tolerance for extreme values
            atol=2e-1,
            msg=f"RMS norm mismatch for extreme value test case {idx}",
        )


@skip_unsupported
def test_rms_norm_formula(default_vllm_config):
    """
    Test that RMS norm follows the correct mathematical formula.

    Verifies: output = input / sqrt(mean(input^2) + eps) * weight
    """
    device = torch.device(DEVICE_TYPE)
    dtype = torch.float32  # Use float32 for higher precision in formula check
    eps = 1e-6
    hidden_size = 1024

    torch.manual_seed(42)
    input_tensor = torch.randn(8, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Compute expected output using the formula
    variance = (input_tensor.pow(2).mean(dim=-1, keepdim=True)).to(dtype)
    expected_output = input_tensor * torch.rsqrt(variance + eps) * weight

    # Batch-invariant implementation
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

    # Compare against formula
    torch.testing.assert_close(
        triton_output,
        expected_output,
        rtol=1e-4,
        atol=1e-4,
        msg="Triton RMS norm doesn't match expected formula",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [128, 1024, 4096, 16384])
def test_rms_norm_different_hidden_sizes(default_vllm_config, hidden_size: int):
    """
    Test RMS norm with various hidden sizes to ensure block size handling.

    The Triton kernel uses a fixed BLOCK_SIZE=1024, so this tests that it
    correctly handles hidden sizes both smaller and larger than the block size.
    """
    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16
    eps = 1e-6
    batch_size = 16

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()
    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

    # Use looser tolerance for bfloat16
    rtol, atol = 1e-1, 1e-1  # 10% tolerance for bfloat16

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"RMS norm mismatch for hidden_size={hidden_size}",
    )


@skip_unsupported
def test_rms_norm_determinism(default_vllm_config):
    """
    Test that batch-invariant RMS norm produces deterministic results.

    Runs the same input through the kernel multiple times and verifies
    identical outputs.
    """
    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16
    eps = 1e-6
    hidden_size = 4096
    batch_size = 32

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Run multiple times
    outputs = []
    for _ in range(5):
        output = rms_norm_batch_invariant(input_tensor.clone(), weight, eps=eps)
        outputs.append(output)

    # All outputs should be identical
    reference = outputs[0]
    for idx, output in enumerate(outputs[1:], start=1):
        torch.testing.assert_close(
            output,
            reference,
            rtol=0.0,
            atol=0.0,
            msg=f"RMS norm not deterministic: run {idx} differs from reference",
        )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_batch_invariance(dtype):
    """Same row gives identical rms_norm result regardless of batch neighbors.

    This verifies that the output for a given row is independent of what other
    rows are present in the batch — the core batch-invariance property.
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)
    hidden_size = 2048
    eps = 1e-6

    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    row = torch.randn(1, hidden_size, dtype=dtype, device=device)

    # Compute rms_norm on the single row alone
    out_single = rms_norm_batch_invariant(row, weight, eps=eps)

    # Embed the same row in a larger batch with random neighbors
    batch = torch.randn(8, hidden_size, dtype=dtype, device=device)
    batch[4] = row[0]
    out_batch = rms_norm_batch_invariant(batch, weight, eps=eps)

    assert torch.equal(out_single[0], out_batch[4]), (
        "rms_norm output for a row differs when batch context changes"
    )


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running quick smoke test of RMS norm implementations...")

    device = torch.device(DEVICE_TYPE)
    batch_size = 8
    hidden_size = 4096
    dtype = torch.bfloat16
    eps = 1e-6

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()
    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

    # Compare
    max_diff = (triton_output - standard_output).abs().max().item()
    mean_diff = (triton_output - standard_output).abs().mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Standard output sample: {standard_output[0, :5].tolist()}")
    print(f"Triton output sample: {triton_output[0, :5].tolist()}")

    if max_diff < 1e-3:
        print("✓ Smoke test passed!")
    else:
        print("✗ Smoke test failed - differences too large")
