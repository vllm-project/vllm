# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test batch-invariant RMS normalization against a PyTorch reference."""

import pytest
import torch
from utils import skip_if_not_cuda, skip_unsupported

from vllm.model_executor.layers.batch_invariant import (
    rms_norm_batch_invariant,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


def _rms_norm_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute RMSNorm independently using PyTorch operations."""
    input_fp32 = input_tensor.float()
    output = input_fp32 * torch.rsqrt(
        input_fp32.square().mean(dim=-1, keepdim=True) + eps
    )
    return (output * weight.float()).to(input_tensor.dtype)


@skip_if_not_cuda
@pytest.mark.parametrize("batch_size", [1, 4, 64, 300])
@pytest.mark.parametrize("hidden_size", [512, 2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("seed", list(range(4)))
def test_rms_norm_batch_invariant_vs_reference(
    default_vllm_config,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
    seed: int,
):
    """
    Compare batch-invariant Triton RMS norm against a PyTorch reference.

    Tests that the Triton-based batch-invariant RMS norm produces numerically
    equivalent results to an independent implementation across configurations.
    """
    device = torch.device(DEVICE_TYPE)

    # Create test input and weight
    torch.manual_seed(seed)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    reference_output = _rms_norm_reference(input_tensor, weight, eps)

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
        reference_output,
        rtol=rtol,
        atol=atol,
        msg=f"RMS norm mismatch for batch_size={batch_size}, "
        f"hidden_size={hidden_size}, "
        f"dtype={dtype}, eps={eps}, seed={seed}",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("n_extra", [3, 299])
@pytest.mark.parametrize("seed", list(range(16)))
def test_fused_add_rms_norm_batch_invariant_residual_path(
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
    n_extra: int,
    seed: int,
):
    """
    Test the batch-invariant fused residual-add + RMSNorm helper directly.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(seed)
    x_single = torch.randn(1, hidden_size, dtype=dtype, device=device)
    residual_single = torch.randn(1, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    x_batch = torch.cat(
        [
            x_single,
            torch.randn(n_extra, hidden_size, dtype=dtype, device=device),
        ],
        dim=0,
    )
    residual_batch = torch.cat(
        [
            residual_single,
            torch.randn(n_extra, hidden_size, dtype=dtype, device=device),
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
    ref_out = _rms_norm_reference(merged_single, weight, eps)

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
        "PyTorch RMSNorm reference",
    )


FP8_DTYPE = current_platform.fp8_dtype()

# The large launch (num_tokens=300 >= 256) drops an un-pinned kernel to block
# 256, while the small launch (255 rows) stays under the threshold and keeps the
# larger block (1024, or 512 for per-block quant). Under the pin the two launches
# use the same block, so the shared first 255 rows must match bit-for-bit; 255 is
# the most rows a single small launch can hold (< 256, and <= 256 for per-block).
_LARGE_TOKENS = 300
_SMALL_TOKENS = 255


def _assert_rows_bit_identical(small, large, msg):
    if small.dtype == FP8_DTYPE:
        assert torch.equal(small.view(torch.uint8), large.view(torch.uint8)), msg
    else:
        torch.testing.assert_close(small, large, rtol=0.0, atol=0.0, msg=msg)


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", list(range(4)))
def test_rms_norm_batch_invariant_nonresidual_kernel(
    hidden_size: int, dtype: torch.dtype, seed: int
):
    """C++ ``rms_norm`` (no residual) must be batch invariant across the block
    threshold. Reached in compiled mode with ``ir_op_priority.rms_norm=["vllm_c"]``
    (default priority is ``native``/inductor codegen when compiling).
    """
    import vllm._custom_ops as ops

    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(seed)
    rows = torch.randn(_LARGE_TOKENS, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    def rms_norm(x):
        out = torch.empty_like(x)
        ops.rms_norm(out, x, weight, 1e-6)
        return out

    large = rms_norm(rows.clone())
    small = rms_norm(rows[:_SMALL_TOKENS].clone())
    _assert_rows_bit_identical(
        small,
        large[:_SMALL_TOKENS],
        "rms_norm output depends on num_tokens (block size)",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", list(range(4)))
@pytest.mark.parametrize("add_residual", [False, True])
def test_rms_norm_static_fp8_quant_batch_invariant(
    hidden_size: int, dtype: torch.dtype, seed: int, add_residual: bool
):
    """C++ static per-tensor fp8-quant RMSNorm must be batch invariant across
    the block threshold. Covers ``rms_norm_static_fp8_quant`` and, with
    ``add_residual``, ``fused_add_rms_norm_static_fp8_quant`` (the compiled fp8
    path where ``RMSNormQuantFusionPass`` rewrites norm + quant into them).
    """
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(seed)
    rows = torch.randn(_LARGE_TOKENS, hidden_size, dtype=dtype, device=device)
    residual = (
        torch.randn(_LARGE_TOKENS, hidden_size, dtype=dtype, device=device)
        if add_residual
        else None
    )
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    quant_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    def quant(x, res):
        out = torch.empty_like(x, dtype=FP8_DTYPE)
        if add_residual:
            torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                out, x, res, weight, quant_scale, 1e-6
            )
        else:
            torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, quant_scale, 1e-6)
        return out

    large = quant(rows.clone(), residual.clone() if residual is not None else None)
    small = quant(
        rows[:_SMALL_TOKENS].clone(),
        residual[:_SMALL_TOKENS].clone() if residual is not None else None,
    )
    _assert_rows_bit_identical(
        small,
        large[:_SMALL_TOKENS],
        "static-fp8-quant RMSNorm output depends on num_tokens (block size)",
    )


@skip_if_not_cuda
@pytest.mark.parametrize("hidden_size", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", list(range(4)))
def test_rms_norm_per_block_quant_batch_invariant(
    hidden_size: int, dtype: torch.dtype, seed: int
):
    """C++ ``rms_norm_per_block_quant`` must be batch invariant across the
    block threshold (compiled fp8 block-quant path; block pinned to 512)."""
    import vllm._custom_ops as ops

    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(seed)
    rows = torch.randn(_LARGE_TOKENS, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    group_size = [1, 128]

    def per_block_quant(x):
        return ops.rms_norm_per_block_quant(x, weight, 1e-6, FP8_DTYPE, group_size)

    out_large, scale_large = per_block_quant(rows.clone())
    out_small, scale_small = per_block_quant(rows[:_SMALL_TOKENS].clone())
    _assert_rows_bit_identical(
        out_small,
        out_large[:_SMALL_TOKENS],
        "rms_norm_per_block_quant output depends on num_tokens (block size)",
    )
    torch.testing.assert_close(
        scale_small,
        scale_large[:_SMALL_TOKENS],
        rtol=0.0,
        atol=0.0,
        msg="rms_norm_per_block_quant scales depend on num_tokens (block size)",
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

    reference_output = _rms_norm_reference(input_tensor, weight, eps)

    # Batch-invariant implementation
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

    # Use looser tolerance for bfloat16
    rtol, atol = 1e-1, 1e-1  # 10% tolerance for bfloat16

    torch.testing.assert_close(
        triton_output,
        reference_output,
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
        reference_output = _rms_norm_reference(input_tensor, weight, eps)

        # Batch-invariant implementation
        triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

        # Check for NaN or Inf
        assert not torch.isnan(reference_output).any(), (
            f"Reference RMS norm produced NaN for test case {idx}"
        )
        assert not torch.isinf(reference_output).any(), (
            f"Reference RMS norm produced Inf for test case {idx}"
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
            reference_output,
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

    reference_output = _rms_norm_reference(input_tensor, weight, eps)

    # Batch-invariant implementation
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

    # Use looser tolerance for bfloat16
    rtol, atol = 1e-1, 1e-1  # 10% tolerance for bfloat16

    torch.testing.assert_close(
        triton_output,
        reference_output,
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

    reference_output = _rms_norm_reference(input_tensor, weight, eps)

    # Batch-invariant implementation
    triton_output = rms_norm_batch_invariant(input_tensor, weight, eps=eps)

    # Compare
    max_diff = (triton_output - reference_output).abs().max().item()
    mean_diff = (triton_output - reference_output).abs().mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Reference output sample: {reference_output[0, :5].tolist()}")
    print(f"Triton output sample: {triton_output[0, :5].tolist()}")

    if max_diff < 1e-3:
        print("✓ Smoke test passed!")
    else:
        print("✗ Smoke test failed - differences too large")
