# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.quantization.triton_quantization import (
    silu_and_mul_per_block_quant_triton,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.platforms import current_platform

DTYPES = [torch.float16, torch.bfloat16]
QUANT_DTYPES = [torch.float8_e4m3fn, torch.int8]
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, *VEC_HIDDEN_SIZES, 2048, 5120]],
    *[(16, i) for i in [64, *VEC_HIDDEN_SIZES, 5120]],
    *[(128, i) for i in [64, *VEC_HIDDEN_SIZES]],
    *[(512, i) for i in [64, 5120]],
]

SCALE_UBS = [False]  # Scale upper bound not supported yet
GROUP_SIZES = [64, 128]
IS_SCALE_TRANSPOSED = [False, True]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]


def as_float32_tensor(x: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device="cuda")


def ref_silu_and_mul_per_block_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None,
    group_size: int,  # Changed from list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation: unfused SiLU+Mul then group quantization."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)

    # SiLU(gate) * up
    silu_out = F.silu(gate) * up

    # Group quantize
    if quant_dtype == current_platform.fp8_dtype():
        torch_out, scales = per_token_group_quant_fp8(
            silu_out, group_size=group_size, use_ue8m0=False
        )
    elif quant_dtype == torch.int8:
        torch_out, scales = per_token_group_quant_int8(silu_out, group_size=group_size)
    else:
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}")

    return torch_out, scales


def ref_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None,
    group_size: int,  # Changed from list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    return ref_silu_and_mul_per_block_quant(x, quant_dtype, scale_ub, group_size)


def ops_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None,
    group_size: int,  # Changed from list[int]
    is_scale_transposed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused kernel implementation."""
    out, scales = ops.silu_and_mul_per_block_quant(
        x, group_size, quant_dtype, scale_ub, is_scale_transposed
    )

    # DON'T transpose - keep scales in native layout
    # Remove these lines:
    # if is_scale_transposed:
    #     scales = scales.t().contiguous()

    return out, scales


def triton_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton kernel implementation."""
    out, scales = silu_and_mul_per_block_quant_triton(
        x, group_size, quant_dtype, scale_ub, is_scale_transposed
    )
    return out, scales


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("is_scale_transposed", IS_SCALE_TRANSPOSED)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    group_size: int,  # Changed from list[int]
    is_scale_transposed: bool,
    seed: int,
    device: str,
) -> None:
    """Test SiLU+Mul+Block Quantization kernel correctness."""
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    if hidden_size % group_size != 0:
        # Skip invalid configurations
        return

    if has_scale_ub:
        # Scale upper bound not implemented yet
        pytest.skip("Scale upper bound not yet supported")

    # Make inputs: [num_tokens, hidden_size * 2] for [gate || up]
    scale = 1 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device) * scale
    scale_ub = None

    # Reference implementation
    ref_out, ref_scales = ref_impl(x, quant_dtype, scale_ub, group_size)

    # Fused kernel implementation
    ops_out, ops_scales = ops_impl(
        x, quant_dtype, scale_ub, group_size, is_scale_transposed
    )

    # ========== ADD THESE CHECKS ==========
    # Check for NaN/Inf in outputs
    assert not torch.isnan(ops_out.float()).any(), "Kernel output contains NaN values"
    assert not torch.isinf(ops_out.float()).any(), "Kernel output contains Inf values"
    assert not torch.isnan(ops_scales).any(), "Kernel scales contain NaN values"
    assert not torch.isinf(ops_scales).any(), "Kernel scales contain Inf values"

    # Also check reference for sanity
    assert not torch.isnan(ref_out.float()).any(), (
        "Reference output contains NaN values"
    )
    assert not torch.isnan(ref_scales).any(), "Reference scales contain NaN values"
    # ======================================

    # Check output dtype
    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype

    # Check scale correctness - normalize layout before comparison
    ops_scales_for_comparison = ops_scales.t() if is_scale_transposed else ops_scales

    assert torch.allclose(
        ref_scales, ops_scales_for_comparison, rtol=1e-5, atol=1e-5
    ), (
        f"Scale mismatch: max diff = "
        f"{(ref_scales - ops_scales_for_comparison).abs().max()}"
    )

    # Check output correctness
    a = ref_out.to(dtype=torch.float32)
    b = ops_out.to(dtype=torch.float32)
    ok = torch.allclose(a, b, atol=1.0, rtol=0.0)

    if not ok:
        # Fallback: compare dequantized values
        # Both ref_scales and ops_scales are now in SAME layout

        # Normalize both to [num_tokens, num_groups]
        if is_scale_transposed:
            # ops_scales is [num_groups, num_tokens], ref is [num_tokens, num_groups]
            ops_scales_row_major = ops_scales.t()
            ref_scales_row_major = ref_scales
        else:
            # Both are [num_tokens, num_groups]
            ops_scales_row_major = ops_scales
            ref_scales_row_major = ref_scales

        # Expand to [num_tokens, hidden_size]
        ref_scales_expanded = ref_scales_row_major.repeat_interleave(group_size, dim=1)
        ops_scales_expanded = ops_scales_row_major.repeat_interleave(group_size, dim=1)

        # ========== ADD DEBUG PRINTS HERE ==========
        print("\n=== DEBUG INFO ===")
        print(
            f"num_tokens={num_tokens}, hidden_size={hidden_size}, "
            f"group_size={group_size}"
        )
        print(f"is_scale_transposed={is_scale_transposed}")
        print(f"a.shape: {a.shape}")
        print(f"b.shape: {b.shape}")
        print(f"ref_scales.shape: {ref_scales.shape}")
        print(f"ops_scales.shape: {ops_scales.shape}")
        print(f"ref_scales_row_major.shape: {ref_scales_row_major.shape}")
        print(f"ops_scales_row_major.shape: {ops_scales_row_major.shape}")
        print(f"ref_scales_expanded.shape: {ref_scales_expanded.shape}")
        print(f"ops_scales_expanded.shape: {ops_scales_expanded.shape}")
        # ===========================================

        a_deq = a * ref_scales_expanded
        b_deq = b * ops_scales_expanded
        ok = torch.allclose(a_deq, b_deq, rtol=5e-2, atol=5e-2)

        if not ok:
            max_diff = (a_deq - b_deq).abs().max()
            print(f"Max dequantized difference: {max_diff}")

    assert ok, "Output values don't match within tolerance"

    # Test opcheck for correctness verification
    output = torch.empty(num_tokens, hidden_size, device=device, dtype=quant_dtype)
    num_groups = hidden_size // group_size
    if is_scale_transposed:
        scales = torch.empty(num_groups, num_tokens, device=device, dtype=torch.float32)
    else:
        scales = torch.empty(num_tokens, num_groups, device=device, dtype=torch.float32)

    opcheck(
        torch.ops._C.silu_and_mul_per_block_quant,
        (output, x, scales, group_size, scale_ub, is_scale_transposed),
    )


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("group_size", [128])  # Changed from [[1, 128]]
def test_silu_block_quant_shapes(
    default_vllm_config,
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    group_size: int,  # Changed type
):
    """Test that output shapes are correct."""
    torch.set_default_device("cuda")

    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device="cuda")

    # Test row-major scales
    out, scales = ops.silu_and_mul_per_block_quant(
        x,
        group_size=group_size,
        quant_dtype=torch.float8_e4m3fn,
        is_scale_transposed=False,
    )

    assert out.shape == (num_tokens, hidden_size)
    assert scales.shape == (num_tokens, hidden_size // group_size)

    # Test column-major scales
    out, scales = ops.silu_and_mul_per_block_quant(
        x,
        group_size=group_size,
        quant_dtype=torch.float8_e4m3fn,
        is_scale_transposed=True,
    )

    assert out.shape == (num_tokens, hidden_size)
    assert scales.shape == (hidden_size // group_size, num_tokens)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [1024, 5120, 14336])
def test_silu_block_quant_edge_cases(
    default_vllm_config, dtype: torch.dtype, batch_size: int, hidden_size: int
):
    """Test edge cases: single token, large batch, large hidden size."""
    torch.set_default_device("cuda")

    x = torch.randn(batch_size, hidden_size * 2, dtype=dtype, device="cuda")

    out, scales = ops.silu_and_mul_per_block_quant(
        x,
        group_size=128,  # Changed from [1, 128]
        quant_dtype=torch.float8_e4m3fn,
        is_scale_transposed=False,
    )

    assert out.shape == (batch_size, hidden_size)
    assert out.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32

    # Check no NaN or Inf
    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(scales).any()
    assert not torch.isinf(scales).any()


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("is_scale_transposed", IS_SCALE_TRANSPOSED)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_triton(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    group_size: int,
    is_scale_transposed: bool,
    seed: int,
    device: str,
) -> None:
    """Test SiLU+Mul+Block Quantization TRITON kernel correctness."""
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    if hidden_size % group_size != 0:
        return

    if has_scale_ub:
        pytest.skip("Scale upper bound not yet supported")

    scale = 1 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device) * scale
    scale_ub = None

    # Reference implementation
    ref_out, ref_scales = ref_impl(x, quant_dtype, scale_ub, group_size)

    # Triton kernel implementation
    triton_out, triton_scales = triton_impl(
        x, quant_dtype, scale_ub, group_size, is_scale_transposed
    )

    # Assertions
    assert not torch.isnan(triton_out.float()).any()
    assert not torch.isinf(triton_out.float()).any()
    assert not torch.isnan(triton_scales).any()
    assert not torch.isinf(triton_scales).any()

    # Normalize scales layout for comparison
    triton_scales_cmp = triton_scales.t() if is_scale_transposed else triton_scales

    # Check scales
    # Triton might have slightly different rounding, reasonable tolerance
    assert torch.allclose(ref_scales, triton_scales_cmp, rtol=1e-3, atol=1e-3), (
        f"Scale mismatch: max diff = {(ref_scales - triton_scales_cmp).abs().max()}"
    )

    # Check output
    # Dequantize to check values because FP8 exact match is hard

    # Normalize layout to [num_tokens, num_groups] for expansion
    triton_scales_row = triton_scales.t() if is_scale_transposed else triton_scales

    ref_scales_expanded = ref_scales.repeat_interleave(group_size, dim=1)
    triton_scales_expanded = triton_scales_row.repeat_interleave(group_size, dim=1)

    ref_deq = ref_out.float() * ref_scales_expanded
    triton_deq = triton_out.float() * triton_scales_expanded

    assert torch.allclose(ref_deq, triton_deq, rtol=0.1, atol=0.1), (
        f"Output mismatch: max diff = {(ref_deq - triton_deq).abs().max()}"
    )
