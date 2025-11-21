# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools

import pytest
import torch

from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    cutlass_scaled_mm,
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    get_col_major_tma_aligned_tensor,
    per_block_cast_to_fp8,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.import_utils import has_deep_gemm

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()

# Test configurations
DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
NUM_TOKENS = [7, 2050]
D = [512, 4096, 5120, 13824]
GROUP_SIZE = [64, 128, 512]
M = [1, 7, 8, 83, 84, 4096]
N = [128, 512, 7168, 7748, 13824]
K = [256, 3884, 4096, 13824, 16384]
# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
BLOCK_SIZE = [[128, 128]]
OUT_DTYPES = [torch.bfloat16]  # [torch.float32, torch.half, torch.bfloat16]
SEEDS = [0]

# Skip all tests if CUDA is not available
pytest.importorskip("torch.cuda")


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "num_tokens,d,dtype,group_size,seed",
    itertools.product(NUM_TOKENS, D, DTYPES, GROUP_SIZE, SEEDS),
)
@torch.inference_mode()
def test_per_token_group_quant_fp8(num_tokens, d, dtype, group_size, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype)

    ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size)
    out, scale = per_token_group_quant_fp8(x, group_size)

    assert torch.allclose(out.to(torch.float32), ref_out.to(torch.float32), rtol=0.15)
    assert torch.allclose(scale, ref_scale)


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@torch.inference_mode()
def test_w8a8_block_fp8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(current_platform.fp8_dtype())
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(current_platform.fp8_dtype())

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(current_platform.fp8_dtype())

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = w8a8_triton_block_scaled_mm(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUTLASS only supported on CUDA platform."
)
@torch.inference_mode()
def test_w8a8_block_fp8_cutlass_matmul():
    # Test simple case where weight.shape % 128 != 0,
    # like in DSV3 kv_a_proj_with_mqa
    M = 32
    N = 576
    K = 7168
    block_size = [128, 128]
    out_dtype = torch.bfloat16
    seed = 0

    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale
    # Hopper requires row-major format for scales
    Bs_cutlass = Bs.T.contiguous() if current_platform.is_device_capability(90) else Bs

    A_fp8, As = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=False
    )
    # CUTLASS uses column-major format for scales
    A_fp8_cutlass, As_cutlass = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True
    )

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = cutlass_scaled_mm(
        A_fp8_cutlass, B_fp8, As_cutlass, Bs_cutlass, block_size, out_dtype
    )

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGemm kernels not available.")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    # only aligned sizes are supported by deepgemm
    if not should_use_deepgemm_for_fp8_linear(
        output_dtype=out_dtype, weight=B_fp32, supports_deep_gemm=True
    ):
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    A_fp8, As_fp8 = per_token_group_quant_fp8(A_fp32, block_size[1])
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_fp32, block_size=block_size)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    # Transpose earlier so that the testing will not trigger transposing kernels
    As_fp8 = get_col_major_tma_aligned_tensor(As_fp8)

    out = torch.zeros((M, N), device="cuda", dtype=out_dtype)

    assert As_fp8.shape == (M, (K + 127) // 128), (
        f"{As_fp8.shape} != {(M, (K + 127) // 128)}"
    )

    fp8_gemm_nt((A_fp8, As_fp8), (B_fp8, Bs_fp8), out)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@torch.inference_mode()
def test_flashinfer_block_gemm_matmul(M, N, K, block_size, out_dtype, seed):
    """
    Test FlashInfer FP8 block-scale GEMM through W8A8BlockFp8LinearOp.

    This tests the FP8 + FP8 → BF16 path (W8A8 full quantization).
    Matches TensorRT-LLM's test_fp8_block_scale_gemm behavior.
    """
    import os

    from vllm.utils.flashinfer import has_flashinfer_block_gemm

    if not has_flashinfer_block_gemm():
        pytest.skip(
            "FlashInfer block GEMM not available (requires SM90+ and FlashInfer)"
        )

    # Skip tests for dimensions that don't have pre-compiled kernels in FlashInfer
    # These cause CUDA runtime errors
    if K == 3884 or N == 7748:
        pytest.skip(f"FlashInfer does not have pre-compiled kernels for K={K} or N={N}")

    # Enable FlashInfer backend (required for W8A8BlockFp8LinearOp to use FlashInfer)
    os.environ["VLLM_USE_FLASHINFER_FP8_LINEAR"] = "1"
    # Reload envs module to pick up the env var change
    import importlib

    from vllm import envs

    importlib.reload(envs)

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        W8A8BlockFp8LinearOp,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
    )

    torch.manual_seed(seed)

    # Create BF16 inputs (normalized like TRT-LLM)
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16) / K
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16) / K

    # Quantize weight with per-block scales
    B_fp8, Bs = per_block_cast_to_fp8(B_bf16, block_size=block_size)

    # Create W8A8BlockFp8LinearOp to handle input quantization
    block_n, block_k = block_size[0], block_size[1]
    weight_group_shape = GroupShape(block_n, block_k)
    act_quant_group_shape = GroupShape(1, block_k)  # Per-token quantization

    linear_op = W8A8BlockFp8LinearOp(
        weight_group_shape=weight_group_shape,
        act_quant_group_shape=act_quant_group_shape,
        cutlass_block_fp8_supported=False,  # Disable CUTLASS
        use_aiter_and_is_supported=False,  # Disable AITER
    )

    # Verify FlashInfer backend is selected
    assert linear_op.w8a8_blockscale_op == linear_op._run_flashinfer, (
        "FlashInfer backend not selected! "
        "Make sure VLLM_USE_FLASHINFER_FP8_LINEAR=1 is set before running tests."
    )

    # Compute reference: BF16 × BF16 matmul (before quantization)
    ref_out = torch.matmul(A_bf16, B_bf16.T)

    # Run W8A8 FlashInfer GEMM (input will be quantized internally)
    out = linear_op.apply(
        input=A_bf16,
        weight=B_fp8,
        weight_scale=Bs,
        input_scale=None,  # Will quantize dynamically
        bias=None,
    )

    # Compare results using TensorRT-LLM's calc_diff metric
    # This measures normalized similarity: sim = 2*<x,y> / (||x||² + ||y||²)
    out_fp64 = out.to(torch.float64)
    ref_fp64 = ref_out.to(torch.float64)
    denominator = (out_fp64 * out_fp64 + ref_fp64 * ref_fp64).sum()
    sim = 2 * (out_fp64 * ref_fp64).sum() / denominator
    diff = 1 - sim

    # W8A8 threshold from TensorRT-LLM: diff < 0.001 (99.9% similarity)
    assert diff < 0.001, (
        f"Similarity difference {diff:.6f} exceeds threshold (similarity: {sim:.6f})"
    )


@pytest.mark.parametrize(
    "M,N,K,block_size,seed",
    [
        (1, 1024, 4096, [128, 128], 0),
        (32, 4096, 512, [128, 128], 0),
        (128, 1024, 4096, [128, 128], 0),
    ],
)
@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (torch.bfloat16, torch.bfloat16),  # BF16 + BF16 (internal quantization)
        (torch.bfloat16, torch.float8_e4m3fn),  # BF16 + FP8 (weight-only)
        (torch.float8_e4m3fn, torch.float8_e4m3fn),  # FP8 + FP8 (W8A8)
    ],
)
@torch.inference_mode()
def test_flashinfer_block_gemm_dtypes(
    M, N, K, block_size, input_dtype, weight_dtype, seed
):
    """
    Test all three supported dtype combinations for FlashInfer FP8 block-scale GEMM.

    Tests:
    - BF16 + BF16 → BF16: Both inputs BF16, internal quantization
    - BF16 + FP8 → BF16: Weight-only quantization
    - FP8 + FP8 → BF16: W8A8 full quantization

    This mirrors FlashInfer's own test_fp8_blockscale_gemm_dtypes and TRT-LLM's tests.
    """
    from vllm.utils.flashinfer import has_flashinfer_block_gemm

    if not has_flashinfer_block_gemm():
        pytest.skip(
            "FlashInfer block GEMM not available (requires SM90+ and FlashInfer)"
        )

    from vllm.model_executor.layers.quantization.utils.flashinfer_block_gemm import (
        flashinfer_block_gemm,
    )

    # Add debug output to verify test execution
    print(f"\n{'=' * 80}")
    print(f"TEST: M={M}, N={N}, K={K} | Input: {input_dtype}, Weight: {weight_dtype}")
    print(f"{'=' * 80}")

    torch.manual_seed(seed)

    # Create BF16 data for reference (same as FlashInfer tests)
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16)

    # Quantize input based on dtype
    if input_dtype == torch.float8_e4m3fn:
        input_tensor, input_scale = per_token_group_quant_fp8(input_bf16, block_size[1])
    else:
        input_tensor, input_scale = input_bf16, None

    # Quantize weight based on dtype
    if weight_dtype == torch.float8_e4m3fn:
        weight_tensor, weight_scale = per_block_cast_to_fp8(
            weight_bf16, block_size=block_size
        )
    else:
        weight_tensor, weight_scale = weight_bf16, None

    # Run FlashInfer FP8 block-scale GEMM
    output = flashinfer_block_gemm(
        input=input_tensor,
        weight=weight_tensor,
        scales_a=input_scale,
        scales_b=weight_scale,
        out_dtype=torch.bfloat16,
    )

    # Verify output properties
    assert output.shape == (M, N), f"Expected shape {(M, N)}, got {output.shape}"
    assert output.dtype == torch.bfloat16, f"Expected BF16 output, got {output.dtype}"

    # Compute reference based on dtype combination
    if input_dtype == torch.float8_e4m3fn and weight_dtype == torch.float8_e4m3fn:
        # W8A8: Compare against dequantized FP8 reference (tests kernel correctness)
        block_n, block_k = block_size[0], block_size[1]
        k_tiles = (K + block_k - 1) // block_k
        n_tiles = (N + block_n - 1) // block_n

        input_dequant = torch.zeros_like(input_bf16)
        for i in range(M):
            for k_tile in range(k_tiles):
                start, end = k_tile * block_k, min((k_tile + 1) * block_k, K)
                input_dequant[i, start:end] = (
                    input_tensor[i, start:end].to(torch.bfloat16)
                    * input_scale[i, k_tile]
                )

        weight_dequant = torch.zeros_like(weight_bf16)
        for j in range(N):
            for k_tile in range(k_tiles):
                start, end = k_tile * block_k, min((k_tile + 1) * block_k, K)
                weight_dequant[j, start:end] = (
                    weight_tensor[j, start:end].to(torch.bfloat16)
                    * weight_scale[j // block_n, k_tile]
                )

        reference = torch.matmul(input_dequant, weight_dequant.T)

        # W8A8: Use TRT-LLM's calc_diff metric with strict threshold
        out_fp64 = output.to(torch.float64)
        ref_fp64 = reference.to(torch.float64)
        denominator = (out_fp64 * out_fp64 + ref_fp64 * ref_fp64).sum()
        sim = 2 * (out_fp64 * ref_fp64).sum() / denominator
        diff = 1 - sim

        # W8A8 achieves very high accuracy: diff < 0.001 (99.9% similarity)
        assert diff < 0.001, (
            f"W8A8 similarity difference {diff:.6f} too high (expected < 0.001, similarity: {sim:.6f})"
        )
    else:
        # BF16+BF16 or BF16+FP8: Compare against original BF16 reference
        reference = torch.matmul(input_bf16, weight_bf16.T)

        out_fp64 = output.to(torch.float64)
        ref_fp64 = reference.to(torch.float64)
        denominator = (out_fp64 * out_fp64 + ref_fp64 * ref_fp64).sum()
        sim = 2 * (out_fp64 * ref_fp64).sum() / denominator
        diff = 1 - sim

        if input_dtype == torch.bfloat16 and weight_dtype == torch.bfloat16:
            # BF16+BF16: Highest accuracy (internal quantization)
            threshold = 0.001
            threshold_desc = "0.1%"
        elif input_dtype == torch.bfloat16 and weight_dtype == torch.float8_e4m3fn:
            # BF16+FP8: Weight-only quantization, higher error
            threshold = 0.01
            threshold_desc = "1%"
        else:
            # Other combinations
            threshold = 0.01
            threshold_desc = "1%"

        assert diff < threshold, (
            f"Similarity difference {diff:.6f} too high for "
            f"{input_dtype} + {weight_dtype} (expected < {threshold_desc}, similarity: {sim:.6f})"
        )
