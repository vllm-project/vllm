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
    get_tma_aligned_size,
    per_block_cast_to_fp8,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    has_flashinfer_fp8_blockscale_gemm,
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
COLUMN_MAJOR_SCALES = [True, False]
TMA_ALIGNED_SCALES = [True, False]
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
    "num_tokens,d,dtype,group_size,column_major_scales,tma_aligned_scales,seed",
    itertools.product(
        NUM_TOKENS,
        D,
        DTYPES,
        GROUP_SIZE,
        COLUMN_MAJOR_SCALES,
        TMA_ALIGNED_SCALES,
        SEEDS,
    ),
)
@torch.inference_mode()
def test_per_token_group_quant_fp8(
    num_tokens, d, dtype, group_size, column_major_scales, tma_aligned_scales, seed
):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype)

    ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size)
    out, scale = per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major_scales,
        tma_aligned_scales=tma_aligned_scales,
    )

    assert torch.allclose(out.to(torch.float32), ref_out.to(torch.float32), rtol=0.15)
    assert torch.allclose(scale, ref_scale)

    if column_major_scales:
        assert scale.stride()[-2] == 1
        if tma_aligned_scales:
            assert scale.stride()[-1] == get_tma_aligned_size(num_tokens, 4)


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

    A_fp8, As_fp8 = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True, tma_aligned_scales=True
    )
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_fp32, block_size=block_size)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    out = torch.zeros((M, N), device="cuda", dtype=out_dtype)

    assert As_fp8.shape == (M, (K + 127) // 128), (
        f"{As_fp8.shape} != {(M, (K + 127) // 128)}"
    )

    fp8_gemm_nt((A_fp8, As_fp8), (B_fp8, Bs_fp8), out)

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
@torch.inference_mode()
def test_w8a8_block_fp8_flashinfer_matmul(M, N, K, block_size, out_dtype, seed):
    if not has_flashinfer_fp8_blockscale_gemm():
        pytest.skip(
            "FlashInfer block GEMM not available (requires SM90+ and FlashInfer)"
        )
    # only aligned sizes
    if K % 128 != 0 or N % 64 != 0:
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_bf16 = (torch.rand(M, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    B_bf16 = (torch.rand(N, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max

    A_fp8, As_fp8 = per_token_group_quant_fp8(A_bf16, block_size[1], use_ue8m0=False)
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_bf16, block_size, use_ue8m0=False)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    out = flashinfer_fp8_blockscale_gemm(
        input=A_bf16,
        weight=B_fp8,
        input_scale=None,
        weight_scale=Bs,
        out_dtype=out_dtype,
    )

    rel_diff = torch.mean(
        torch.abs(out.to(torch.bfloat16) - ref_out.to(torch.bfloat16))
    ) / torch.mean(torch.abs(ref_out.to(torch.bfloat16)))
    assert rel_diff < 0.001
