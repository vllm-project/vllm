# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.platform_utils import get_cu_count

DTYPES = [torch.bfloat16, torch.float16]
BIAS_MODES = [0, 1, 2]
# Specific (N, K, M) combinations for targeted testing
NKM_FACTORS_LLMM1 = [
    # Small, medium, large cases
    (1, 8, 16),
    (1, 32, 64),
    (1, 128, 256),
    (1, 512, 1024),
    (1, 2048, 4096),
    # Edge cases with specific K sizes
    (1, 6144, 1024),
    (1, 8192, 2048),
    # Very large case
    (1, 4096, 8192),
]

NKM_FACTORS_WVSPLITK = [
    # Different batch sizes with key dimensions
    (1, 16, 16),
    (1, 64, 64),
    (2, 256, 256),
    (3, 1024, 1024),
    (4, 4096, 4096),
    # Extended K values
    (1, 9216, 512),
    (2, 10240, 1024),
    (4, 16384, 8192),
    # Minimum M constraint validation (m >= 8)
    (1, 64, 8),
    (2, 128, 8),
    (4, 256, 8),
]

N_FACTORS_WVSPLITKRC = [
    13,
    16,
    17,
    25,
    29,
    31,
    32,
    41,
    51,
    64,
    71,
    81,
    91,
    103,
    117,
    128,
]
K_FACTORS_WVSPLITKRC = [2880, 2880 + 8, 3072, 3072 + 8]
M_FACTORS_WVSPLITKRC = [128, 128 + 64, 256, 256 + 64, 640, 640 + 64]

NKM_FACTORS_WVSPLITK_FP8 = [
    # FP8-specific cases with K % 16 == 0
    (1, 16, 16),
    (1, 64, 64),
    (2, 512, 512),
    (3, 2048, 2048),
    (4, 4096, 4096),
    (4, 16400, 2048),
    # Extended FP8 dimensions not covered by WVSPLITK
    (1, 14336, 1024),
    (2, 24576, 2048),
    (4, 32768, 28672),
]

SEEDS = [0]


def pad_weights_fp8(weight):
    num_pad = 256 // weight.element_size()
    import torch.nn.functional as F

    return F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]


@pytest.mark.parametrize("n", N_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("k", K_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("m", M_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_rocm_wvsplitkrc_kernel(n, k, m, dtype, seed, bias_mode):
    torch.manual_seed(seed)
    cu_count = get_cu_count()

    if (((m + 15) // 16) * ((k + 511) // 512)) * (
        1 if (n <= 16) else (2 if (n <= 32) else 4)
    ) > cu_count * 4:
        pytest.skip("Too large for wvSplitKrc")

    xavier = math.sqrt(2 / k)  # normalize to avoid large output-bias deltas
    A = (torch.rand(n, k, dtype=dtype, device="cuda") - 0.5) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") - 0.5) * xavier

    BIAS = None
    if bias_mode == 1:
        BIAS = torch.rand(m, dtype=dtype, device="cuda") - 0.5
    elif bias_mode == 2:
        BIAS = torch.rand(n, m, dtype=dtype, device="cuda") - 0.5

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitKrc(B, A.view(-1, A.size(-1)), cu_count, BIAS)

    assert torch.allclose(out, ref_out, rtol=0.04 if dtype == torch.float16 else 0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_LLMM1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows_per_block", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_llmm1_kernel(n, k, m, dtype, rows_per_block, seed):
    torch.manual_seed(seed)
    # TODO: Zero-centering the inputs causes errors for LLMM1!
    #      Without that the numbers quickly saturate, and may
    #      be giving false matches.
    A = torch.rand(n, k, dtype=dtype, device="cuda")
    B = torch.rand(m, k, dtype=dtype, device="cuda")

    ref_out = torch.matmul(A, B.t())
    out = ops.LLMM1(B, A, rows_per_block)

    assert torch.allclose(out, ref_out, rtol=0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
def test_rocm_wvsplitk_kernel(n, k, m, dtype, seed):
    torch.manual_seed(seed)
    cu_count = get_cu_count()

    A = torch.rand(n, k, dtype=dtype, device="cuda") - 0.5
    B = torch.rand(m, k, dtype=dtype, device="cuda") - 0.5

    ref_out = torch.nn.functional.linear(A, B)
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu_count)

    assert torch.allclose(out, ref_out, rtol=0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
def test_rocm_wvsplitk_bias1D_kernel(n, k, m, dtype, seed):
    torch.manual_seed(seed)
    cu_count = get_cu_count()

    xavier = math.sqrt(2 / k)  # normalize to avoid large output-bias deltas
    A = (torch.rand(n, k, dtype=dtype, device="cuda") - 0.5) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") - 0.5) * xavier
    BIAS = torch.rand(m, dtype=dtype, device="cuda") - 0.5

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu_count, BIAS)

    assert torch.allclose(out, ref_out, rtol=0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
def test_rocm_wvsplitk_bias2D_kernel(n, k, m, dtype, seed):
    torch.manual_seed(seed)
    cu_count = get_cu_count()

    xavier = math.sqrt(2 / k)  # normalize to avoid large output-bias deltas
    A = (torch.rand(n, k, dtype=dtype, device="cuda") - 0.5) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") - 0.5) * xavier
    BIAS = torch.rand(n, m, dtype=dtype, device="cuda") - 0.5

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu_count, BIAS)

    assert torch.allclose(out, ref_out, rtol=0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_FP8)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="only test for rocm fp8",
)
def test_rocm_wvsplitk_fp8_kernel(n, k, m, dtype, seed, padded):
    torch.manual_seed(seed)

    A = torch.rand(n, k, device="cuda") - 0.5
    B = torch.rand(m, k, device="cuda") - 0.5

    A, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B, scale_b = ref_dynamic_per_tensor_fp8_quant(B)
    if padded:
        B = pad_weights_fp8(B)

    ref_out = torch._scaled_mm(
        A, B.t(), out_dtype=dtype, scale_a=scale_a, scale_b=scale_b
    )
    out = ops.wvSplitKQ(
        B,
        A,
        dtype,
        scale_a,
        scale_b,
        get_cu_count(),
    )

    assert torch.allclose(out, ref_out, rtol=0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_FP8)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="only test for rocm fp8",
)
def test_rocm_wvsplitk_fp8_bias1D_kernel(n, k, m, dtype, seed, padded):
    torch.manual_seed(seed)

    xavier = math.sqrt(2 / k)  # normalize to avoid large output-bias deltas
    A = (torch.rand(n, k, device="cuda") - 0.5) * xavier
    B = (torch.rand(m, k, device="cuda") - 0.5) * xavier
    BIAS = torch.rand(m, dtype=dtype, device="cuda") - 0.5

    A, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B, scale_b = ref_dynamic_per_tensor_fp8_quant(B)
    if padded:
        B = pad_weights_fp8(B)

    ref_out = torch._scaled_mm(
        A, B.t(), out_dtype=dtype, scale_a=scale_a, scale_b=scale_b, bias=BIAS
    )
    out = ops.wvSplitKQ(
        B,
        A,
        dtype,
        scale_a,
        scale_b,
        get_cu_count(),
        BIAS,
    )

    assert torch.allclose(out, ref_out, rtol=0.01)
