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

NKM_FACTORS_WVSPLITKRC = [
    (16, 2880, 128),
    (16, 2880, 640),
    (17, 2880, 128),
    (17, 2880, 640),
    (25, 2880, 128),
    (25, 2880, 640),
    (31, 2880, 128),
    (31, 2880, 640),
    (32, 2880, 128),
    (32, 2880, 640),
    (40, 2880, 128),
    (40, 2880, 640),
    (60, 2880, 128),
    (60, 2880, 640),
    (64, 2880, 128),
    (64, 2880, 640),
    (81, 2880, 128),
    (81, 2880, 640),
    (98, 2880, 128),
    (98, 2880, 640),
    (128, 2880, 128),
    (128, 2880, 640),
]

NKM_FACTORS_WVSPLITK_FP8 = [
    # FP8-specific cases with K % 16 == 0
    (1, 16, 16),
    (1, 32, 16 + 16),
    (1, 64, 64),
    (1, 64, 64 + 16),
    (1, 64 + 16, 64),
    (1, 64 + 16, 64 + 16),
    (4, 64, 64),
    (4, 64, 64 + 16),
    (4, 64 + 16, 64),
    (4, 64 + 16, 64 + 16),
    (2, 512, 512),
    (3, 512, 512),
    (3, 512, 512 + 16),
    (4, 512, 512),
    (3, 2048, 2048),
    (3, 2048, 2048 + 16),
    (4, 2048 + 16, 2048),
    (4, 2048 + 16, 2048 + 16),
    (4, 4096, 4096),
    (4, 16400, 2048),
    (4, 16400, 2048 + 16),
    # Extended FP8 dimensions not covered by WVSPLITK
    (1, 14336, 1024),
    (2, 24576, 2048),
    (4, 32768, 28672),
    (4, 32768 * 2, 28672),
    (4, 32768 * 2, 28672 + 16),
    (4, 32768 * 2 + 16, 28672),
    (4, 32768 * 2 + 16, 28672 + 16),
]

SEEDS = [0]


def pad_fp8(weight):
    num_pad = 256 // weight.element_size()
    import torch.nn.functional as F

    return F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_rocm_wvsplitkrc_kernel(n, k, m, dtype, seed, bias_mode):
    torch.manual_seed(seed)
    cu_count = get_cu_count()

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

    assert torch.allclose(out, ref_out, rtol=0.01)


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


@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_FP8)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("padded_b", [False, True])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="only test for rocm fp8",
)
def test_rocm_wvsplitk_fp8_kernel(
    xnorm, n, k, m, dtype, seed, padded_a, padded_b, biased
):
    torch.manual_seed(seed)

    xavier = math.sqrt(2 / k) if xnorm else 1  # normalize to avoid large deltas
    A = (torch.rand(n, k, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, device="cuda") * 2 - 1) * xavier

    A, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B, scale_b = ref_dynamic_per_tensor_fp8_quant(B)
    if padded_b:
        B = pad_fp8(B)
    if padded_a:
        A = pad_fp8(A)

    BIAS = None if (not biased) else (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1)

    ref_out = torch._scaled_mm(
        A, B.t(), out_dtype=dtype, scale_a=scale_a, scale_b=scale_b, bias=BIAS
    )
    out = ops.wvSplitKQ(B, A, dtype, scale_a, scale_b, get_cu_count(), BIAS)

    if xnorm:
        assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-8)
    else:
        assert torch.allclose(out, ref_out, 0.01)
