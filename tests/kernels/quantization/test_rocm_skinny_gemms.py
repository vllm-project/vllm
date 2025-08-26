# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.platforms import current_platform

DTYPES = [torch.bfloat16, torch.float16]
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

NKM_FACTORS_WVSPLITK_FP8 = [
    # FP8-specific cases with K % 16 == 0
    (1, 16, 16),
    (1, 64, 64),
    (2, 512, 512),
    (3, 2048, 2048),
    (4, 4096, 4096),
    # Extended FP8 dimensions not covered by WVSPLITK
    (1, 14336, 1024),
    (2, 24576, 2048),
    (4, 32768, 28672),
]

SEEDS = [0]


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_LLMM1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows_per_block", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="only test for rocm")
@torch.inference_mode()
def test_rocm_llmm1_kernel(n, k, m, dtype, rows_per_block, seed):
    torch.manual_seed(seed)
    A = torch.rand(n, k, dtype=dtype, device="cuda")
    B = torch.rand(m, k, dtype=dtype, device="cuda")

    ref_out = torch.matmul(A, B.t())
    out = ops.LLMM1(B, A, rows_per_block)

    assert torch.allclose(out, ref_out, rtol=0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="only test for rocm")
def test_rocm_wvsplitk_kernel(n, k, m, dtype, seed):
    torch.manual_seed(seed)
    cu_count = current_platform.get_cu_count()

    A = torch.rand(n, k, dtype=dtype, device="cuda")
    B = torch.rand(m, k, dtype=dtype, device="cuda")

    ref_out = torch.matmul(A, B.t())
    out = ops.wvSplitK(B, A, cu_count)

    assert torch.allclose(out, ref_out, rtol=0.01)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_FP8)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="only test for rocm fp8")
def test_rocm_wvsplitk_fp8_kernel(n, k, m, dtype, seed):
    torch.manual_seed(seed)

    A = torch.rand(n, k, device="cuda")
    B = torch.rand(m, k, device="cuda")

    A, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B, scale_b = ref_dynamic_per_tensor_fp8_quant(B)

    ref_out = torch._scaled_mm(A,
                               B.t(),
                               out_dtype=dtype,
                               scale_a=scale_a,
                               scale_b=scale_b)
    out = ops.wvSplitKQ(B, A, dtype, scale_a, scale_b,
                        current_platform.get_cu_count())

    assert torch.allclose(out, ref_out, rtol=0.01)
