# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.platforms import current_platform

DTYPES = [torch.bfloat16, torch.float16]
M = [16, 32, 64, 128, 256, 512, 1024, 4096, 8192]
K = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192]  # k % 8 == 0
N = [1, 2, 3, 4]
SEEDS = [0]


@pytest.mark.parametrize("n", [1])  # only test for batch size 1
@pytest.mark.parametrize("k", K)
@pytest.mark.parametrize("m", M)
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


@pytest.mark.parametrize("n", N)  # only test for batch size <= 4
@pytest.mark.parametrize("k", K + [9216, 10240, 16384])
@pytest.mark.parametrize("m", [8] + M)  # m >= 8
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


@pytest.mark.parametrize("n", N)  # only test for batch size <= 4
@pytest.mark.parametrize("k", K[1:] + [14336, 24576, 32768])  # k % 16 == 0
@pytest.mark.parametrize("m", M + [28672])  # m >= 16
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
