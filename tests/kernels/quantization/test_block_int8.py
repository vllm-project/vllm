# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/blob/main/test/srt/test_block_int8.py
import itertools

import pytest
import torch

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    w8a8_block_int8_matmul,
)
from vllm.platforms import current_platform

_cap = current_platform.get_device_capability()
if _cap is not None and _cap < (7, 0):
    pytest.skip("INT8 Triton requires CUDA 7.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()

DTYPES = [torch.half, torch.bfloat16]
M = [1, 33, 64, 222]
N = [128, 1024]
K = [256, 4096]
# BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
BLOCK_SIZE = [[128, 128]]
SEEDS = [0]


@pytest.fixture(autouse=True, scope="module")
def setup_cuda():
    """Sets the default CUDA device for all tests in this module."""
    torch.set_default_device("cuda")


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, DTYPES, SEEDS),
)
@torch.inference_mode()
def test_w8a8_block_int8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    int8_info = torch.iinfo(torch.int8)
    int8_max, int8_min = int8_info.max, int8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * int8_max
    A_fp8 = A_fp32.clamp(min=int8_min, max=int8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * int8_max
    B_fp8 = B_fp32.clamp(min=int8_min, max=int8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = w8a8_block_int8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


# Tensor-descriptor (TD) operand-load path for _w8a8_block_int8_matmul. The
# block-scale multiply lands on the [M, N] accumulator (not the operands), so the
# TD path must be bit-exact vs the plain masked-load path.
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="No GPU device available",
)
@pytest.mark.parametrize(
    "M,N,K", [(1, 4096, 4096), (64, 4096, 4096), (256, 2048, 4096)]
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_w8a8_block_int8_matmul_td_matches_plain(M, N, K, out_dtype):
    dev = current_platform.device_type
    torch.manual_seed(0)
    block_size = [128, 128]
    bn, bk = block_size
    A = torch.randint(-8, 8, (M, K), device=dev, dtype=torch.int8)
    B = torch.randint(-8, 8, (N, K), device=dev, dtype=torch.int8)
    As = torch.rand(M, K // bk, device=dev, dtype=torch.float32) * 1e-2
    Bs = torch.rand(N // bn, K // bk, device=dev, dtype=torch.float32) * 1e-2

    out_plain = w8a8_block_int8_matmul(
        A, B, As, Bs, block_size, out_dtype, use_td=False
    )
    out_td = w8a8_block_int8_matmul(A, B, As, Bs, block_size, out_dtype, use_td=True)
    torch.testing.assert_close(out_td, out_plain, rtol=0, atol=0)
