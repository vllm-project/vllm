# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/blob/main/test/srt/test_block_int8.py
import itertools

import pytest
import torch

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    w8a8_block_int8_matmul)
from vllm.platforms import current_platform

if current_platform.get_device_capability() < (7, 0):
    pytest.skip("INT8 Triton requires CUDA 7.0 or higher",
                allow_module_level=True)

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


# For test
def native_per_token_group_quant_int8(x,
                                      group_size,
                                      eps=1e-10,
                                      dtype=torch.int8):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch.

    It converts the tensor values into int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    """
    assert (x.shape[-1] % group_size == 0
            ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_min = iinfo.min
    int8_max = iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    # Use float32 for scale calculation for stability
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_.to(torch.float32) / x_s).round().clamp(
        min=int8_min, max=int8_max).to(dtype)  # Round before clamping
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s


# For test
def torch_w8a8_block_int8_moe(a, w1, w2, w1_s, w2_s, score, topk, block_shape):
    """This function performs fused moe with block-wise quantization using
    native torch."""
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_int8(a, block_k)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_matmul(a_q[mask],
                                                 w1[i],
                                                 a_s[mask],
                                                 w1_s[i],
                                                 block_shape,
                                                 output_dtype=a.dtype)
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_int8(
                act_out, block_k)
            act_out = act_out.to(torch.float32)
            out[mask] = native_w8a8_block_matmul(act_out_q,
                                                 w2[i],
                                                 act_out_s,
                                                 w2_s[i],
                                                 block_shape,
                                                 output_dtype=a.dtype)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


DTYPES = [torch.half, torch.bfloat16]
M = [1, 33, 64, 222]
N = [128, 1024]
K = [256, 4096]
E = [8, 24]
TOP_KS = [2, 6]
# BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
BLOCK_SIZE = [[128, 128]]
SEEDS = [0]


@pytest.fixture(autouse=True, scope="module")
def setup_cuda():
    """Sets the default CUDA device for all tests in this module."""
    torch.set_default_device("cuda")


@pytest.mark.parametrize("M,N,K,block_size,out_dtype,seed",
                         itertools.product(M, N, K, BLOCK_SIZE, DTYPES, SEEDS))
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

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size,
                                       out_dtype)
    out = w8a8_block_int8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.001


@pytest.mark.parametrize(
    "M, N, K, E, topk, block_size, dtype, seed",
    itertools.product(M, N, K, E, TOP_KS, BLOCK_SIZE, DTYPES, SEEDS))
@torch.inference_mode()
def test_w8a8_block_int8_fused_moe(M, N, K, E, topk, block_size, dtype, seed):
    """Tests the fused_moe kernel with W8A8 INT8 block quantization against a
    native torch reference."""
    torch.manual_seed(seed)
    # Use a smaller factor for scale initialization to prevent large
    # values/overflow especially when output dtype might be float16
    factor_for_scale = 1e-2
    int8_info = torch.iinfo(torch.int8)
    int8_max, int8_min = int8_info.max, int8_info.min

    a = torch.randn((M, K), dtype=dtype) / 10

    w1_fp32 = (torch.rand(
        (E, 2 * N, K), dtype=torch.float32) - 0.5) * 2 * int8_max
    w1 = w1_fp32.clamp(min=int8_min, max=int8_max).to(torch.int8)

    w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2 * int8_max
    w2 = w2_fp32.clamp(min=int8_min, max=int8_max).to(torch.int8)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = (2 * N + block_n - 1) // block_n
    n_tiles_w2 = (K + block_n - 1) // block_n
    k_tiles_w1 = (K + block_k - 1) // block_k
    k_tiles_w2 = (N + block_k - 1) // block_k

    w1_s = (torch.rand(
        (E, n_tiles_w1, k_tiles_w1), dtype=torch.float32) * factor_for_scale)
    w2_s = (torch.rand(
        (E, n_tiles_w2, k_tiles_w2), dtype=torch.float32) * factor_for_scale)

    score = torch.randn((M, E), dtype=dtype)

    # Set the context to avoid lots of warning spam.
    with set_current_vllm_config(vllm_config):
        out = fused_moe(
            a,
            w1,
            w2,
            score,
            topk,
            renormalize=False,
            use_int8_w8a8=True,
            w1_scale=w1_s,
            w2_scale=w2_s,
            block_shape=block_size,
        )
        ref_out = torch_w8a8_block_int8_moe(a, w1, w2, w1_s, w2_s, score, topk,
                                            block_size)

    # Check results
    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.06
