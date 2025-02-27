# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/sgl-project/sglang/pull/2575
# TODO: try/catch this?
import deep_gemm

import itertools
import pytest
import torch

from typing import Tuple

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8, w8a8_block_fp8_matmul)
from vllm.platforms import current_platform

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher",
                allow_module_level=True)

# Test configurations
DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
NUM_TOKENS = [7, 83, 2048]
D = [512, 4096, 5120, 13824]
GROUP_SIZE = [64, 128, 256, 512]
#M = [1, 7, 83, 512, 2048]

M = [1, 8, 84, 512, 2048, 4096]
N = [128, 512, 1024, 4096, 7748, 13824, 7168]
K = [256, 4096, 5120, 3884, 13824, 16384]

#M = [128]
#N = [24576]
#K = [1536]

# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
#M_moe = [1, 7, 83, 512, 2048]
M_moe = [1, 8, 84, 512, 2048]
N_moe = [4608]  # [128, 4608, 13824]
K_moe = [7168]  # [256, 7168, 13824]
BLOCK_SIZE = [[128, 128]]
E = [8, 24]  # [8, 24, 128, 256]
TOP_KS = [2]  # [1, 2, 6]
OUT_DTYPES = [torch.bfloat16]  # [torch.float32, torch.half, torch.bfloat16]
SEEDS = [0]


def native_per_token_group_quant_fp8(x,
                                     group_size,
                                     eps=1e-10,
                                     dtype=torch.float8_e4m3fn):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, ("the last dimension of `x` cannot "
                                           "be divisible by `group_size`")
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s


def native_w8a8_block_fp8_matmul(A,
                                 B,
                                 As,
                                 Bs,
                                 block_size,
                                 output_dtype=torch.float16):
    """Matrix multiplication with block-wise quantization using native torch."""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [
        A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)
    ]
    B_tiles = [[
        B[
            j * block_n:min((j + 1) * block_n, N),
            i * block_k:min((i + 1) * block_k, K),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [
        C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)
    ]
    As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, score, topk, block_shape):
    """Fused moe with block-wise quantization using native torch."""
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
    a_q = a_q.to(torch.float32)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_fp8_matmul(a_q[mask],
                                                     w1[i],
                                                     a_s[mask],
                                                     w1_s[i],
                                                     block_shape,
                                                     output_dtype=a.dtype)
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_fp8(
                act_out, block_k)
            act_out = act_out.to(torch.float32)
            out[mask] = native_w8a8_block_fp8_matmul(act_out_q,
                                                     w2[i],
                                                     act_out_s,
                                                     w2_s[i],
                                                     block_shape,
                                                     output_dtype=a.dtype)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


# Skip all tests if CUDA is not available
pytest.importorskip("torch.cuda")


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


@pytest.mark.parametrize(
    "num_tokens,d,dtype,group_size,seed",
    itertools.product(NUM_TOKENS, D, DTYPES, GROUP_SIZE, SEEDS))
@torch.inference_mode()
def test_per_token_group_quant_fp8(num_tokens, d, dtype, group_size, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype)

    ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size)
    out, scale = per_token_group_quant_fp8(x, group_size)

    assert torch.allclose(out.to(torch.float32),
                          ref_out.to(torch.float32),
                          rtol=0.15)
    assert torch.allclose(scale, ref_scale)


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS))
@torch.inference_mode()
def test_w8a8_block_fp8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    ref_out = native_w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size,
                                           out_dtype)
    out = w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.001


@pytest.mark.parametrize(
    "M,N,K,E,topk,block_size,dtype,seed",
    itertools.product(M_moe, N_moe, K_moe, E, TOP_KS, BLOCK_SIZE, DTYPES,
                      SEEDS))
@torch.inference_mode()
def test_w8a8_block_fp8_fused_moe(M, N, K, E, topk, block_size, dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a = torch.randn((M, K), dtype=dtype) / 10

    w1_bf16 = (torch.rand(
        (E, 2 * N, K), dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    w1 = w1_bf16.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    del w1_bf16

    w2_bf16 = (torch.rand((E, K, N), dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    w2 = w2_bf16.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    del w2_bf16

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = (2 * N + block_n - 1) // block_n
    n_tiles_w2 = (K + block_n - 1) // block_n
    k_tiles_w1 = (K + block_k - 1) // block_k
    k_tiles_w2 = (N + block_k - 1) // block_k

    w1_s = torch.rand(
        (E, n_tiles_w1, k_tiles_w1), dtype=torch.float32) * factor_for_scale
    w2_s = torch.rand(
        (E, n_tiles_w2, k_tiles_w2), dtype=torch.float32) * factor_for_scale

    score = torch.randn((M, E), dtype=dtype)

    out = fused_moe(
        a,
        w1,
        w2,
        score,
        topk,
        renormalize=False,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=block_size,
    )
    ref_out = torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, score, topk,
                                       block_size)

    print(f"{out.sum()=}")
    print(f"{ref_out.sum()=}")

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.03


#########################################################################################

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((deep_gemm.cell_div(m, 128) * 128, deep_gemm.cell_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS))
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_matmul(M, N, K, block_size, out_dtype, seed):
    # only aligned sizes
    if M % 4 != 0 or K % 128 != 0 or N % 64 != 0:
        return

    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    A_fp8_dg, As_dg = per_token_group_quant_fp8(A_fp32, block_k)
    B_fp8_dg, Bs_dg = per_block_cast_to_fp8(B_fp32)

    As = As_dg.to(torch.float32)
    Bs = Bs_dg.to(torch.float32)

    ref_out = native_w8a8_block_fp8_matmul(A_fp8_dg, B_fp8_dg, As, Bs, block_size,
                                           out_dtype)

    #A_fp8_dg, As_dg = per_token_group_quant_fp8(A_fp32, block_k)
    #B_fp8_dg, Bs_dg = per_block_cast_to_fp8(B_fp32)

    # Transpose earlier so that the testing will not trigger transposing kernels
    As_dg = deep_gemm.get_col_major_tma_aligned_tensor(As_dg)

    out = torch.zeros((M, N), device='cuda', dtype=out_dtype)

    assert As_dg.shape == (M, (K + 127) // 128), f"{As_dg.shape} != {(M, (K + 127) // 128)}"

    deep_gemm.gemm_fp8_fp8_bf16_nt((A_fp8_dg, As_dg), (B_fp8_dg, Bs_dg), out)

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.001


###################################################################################

def construct_grouped(
    num_groups: int,
    m: int,
    k: int,
    n: int,
    is_masked: bool
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out


# ref_out = torch.einsum('gmk,gnk->gmn', x, y)

from vllm.model_executor.layers.fused_moe import fused_topk, grouped_topk

def deep_gemm_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, score, topk, block_shape):
    """Fused moe with block-wise quantization using native torch."""
    M, K = a.shape
    print(f"before {a.shape}")
    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.to(dtype=torch.int32).view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = per_token_group_quant_fp8(a, block_k)

    num_groups = w1.shape[0]
    for i in range(num_groups):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1[i].to(dtype=torch.bfloat16))
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2[i].to(dtype=torch.bfloat16))

    print(f"{M}, {num_groups}, {a.shape}")

    m_indices = torch.arange(0, num_groups, device=a.device, dtype=torch.int)
    m_indices = m_indices.unsqueeze(-1).expand(num_groups, a.shape[0]//num_groups).contiguous().view(-1)

    inter_out = torch.zeros(a_q.shape[0], w1.shape[1], dtype=torch.bfloat16, device=a.device)

    print("FIRST GEMM")

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous((a_q, a_s),
                                                        (w1, w1_s),
                                                        inter_out,
                                                        topk_ids)

    act_out = SiluAndMul().forward_native(inter_out)
    act_out_q, act_out_s = per_token_group_quant_fp8(act_out, block_k)

    out = torch.zeros(M * topk, w2.shape[1], dtype=torch.bfloat16, device=a.device)

    print("SECOND GEMM")

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous((act_out_q, act_out_s),
                                                        (w2, w2_s),
                                                        out,
                                                        topk_ids)

    return (out.view(M, -1, w2.shape[1]) *
            topk_weight.view(M, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize(
    "M,N,K,E,topk,block_size,dtype,seed",
    itertools.product(M_moe, N, K_moe, E, TOP_KS, BLOCK_SIZE, DTYPES,
                      SEEDS))
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_fused_moe(M, N, K, E, topk, block_size, dtype, seed):

    # only aligned sizes
    if M % 4 != 0 or K % 128 != 0 or N % 64 != 0:
        return

    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a = torch.randn((M, K), dtype=dtype) / 10

    w1_bf16 = (torch.rand(
        (E, 2 * N, K), dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    w1 = w1_bf16.clamp(min=fp8_min, max=fp8_max)
    del w1_bf16

    w2_bf16 = (torch.rand((E, K, N), dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    w2 = w2_bf16.clamp(min=fp8_min, max=fp8_max)
    del w2_bf16

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = (2 * N + block_n - 1) // block_n
    n_tiles_w2 = (K + block_n - 1) // block_n
    k_tiles_w1 = (K + block_k - 1) // block_k
    k_tiles_w2 = (N + block_k - 1) // block_k

    w1_s = torch.rand(
        (E, n_tiles_w1, k_tiles_w1), dtype=torch.float32) * factor_for_scale
    w2_s = torch.rand(
        (E, n_tiles_w2, k_tiles_w2), dtype=torch.float32) * factor_for_scale

    score = torch.randn((M, E), dtype=dtype)

    w1 = w1.to(torch.float8_e4m3fn)
    w2 = w2.to(torch.float8_e4m3fn)

    ref_out = deep_gemm_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, score, topk,
                                           block_size)

    out = fused_moe(
        a,
        w1,
        w2,
        score,
        topk,
        renormalize=False,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=block_size,
    )
    print(f"{out.sum()=}")
    print(f"{ref_out.sum()=}")

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.03
