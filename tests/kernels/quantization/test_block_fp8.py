# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools

import pytest
import torch

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    _valid_deep_gemm_shape, deep_gemm_moe_fp8)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, modular_triton_fused_moe)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8, w8a8_block_fp8_matmul)
from vllm.platforms import current_platform

dg_available = False
try:
    import deep_gemm
    dg_available = True
except ImportError:
    pass

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher",
                allow_module_level=True)

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

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
M_moe = [1, 2, 7, 83, 128, 2048, 1024 * 128]
M_moe_dg = [128, 192, 1335, 2048]
N_moe = [128, 256, 1024, 4608]  # [13824]
K_moe = [256, 512, 7168]  # [13824]
BLOCK_SIZE = [[128, 128]]
E = [2, 8, 16, 24]  # [128, 256]
TOP_KS = [1, 2, 6]
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
            inter_out = native_w8a8_block_matmul(a_q[mask],
                                                 w1[i],
                                                 a_s[mask],
                                                 w1_s[i],
                                                 block_shape,
                                                 output_dtype=a.dtype)
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_fp8(
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

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size,
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
    if topk > E:
        pytest.skip(f"Skipping test; topk={topk} > E={E}")

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

    m_fused_moe = modular_triton_fused_moe(use_fp8_w8a8=True,
                                           use_int8_w8a8=False,
                                           use_int8_w8a16=False,
                                           use_int4_w4a16=False,
                                           per_channel_quant=False,
                                           block_shape=block_size)

    # Set the context to avoid lots of warning spam.
    with set_current_vllm_config(vllm_config):
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

        topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
        m_out = m_fused_moe(a,
                            w1,
                            w2,
                            topk_weights,
                            topk_ids,
                            global_num_experts=E,
                            w1_scale=w1_s,
                            w2_scale=w2_s)

    #print(f"{out.sum()=}")
    #print(f"{ref_out.sum()=}")

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.03

    rel_diff = (torch.mean(
        torch.abs(m_out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.03


def per_block_cast_to_fp8(
        x: torch.Tensor,
        block_size_n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (deep_gemm.ceil_div(m, 128) * 128,
         deep_gemm.ceil_div(n, block_size_n) * block_size_n),
        dtype=x.dtype,
        device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS))
@pytest.mark.skipif(not dg_available, reason="DeepGemm kernels not available.")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_matmul(M, N, K, block_size, out_dtype, seed):
    # only aligned sizes
    if M % 4 != 0 or K % 128 != 0 or N % 64 != 0:
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    _, block_k = block_size[0], block_size[1]

    A_fp8, As_fp8 = per_token_group_quant_fp8(A_fp32, block_k)
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_fp32)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size,
                                       out_dtype)

    # Transpose earlier so that the testing will not trigger transposing kernels
    As_fp8 = deep_gemm.get_col_major_tma_aligned_tensor(As_fp8)

    out = torch.zeros((M, N), device='cuda', dtype=out_dtype)

    assert As_fp8.shape == (M, (K + 127) //
                            128), f"{As_fp8.shape} != {(M, (K + 127) // 128)}"

    deep_gemm.gemm_fp8_fp8_bf16_nt((A_fp8, As_fp8), (B_fp8, Bs_fp8), out)

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.001


def fp8_perm(m, idx):
    if torch.is_floating_point(m) and torch.finfo(m.dtype).bits == 8:
        return m.view(dtype=torch.uint8)[idx, ...].view(dtype=m.dtype)
    else:
        return m[idx, ...]


def _moe_permute(a, a_s, topk_ids, num_groups, topk, block_m):
    M, K = a.shape

    sorted_token_ids, m_indices, num_pad = moe_align_block_size(
        topk_ids, block_m, num_groups, None, pad_sorted_ids=True)

    num_tokens = topk * M

    sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)
    m_indices = torch.repeat_interleave(m_indices, block_m, dim=0)
    inv_perm = torch.argsort(sorted_token_ids)[:M * topk]

    a = fp8_perm(a, sorted_token_ids // topk)
    if a_s is not None:
        a_s = a_s[sorted_token_ids // topk]

    return a, a_s, m_indices, inv_perm


def _moe_unpermute(out, inv_perm, topk, K, topk_weight):
    M = topk_weight.shape[0]
    out = out[inv_perm, ...]
    tmp_out = out.view(-1, topk, K)
    return (tmp_out * topk_weight.view(M, -1, 1).to(out.dtype)).sum(dim=1)


def deep_gemm_w8a8_block_fp8_moe(M, K, a, w1, w2, w1_s, w2_s, score, topk,
                                 block_shape):
    """Fused moe with block-wise quantization using DeepGemm grouped gemm."""
    num_groups = w1.shape[0]
    M, K = a.shape
    N = w2.shape[-1]

    topk_weight, topk_ids, token_expert_indices = fused_topk(
        a, score.float(), topk, False)

    block_m = deep_gemm.get_m_alignment_for_contiguous_layout()

    _, block_k = block_shape[0], block_shape[1]

    a_q, a_s = per_token_group_quant_fp8(a, block_m)

    a_q, a_s, m_indices, inv_perm = _moe_permute(a_q, a_s, topk_ids,
                                                 num_groups, topk, block_m)

    inter_out = torch.zeros((a_q.shape[0], N * 2),
                            dtype=torch.bfloat16,
                            device=a.device)

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous((a_q, a_s), (w1, w1_s),
                                                        inter_out, m_indices)

    act_out = SiluAndMul().forward_native(inter_out)
    act_out_q, act_out_s = per_token_group_quant_fp8(act_out, block_k)

    out = torch.zeros(a_q.shape[0], K, dtype=torch.bfloat16, device=a.device)

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        (act_out_q, act_out_s), (w2, w2_s), out, m_indices)

    final_out = _moe_unpermute(out, inv_perm, topk, K, topk_weight)

    return final_out


@pytest.mark.parametrize(
    "M,N,K,E,topk,seed",
    itertools.product(M_moe_dg, N_moe, K_moe, E, TOP_KS, SEEDS))
@pytest.mark.skipif(not dg_available, reason="DeepGemm kernels not available.")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_fused_moe(M, N, K, E, topk, seed):

    block_m = deep_gemm.get_m_alignment_for_contiguous_layout()
    block_size = [block_m, block_m]
    dtype = torch.bfloat16

    if topk > E:
        pytest.skip(f"Skipping test: topk={topk} > E={E}")

    if not _valid_deep_gemm_shape(M, N, K):
        pytest.skip(f"Skipping test: invalid size m={M}, n={N}, k={K}")

    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a = torch.randn((M, K), dtype=dtype) / 10

    w1_bf16 = ((torch.rand((E, 2 * N, K), dtype=torch.bfloat16) - 0.5) * 2 *
               fp8_max).clamp(min=fp8_min, max=fp8_max)

    w2_bf16 = ((torch.rand((E, K, N), dtype=torch.bfloat16) - 0.5) * 2 *
               fp8_max).clamp(min=fp8_min, max=fp8_max)

    score = torch.randn((M, E), dtype=dtype)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = ((2 * N) + block_n - 1) // block_n
    k_tiles_w1 = (K + block_k - 1) // block_k
    n_tiles_w2 = (K + block_n - 1) // block_n
    k_tiles_w2 = (N + block_k - 1) // block_k

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)

    w1_s = torch.empty((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
    w2_s = torch.empty((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)

    w1_s = deep_gemm.get_col_major_tma_aligned_tensor(w1_s).contiguous()
    w2_s = deep_gemm.get_col_major_tma_aligned_tensor(w2_s).contiguous()

    assert w1_s.shape == (E, (2 * N + 127) // 128, (K + 127) // 128)
    assert (w2.shape[-2] + block_n - 1) // block_n == w2_s.shape[-2]

    for i in range(E):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i])
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i])

    # Set the context to avoid lots of warning spam.
    with set_current_vllm_config(vllm_config):
        if M >= 128:
            ref_out = deep_gemm_w8a8_block_fp8_moe(M, K, a, w1, w2, w1_s, w2_s,
                                                   score, topk, block_size)
        else:
            ref_out = torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, score,
                                               topk, block_size)

        topk_weights, topk_ids, token_expert_indices = fused_topk(
            a, score.float(), topk, False)

        out = deep_gemm_moe_fp8(a, w1, w2, w1_s, w2_s, topk_weights, topk_ids)

    #print(f"{out.sum()=}")
    #print(f"{ref_out.sum()=}")

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))

    assert rel_diff < 0.03
