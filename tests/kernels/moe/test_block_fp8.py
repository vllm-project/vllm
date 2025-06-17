# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools

import pytest
import torch

from tests.kernels.quant_utils import (native_per_token_group_quant_fp8,
                                       native_w8a8_block_matmul,
                                       per_block_cast_to_fp8)
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
    per_token_group_quant_fp8)
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
# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
M = [1, 2, 83, 128, 2048, 40000]
M_dg = [128, 192, 1335, 2048]
N = [128, 256, 1024, 4608]  # [13824]
K = [256, 512, 7168]  # [13824]
BLOCK_SIZE = [[128, 128]]
E = [2, 8, 16, 24]  # [128, 256]
TOP_KS = [1, 2, 6]
OUT_DTYPES = [torch.bfloat16]  # [torch.float32, torch.half, torch.bfloat16]
SEEDS = [0]


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
    "M,N,K,E,topk,block_size,dtype,seed",
    itertools.product(M, N, K, E, TOP_KS, BLOCK_SIZE, DTYPES, SEEDS))
@torch.inference_mode()
def test_w8a8_block_fp8_fused_moe(M, N, K, E, topk, block_size, dtype, seed,
                                  monkeypatch):
    if topk > E:
        pytest.skip(f"Skipping test; topk={topk} > E={E}")

    torch.manual_seed(seed)

    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")

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
                                           per_act_token_quant=False,
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


@pytest.mark.parametrize("M,N,K,E,topk,seed",
                         itertools.product(M_dg, N, K, E, TOP_KS, SEEDS))
@pytest.mark.skipif(not dg_available, reason="DeepGemm kernels not available.")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_fused_moe(M, N, K, E, topk, seed,
                                            monkeypatch):
    if topk > E:
        pytest.skip(f"Skipping test: topk={topk} > E={E}")

    if not _valid_deep_gemm_shape(M, N, K):
        pytest.skip(f"Skipping test: invalid size m={M}, n={N}, k={K}")

    chunk_size = 1024

    torch.manual_seed(seed)

    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", str(chunk_size))

    block_m = deep_gemm.get_m_alignment_for_contiguous_layout()
    block_size = [block_m, block_m]
    dtype = torch.bfloat16

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

    use_compile = (chunk_size < M and N >= 1024 and K >= 1024
                   and current_platform.is_cuda_alike())
    use_cudagraph = use_compile

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

        if use_compile:
            deep_gemm_moe_fp8_fn = torch.compile(deep_gemm_moe_fp8,
                                                 backend="inductor",
                                                 fullgraph=True)
        else:
            deep_gemm_moe_fp8_fn = deep_gemm_moe_fp8

        out = deep_gemm_moe_fp8_fn(a, w1, w2, w1_s, w2_s, topk_weights,
                                   topk_ids)

        if use_cudagraph:
            out.fill_(0)
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = deep_gemm_moe_fp8_fn(a, w1, w2, w1_s, w2_s, topk_weights,
                                           topk_ids)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()

    #print(f"{out.sum()=}")
    #print(f"{ref_out.sum()=}")

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))

    assert rel_diff < 0.03
