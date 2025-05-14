# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import pytest
import torch
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import vllm.model_executor.layers.fused_moe  # noqa
from tests.kernels.utils import opcheck, stack_and_dev, torch_moe
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_fp4_like)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize, marlin_quantize)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

NUM_EXPERTS = [8, 64]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]


@pytest.mark.parametrize("m", [1, 33, 64, 222, 1024 * 128])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [True, False])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    padding: bool,
):
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0,
                              e, (local_e, ),
                              device="cuda",
                              dtype=torch.int32)
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    torch_output = torch_moe(a, w1, w2, score, topk, e_map)
    iterative_output = iterative_moe(a,
                                     w1,
                                     w2,
                                     score,
                                     topk,
                                     global_num_experts=e,
                                     expert_map=e_map,
                                     renormalize=False)

    # Pad the weight if moe padding is enabled
    if padding:
        w1 = F.pad(w1, (0, 128), "constant", 0)[..., 0:-128]
        torch.cuda.empty_cache()
        w2 = F.pad(w2, (0, 128), "constant", 0)[..., 0:-128]
        torch.cuda.empty_cache()

    triton_output = fused_moe(a,
                              w1,
                              w2,
                              score,
                              topk,
                              global_num_experts=e,
                              expert_map=e_map,
                              renormalize=False)
    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    torch.testing.assert_close(iterative_output,
                               torch_output,
                               atol=2e-2,
                               rtol=0)


@pytest.mark.parametrize("m", [1, 32, 222])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("has_zp", [True, False])
@pytest.mark.parametrize("weight_bits", [4, 8])
def test_fused_moe_wn16(m: int, n: int, k: int, e: int, topk: int,
                        ep_size: int, dtype: torch.dtype, group_size: int,
                        has_zp: bool, weight_bits: int):
    print(m, n, k, e, topk, dtype, group_size, has_zp, weight_bits)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qweight = torch.empty((e, 2 * n, k // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size),
                            device="cuda",
                            dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size),
                            device="cuda",
                            dtype=dtype)
    w1_qzeros = torch.empty((e, 2 * n // pack_factor, k // group_size),
                            device="cuda",
                            dtype=torch.uint8)
    w2_qzeros = torch.empty((e, k // pack_factor, n // group_size),
                            device="cuda",
                            dtype=torch.uint8)

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w1, w1_ref, w1_qweight, w1_scales, w1_qzeros
        else:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w2, w2_ref, w2_qweight, w2_scales, w2_qzeros
        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False)
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T
        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
            if has_zp:
                qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

        w_ref[expert_id] = weight
        w_qweight[expert_id] = qweight
        w_scales[expert_id] = scales
        if has_zp:
            w_qzeros[expert_id] = qzeros

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0,
                              e, (local_e, ),
                              device="cuda",
                              dtype=torch.int32)
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1_ref = w1_ref[e_ids]
        w2_ref = w2_ref[e_ids]
        w1_qweight = w1_qweight[e_ids]
        w2_qweight = w2_qweight[e_ids]
        w1_scales = w1_scales[e_ids]
        w2_scales = w2_scales[e_ids]
        w1_qzeros = w1_qzeros[e_ids]
        w2_qzeros = w2_qzeros[e_ids]
    else:
        e_map = None

    triton_output = fused_moe(a,
                              w1_qweight,
                              w2_qweight,
                              score,
                              topk,
                              renormalize=False,
                              use_int4_w4a16=weight_bits == 4,
                              use_int8_w8a16=weight_bits == 8,
                              global_num_experts=e,
                              expert_map=e_map,
                              w1_scale=w1_scales,
                              w2_scale=w2_scales,
                              w1_zp=w1_qzeros if has_zp else None,
                              w2_zp=w2_qzeros if has_zp else None,
                              block_shape=[0, group_size])
    torch_output = torch_moe(a, w1_ref, w2_ref, score, topk, e_map)
    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False])
@torch.inference_mode()
def test_mixtral_moe(dtype: torch.dtype, padding: bool, use_rocm_aiter: bool,
                     monkeypatch):
    """Make sure our Mixtral MoE implementation agrees with the one from
    huggingface."""

    # clear the cache before every test
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        is_rocm_aiter_moe_enabled)
    is_rocm_aiter_moe_enabled.cache_clear()
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

        if dtype == torch.float32:
            pytest.skip("AITER ROCm test skip for float32")

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        params_dtype=dtype,
        tp_size=1,
        dp_size=1,
    ).cuda()

    # Load the weights
    vllm_moe.gate.weight.data[:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.experts.w13_weight[i][:] = torch.cat(weights, dim=0)
        vllm_moe.experts.w2_weight[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    hf_inputs = torch.randn((1, 64, config.hidden_size)).to(dtype).to("cuda")
    # vLLM uses 1D query [num_tokens, hidden_dim]
    vllm_inputs = hf_inputs.flatten(0, 1)

    # Pad the weight if moe padding is enabled
    if padding:
        vllm_moe.experts.w13_weight = Parameter(F.pad(
            vllm_moe.experts.w13_weight, (0, 128), "constant", 0)[..., 0:-128],
                                                requires_grad=False)
        torch.cuda.empty_cache()
        vllm_moe.experts.w2_weight = Parameter(F.pad(
            vllm_moe.experts.w2_weight, (0, 128), "constant", 0)[..., 0:-128],
                                               requires_grad=False)
        torch.cuda.empty_cache()

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(hf_inputs)
    vllm_states = vllm_moe.forward(vllm_inputs)

    mixtral_moe_tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    if use_rocm_aiter:
        # The values of rtol and atol are set based on the tests in ROCM AITER package. # noqa: E501
        # https://github.com/ROCm/aiter/blob/dfed377f4be7da96ca2d75ac0761f569676f7240/op_tests/test_moe.py#L174  # noqa: E501
        torch.testing.assert_close(hf_states.flatten(0, 1),
                                   vllm_states,
                                   rtol=0.01,
                                   atol=100)
    else:
        torch.testing.assert_close(hf_states.flatten(0, 1),
                                   vllm_states,
                                   rtol=mixtral_moe_tol[dtype],
                                   atol=mixtral_moe_tol[dtype])


def marlin_moe_generate_valid_test_cases():
    import itertools
    m_list = [1, 123, 666]
    n_list = [128, 1024]
    k_list = [256, 2048]
    e_list = [4, 12]
    topk_list = [2, 3]
    ep_size_list = [1, 4]
    dtype_list = [torch.half, torch.bfloat16]
    group_size_list = [-1, 16, 32, 128]
    act_order_list = [True, False]
    quant_type_list = [
        scalar_types.float4_e2m1f,
        scalar_types.float8_e4m3fn,
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.uint8b128,
    ]
    is_k_full_list = [True, False]

    all_combinations = itertools.product(m_list, n_list, k_list, e_list,
                                         topk_list, ep_size_list, dtype_list,
                                         group_size_list, act_order_list,
                                         quant_type_list, is_k_full_list)

    def is_invalid(m, n, k, e, topk, ep_size, dtype, group_size, act_order,
                   quant_type, is_k_full):

        if quant_type == scalar_types.float8_e4m3fn and \
                group_size not in [-1, 128]:
            return False
        if quant_type == scalar_types.float4_e2m1f and group_size != 16:
            return False
        if quant_type != scalar_types.float4_e2m1f and group_size == 16:
            return False

        # Filter act_order
        if act_order:
            if group_size in (-1, k, n):
                return False
            if quant_type not in [scalar_types.uint4b8]:
                return False
        elif not is_k_full:
            return False

        return True

    cases = []
    for case in all_combinations:
        if is_invalid(*case):
            cases.append(case)
    return cases


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(("m, n, k, e, topk, ep_size, dtype, group_size,"
                          "act_order, quant_type, is_k_full"),
                         marlin_moe_generate_valid_test_cases())
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_fused_marlin_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    group_size: int,
    act_order: bool,
    quant_type: ScalarType,
    is_k_full: bool,
):
    torch.cuda.manual_seed(0)
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    if quant_type == scalar_types.float8_e4m3fn:
        if group_size not in [-1, 128]:
            return
        if act_order:
            return

    # Filter act_order
    if act_order:
        if quant_type == scalar_types.float8_e4m3fn:
            return
        if group_size == -1:
            return
        if group_size in (k, n):
            return
        if has_zp:
            return
    else:
        if not is_k_full:
            return

    if quant_type == scalar_types.float4_e2m1f and group_size != 16:
        return
    if quant_type != scalar_types.float4_e2m1f and group_size == 16:
        return

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randperm(e, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    w_ref1_l = []
    qweight1_l = []
    scales1_l = []
    global_scale1_l = []
    zeros1_l = []
    g_idx1_l = []
    sort_indices1_l = []

    for i in range(w1.shape[0]):
        if quant_type == scalar_types.float4_e2m1f:
            w_ref1, qweight1, scales1, global_scale1 = \
                rand_marlin_weight_fp4_like(w1[i], group_size)

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            global_scale1_l.append(global_scale1)
        elif quant_type == scalar_types.float8_e4m3fn:
            w_ref1, qweight1, scales1 = marlin_quant_fp8_torch(
                w1[i], group_size)
            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
        elif has_zp:
            w_ref1, qweight1, scales1, zeros1 = awq_marlin_quantize(
                w1[i].transpose(1, 0), quant_type, group_size)

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            zeros1_l.append(zeros1)
        else:
            test_perm = torch.randperm(k)
            w_ref1, qweight1, scales1, g_idx1, sort_indices1, _ = \
                marlin_quantize(w1[i].transpose(1, 0), quant_type,
                                group_size, act_order, test_perm)

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            g_idx1_l.append(g_idx1)
            sort_indices1_l.append(sort_indices1)

    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweight1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    global_scale1 = stack_and_dev(global_scale1_l) if global_scale1_l else None
    g_idx1 = stack_and_dev(g_idx1_l) if g_idx1_l else None
    zeros1 = stack_and_dev(zeros1_l) if zeros1_l else None
    sort_indices1 = stack_and_dev(sort_indices1_l) if sort_indices1_l else None

    w_ref2_l = []
    qweight2_l = []
    scales2_l = []
    global_scale2_l = []
    zeros2_l = []
    g_idx2_l = []
    sort_indices2_l = []

    for i in range(w2.shape[0]):
        if quant_type == scalar_types.float4_e2m1f:
            w_ref2, qweight2, scales2, global_scale2 = \
                rand_marlin_weight_fp4_like(w2[i], group_size)

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            global_scale2_l.append(global_scale2)
        elif quant_type == scalar_types.float8_e4m3fn:
            w_ref2, qweight2, scales2 = marlin_quant_fp8_torch(
                w2[i], group_size)
            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
        elif has_zp:
            w_ref2, qweight2, scales2, zeros2 = awq_marlin_quantize(
                w2[i].transpose(1, 0), quant_type, group_size)

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            zeros2_l.append(zeros2)
        else:
            test_perm = torch.randperm(n)
            w_ref2, qweight2, scales2, g_idx2, sort_indices2, _ = \
                marlin_quantize(w2[i].transpose(1, 0), quant_type,
                                group_size, act_order, test_perm)

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            g_idx2_l.append(g_idx2)
            sort_indices2_l.append(sort_indices2)

    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweight2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    global_scale2 = stack_and_dev(global_scale2_l) if global_scale2_l else None
    g_idx2 = stack_and_dev(g_idx2_l) if g_idx2_l else None
    zeros2 = stack_and_dev(zeros2_l) if zeros2_l else None
    sort_indices2 = stack_and_dev(sort_indices2_l) if sort_indices2_l else None

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    torch_output = torch_moe(a, w_ref1, w_ref2, score, topk, e_map)

    marlin_output = torch.ops.vllm.fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=e_map,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        quant_type_id=quant_type.id,
        is_k_full=is_k_full)

    torch.testing.assert_close(marlin_output, torch_output, atol=5e-2, rtol=0)


def test_moe_align_block_size_opcheck():
    num_experts = 4
    block_size = 4
    topk_ids = torch.randint(0,
                             num_experts, (3, 4),
                             dtype=torch.int32,
                             device='cuda')

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    opcheck(torch.ops._moe_C.moe_align_block_size,
            (topk_ids, num_experts, block_size, sorted_ids, expert_ids,
             num_tokens_post_pad))
