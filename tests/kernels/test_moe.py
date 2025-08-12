"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import vllm.model_executor.layers.fused_moe  # noqa
from tests.kernels.utils import (compute_max_diff, opcheck, stack_and_dev,
                                 torch_moe, torch_moe_single)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


@pytest.mark.parametrize("m", [1024 * 128, 512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
    torch_output = torch_moe(a, w1, w2, score, topk)
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mixtral_moe(dtype: torch.dtype):
    """Make sure our Mixtral MoE implementation agrees with the one from
    huggingface."""

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

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(hf_inputs)
    vllm_states = vllm_moe.forward(vllm_inputs)

    mixtral_moe_tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    torch.testing.assert_close(hf_states.flatten(0, 1),
                               vllm_states,
                               rtol=mixtral_moe_tol[dtype],
                               atol=mixtral_moe_tol[dtype])


@pytest.mark.parametrize("m", [64, 512, 222, 33, 1])
@pytest.mark.parametrize("n", [128, 2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024, 512])
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("is_k_full", [True, False])
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_fused_marlin_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    act_order: bool,
    num_bits: int,
    is_k_full: bool,
):
    current_platform.seed_everything(7)

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size in (k, n):
            return
    else:
        if not is_k_full:
            return

    quant_type = (scalar_types.uint4b8
                  if num_bits == 4 else scalar_types.uint8b128)
    dtype = torch.float16
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    w_ref1_l = []
    qweight1_l = []
    scales1_l = []
    g_idx1_l = []
    sort_indices1_l = []

    for i in range(w1.shape[0]):
        test_perm = torch.randperm(k)
        w_ref1, qweight1, scales1, g_idx1, sort_indices1, _ = marlin_quantize(
            w1[i].transpose(1, 0), quant_type, group_size, act_order,
            test_perm)
        w_ref1_l.append(w_ref1)
        qweight1_l.append(qweight1)
        scales1_l.append(scales1)
        g_idx1_l.append(g_idx1)
        sort_indices1_l.append(sort_indices1)

    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweight1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    g_idx1 = stack_and_dev(g_idx1_l)
    sort_indices1 = stack_and_dev(sort_indices1_l)

    w_ref2_l = []
    qweight2_l = []
    scales2_l = []
    g_idx2_l = []
    sort_indices2_l = []

    for i in range(w2.shape[0]):
        test_perm = torch.randperm(n)
        w_ref2, qweight2, scales2, g_idx2, sort_indices2, _ = marlin_quantize(
            w2[i].transpose(1, 0), quant_type, group_size, act_order,
            test_perm)
        w_ref2_l.append(w_ref2)
        qweight2_l.append(qweight2)
        scales2_l.append(scales2)
        g_idx2_l.append(g_idx2)
        sort_indices2_l.append(sort_indices2)

    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweight2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    g_idx2 = stack_and_dev(g_idx2_l)
    sort_indices2 = stack_and_dev(sort_indices2_l)

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(a, score, topk, False)

    triton_output = fused_moe(
        a,
        w_ref1.transpose(1, 2).contiguous(),
        w_ref2.transpose(1, 2).contiguous(),
        score,
        topk,
        renormalize=False,
    )
    marlin_output = torch.ops.vllm.fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        num_bits=num_bits,
        is_k_full=is_k_full,
    )

    assert compute_max_diff(marlin_output, triton_output) < 4e-2

    if ops.supports_moe_ops:
        token_expert_indicies = torch.empty(m,
                                            topk,
                                            dtype=torch.int32,
                                            device=a.device)

        opcheck(torch.ops._moe_C.topk_softmax, (
            topk_weights,
            topk_ids,
            token_expert_indicies,
            score.float(),
        ))

        block_size_m = 4

        sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m,
                                                      e)

        max_workspace_size = ((m + 255) // 256) * (max(2 * n, k) // 64) * 16
        workspace = torch.zeros(max_workspace_size,
                                dtype=torch.int,
                                device="cuda",
                                requires_grad=False)

        zp = torch.empty((0, 0),
                         dtype=dtype,
                         device="cuda",
                         requires_grad=False)
        opcheck(torch.ops._moe_C.marlin_gemm_moe,
                (a, qweight1, sorted_token_ids, topk_weights, topk_ids,
                 scales1, zp, g_idx1, sort_indices1, workspace, quant_type.id,
                 m, 2 * n, k, True, e, topk, block_size_m, True, False))


@pytest.mark.skip("This test is here for the sake of debugging, "
                  "don't run it in automated tests.")
@pytest.mark.parametrize("m", [64, 512, 222, 33, 1])
@pytest.mark.parametrize("n", [128, 2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024, 512])
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("is_k_full", [True, False])
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_single_marlin_moe_multiply(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    act_order: bool,
    num_bits: int,
    is_k_full: bool,
):

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size == k:
            return
    else:
        if not is_k_full:
            return

    quant_type = (scalar_types.uint4b8
                  if num_bits == 4 else scalar_types.uint8b128)
    dtype = torch.float16
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10

    w_ref_l = []
    qweights_l = []
    scales_l = []
    g_idx_l = []
    sort_indices_l = []

    for i in range(w.shape[0]):
        test_perm = torch.randperm(k)
        w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
            w[i].transpose(1, 0), quant_type, group_size, act_order, test_perm)
        w_ref_l.append(w_ref)
        qweights_l.append(qweight)
        scales_l.append(scales)
        g_idx_l.append(g_idx)
        sort_indices_l.append(sort_indices)

    w_ref = stack_and_dev(w_ref_l)
    qweight = stack_and_dev(qweights_l).contiguous()
    scales = stack_and_dev(scales_l)
    g_idx = stack_and_dev(g_idx_l)
    sort_indices = stack_and_dev(sort_indices_l)

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    marlin_output = torch.ops.vllm.single_marlin_moe(
        a,
        qweight,
        scales,
        score,
        topk,
        renormalize=False,
        g_idx=g_idx,
        sort_indices=sort_indices,
        num_bits=num_bits,
        is_k_full=is_k_full,
    )

    torch_output = torch_moe_single(a, w_ref.transpose(1, 2), score, topk)

    assert compute_max_diff(marlin_output, torch_output) < 1e-2


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
