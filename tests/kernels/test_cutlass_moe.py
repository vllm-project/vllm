import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from typing import List

import vllm.model_executor.layers.fused_moe  # noqa
from tests.kernels.utils import (compute_max_diff, opcheck, stack_and_dev,
                                 torch_moe, torch_moe_single)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size)
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]

# TODO move to a better file later
# TODO handle scores
def cutlass_moe(a: torch.Tensor,
                a_q: torch.Tensor,
                a_scale: torch.Tensor,
                w1_qs: List[torch.Tensor],
                w2_qs: List[torch.Tensor],
                w1_scales: List[torch.Tensor],
                w2_scales: List[torch.Tensor],
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                m: int, n: int, k: int,
):
    # TODO look at the code in benchmark_grouped_gemm_cutlass.py
    # and get the relevant parts
    # (also the fused_moe function)

    # print(a.shape, a_scale.shape)
    # print(w1_qs[0].shape, w1_scales[0].shape)
    # print(w2_qs[0].shape, w2_scales[0].shape)

    num_groups = len(w1_qs)
    topk = topk_ids.shape[1]
    num_tokens = topk_ids.shape[0]
    # print("tk_cut:", topk_ids)

    # TODO make this GPU only
    # occurrences = [0] * num_groups
    # expert_offsets = [0] * (num_groups + 1)
    # for id in topk_ids.cpu().flatten():
    #     occurrences[id] += 1
    # for e in range(num_groups):
    #     expert_offsets[e + 1] = expert_offsets[e] + occurrences[e]

    # TODO duplicate A rows topk times
    # compute sorted_token_ids (argsort?)
    # shuffle A according to this so each group input is contiguous

    # TODO
    # get a_ptrs = a + expert_indices[:-1]

    a_ptrs = torch.empty((num_groups), dtype=torch.int64, device="cuda")
    expert_offsets = torch.empty((num_groups + 1), dtype=torch.int64, device="cuda")
    # TODO might need to call it from inside cutlass code?
    # help(ops)

    # print(a_ptrs)
    # print(rep_a_q)
    # print(topk_ids)
    # print(expert_offsets)
    # print(num_groups)

    # print(topk_ids)
    a_map = topk_ids.flatten().argsort()
    rep_a_q = a_q.repeat_interleave(topk, dim=0)

    torch.ops._C.compute_expert_offsets(a_ptrs, rep_a_q, topk_ids.cuda(),
                                        expert_offsets, num_groups)
    # print(expert_offsets)
    # print(a_ptrs)
    # print(expert_offsets)

    # print("a_map:", a_map)
    # print("rep_a_q:", rep_a_q)

    a_q_s = []
    a_scales_s = []
    c_s1 = []
    c_s2 = []
    for e in range(num_groups):
        expert_map = a_map[expert_offsets[e]:expert_offsets[e+1]]
        cut_out = rep_a_q.view(dtype=torch.uint8)[expert_map].view(
            dtype=a_q.dtype)
        a_q_s.append(cut_out.clone())
        # print("CU:", expert_map, cut_out)
        #TODO if we have 1 scale per token, we need to do a_scale[expert_map]
        a_scales_s.append(a_scale.clone())
        c_s1.append(torch.zeros((cut_out.shape[0], n * 2), device="cuda",
                               dtype=torch.half))
        c_s2.append(torch.zeros((cut_out.shape[0], k), device="cuda",
                               dtype=torch.half))
    # print("a_q_s:", a_q_s[0].shape)
    # print("a_scales_s:", a_scales_s[0].shape)
    # print("cs:", c_s[0].shape)
    # print("w1_qs:", w1_qs[0].shape)
    # print("w1_scales", w1_scales[0].shape)

    # print("a_q_s:", a_q_s)
    # print("a_scales_s:", a_scales_s)
    # print(w1_qs)
    # print(w1_scales)
    torch.ops._C.cutlass_grouped_mm(c_s1, a_q_s, w1_qs,
                                    a_scales_s, w1_scales)
    # c_s1 = [c.reshape((-1, n)) for c in c_s1]
    # print([w.stride() for w in w1_qs])

    # print(c_s1)

    ### UNCOMMENT THIS TO DO ONLY A SINGLE MUL
    # intermediate1 = torch.empty((m * topk, n * 2), device="cuda", dtype=torch.half)
    # for e in range(num_groups):
    #     expert_map = a_map[expert_offsets[e]:expert_offsets[e+1]]
    #     intermediate1[expert_map] = c_s1[e]
    # return intermediate1.reshape(m, topk, n * 2).sum(dim=1)
    ###

    # # print(out)
    # intermediate2 = torch.empty((m * topk, n), device="cuda", dtype=torch.half)
    # torch.ops._C.silu_and_mul(intermediate2, intermediate1)

    intermediate2 = []
    intermediate2_scales = []
    for e in range(num_groups):
        inter2 = torch.empty((c_s1[e].shape[0], n), device="cuda", dtype=torch.half)
        torch.ops._C.silu_and_mul(inter2, c_s1[e])
        inter2_v, inter2_s = ops.scaled_fp8_quant(inter2)
        # print("cutlass:", inter2)
        intermediate2.append(inter2_v)
        intermediate2_scales.append(inter2_s.reshape((1, 1)))

    # print(m, k, n, a_q_s[0].shape, w2_qs[0].shape, "->", intermediate2[0].shape)
    # print(m, k, n, intermediate2[0].shape, w2_qs[0].shape, intermediate2_scales[0].shape, w2_scales[0].shape)
    torch.ops._C.cutlass_grouped_mm(c_s2, intermediate2, w2_qs,
                                    intermediate2_scales, w2_scales)
    # print("cutlass:", c_s2)
    intermediate3 = torch.empty((m * topk, k), device="cuda", dtype=torch.half)
    for e in range(num_groups):
        expert_map = a_map[expert_offsets[e]:expert_offsets[e+1]]
        intermediate3[expert_map] = c_s2[e]
    
    # print("cutlass:", intermediate3.view(m, topk, k))
    # print("cutlass:", topk_weights.view(m, topk, 1).half())
    out = (intermediate3.reshape(m, topk, k) *
           topk_weights.view(m, topk, 1).half()).sum(dim=1)
    # return intermediate3.reshape(m, topk, k).sum(dim=1)
    return out

# @pytest.mark.parametrize("m", [1, 33, 64, 222])
# @pytest.mark.parametrize("n", [128, 2048])
# @pytest.mark.parametrize("k", [128, 1024])
# @pytest.mark.parametrize("e", NUM_EXPERTS)
# @pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("m", [16])
@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("k", [16])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
def test_cutlass_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
):
    current_platform.seed_everything(7)

    dtype = torch.half

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    a_q, a_scale = ops.scaled_fp8_quant(a)

    # print(a)
    # print(a_q)
    # print(a_scale)

    w1_qs = []
    w2_qs = []
    w1_scales = []
    w2_scales = []

    for expert in range(e):
        w1_q, w1_scale = ops.scaled_fp8_quant(w1[expert])
        w2_q, w2_scale = ops.scaled_fp8_quant(w2[expert])
        w1_qs.append(w1_q.t())
        w2_qs.append(w2_q.t())
        w1_scales.append(w1_scale.reshape((1, 1)))
        w2_scales.append(w2_scale.reshape((1, 1)))

    # (assume score is a vector of ones for now)
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    # e_range = torch.full((m, e), 1.0 / e)
    # topk_ids = torch.multinomial(e_range, topk).int().sort()[0]
    # topk_weights = torch.rand((m, topk))

    topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

    # torch_output = torch_moe(a, w1, w2, score, topk)
    a_d = (a_q.float() * a_scale).half()
    w1_d = torch.empty_like(w1)
    w2_d = torch.empty_like(w2)
    for expert in range(e):
        w1_d[expert] = (w1_qs[expert].t().float() * w1_scales[expert]).half()
        w2_d[expert] = (w2_qs[expert].t().float() * w2_scales[expert]).half()
    torch_output = torch_moe(a_d, w1_d, w2_d, score, topk)
    # torch_output = torch_moe_single(a_d, w1_d, score, topk)
    cutlass_output = cutlass_moe(a, a_q, a_scale, w1_qs, w2_qs, w1_scales,
                                 w2_scales, topk_weights, topk_ids,
                                 m, n, k)
    
    # print(torch_output.shape)
    # print(cutlass_output.shape)
    print(torch_output)
    print(cutlass_output)
    print(torch_output / cutlass_output)

    torch.testing.assert_close(torch_output,
                               cutlass_output,
                               atol=5e-2,
                               rtol=1e-2)
