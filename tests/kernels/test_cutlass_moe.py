import pytest
import torch

from typing import List

from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.platforms import current_platform
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]


# TODO move to a better file later
# TODO handle scores
def cutlass_moe(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    w1_qs: List[torch.Tensor],
    w2_qs: List[torch.Tensor],
    w1_scales: List[torch.Tensor],
    w2_scales: List[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    m: int,
    n: int,
    k: int,
):
    num_groups = len(w1_qs)
    topk = topk_ids.shape[1]

    a_ptrs = torch.empty((num_groups), dtype=torch.int64, device="cuda")
    expert_offsets = torch.empty((num_groups + 1),
                                 dtype=torch.int64,
                                 device="cuda")

    a_map = topk_ids.flatten().argsort()
    rep_a_q = a_q.repeat_interleave(topk, dim=0)

    torch.ops._C.compute_expert_offsets(a_ptrs, rep_a_q, topk_ids.cuda(),
                                        expert_offsets, num_groups)

    a_q_s = []
    a_scales_s = []
    c_s1 = []
    c_s2 = []
    for e in range(num_groups):
        expert_map = a_map[expert_offsets[e]:expert_offsets[e + 1]]
        cut_out = rep_a_q.view(dtype=torch.uint8)[expert_map].view(
            dtype=a_q.dtype)
        a_q_s.append(cut_out.clone())
        a_scales_s.append(a_scale.clone())
        c_s1.append(
            torch.zeros((cut_out.shape[0], n * 2),
                        device="cuda",
                        dtype=torch.half))
        c_s2.append(
            torch.zeros((cut_out.shape[0], k), device="cuda",
                        dtype=torch.half))

    torch.ops._C.cutlass_grouped_mm(c_s1, a_q_s, w1_qs, a_scales_s, w1_scales)

    # ### UNCOMMENT THIS TO DO ONLY A SINGLE MUL
    # intermediate1 = torch.empty((m * topk, n * 2),
    #                             device="cuda",
    #                             dtype=torch.half)
    # for e in range(num_groups):
    #     expert_map = a_map[expert_offsets[e]:expert_offsets[e+1]]
    #     intermediate1[expert_map] = c_s1[e]
    # return intermediate1.reshape(m, topk, n * 2).sum(dim=1)
    # ###

    full_groups = []

    intermediate2 = []
    intermediate2_scales = []
    for e in range(num_groups):
        if c_s1[e].shape[0] != 0:
            full_groups.append(e)
            inter2 = torch.empty((c_s1[e].shape[0], n),
                                 device="cuda",
                                 dtype=torch.half)
            torch.ops._C.silu_and_mul(inter2, c_s1[e])
            inter2_v, inter2_s = ops.scaled_fp8_quant(inter2)
            intermediate2.append(inter2_v)
            intermediate2_scales.append(inter2_s.reshape((1, 1)))

    def filter_list(items: List, idxs: List):
        return [items[idx] for idx in idxs]

    torch.ops._C.cutlass_grouped_mm(filter_list(c_s2,
                                                full_groups), intermediate2,
                                    filter_list(w2_qs, full_groups),
                                    intermediate2_scales,
                                    filter_list(w2_scales, full_groups))
    intermediate3 = torch.empty((m * topk, k), device="cuda", dtype=torch.half)
    for e in range(num_groups):
        expert_map = a_map[expert_offsets[e]:expert_offsets[e + 1]]
        intermediate3[expert_map] = c_s2[e]

    out = (intermediate3.reshape(m, topk, k) *
           topk_weights.view(m, topk, 1).half()).sum(dim=1)
    return out


@pytest.mark.parametrize("m", [16, 32, 64, 224])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
def test_cutlass_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
):
    current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):

        dtype = torch.half

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        a_q, a_scale = ops.scaled_fp8_quant(a)

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

        score = torch.randn((m, e), device="cuda", dtype=dtype)

        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        a_d = (a_q.float() * a_scale).half()
        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_qs[expert].t().float() *
                            w1_scales[expert]).half()
            w2_d[expert] = (w2_qs[expert].t().float() *
                            w2_scales[expert]).half()
        torch_output = torch_moe(a_d, w1_d, w2_d, score, topk)
        cutlass_output = cutlass_moe(a_q, a_scale, w1_qs, w2_qs, w1_scales,
                                     w2_scales, topk_weights, topk_ids, m, n,
                                     k)

        # print(torch_output)
        # print(cutlass_output)
        # print(torch_output / cutlass_output)

        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)
