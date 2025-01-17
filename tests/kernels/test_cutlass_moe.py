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
):
    # TODO look at the code in benchmark_grouped_gemm_cutlass.py
    # and get the relevant parts
    # (also the fused_moe function)

    num_groups = len(w1_qs)
    topk = topk_ids.shape[1]
    num_tokens = topk_ids.shape[0]

    # TODO make this GPU only
    occurrences = [0] * num_groups
    expert_offsets = [0] * (num_groups + 1)
    for id in topk_ids.cpu().flatten():
        occurrences[id] += 1
    for e in range(num_groups):
        expert_offsets[e + 1] = expert_offsets[e] + occurrences[e]

    # TODO duplicate A rows topk times
    # compute sorted_token_ids (argsort?)
    # shuffle A according to this so each group input is contiguous

    # print(topk_ids)
    # print(expert_offsets)
    a_map = topk_ids.flatten().argsort()
    rep_a_q = a_q.repeat_interleave(topk, dim=0)

    print(a_map)
    print(rep_a_q)

    a_q_s = []
    for e in range(num_groups):
        a_q_s.append(rep_a_q[a_map[expert_offsets[e]:expert_offsets[e+1]]])
    print(a_q_s)
    return
    # get a_map and expert_indices on device

    # TODO shuffle rep_a_q according to a_map
    # get a_ptrs = a + expert_indices[:-1]

    a_ptrs = torch.empty((num_groups), dtype=torch.int64, device="cuda")
    expert_offsets = torch.empty((num_groups + 1), dtype=torch.int64, device="cuda")
    # TODO might need to call it from inside cutlass code?
    # help(ops)

    # print(a_ptrs)
    # print(rep_a_q)
    print(topk_ids)
    # print(expert_offsets)
    # print(num_groups)
    torch.ops._C.compute_expert_offsets(a_ptrs, rep_a_q, topk_ids.cuda(),
                                        expert_offsets, num_groups)
    print(a_ptrs)
    print(expert_offsets)

# @pytest.mark.parametrize("m", [1, 33, 64, 222])
# @pytest.mark.parametrize("n", [128, 2048])
# @pytest.mark.parametrize("k", [128, 1024])
# @pytest.mark.parametrize("e", NUM_EXPERTS)
# @pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("k", [128])
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

    dtype = torch.bfloat16

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
        w1_qs.append(w1_q)
        w2_qs.append(w2_q)
        w1_scales.append(w1_scale)
        w2_scales.append(w2_scale)

    # (assume score is a vector of ones for now)
    score = torch.ones((m, e), device="cuda", dtype=dtype)

    e_range = torch.full((m, e), 1.0 / e)
    topk_ids = torch.multinomial(e_range, topk).int().sort()[0]
    topk_weights = torch.rand((m, topk))

    torch_output = torch_moe(a, w1, w2, score, topk)
    cutlass_output = cutlass_moe(a, a_q, a_scale, w1_qs, w2_qs, w1_scales,
                                 w2_scales, topk_weights, topk_ids)

    # torch.testing.assert_close(torch_output,
    #                            cutlass_output,
    #                            atol=2e-2,
    #                            rtol=0)
