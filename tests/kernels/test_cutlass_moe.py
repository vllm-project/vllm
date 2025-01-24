import pytest
import torch

from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk, cutlass_moe
from vllm.platforms import current_platform
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]


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
