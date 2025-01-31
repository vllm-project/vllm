import pytest
import torch

from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import (cutlass_moe,
                                                            fused_topk)
from vllm.platforms import current_platform

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]


def run(a_q: torch.Tensor, a_scale: torch.Tensor, w1_q: torch.Tensor,
        w2_q: torch.Tensor, w1_scale: torch.Tensor, w2_scale: torch.Tensor,
        topk_weights: torch.Tensor, topk_ids: torch.Tensor, m: int, n: int,
        k: int, e: int):
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):
        return cutlass_moe(a_q, a_scale, w1_q, w2_q, w1_scale, w2_scale,
                           topk_weights, topk_ids, m, n, k, e)


@pytest.mark.parametrize("m", [2, 16, 32, 64, 224])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
def test_cutlass_moe_no_graph(
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

        w1_q = torch.empty((e, 2 * n, k),
                           device="cuda",
                           dtype=torch.float8_e4m3fn)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty((e, 1, 1), device="cuda", dtype=torch.float32)
        w2_scale = torch.empty((e, 1, 1), device="cuda", dtype=torch.float32)

        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(w1[expert])
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(w2[expert])
        w1_q = w1_q.transpose(1, 2)
        w2_q = w2_q.transpose(1, 2)
        a_d = (a_q.float() * a_scale).half()
        w1_d = (w1_q.transpose(1, 2).float() * w1_scale).half()
        w2_d = (w2_q.transpose(1, 2).float() * w2_scale).half()

        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].t().float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].t().float() * w2_scale[expert]).half()

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        torch_output = torch_moe(a_d, w1_d, w2_d, score, topk)
        cutlass_output = cutlass_moe(a_q, a_scale, w1_q, w2_q, w1_scale,
                                     w2_scale, topk_weights, topk_ids, m, n, k,
                                     e)

        print(torch_output)
        print(cutlass_output)
        print("*")

        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)


@pytest.mark.parametrize("m", [2, 16, 32, 64, 224])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
def test_cutlass_moe_cuda_graph(
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

        w1_q = torch.empty((e, 2 * n, k),
                           device="cuda",
                           dtype=torch.float8_e4m3fn)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty((e, 1, 1), device="cuda", dtype=torch.float32)
        w2_scale = torch.empty((e, 1, 1), device="cuda", dtype=torch.float32)

        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(w1[expert])
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(w2[expert])
        w1_q = w1_q.transpose(1, 2)
        w2_q = w2_q.transpose(1, 2)
        a_d = (a_q.float() * a_scale).half()
        w1_d = (w1_q.transpose(1, 2).float() * w1_scale).half()
        w2_d = (w2_q.transpose(1, 2).float() * w2_scale).half()

        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].t().float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].t().float() * w2_scale[expert]).half()

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        torch_output = torch_moe(a_d, w1_d, w2_d, score, topk)

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cutlass_output = run(a_q, a_scale, w1_q, w2_q, w1_scale, w2_scale,
                                 topk_weights, topk_ids, m, n, k, e)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        print(torch_output)
        print(cutlass_output)
        # print((cutlass_output - torch_output) / torch_output)
        print("*")

        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)
