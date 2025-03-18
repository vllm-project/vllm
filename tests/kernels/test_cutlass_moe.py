# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import (cutlass_moe_fp8,
                                                            fused_experts,
                                                            fused_topk)
from vllm.platforms import current_platform

NUM_EXPERTS = [32, 40, 64]
TOP_KS = [6, 8]


def run(a_q: torch.Tensor, a_scale: torch.Tensor, w1_q: torch.Tensor,
        w2_q: torch.Tensor, w1_scale: torch.Tensor, w2_scale: torch.Tensor,
        topk_weights: torch.Tensor, topk_ids: torch.Tensor,
        ab_strides1: torch.Tensor, c_strides1: torch.Tensor,
        ab_strides2: torch.Tensor, c_strides2: torch.Tensor):
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):
        return cutlass_moe_fp8(a_q, a_scale, w1_q, w2_q, w1_scale, w2_scale,
                               topk_weights, topk_ids, ab_strides1, c_strides1,
                               ab_strides2, c_strides2)


@pytest.mark.parametrize("m", [2, 16, 32, 64, 224, 512, 163840])
@pytest.mark.parametrize("n", [1024, 2048, 3072])
@pytest.mark.parametrize("k", [1024, 1536, 2048])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()),
    reason="Grouped gemm is not supported on this GPU type.")
def test_cutlass_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
):
    current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):

        dtype = torch.half

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        a_q, a_scale = ops.scaled_fp8_quant(
            a, use_per_token_if_dynamic=per_act_token)

        n_b_scales = 2 * n if per_out_ch else 1
        k_b_scales = k if per_out_ch else 1

        w1_q = torch.empty((e, 2 * n, k),
                           device="cuda",
                           dtype=torch.float8_e4m3fn)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty((e, n_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)

        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                w1[expert], use_per_token_if_dynamic=per_out_ch)
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                w2[expert], use_per_token_if_dynamic=per_out_ch)
        w1_q = w1_q.transpose(1, 2)
        w2_q = w2_q.transpose(1, 2)
        a_d = (a_q.float() * a_scale).half()

        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].t().float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].t().float() * w2_scale[expert]).half()

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        torch_output = torch_moe(a_d, w1_d, w2_d, score, topk, None)
        cutlass_output = cutlass_moe_fp8(a_q, a_scale, w1_q, w2_q, w1_scale,
                                         w2_scale, topk_weights, topk_ids,
                                         ab_strides1, c_strides1, ab_strides2,
                                         c_strides2)

        print(torch_output)
        print(cutlass_output)
        print("*")

        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)


@pytest.mark.parametrize("m", [2, 16, 32, 64, 224, 512, 163840])
@pytest.mark.parametrize("n", [1024, 2048, 3072])
@pytest.mark.parametrize("k", [1024, 1536, 2048])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()),
    reason="Grouped gemm is not supported on this GPU type.")
def test_cutlass_moe_cuda_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
):
    current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):

        dtype = torch.half

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        a_q, a_scale = ops.scaled_fp8_quant(
            a, use_per_token_if_dynamic=per_act_token)

        n_b_scales = 2 * n if per_out_ch else 1
        k_b_scales = k if per_out_ch else 1

        w1_q = torch.empty((e, 2 * n, k),
                           device="cuda",
                           dtype=torch.float8_e4m3fn)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty((e, n_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)

        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                w1[expert], use_per_token_if_dynamic=per_out_ch)
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                w2[expert], use_per_token_if_dynamic=per_out_ch)
        w1_q = w1_q.transpose(1, 2)
        w2_q = w2_q.transpose(1, 2)
        a_d = (a_q.float() * a_scale).half()

        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].t().float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].t().float() * w2_scale[expert]).half()

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        torch_output = torch_moe(a_d, w1_d, w2_d, score, topk, None)

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cutlass_output = run(a_q, a_scale, w1_q, w2_q, w1_scale, w2_scale,
                                 topk_weights, topk_ids, ab_strides1,
                                 c_strides1, ab_strides2, c_strides2)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        print(torch_output)
        print(cutlass_output)
        print("*")

        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)


@pytest.mark.skip("profiling only")
@pytest.mark.parametrize("m", [2, 16, 32, 64, 224])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
def test_cutlass_moe_profile(
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
        w1_q_notransp = w1_q.clone()
        w2_q_notransp = w2_q.clone()
        w1_q = w1_q.transpose(1, 2)
        w2_q = w2_q.transpose(1, 2)

        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].t().float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].t().float() * w2_scale[expert]).half()

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        # ruff: noqa: SIM117
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True) as prof_cutlass:
            with torch.profiler.record_function("cutlass_output"):
                cutlass_output = cutlass_moe_fp8(a_q, a_scale, w1_q, w2_q,
                                                 w1_scale, w2_scale,
                                                 topk_weights, topk_ids,
                                                 ab_strides1, c_strides1,
                                                 ab_strides2, c_strides2)
        print("profile cutlass:")
        print(
            prof_cutlass.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=50))

        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True) as prof_triton:
            with torch.profiler.record_function("triton_output"):
                triton_output = fused_experts(a,
                                              w1_q_notransp,
                                              w2_q_notransp,
                                              topk_weights,
                                              topk_ids,
                                              use_fp8_w8a8=True,
                                              w1_scale=w1_scale,
                                              w2_scale=w2_scale,
                                              a1_scale=a_scale)

        print("profile triton:")
        print(
            prof_triton.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=50))

        # Uncomment to produce trace files
        # cutlass_trace_name = f"trace_cutlass-{m}x{n}x{k}-{e}x{topk}.json"
        # triton_trace_name = f"trace_triton-{m}x{n}x{k}-{e}x{topk}.json"
        # prof_cutlass.export_chrome_trace(cutlass_trace_name)
        # prof_triton.export_chrome_trace(triton_trace_name)

        torch.testing.assert_close(triton_output,
                                   cutlass_output,
                                   atol=5e-2,
                                   rtol=1e-2)
