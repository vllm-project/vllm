# SPDX-License-Identifier: Apache-2.0

import dataclasses
from itertools import product
from typing import Optional

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES_MOE

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_experts,
                                                            fused_topk)
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = [
    "nm-testing/Mixtral-8x7B-Instruct-v0.1", "nm-testing/deepseekv2-lite",
    "ibm-granite/granite-3.0-1b-a400m", "ibm-granite/granite-3.0-3b-a800m"
]
DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]

PER_ACT_TOKEN_OPTS = [False, True]
PER_OUT_CH_OPTS = [False, True]


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def is_16bit(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 2


def is_8bit(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1


@dataclasses.dataclass
class MOETensors:
    a: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_t: torch.Tensor  # Transposed w1 for cutlass_moe
    w2_t: torch.Tensor  # Transposed w2 for cutlass_moe
    ab_strides1: torch.Tensor
    c_strides1: torch.Tensor
    ab_strides2: torch.Tensor
    c_strides2: torch.Tensor
    # quantized
    a_q: Optional[torch.Tensor] = None  # a -> a_q
    w1_q: Optional[torch.Tensor] = None  # w1 -> w1_q
    w2_q: Optional[torch.Tensor] = None  # w2 -> w2_q
    a_scale: Optional[torch.Tensor] = None
    w1_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None

    @staticmethod
    def make_moe_tensors(in_dtype: torch.dtype, m: int, k: int, n: int, e: int,
                         per_act_token: bool,
                         per_out_channel: bool) -> "MOETensors":

        # For fp8, use torch.half to create 16bit tensors that can be later
        # quantized into fp8.
        dtype = in_dtype if is_16bit(in_dtype) else torch.half

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

        if is_16bit(in_dtype):
            assert not (per_act_token or per_out_channel)
            return MOETensors(a=a,
                              w1=w1,
                              w2=w2,
                              w1_t=w1.transpose(1, 2),
                              w2_t=w2.transpose(1, 2),
                              ab_strides1=ab_strides1,
                              c_strides1=c_strides1,
                              ab_strides2=ab_strides2,
                              c_strides2=c_strides2)

        assert in_dtype == torch.float8_e4m3fn
        q_dtype = torch.float8_e4m3fn
        # a -> a_q, w1 -> w1_q, w2 -> w2_q
        n_b_scales = 2 * n if per_out_channel else 1
        k_b_scales = k if per_out_channel else 1
        # Get the right scale for tests.
        _, a_scale = ops.scaled_fp8_quant(
            a, use_per_token_if_dynamic=per_act_token)
        a_q, _ = ops.scaled_fp8_quant(a,
                                      a_scale,
                                      use_per_token_if_dynamic=per_act_token)
        w1_q = torch.empty((e, 2 * n, k), device="cuda", dtype=q_dtype)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=q_dtype)

        w1_scale = torch.empty((e, n_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                w1[expert], use_per_token_if_dynamic=per_out_channel)
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                w2[expert], use_per_token_if_dynamic=per_out_channel)

        return MOETensors(a=a,
                          w1=w1,
                          w2=w2,
                          w1_t=w1.transpose(1, 2),
                          w2_t=w2.transpose(1, 2),
                          ab_strides1=ab_strides1,
                          c_strides1=c_strides1,
                          ab_strides2=ab_strides2,
                          c_strides2=c_strides2,
                          a_q=a_q,
                          w1_q=w1_q,
                          w2_q=w2_q,
                          a_scale=a_scale,
                          w1_scale=w1_scale,
                          w2_scale=w2_scale)

    def as_8bit_tensors(self) -> "MOETensors":
        assert all([
            x is not None for x in
            [self.w1_q, self.w2_q, self.w1_scale, self.w2_scale, self.a_scale]
        ])
        return MOETensors(a=self.a,
                          w1=self.w1_q,
                          w2=self.w2_q,
                          w1_t=self.w1_q.transpose(1, 2),
                          w2_t=self.w2_q.transpose(1, 2),
                          ab_strides1=self.ab_strides1,
                          c_strides1=self.c_strides1,
                          ab_strides2=self.ab_strides2,
                          c_strides2=self.c_strides2,
                          a_q=None,
                          w1_q=None,
                          w2_q=None,
                          a_scale=self.a_scale,
                          w1_scale=self.w1_scale,
                          w2_scale=self.w2_scale)

    def as_16bit_tensors(self) -> "MOETensors":
        return MOETensors(a=self.a,
                          w1=self.w1,
                          w2=self.w2,
                          w1_t=self.w1.transpose(1, 2),
                          w2_t=self.w2.transpose(1, 2),
                          ab_strides1=self.ab_strides1,
                          c_strides1=self.c_strides1,
                          ab_strides2=self.ab_strides2,
                          c_strides2=self.c_strides2,
                          a_q=None,
                          w1_q=None,
                          w2_q=None,
                          a_scale=None,
                          w1_scale=None,
                          w2_scale=None)


def bench_run(results: list[benchmark.Measurement], dtype: torch.dtype,
              model: str, num_experts: int, topk: int, per_act_token: bool,
              per_out_ch: bool, mkn: tuple[int, int, int]):
    label = "Quant Matmul" if dtype == torch.float8_e4m3fn else "Matmul"

    sub_label = (
        "{}, num_experts={}, topk={}, per_act_token={} per_out_ch={}, "
        "MKN=({})".format(model, num_experts, topk, per_act_token, per_out_ch,
                          mkn))

    print(f"Testing: {sub_label}")

    (m, k, n) = mkn
    tensors = MOETensors.make_moe_tensors(dtype,
                                          m=m,
                                          k=k,
                                          n=n,
                                          e=num_experts,
                                          per_act_token=per_act_token,
                                          per_out_channel=per_out_ch)
    tensors = tensors.as_8bit_tensors() if is_8bit(
        dtype) else tensors.as_16bit_tensors()

    score_dtype = torch.half if is_8bit(dtype) else dtype
    score = torch.randn((m, num_experts), device="cuda", dtype=score_dtype)
    topk_weights, topk_ids = fused_topk(tensors.a,
                                        score,
                                        topk,
                                        renormalize=False)

    def run_triton_moe(tensors: MOETensors, topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor, num_repeats: int):
        use_fp8_w8a8 = (tensors.a_scale is not None
                        and tensors.w1_scale is not None)
        for _ in range(num_repeats):
            fused_experts(tensors.a,
                          tensors.w1,
                          tensors.w2,
                          topk_weights,
                          topk_ids,
                          use_fp8_w8a8=use_fp8_w8a8,
                          w1_scale=tensors.w1_scale,
                          w2_scale=tensors.w2_scale,
                          a1_scale=tensors.a_scale,
                          a2_scale=tensors.a_scale)

    def run_cutlass_moe(tensors: MOETensors, topk_weights: torch.Tensor,
                        topk_ids: torch.Tensor, num_repeats: int):
        for _ in range(num_repeats):
            cutlass_moe(tensors.a,
                        tensors.w1_t,
                        tensors.w2_t,
                        topk_weights,
                        topk_ids,
                        tensors.ab_strides1,
                        tensors.c_strides1,
                        tensors.ab_strides2,
                        tensors.c_strides2,
                        w1_scale=tensors.w1_scale,
                        w2_scale=tensors.w2_scale,
                        a1_scale=tensors.a_scale)

    def run_cutlass_from_graph(tensors: MOETensors, topk_weights: torch.Tensor,
                               topk_ids: torch.Tensor):
        with set_current_vllm_config(
                VllmConfig(parallel_config=ParallelConfig(
                    pipeline_parallel_size=1))):
            return cutlass_moe(tensors.a,
                               tensors.w1_t,
                               tensors.w2_t,
                               topk_weights,
                               topk_ids,
                               tensors.ab_strides1,
                               tensors.c_strides1,
                               tensors.ab_strides2,
                               tensors.c_strides2,
                               w1_scale=tensors.w1_scale,
                               w2_scale=tensors.w2_scale,
                               a1_scale=tensors.a_scale)

    def run_triton_from_graph(tensors: MOETensors, topk_weights: torch.Tensor,
                              topk_ids: torch.Tensor):
        use_fp8_w8a8 = (tensors.a_scale is not None
                        and tensors.w1_scale is not None)
        with set_current_vllm_config(
                VllmConfig(parallel_config=ParallelConfig(
                    pipeline_parallel_size=1))):
            return fused_experts(tensors.a,
                                 tensors.w1,
                                 tensors.w2,
                                 topk_weights,
                                 topk_ids,
                                 use_fp8_w8a8=use_fp8_w8a8,
                                 w1_scale=tensors.w1_scale,
                                 w2_scale=tensors.w2_scale,
                                 a1_scale=tensors.a_scale,
                                 a2_scale=tensors.a_scale)

    def replay_graph(graph, num_repeats):
        for _ in range(num_repeats):
            graph.replay()
        torch.cuda.synchronize()

    cutlass_stream = torch.cuda.Stream()
    cutlass_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cutlass_graph, stream=cutlass_stream):
        run_cutlass_from_graph(tensors, topk_weights, topk_ids)
    torch.cuda.synchronize()

    if not per_act_token and not per_out_ch:
        triton_stream = torch.cuda.Stream()
        triton_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(triton_graph, stream=triton_stream):
            run_triton_from_graph(tensors, topk_weights, topk_ids)
        torch.cuda.synchronize()
    else:
        triton_graph = []

    min_run_time = 5
    num_warmup = 5
    num_runs = 25

    globals = {
        # Baseline params
        "score": score,
        "topk": topk,
        "tensors": tensors,
        # cuda graph params
        "cutlass_graph": cutlass_graph,
        "triton_graph": triton_graph,
        # Gen params
        "a": tensors.a,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "num_runs": num_runs,
        # Kernels
        "run_triton_moe": run_triton_moe,
        "run_cutlass_moe": run_cutlass_moe,
        "replay_graph": replay_graph,
    }

    if not per_act_token and not per_out_ch:
        # Warmup
        run_triton_moe(tensors, topk_weights, topk_ids, num_warmup)

        results.append(
            benchmark.Timer(
                stmt=
                "run_triton_moe(tensors, topk_weights, topk_ids, num_runs)",  # noqa: E501
                globals=globals,
                label=label,
                sub_label=sub_label,
                description="triton_moe",
            ).blocked_autorange(min_run_time=min_run_time))

        # Warmup
        replay_graph(triton_graph, num_warmup)

        results.append(
            benchmark.Timer(
                stmt="replay_graph(triton_graph, num_runs)",
                globals=globals,
                label=label,
                sub_label=sub_label,
                description="triton_moe_cuda_graphs",
            ).blocked_autorange(min_run_time=min_run_time))

    # Warmup
    run_cutlass_moe(tensors, topk_weights, topk_ids, num_warmup)

    results.append(
        benchmark.Timer(
            stmt=
            "run_cutlass_moe(tensors, topk_weights, topk_ids, num_runs)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="cutlass_moe",
        ).blocked_autorange(min_run_time=min_run_time))

    # Warmup
    replay_graph(cutlass_graph, num_warmup)

    results.append(
        benchmark.Timer(
            stmt="replay_graph(cutlass_graph, num_runs)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="cutlass_moe_cuda_graphs",
        ).blocked_autorange(min_run_time=min_run_time))


def main(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    results: list[benchmark.Measurement] = []

    quant_schemes = product(PER_ACT_TOKEN_OPTS, PER_OUT_CH_OPTS) if is_8bit(
        args.dtype) else [(False, False)]

    for model in args.models:
        for tp in args.tp_sizes:
            for layer in WEIGHT_SHAPES_MOE[model]:
                num_experts = layer[0]
                topk = layer[1]
                size_k = layer[2]
                size_n = layer[3] // tp

                if len(args.limit_k) > 0 and size_k not in args.limit_k:
                    continue

                if len(args.limit_n) > 0 and size_n not in args.limit_n:
                    continue

                for per_act_token, per_out_ch in quant_schemes:
                    for size_m in args.batch_sizes:
                        mkn = (size_m, size_k, size_n)
                        bench_run(results, args.dtype, model, num_experts,
                                  topk, per_act_token, per_out_ch, mkn)

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":

    def str_to_dtype(dtype_str: str) -> torch.dtype:
        if dtype_str == "fp8":
            return torch.float8_e4m3fn
        if dtype_str == "fp16":
            return torch.float16
        if dtype_str == "bf16":
            return torch.bfloat16
        raise ValueError(f"Unrecognized dtype str {dtype_str}")

    parser = FlexibleArgumentParser(description="""
        Benchmark Cutlass MOE layer against Triton MOE Layer. \n
        Example : python3 benchmarks/kernels/benchmark_cutlass_moe.py 
                     --dtype bf16 
                     --models nm-testing/Mixtral-8x7B-Instruct-v0.1 
                     --batch-sizes 1 16 32
                     """)
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES_MOE.keys(),
    )
    parser.add_argument("--dtype",
                        type=str_to_dtype,
                        required=True,
                        help="Please choose one from fp8, fp16 or bf16")
    parser.add_argument("--tp-sizes",
                        nargs="+",
                        type=int,
                        default=DEFAULT_TP_SIZES)
    parser.add_argument("--batch-sizes",
                        nargs="+",
                        type=int,
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--limit-k", nargs="+", type=int, default=[])
    parser.add_argument("--limit-n", nargs="+", type=int, default=[])
    parser.add_argument("--limit-num-groups", nargs="+", type=int, default=[])
    parser.add_argument("--limit-per-act-token",
                        nargs="+",
                        type=int,
                        default=[])
    parser.add_argument("--limit-per-out-ch", nargs="+", type=int, default=[])

    args = parser.parse_args()
    main(args)
