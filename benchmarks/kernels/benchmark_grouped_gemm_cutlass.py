from typing import List, Tuple

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES_MOE

from vllm import _custom_ops as ops
from vllm.utils import FlexibleArgumentParser
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe, fused_topk, fused_experts)

DEFAULT_MODELS = ["nm-testing/Mixtral-8x7B-Instruct-v0.1"]
                #   "nm-testing/deepseekv2-lite",
                #   "ibm-granite/granite-3.0-1b-a400m",
                #   "ibm-granite/granite-3.0-3b-a800m"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]

NUM_GROUPS_OPTS = [8]
PER_ACT_TOKEN_OPTS = [False, True]
PER_OUT_CH_OPTS = [False, True]

def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)

def grouped_gemm(a_g_tensors: List[torch.Tensor],
                 b_g_tensors: List[torch.Tensor],
                 out_g_tensors: List[torch.Tensor],
                 a_scales_tensors: List[torch.Tensor],
                 b_scales_tensors: List[torch.Tensor]):
    ops.cutlass_grouped_mm(out_g_tensors, a_g_tensors, b_g_tensors,
                           a_scales_tensors, b_scales_tensors)

def baseline_gemm(num_groups: int, a_tensors: List[torch.Tensor],
                  b_tensors: List[torch.Tensor],
                  out_tensors: List[torch.Tensor]):
    for g in range(num_groups):
        a = a_tensors[g]
        b = b_tensors[g]
        out = torch.mm(a, b)
        out_tensors[g] = out

def cutlass_fused(a_tensors: List[torch.Tensor],
                  w1_tensors: List[torch.Tensor],
                  w2_tensors: List[torch.Tensor],
                  c1_tensors: List[torch.Tensor],
                  c2_tensors: List[torch.Tensor],
                  c2_tensors_fp8: List[torch.Tensor],
                  c3_tensors: List[torch.Tensor],
                  a_scales: List[torch.Tensor],
                  w1_scales: List[torch.Tensor],
                  w2_scales: List[torch.Tensor],
                  c2_scales: List[torch.Tensor],
                  num_groups: int):
    # output_dtype = c3_tensors[0].dtype
    N = c2_tensors[0].shape[1]
    ops.cutlass_grouped_mm(c1_tensors, a_tensors, w1_tensors,
                           a_scales, w1_scales)
    # TODO make this work as it should
    for idx in range(num_groups):
        torch.ops._C.silu_and_mul(c2_tensors[idx], c1_tensors[idx].view(-1, N))
        print(c2_tensors[idx])
        c2_tensors_fp8[idx] = to_fp8(c2_tensors[idx].half())
    ops.cutlass_grouped_mm(c3_tensors, c2_tensors, w2_tensors,
                           c2_scales, w2_scales)
        
def bench_run(results: List[benchmark.Measurement], model: str, num_groups: int,
              per_act_token: bool, per_out_ch: bool,
              mkn: List[Tuple[int, int, int]]):
    label = "Quant Matmul"

    sub_label = ("{}, num_groups={}, per_act_token={} per_out_ch={}, "
                 "MKN=({})".format(model, num_groups, per_act_token,
                                         per_out_ch, mkn))

    print(f"Testing: {sub_label}")

    device = "cuda"
    out_dtype = torch.half
    
    a_tensors = []
    b_tensors = []
    a_g_tensors = []
    b_g_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    out_tensors = []
    out_g_tensors = []
    baseline_tensors = []

    for g in range(num_groups):
        m_g = mkn[g][0]
        k_g = mkn[g][1]
        n_g = mkn[g][2]

        m_a_scales = m_g if per_act_token else 1
        n_b_scales = n_g if per_out_ch else 1

        a = torch.randn((m_g, k_g), device=device)
        b = torch.randn((n_g, k_g), device=device).t()
        c = torch.zeros((m_g, n_g), device=device, dtype=torch.bfloat16)

        a_g = to_fp8(a)
        b_g = to_fp8(b)
        c_g = torch.zeros((m_g, n_g), device=device, dtype=out_dtype)

        scale_a = (torch.randn((m_a_scales, 1), device=device,
                            dtype=torch.float32))
        scale_b = (torch.randn((1, n_b_scales), device=device,
                            dtype=torch.float32))

        a_tensors.append(a.to(dtype=torch.bfloat16))
        b_tensors.append(b.to(dtype=torch.bfloat16))
        out_tensors.append(c)
        a_g_tensors.append(a_g)
        b_g_tensors.append(b_g)
        out_g_tensors.append(c_g)
        baseline_tensors.append(c_g)
        a_scales_tensors.append(scale_a)
        b_scales_tensors.append(scale_b)

    globals = {
        # Gen params
        "a_tensors": a_tensors,
        "b_tensors": b_tensors,
        "a_g_tensors": a_g_tensors,
        "b_g_tensors": b_g_tensors,
        "out_g_tensors": out_g_tensors,
        "out_tensors": out_tensors,
        "baseline_tensors": baseline_tensors,
        "a_scales_tensors": a_scales_tensors,
        "b_scales_tensors": b_scales_tensors,
        "num_groups": num_groups,
        # Kernels
        "grouped_gemm": grouped_gemm,
        "baseline_gemm": baseline_gemm,
    }

    min_run_time = 1
    num_warmup = 5

    # Warmup pytorch
    for _ in range(num_warmup):
        grouped_gemm(a_g_tensors, b_g_tensors, out_g_tensors, a_scales_tensors,
                     b_scales_tensors)

    results.append(
        benchmark.Timer(
            stmt="grouped_gemm(a_g_tensors, b_g_tensors, out_g_tensors, a_scales_tensors, b_scales_tensors)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="grouped_gemm",
        ).blocked_autorange(min_run_time=min_run_time))

    # Warmup pytorch
    for _ in range(num_warmup):
        baseline_gemm(num_groups, a_tensors, b_tensors, out_tensors)

    results.append(
        benchmark.Timer(
            stmt=
            "output = baseline_gemm(num_groups, a_tensors, b_tensors, out_tensors)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="baseline_gemm",
        ).blocked_autorange(min_run_time=min_run_time))

def bench_run_moe(results: List[benchmark.Measurement], model: str, num_groups: int,
              per_act_token: bool, per_out_ch: bool,
              mkn: List[Tuple[int, int, int]]):
    label = "Quant Matmul"

    sub_label = ("{}, num_groups={}, per_act_token={} per_out_ch={}, "
                 "MKN=({})".format(model, num_groups, per_act_token,
                                         per_out_ch, mkn))

    print(f"Testing: {sub_label}")

    device = "cuda"
    out_dtype = torch.bfloat16

    def to_fp8(tensor: torch.Tensor):
        finfo = torch.finfo(torch.float8_e4m3fn)
        return torch.round(tensor.clamp(
            min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)

    m_tot = sum([elem[0] for elem in mkn])
    k_g = mkn[0][1]
    n_g = mkn[0][2]
    
    a_tensors = []
    w1_tensors = []
    w2_tensors = []
    c1_tensors = []
    c2_tensors = []
    c2_tensors_fp8 = []
    c3_tensors = []
    a_scales = []
    w1_scales = []
    w2_scales = []
    c2_scales = []

    a = torch.randn((m_tot, k_g), device=device, dtype=out_dtype)
    w1 = torch.randn((num_groups, 2 * n_g, k_g), device=device, dtype=out_dtype)
    w2 = torch.randn((num_groups, k_g, n_g), device=device, dtype=out_dtype)
    scored_output = torch.randn((m_tot, num_groups), device="cuda", dtype=out_dtype)
    topk = 2
    # triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)

    #TODO grouped topk for deepseek
    topk_weights, topk_ids = fused_topk(a, scored_output, topk, renormalize=True)
    fused_experts(a, w1, w2, topk_weights, topk_ids)
    topk_ids_cpu = topk_ids.cpu()

    occurrences = [0] * num_groups
    expert_offsets = [0] * (num_groups + 1)
    for id in topk_ids_cpu.flatten():
        occurrences[id] += 1

    for e in range(num_groups):
        expert_offsets[e + 1] = expert_offsets[e] + occurrences[e]
    
    print(expert_offsets, m_tot)

    a = torch.randn((m_tot, k_g))
    a_group[0] = a[sorted_token_ids[0]]

    # TODO
    # create full input tensor m_tot x k_g x topk
    # get shuffle data like sorted_token_ids etc.
    # create view

    for g in range(num_groups):
        m_g = occurrences[g]
        a_g = to_fp8(torch.randn((m_g, k_g), device=device))
        w1_g = to_fp8(torch.randn((2 * n_g, k_g), device=device).t())
        w2_g = to_fp8(torch.randn((k_g, n_g), device=device).t())
        c1_g = torch.zeros((m_g, 2 * n_g), device=device, dtype=torch.bfloat16)
        c2_g = torch.zeros((m_g, n_g), device=device, dtype=torch.bfloat16)
        c2_g_fp8 = to_fp8(torch.zeros((m_g, n_g), device=device))
        c3_g = torch.zeros((m_g, k_g), device=device, dtype=torch.bfloat16)
        # m_a_scales = m_g if per_act_token else 1
        # n_b_scales = n_g if per_out_ch else 1
        m_scales = 1
        n2_scales = 1
        k_scales = 1
        scale_a = (torch.randn((m_scales, 1), device=device,
                               dtype=torch.float32))
        scale_w1 = (torch.randn((n2_scales, 1), device=device,
                               dtype=torch.float32))
        scale_w2 = (torch.randn((k_scales, 1), device=device,
                               dtype=torch.float32))
        scale_c2 = (torch.randn((m_scales, 1), device=device,
                               dtype=torch.float32))
        
        a_tensors.append(a_g)
        w1_tensors.append(w1_g)
        w2_tensors.append(w2_g)
        c1_tensors.append(c1_g)
        c2_tensors.append(c2_g)
        c2_tensors_fp8.append(c2_g_fp8)
        c3_tensors.append(c3_g)
        a_scales.append(scale_a)
        w1_scales.append(scale_w1)
        w2_scales.append(scale_w2)
        c2_scales.append(scale_c2)

    globals = {
        # Gen params
        "num_groups": num_groups,
        # Grouped gemm params
        "a_tensors": a_tensors,
        "w1_tensors": w1_tensors,
        "w2_tensors": w2_tensors,
        "c1_tensors": c1_tensors,
        "c2_tensors": c2_tensors,
        "c2_tensors_fp8": c2_tensors_fp8,
        "c3_tensors": c3_tensors,
        "a_scales": a_scales,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "c2_scales": c2_scales,
        # Triton params (fused_moe)
        "a": a,
        "w1": w1,
        "w2": w2,
        "scored_output": scored_output,
        "topk": topk,
        # Kernels
        "fused_moe": fused_moe,
        "cutlass_fused": cutlass_fused,
    }

    min_run_time = 1
    num_warmup = 5

    # Warmup triton
    for _ in range(num_warmup):
        fused_moe(a, w1, w2, scored_output, topk, renormalize=False)

    results.append(
        benchmark.Timer(
            stmt="fused_moe(a, w1, w2, scored_output, topk, renormalize=False)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="grouped_gemm",
        ).blocked_autorange(min_run_time=min_run_time))

    # Warmup cutlass
    for _ in range(num_warmup):
        cutlass_fused(a_tensors, w1_tensors, w2_tensors, c1_tensors, c2_tensors,
                      c2_tensors_fp8, c3_tensors, a_scales, w1_scales,
                      w2_scales, c2_scales, num_groups)

    results.append(
        benchmark.Timer(
            stmt=
            "cutlass_fused(a_tensors, w1_tensors, w2_tensors, c1_tensors, c2_tensors, c2_tensors_fp8, c3_tensors, a_scales, w1_scales, w2_scales, c2_scales, num_groups)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="baseline_gemm",
        ).blocked_autorange(min_run_time=min_run_time))

def main(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    results: List[benchmark.Measurement] = []

    for model in args.models:
        for layer in WEIGHT_SHAPES_MOE[model]:
            num_groups = layer[0]
            size_k = layer[1]
            size_n = layer[2]

            if len(args.limit_k) > 0 and size_k not in args.limit_k:
                continue

            if len(args.limit_n) > 0 and size_n not in args.limit_n:
                continue

            for per_act_token in PER_ACT_TOKEN_OPTS:
                for per_out_ch in PER_OUT_CH_OPTS:
                    for size_m in DEFAULT_BATCH_SIZES:
                        mkn = [(size_m, size_k, size_n)] * num_groups
                        bench_run_moe(results, model, num_groups, per_act_token,
                                    per_out_ch, mkn)

    compare = benchmark.Compare(results)
    compare.print()


# For quick benchmarking use:
#   python benchmark_marlin.py --batch-sizes 1 16 32 --limit-k 4096 --limit-n 4096 ...
#
if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark Marlin across specified models/shapes/batches")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES_MOE.keys(),
    )
    parser.add_argument("--batch-sizes",
                        nargs="+",
                        type=int,
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--limit-k", nargs="+", type=int, default=[])
    parser.add_argument("--limit-n", nargs="+", type=int, default=[])
    parser.add_argument("--limit-num-groups", nargs="+", type=int, default=[])
    parser.add_argument("--limit-per-act-token", nargs="+", type=int, default=[])
    parser.add_argument("--limit-per-out-ch", nargs="+", type=int, default=[])

    args = parser.parse_args()
    main(args)
