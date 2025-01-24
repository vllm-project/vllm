from typing import List, Tuple

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES_MOE

from vllm import _custom_ops as ops
from vllm.utils import FlexibleArgumentParser
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_topk,
                                                            cutlass_moe,
                                                            fused_experts)

DEFAULT_MODELS = [
    "nm-testing/Mixtral-8x7B-Instruct-v0.1", "nm-testing/deepseekv2-lite",
    "ibm-granite/granite-3.0-1b-a400m", "ibm-granite/granite-3.0-3b-a800m"
]
DEFAULT_BATCH_SIZES = [16, 32, 64, 128, 256, 512]

NUM_GROUPS_OPTS = [8]  #[8, 64]
PER_ACT_TOKEN_OPTS = [False]  #[False, True]
PER_OUT_CH_OPTS = [False]  #[False, True]
TOPKS = [2, 6]


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


# TODO marlin baseline
def bench_run(results: List[benchmark.Measurement], model: str,
              num_experts: int, topk: int, per_act_token: bool,
              per_out_ch: bool, mkn: Tuple[int, int, int]):
    label = "Quant Matmul"

    sub_label = ("{}, num_experts={}, per_act_token={} per_out_ch={}, "
                 "MKN=({})".format(model, num_experts, per_act_token,
                                   per_out_ch, mkn))

    print(f"Testing: {sub_label}")

    (m, k, n) = mkn

    dtype = torch.half

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((num_experts, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device="cuda", dtype=dtype) / 10

    a_q, a_scale = ops.scaled_fp8_quant(a)

    w1_qs = []
    w2_qs = []
    w1_scales = []
    w2_scales = []

    for expert in range(num_experts):
        w1_q, w1_scale = ops.scaled_fp8_quant(w1[expert])
        w2_q, w2_scale = ops.scaled_fp8_quant(w2[expert])
        w1_qs.append(w1_q.t())
        w2_qs.append(w2_q.t())
        w1_scales.append(w1_scale.reshape((1, 1)))
        w2_scales.append(w2_scale.reshape((1, 1)))

    score = torch.randn((m, num_experts), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

    globals = {
        # Baseline params
        "a": a,
        "w1": w1,
        "w2": w2,
        "score": score,
        "topk": topk,
        # Cutlass params
        "a_q": a_q,
        "a_scale": a_scale,
        "w1_qs": w1_qs,
        "w2_qs": w2_qs,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "m": m,
        "n": n,
        "k": k,
        # Gen params
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        # Kernels
        "fused_experts": fused_experts,
        "cutlass_moe": cutlass_moe,
    }

    min_run_time = 1
    num_warmup = 5

    # Warmup pytorch
    for _ in range(num_warmup):
        fused_experts(a, w1, w2, topk_weights, topk_ids)

    results.append(
        benchmark.Timer(
            stmt="fused_experts(a, w1, w2, topk_weights, topk_ids)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="baseline_gemm",
        ).blocked_autorange(min_run_time=min_run_time))

    # Warmup pytorch
    for _ in range(num_warmup):
        cutlass_moe(a_q, a_scale, w1_qs, w2_qs, w1_scales, w2_scales,
                    topk_weights, topk_ids, m, n, k)

    results.append(
        benchmark.Timer(
            stmt=
            "cutlass_moe(a_q, a_scale, w1_qs, w2_qs, w1_scales, w2_scales, topk_weights, topk_ids, m, n, k)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="grouped_gemm",
        ).blocked_autorange(min_run_time=min_run_time))


def main(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    results: List[benchmark.Measurement] = []

    for model in args.models:
        for layer in WEIGHT_SHAPES_MOE[model]:
            num_experts = layer[0]
            size_k = layer[1]
            size_n = layer[2]

            if len(args.limit_k) > 0 and size_k not in args.limit_k:
                continue

            if len(args.limit_n) > 0 and size_n not in args.limit_n:
                continue

            for per_act_token in PER_ACT_TOKEN_OPTS:
                for per_out_ch in PER_OUT_CH_OPTS:
                    for topk in TOPKS:
                        for size_m in DEFAULT_BATCH_SIZES:
                            mkn = (size_m, size_k, size_n)
                            bench_run(results, model, num_experts, topk,
                                      per_act_token, per_out_ch, mkn)

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
    parser.add_argument("--limit-per-act-token",
                        nargs="+",
                        type=int,
                        default=[])
    parser.add_argument("--limit-per-out-ch", nargs="+", type=int, default=[])

    args = parser.parse_args()
    main(args)
