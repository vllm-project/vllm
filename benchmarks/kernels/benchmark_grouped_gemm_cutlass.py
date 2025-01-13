from typing import List, Tuple

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES_MOE

from vllm import _custom_ops as ops
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = ["nm-testing/Mixtral-8x7B-Instruct-v0.1"]
                #   "nm-testing/deepseekv2-lite",
                #   "ibm-granite/granite-3.0-1b-a400m",
                #   "ibm-granite/granite-3.0-3b-a800m"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]

NUM_GROUPS_OPTS = [8]
PER_ACT_TOKEN_OPTS = [False, True]
PER_OUT_CH_OPTS = [False, True]

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

    def to_fp8(tensor: torch.Tensor):
        finfo = torch.finfo(torch.float8_e4m3fn)
        return torch.round(tensor.clamp(
            min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)
    
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
                        bench_run(results, model, num_groups, per_act_token,
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
