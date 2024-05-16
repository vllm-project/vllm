import argparse

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQ_MARLIN_SUPPORTED_GROUP_SIZES, GPTQ_MARLIN_SUPPORTED_NUM_BITS)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MarlinWorkspace, marlin_quantize)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack, quantize_weights, sort_weights)

DEFAULT_MODELS = ["meta-llama/Llama-2-7b-hf/TP1"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]

ACT_ORDER_OPTS = [False, True]
K_FULL_OPTS = [False, True]


def bench_run(results, model, act_order, is_k_full, num_bits, group_size,
              size_m, size_k, size_n):
    label = "Quant Matmul"

    sub_label = ("{}, act={} k_full={}, b={}, g={}, "
                 "MKN=({}x{}x{})".format(model, act_order, is_k_full, num_bits,
                                         group_size, size_m, size_k, size_n))

    print(f"Testing: {sub_label}")

    a = torch.randn(size_m, size_k).to(torch.half).cuda()
    b = torch.rand(size_k, size_n).to(torch.half).cuda()

    a_tmp = (torch.zeros(size_m, size_k).to(torch.half).cuda())

    # Marlin quant
    (
        marlin_w_ref,
        marlin_q_w,
        marlin_s,
        marlin_g_idx,
        marlin_sort_indices,
        marlin_rand_perm,
    ) = marlin_quantize(b, num_bits, group_size, act_order)

    # GPTQ quant
    (w_ref, q_w, s, g_idx,
     rand_perm) = quantize_weights(b, num_bits, group_size, act_order)
    q_w_gptq = gptq_pack(q_w, num_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx"
    # so that group ids are increasing
    repack_sort_indices = torch.empty(0, dtype=torch.int, device=b.device)
    if act_order:
        (q_w, g_idx, repack_sort_indices) = sort_weights(q_w, g_idx)

    # Prepare
    marlin_workspace = MarlinWorkspace(size_n)

    globals = {
        "marlin_w_ref": marlin_w_ref,
        "marlin_q_w": marlin_q_w,
        "marlin_s": marlin_s,
        "marlin_g_idx": marlin_g_idx,
        "marlin_sort_indices": marlin_sort_indices,
        "marlin_rand_perm": marlin_rand_perm,
        "q_w_gptq": q_w_gptq,
        "repack_sort_indices": repack_sort_indices,
        "num_bits": num_bits,
        "group_size": group_size,
        "size_m": size_m,
        "size_n": size_n,
        "size_k": size_k,
        "is_k_full": is_k_full,
        "a": a,
        "a_tmp": a_tmp,
        "gptq_marlin_gemm": ops.gptq_marlin_gemm,
        "gptq_marlin_repack": ops.gptq_marlin_repack,
        "marlin_workspace": marlin_workspace,
    }

    min_run_time = 1

    # Warmup pytorch
    for i in range(5):
        torch.matmul(a, marlin_w_ref)

    results.append(
        benchmark.Timer(
            stmt="torch.matmul(a, marlin_w_ref)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="pytorch_gemm",
        ).blocked_autorange(min_run_time=min_run_time))

    results.append(
        benchmark.Timer(
            stmt=
            "output = gptq_marlin_gemm(a, marlin_q_w, marlin_s, marlin_g_idx, marlin_sort_indices, marlin_workspace.scratch, num_bits, size_m, size_n, size_k, is_k_full)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="gptq_marlin_gemm",
        ).blocked_autorange(min_run_time=min_run_time))

    results.append(
        benchmark.Timer(
            stmt=
            "q_res = gptq_marlin_repack(q_w_gptq, repack_sort_indices, size_k, size_n, num_bits)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="gptq_marlin_repack",
        ).blocked_autorange(min_run_time=min_run_time))


def main(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    results = []

    for model in args.models:
        for layer in WEIGHT_SHAPES[model]:
            size_k = layer[0]
            size_n = layer[1]

            if len(args.limit_k) > 0 and size_k not in args.limit_k:
                continue

            if len(args.limit_n) > 0 and size_n not in args.limit_n:
                continue

            for act_order in ACT_ORDER_OPTS:
                for is_k_full in K_FULL_OPTS:
                    for num_bits in GPTQ_MARLIN_SUPPORTED_NUM_BITS:
                        for group_size in GPTQ_MARLIN_SUPPORTED_GROUP_SIZES:
                            if len(
                                    args.limit_group_size
                            ) > 0 and group_size not in args.limit_group_size:
                                continue

                            # For act_order, the group_size must be less than
                            # size_k
                            if act_order and (group_size == size_k
                                              or group_size == -1):
                                continue

                            for size_m in args.batch_sizes:
                                bench_run(results, model, act_order, is_k_full,
                                          num_bits, group_size, size_m, size_k,
                                          size_n)

    compare = benchmark.Compare(results)
    compare.print()


# For quick benchmarking use:
#   python benchmark_marlin.py --batch-sizes 1 16 32 --limit-k 4096 --limit-n 4096 --limit-group-size 128 # noqa E501
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Marlin across specified models/shapes/batches")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    parser.add_argument("--batch-sizes",
                        nargs="+",
                        type=int,
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--limit-k", nargs="+", type=int, default=[])
    parser.add_argument("--limit-n", nargs="+", type=int, default=[])
    parser.add_argument("--limit-group-size", nargs="+", type=int, default=[])

    args = parser.parse_args()
    main(args)
