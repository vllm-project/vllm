# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL,
    GPTQ_MARLIN_24_MIN_THREAD_N,
    GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES,
    GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES,
)
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
    ALLSPARK_SUPPORTED_QUANT_TYPES,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL,
    GPTQ_MARLIN_MIN_THREAD_N,
    MARLIN_SUPPORTED_GROUP_SIZES,
    query_marlin_supported_quant_types,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    marlin_24_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
    quantize_weights,
    sort_weights,
)
from vllm.scalar_type import ScalarType
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = ["meta-llama/Llama-2-7b-hf/TP1"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

ACT_ORDER_OPTS = [False, True]
K_FULL_OPTS = [False, True]


def bench_run(
    results: list[benchmark.Measurement],
    model: str,
    act_order: bool,
    is_k_full: bool,
    quant_type: ScalarType,
    group_size: int,
    size_m: int,
    size_k: int,
    size_n: int,
):
    label = "Quant Matmul"

    sub_label = "{}, act={} k_full={}, q={}, g={}, MKN=({}x{}x{})".format(
        model, act_order, is_k_full, str(quant_type), group_size, size_m, size_k, size_n
    )

    print(f"Testing: {sub_label}")

    a = torch.randn(size_m, size_k).to(torch.half).cuda()
    b = torch.rand(size_k, size_n).to(torch.half).cuda()

    a_tmp = torch.zeros(size_m, size_k).to(torch.half).cuda()

    # Marlin quant
    (
        marlin_w_ref,
        marlin_q_w,
        marlin_s,
        marlin_g_idx,
        marlin_sort_indices,
        marlin_rand_perm,
    ) = marlin_quantize(b, quant_type, group_size, act_order)

    # Marlin_24 quant
    (marlin_24_w_ref, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s) = (
        marlin_24_quantize(b, quant_type, group_size)
    )

    marlin_zp = torch.empty(0, dtype=torch.int, device=b.device)

    # GPTQ quant
    (w_ref, q_w, s, g_idx, rand_perm) = gptq_quantize_weights(
        b, quant_type, group_size, act_order
    )
    q_w_gptq = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx"
    # so that group ids are increasing
    repack_sort_indices = torch.empty(0, dtype=torch.int, device=b.device)
    if act_order:
        (q_w, g_idx, repack_sort_indices) = sort_weights(q_w, g_idx)

    # Prepare
    marlin_workspace = MarlinWorkspace(
        size_n, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
    )

    marlin_24_workspace = MarlinWorkspace(
        size_n, GPTQ_MARLIN_24_MIN_THREAD_N, GPTQ_MARLIN_24_MAX_PARALLEL
    )
    marlin_zp = torch.zeros_like(marlin_s, dtype=torch.int)

    # AllSpark W8A16 quant
    as_supported_case = (
        quant_type in ALLSPARK_SUPPORTED_QUANT_TYPES
        and group_size == -1
        and not act_order
        and is_k_full
    )
    if as_supported_case:
        properties = torch.cuda.get_device_properties(b.device.index)
        sm_count = properties.multi_processor_count
        sm_version = properties.major * 10 + properties.minor

        supported_arch = sm_version >= 80 and sm_version < 90
        as_supported_case = as_supported_case and supported_arch
        if supported_arch:
            has_zp = False
            w_ref, qw, s, zp = quantize_weights(b, quant_type, group_size, has_zp)
            qw = qw.to(torch.uint8)

            qw_reorder, s_reorder, zp_reorder = ops.allspark_repack_weight(
                qw, s, zp, has_zp
            )
            CUBLAS_M_THRESHOLD = ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD

    globals = {
        # Gen params
        "quant_type": quant_type,
        "group_size": group_size,
        "size_m": size_m,
        "size_n": size_n,
        "size_k": size_k,
        "a": a,
        "a_tmp": a_tmp,
        # Marlin params
        "marlin_w_ref": marlin_w_ref,
        "marlin_q_w": marlin_q_w,
        "marlin_s": marlin_s,
        "marlin_zp": marlin_zp,
        "marlin_g_idx": marlin_g_idx,
        "marlin_sort_indices": marlin_sort_indices,
        "marlin_rand_perm": marlin_rand_perm,
        "marlin_workspace": marlin_workspace,
        "is_k_full": is_k_full,
        # Marlin_24 params
        "marlin_24_w_ref": marlin_24_w_ref,
        "marlin_24_q_w_comp": marlin_24_q_w_comp,
        "marlin_24_meta": marlin_24_meta,
        "marlin_24_s": marlin_24_s,
        "marlin_24_workspace": marlin_24_workspace,
        # GPTQ params
        "q_w_gptq": q_w_gptq,
        "repack_sort_indices": repack_sort_indices,
        # AllSpark W8A16 params
        "qw_reorder": qw_reorder if as_supported_case else None,
        "s_reorder": s_reorder if as_supported_case else None,
        "zp_reorder": zp_reorder if as_supported_case else None,
        "sm_count": sm_count if as_supported_case else None,
        "sm_version": sm_version if as_supported_case else None,
        "CUBLAS_M_THRESHOLD": CUBLAS_M_THRESHOLD if as_supported_case else None,
        # Kernels
        "gptq_marlin_gemm": ops.gptq_marlin_gemm,
        "gptq_marlin_24_gemm": ops.gptq_marlin_24_gemm,
        "gptq_marlin_repack": ops.gptq_marlin_repack,
        "allspark_w8a16_gemm": ops.allspark_w8a16_gemm,
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
        ).blocked_autorange(min_run_time=min_run_time)
    )

    results.append(
        benchmark.Timer(
            stmt="output = gptq_marlin_gemm(a, marlin_q_w, marlin_s, marlin_zp, marlin_g_idx, marlin_sort_indices, marlin_workspace.scratch, quant_type, size_m, size_n, size_k, is_k_full, False, False, False)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="gptq_marlin_gemm_fp16",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    results.append(
        benchmark.Timer(
            stmt="output = gptq_marlin_gemm(a, marlin_q_w, marlin_s, marlin_zp, marlin_g_idx, marlin_sort_indices, marlin_workspace.scratch, quant_type, size_m, size_n, size_k, is_k_full, False, True, False)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="gptq_marlin_gemm_fp32",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    if (
        quant_type in GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES
        and group_size in GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES
    ):
        results.append(
            benchmark.Timer(
                stmt="output = gptq_marlin_24_gemm(a, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s, marlin_24_workspace.scratch, quant_type, size_m, size_n, size_k)",  # noqa: E501
                globals=globals,
                label=label,
                sub_label=sub_label,
                description="gptq_marlin_24_gemm",
            ).blocked_autorange(min_run_time=min_run_time)
        )

    results.append(
        benchmark.Timer(
            stmt="q_res = gptq_marlin_repack(q_w_gptq, repack_sort_indices, size_k, size_n, quant_type.size_bits)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="gptq_marlin_repack",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    if as_supported_case:
        results.append(
            benchmark.Timer(
                stmt="output = allspark_w8a16_gemm(a, qw_reorder, s_reorder, zp_reorder, size_n, group_size, sm_count, sm_version, CUBLAS_M_THRESHOLD, False, True)",  # noqa: E501
                globals=globals,
                label=label,
                sub_label=sub_label,
                description="allspark_w8a16_gemm_fp32",
            ).blocked_autorange(min_run_time=min_run_time)
        )


def main(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    results: list[benchmark.Measurement] = []

    for model in args.models:
        for layer in WEIGHT_SHAPES[model]:
            size_k = layer[0]
            size_n = layer[1]

            if len(args.limit_k) > 0 and size_k not in args.limit_k:
                continue

            if len(args.limit_n) > 0 and size_n not in args.limit_n:
                continue

            for act_order in ACT_ORDER_OPTS:
                if (
                    len(args.limit_act_order) > 0
                    and act_order not in args.limit_act_order
                ):
                    continue

                for is_k_full in K_FULL_OPTS:
                    if (
                        len(args.limit_k_full) > 0
                        and is_k_full not in args.limit_k_full
                    ):
                        continue

                    for quant_type in query_marlin_supported_quant_types(False):
                        if (
                            len(args.limit_num_bits) > 0
                            and quant_type.size_bits not in args.limit_num_bits
                        ):
                            continue

                        for group_size in MARLIN_SUPPORTED_GROUP_SIZES:
                            if (
                                len(args.limit_group_size) > 0
                                and group_size not in args.limit_group_size
                            ):
                                continue

                            # For act_order, the group_size must be less than
                            # size_k
                            if act_order and (group_size == size_k or group_size == -1):
                                continue

                            for size_m in args.batch_sizes:
                                bench_run(
                                    results,
                                    model,
                                    act_order,
                                    is_k_full,
                                    quant_type,
                                    group_size,
                                    size_m,
                                    size_k,
                                    size_n,
                                )

    compare = benchmark.Compare(results)
    compare.print()


# For quick benchmarking use:
#   python benchmark_marlin.py --batch-sizes 1 16 32 --limit-k 4096 --limit-n 4096 --limit-group-size 128 --limit-num-bits 4 --limit-act-order 0 --limit-k-full 1 # noqa E501
#
if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark Marlin across specified models/shapes/batches"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--limit-k", nargs="+", type=int, default=[])
    parser.add_argument("--limit-n", nargs="+", type=int, default=[])
    parser.add_argument("--limit-group-size", nargs="+", type=int, default=[])
    parser.add_argument("--limit-num-bits", nargs="+", type=int, default=[])
    parser.add_argument("--limit-act-order", nargs="+", type=int, default=[])
    parser.add_argument("--limit-k-full", nargs="+", type=int, default=[])

    args = parser.parse_args()
    main(args)
