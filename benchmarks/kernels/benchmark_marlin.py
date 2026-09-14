# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
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
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_SUPPORTED_GROUP_SIZES,
    rand_marlin_weight_fp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
    awq_marlin_quantize,
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
    quantize_weights,
)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_MODELS = ["meta-llama/Llama-2-7b-hf/TP1"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def bench_run(
    results: list[benchmark.Measurement],
    model: str,
    quant_type: ScalarType,
    group_size: int,
    size_m: int,
    size_k: int,
    size_n: int,
):
    label = "Quant Matmul"
    sub_label = "{}, q={}, g={}, MKN=({}x{}x{})".format(
        model, str(quant_type), group_size, size_m, size_k, size_n
    )
    print(f"Testing: {sub_label}")

    a = torch.randn(size_m, size_k).to(torch.half).cuda()
    b = torch.rand(size_k, size_n).to(torch.half).cuda()
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]
    if size_k % group_size != 0:
        return

    repack_supported = group_size in MARLIN_SUPPORTED_GROUP_SIZES
    allspark_supported = (
        quant_type in ALLSPARK_SUPPORTED_QUANT_TYPES and group_size == -1
    )

    def gen_marlin_params():
        # Marlin quant
        marlin_zp = marlin_s2 = None
        if quant_type == scalar_types.float4_e2m1f:
            if group_size != 16:
                return
            marlin_w_ref, marlin_q_w, marlin_s, marlin_s2 = rand_marlin_weight_fp4_like(
                b.T, group_size
            )
        elif quant_type == scalar_types.float8_e4m3fn:
            if group_size not in [-1, 128]:
                return
            marlin_w_ref, marlin_q_w, marlin_s = marlin_quant_fp8_torch(b.T, group_size)
        elif group_size == 16:
            return
        elif has_zp:
            marlin_w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
                b, quant_type, group_size
            )
        else:
            marlin_w_ref, marlin_q_w, marlin_s = marlin_quantize(
                b, quant_type, group_size
            )
        return (
            marlin_w_ref,
            marlin_q_w,
            marlin_s,
            marlin_s2,
            marlin_zp,
        )

    def gen_repack_params():
        q_w_gptq = None
        if repack_supported:
            _, q_w, _ = gptq_quantize_weights(b, quant_type, group_size)
            q_w_gptq = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)
        return q_w_gptq

    def gen_allspark_params():
        qw_reorder = s_reorder = zp_reorder = sm_count = sm_version = (
            CUBLAS_M_THRESHOLD
        ) = None
        nonlocal allspark_supported
        if allspark_supported:
            properties = torch.cuda.get_device_properties(b.device.index)
            sm_count = properties.multi_processor_count
            sm_version = properties.major * 10 + properties.minor

            supported_arch = sm_version >= 80 and sm_version < 90
            allspark_supported = allspark_supported and supported_arch
            if supported_arch:
                w_ref, qw, s, zp = quantize_weights(b, quant_type, group_size, has_zp)
                qw = qw.to(torch.uint8)

                qw_reorder, s_reorder, zp_reorder = ops.allspark_repack_weight(
                    qw, s, zp, has_zp
                )
                CUBLAS_M_THRESHOLD = ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD
        return (
            qw_reorder,
            s_reorder,
            zp_reorder,
            sm_count,
            sm_version,
            CUBLAS_M_THRESHOLD,
        )

    (
        marlin_w_ref,
        marlin_q_w,
        marlin_s,
        marlin_s2,
        marlin_zp,
    ) = gen_marlin_params()
    q_w_gptq = gen_repack_params()
    qw_reorder, s_reorder, zp_reorder, sm_count, sm_version, CUBLAS_M_THRESHOLD = (
        gen_allspark_params()
    )

    # Prepare
    marlin_workspace = MarlinWorkspace(
        size_n, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
    )

    globals = {
        # Gen params
        "quant_type": quant_type,
        "group_size": group_size,
        "size_m": size_m,
        "size_n": size_n,
        "size_k": size_k,
        "a": a,
        # Marlin params
        "marlin_w_ref": marlin_w_ref,
        "marlin_q_w": marlin_q_w,
        "marlin_s": marlin_s,
        "marlin_s2": marlin_s2,
        "marlin_zp": marlin_zp,
        "marlin_workspace": marlin_workspace,
        # GPTQ params
        "q_w_gptq": q_w_gptq,
        # AllSpark W8A16 params
        "qw_reorder": qw_reorder,
        "s_reorder": s_reorder,
        "zp_reorder": zp_reorder,
        "sm_count": sm_count,
        "sm_version": sm_version,
        "CUBLAS_M_THRESHOLD": CUBLAS_M_THRESHOLD,
        # Kernels
        "marlin_gemm": ops.marlin_gemm,
        "gptq_marlin_repack": ops.gptq_marlin_repack,
        "allspark_w8a16_gemm": ops.allspark_w8a16_gemm,
    }

    min_run_time = 1

    # Warmup pytorch
    for _ in range(5):
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
            stmt="output = marlin_gemm(a, None, marlin_q_w, None, marlin_s, None, marlin_s2, marlin_zp, marlin_workspace.scratch, quant_type, size_m, size_n, size_k, False, False, False)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="marlin_gemm",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    results.append(
        benchmark.Timer(
            stmt="output = marlin_gemm(a, None, marlin_q_w, None, marlin_s, None, marlin_s2, marlin_zp, marlin_workspace.scratch, quant_type, size_m, size_n, size_k, False, True, False)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="marlin_gemm_fp32",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    if repack_supported:
        results.append(
            benchmark.Timer(
                stmt="q_res = gptq_marlin_repack(q_w_gptq, size_k, size_n, quant_type.size_bits)",  # noqa: E501
                globals=globals,
                label=label,
                sub_label=sub_label,
                description="gptq_marlin_repack",
            ).blocked_autorange(min_run_time=min_run_time)
        )

    if allspark_supported:
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

            for quant_type in query_marlin_supported_quant_types():
                if (
                    len(args.limit_num_bits) > 0
                    and quant_type.size_bits not in args.limit_num_bits
                ):
                    continue

                for group_size in (
                    MARLIN_SUPPORTED_GROUP_SIZES + FP4_MARLIN_SUPPORTED_GROUP_SIZES
                ):
                    if (
                        len(args.limit_group_size) > 0
                        and group_size not in args.limit_group_size
                    ):
                        continue

                    for size_m in args.batch_sizes:
                        bench_run(
                            results,
                            model,
                            quant_type,
                            group_size,
                            size_m,
                            size_k,
                            size_n,
                        )

    compare = benchmark.Compare(results)
    compare.print()


# For quick benchmarking use:
#   python benchmark_marlin.py --batch-sizes 1 16 32 --limit-k 4096 --limit-n 4096 --limit-group-size 128 --limit-num-bits 4 # noqa E501
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

    args = parser.parse_args()
    main(args)
