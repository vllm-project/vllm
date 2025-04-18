# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import itertools
import math
import os
import pickle as pkl
import time
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N, marlin_permute_scales,
    marlin_zero_points)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows, quantize_weights)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = ["meta-llama/Llama-3-8b", "meta-llama/Llama-2-70b-hf"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_TP_SIZES = [1]

NVTX_PROFILE = os.environ.get("NVTX_PROFILE", False)

if NVTX_PROFILE:
    import nvtx


def terse_type_name(dt):
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.float8_e4m3fn: "fp8",
        torch.bfloat16: "bf16",
        torch.float: "float",
        torch.int: "int",
    }[dt]


@dataclass
class BenchmarkTensors:
    w_ref: torch.Tensor
    a: torch.Tensor

    w_q: torch.Tensor
    group_size: Optional[int]
    wtype: ScalarType
    w_g_s: torch.Tensor
    w_g_zp: Optional[torch.Tensor]
    w_ch_s: Optional[torch.Tensor]
    w_tok_s: Optional[torch.Tensor]


@dataclass
class TypeConfig:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]


def rand_data(shape, dtype=torch.float16, scale=1):
    if dtype.is_floating_point:
        return (scale * torch.rand(shape, device="cuda") - 0.3).to(dtype)
    else:
        return torch.randint(-15, 15, shape, dtype=dtype, device="cuda")


def quantize_and_pack(atype: torch.dtype,
                      w: torch.Tensor,
                      wtype: ScalarType,
                      stype: Optional[torch.dtype],
                      group_size: Optional[int],
                      zero_points: bool = False):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w,
        wtype,
        group_size=group_size,
        zero_points=zero_points,
        # to match how the kernel applies zps
        ref_zero_points_after_scales=True)

    w_q = pack_rows(w_q, wtype.size_bits, *w_q.shape)
    return w_ref, w_q, w_s, w_zp


def create_bench_tensors(shape: Tuple[int, int, int], types: TypeConfig,
                         group_size: Optional[int]) -> List[BenchmarkTensors]:
    m, n, k = shape

    # we want to make sure that weights don't fit into L2 cache between runs so
    #  we construct enough weights to exceed L2 cache, which is 50mb on a H100
    #  so we target total weight size > 2*50mb
    num_weights = math.ceil(2 * 50 * 1024**2 * 8 /
                            (k * n * types.weight_type.size_bits))

    a = rand_data((m, k), types.act_type, scale=5)

    benchmark_tensors: List[BenchmarkTensors] = []
    for _ in range(num_weights):
        w = rand_data((k, n), types.act_type, scale=5)

        if types.group_scale_type is not None:
            w = w.to(types.group_scale_type)
        if w.dtype.itemsize == 1:
            w = w.to(torch.float16)

        w_ref, w_q_packed, w_s, w_zp = quantize_and_pack(
            a.dtype, w, types.weight_type, types.group_scale_type, group_size,
            types.group_zero_type is not None)

        if not a.dtype.is_floating_point:
            aiinfo = torch.iinfo(a.dtype)
            w_ref = w_ref.round().clamp(aiinfo.min, aiinfo.max)

        w_ref = w_ref.to(torch.float32)

        w_ch_s = None if types.channel_scale_type is None else\
            rand_data((n,), types.channel_scale_type)
        w_tok_s = None if types.token_scale_type is None else\
            rand_data((m,), types.token_scale_type)

        benchmark_tensors.append(
            BenchmarkTensors(w_ref=w_ref,
                             a=a,
                             w_q=w_q_packed,
                             wtype=types.weight_type,
                             w_g_s=w_s,
                             w_g_zp=w_zp,
                             group_size=group_size,
                             w_ch_s=w_ch_s,
                             w_tok_s=w_tok_s))

    return benchmark_tensors


def torch_matmul_f16_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    a = bt.a
    w = bt.w_ref.to(bt.a.dtype)  # use float reference tensor
    if a.dtype not in [torch.float16, torch.bfloat16]:
        a = a.to(torch.float16)
        w = w.to(torch.float16)
    return lambda: torch.matmul(a, w)


def cutlass_scaled_mm_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    if bt.w_ch_s is not None and bt.w_tok_s is not None:
        scale_a = bt.w_tok_s.to(torch.float32)
        scale_b = bt.w_ch_s.to(torch.float32)
    else:
        scale_a = torch.tensor(1.0, dtype=torch.float32, device=bt.a.device)
        scale_b = torch.tensor(1.0, dtype=torch.float32, device=bt.a.device)
    w_col_major = bt.w_ref.to(bt.a.dtype).t().contiguous().t()
    return lambda: ops.cutlass_scaled_mm(
        bt.a, w_col_major, scale_a, scale_b, out_dtype=torch.float16)


def marlin_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    device = bt.a.device

    workspace = MarlinWorkspace(bt.w_ref.shape[1], GPTQ_MARLIN_MIN_THREAD_N,
                                GPTQ_MARLIN_MAX_PARALLEL)

    if bt.w_g_zp is None:
        w_zp = torch.empty(0, dtype=torch.int, device=device)
    else:
        w_zp = marlin_zero_points(bt.w_g_zp, bt.w_ref.shape[0],
                                  bt.w_ref.shape[1], bt.wtype.size_bits)

    if bt.group_size is None:
        w_s = torch.tensor([], device="cuda", dtype=torch.half)
    else:
        w_s = marlin_permute_scales(bt.w_g_s, bt.w_ref.shape[0],
                                    bt.w_ref.shape[1], bt.group_size)

    sort_indices = torch.empty(0, dtype=torch.int, device=device)
    g_idx = torch.empty(0, dtype=torch.int, device=device)
    w_q = ops.gptq_marlin_repack(bt.w_q, sort_indices, bt.w_ref.shape[0],
                                 bt.w_ref.shape[1], bt.wtype.size_bits)

    if bt.a.dtype.is_floating_point:
        assert bt.w_ch_s is None
        assert bt.w_tok_s is None
        assert bt.group_size is not None

        fn = lambda: ops.gptq_marlin_gemm(a=bt.a,
                                          b_q_weight=w_q,
                                          b_scales=w_s,
                                          b_zeros=w_zp,
                                          g_idx=g_idx,
                                          perm=sort_indices,
                                          workspace=workspace.scratch,
                                          b_q_type=bt.wtype,
                                          size_m=bt.a.shape[0],
                                          size_n=bt.w_ref.shape[1],
                                          size_k=bt.w_ref.shape[0],
                                          is_k_full=True,
                                          is_zp_float=False)
    else:
        assert bt.a.dtype == torch.int8
        assert bt.wtype == scalar_types.uint4b8

        if bt.w_ch_s is not None:
            s_ch = bt.w_ch_s.to(torch.float32)
        else:
            s_ch = torch.ones(bt.w_ref.shape[1],
                              dtype=torch.float32,
                              device=device)

        if bt.w_tok_s is not None:
            s_tok = bt.w_tok_s.to(torch.float32)
        else:
            s_tok = torch.ones(bt.a.shape[0],
                               dtype=torch.float32,
                               device=device)

        fn = lambda: ops.marlin_qqq_gemm(a=bt.a,
                                         b_q_weight=w_q,
                                         s_group=w_s,
                                         s_tok=s_tok,
                                         s_ch=s_ch,
                                         workspace=workspace.scratch,
                                         size_m=bt.a.shape[0],
                                         size_n=bt.w_ref.shape[1],
                                         size_k=bt.w_ref.shape[0])

    return fn


def machete_create_bench_fn(bt: BenchmarkTensors,
                            out_type=torch.dtype,
                            schedule=None) -> Callable:
    w_q = bt.w_q.t().contiguous().t()  # make col major
    w_q = ops.machete_prepack_B(w_q, bt.a.dtype, bt.wtype,
                                None if bt.w_g_s is None else bt.w_g_s.dtype)

    w_g_zp = bt.w_g_zp
    if w_g_zp is not None:
        w_g_zp = -1 * bt.w_g_s * (w_g_zp.to(bt.w_g_s.dtype))

    return lambda: ops.machete_mm(
        a=bt.a,
        b_q=bt.w_q,
        b_type=bt.wtype,
        b_group_scales=bt.w_g_s,
        b_group_zeros=w_g_zp,
        b_group_size=bt.group_size,
        b_channel_scales=bt.w_ch_s,
        a_token_scales=bt.w_tok_s,
        out_type=out_type,
        schedule=schedule,
    )


# impl

# bench


def bench_fns(label: str, sub_label: str, description: str,
              fns: List[Callable]):

    min_run_time = 1 if not NVTX_PROFILE else 0.1
    res = TBenchmark.Timer(
        stmt="""
        for fn in fns:
            fn()
        """,
        globals={
            "fns": fns
        },
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)

    if NVTX_PROFILE:
        with nvtx.annotate("mm-bench"), nvtx.annotate(
                f"{label}|{sub_label}|{description}"):
            fns[0]()

    return res


_SWEEP_SCHEDULES_RESULTS: Optional[pd.DataFrame] = None
_SWEEP_SCHEDULES_RESULTS_CSV: Optional[str] = None


def bench(types: TypeConfig,
          group_size: int,
          m: int,
          k: int,
          n: int,
          label: str,
          sub_label: str,
          sweep_schedules: bool = True) -> List[TMeasurement]:
    benchmark_tensors = create_bench_tensors((m, n, k), types, group_size)
    sub_label += f", L={len(benchmark_tensors)}"

    name_type_string = f"W{types.weight_type}"+\
                       f"-A{terse_type_name(types.act_type)}"
    if types.group_scale_type is not None:
        name_type_string += f"-GS{terse_type_name(types.group_scale_type)}"
    if types.group_zero_type is not None:
        name_type_string += f"-GZ{terse_type_name(types.group_zero_type)}"
    if group_size is not None:
        name_type_string += f"-G{group_size}"
    if types.channel_scale_type is not None:
        name_type_string += f"-CS{terse_type_name(types.channel_scale_type)}"
    if types.token_scale_type is not None:
        name_type_string += f"-TS{terse_type_name(types.token_scale_type)}"

    timers = []
    # pytorch impl
    timers.append(
        bench_fns(
            label, sub_label, "torch.matmul (fp16)",
            [torch_matmul_f16_create_bench_fn(bt)
             for bt in benchmark_tensors]))

    if types.act_type == torch.int8 or types.act_type == torch.float8_e4m3fn:
        timers.append(
            bench_fns(
                label, sub_label,
                f"cutlass_scaled_mm ({terse_type_name(types.act_type)})", [
                    cutlass_scaled_mm_create_bench_fn(bt)
                    for bt in benchmark_tensors
                ]))

    if types.act_type != torch.float8_e4m3fn:
        timers.append(
            bench_fns(label, sub_label, f"marlin ({name_type_string})",
                      [marlin_create_bench_fn(bt)
                       for bt in benchmark_tensors]))

    # machete
    timers.append(
        bench_fns(label, sub_label, f"machete ({name_type_string})", [
            machete_create_bench_fn(bt, out_type=types.output_type)
            for bt in benchmark_tensors
        ]))

    if sweep_schedules:
        global _SWEEP_SCHEDULES_RESULTS

        print("Finding best schedule for machete")
        best = None
        best_schedule = None
        schedules = ops.machete_supported_schedules(
            a_type=types.act_type,
            b_type=types.weight_type,
            group_scales_type=types.group_scale_type,
            group_zeros_type=types.group_zero_type,
            token_scales_type=types.token_scale_type,
            channel_scales_type=types.channel_scale_type,
            out_type=types.output_type)

        if schedules is None or len(schedules) == 0:
            raise ValueError("No schedules found to sweep")

        for schedule in reversed(schedules):
            schedule_M = int(schedule.split("_")[0].split("x")[1])

            # Prune known bad schedules
            if schedule_M >= 2 * max(m, 16) or schedule_M < m // 4:
                continue

            res = bench_fns(label, sub_label, "machete_best", [
                machete_create_bench_fn(
                    bt, out_type=types.output_type, schedule=schedule)
                for bt in benchmark_tensors
            ])

            results_row = {
                "M": m,
                "K": k,
                "N": n,
                "group_size": group_size,
                "schedule": schedule,
                "median": res.median,
            }
            if _SWEEP_SCHEDULES_RESULTS is None:
                _SWEEP_SCHEDULES_RESULTS = pd.DataFrame(
                    columns=results_row.keys())
            _SWEEP_SCHEDULES_RESULTS.\
                loc[len(_SWEEP_SCHEDULES_RESULTS)] = results_row

            print(f"  {res.median:5.5} ", schedule)
            if not best or res.median < best.median:
                best = res
                best_schedule = schedule
        print("Best schedule:", best_schedule)
        timers.append(best)

    return timers


# runner
def print_timers(timers: List[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(args, MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:
    types = TypeConfig(
        act_type=args.act_type,
        weight_type=scalar_types.uint4b8 if args.group_zero_type is None \
            else scalar_types.uint4,
        output_type=args.out_type,
        group_scale_type=args.group_scale_type,
        group_zero_type=args.group_zero_type,
        channel_scale_type=args.channel_scale_type,
        token_scale_type=args.token_scale_type,
    )

    results: List[TMeasurement] = []
    for m, k, n in MKNs:
        timers = bench(types,
                       args.group_size,
                       m,
                       k,
                       n,
                       f"{args.act_type}-gemm",
                       f"MKN=({m}x{k}x{n})",
                       sweep_schedules=args.sweep_schedules)
        print_timers(timers)
        results.extend(timers)

    return results


# output makers
def make_output(
    data: List[TMeasurement],
    MKNs: Iterable[Tuple[int, int, int]],
    base_description: str,
    timestamp=None,
):

    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


# argparse runners


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args.dtype, args.sweep_schedules, MKNs)

    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    m_start, k_start, n_start = (int(x) for x in args.dim_start.split(","))
    m_end, k_end, n_end = (int(x) for x in args.dim_end.split(","))
    m_increment, k_increment, n_increment = \
        (int(x) for x in args.dim_increment.split(","))
    Ms = list(range(m_start, m_end + 1, m_increment))
    Ks = list(range(k_start, k_end + 1, k_increment))
    Ns = list(range(n_start, n_end + 1, n_increment))
    MKNs = list(product(Ms, Ks, Ns))

    data = run(args.dtype, args.sweep_schedules, MKNs)

    make_output(data, MKNs, f"range_bench-{args.dtype}")


def run_model_bench(args):

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args, MKNs)
        model_bench_data.append(data)

    type_string = f"{args.act_type}"

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {type_string} {model}-TP{tp_size} ====")
        print_timers(data)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    all_results = []
    for d in model_bench_data:
        all_results.extend(d)

    # pickle all data
    with open(f"model_bench-{type_string}-{timestr}.pkl", "wb") as f:
        args_dict = vars(args)
        args_dict.pop("func")
        pkl.dump({
            "args": args_dict,
            "results": all_results,
        }, f)


if __name__ == "__main__":

    def to_torch_dtype(dt):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "int8": torch.int8,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "int": torch.int,
            "float": torch.float,
        }[dt]

    class ToTorchDtype(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, to_torch_dtype(values))

    parser = FlexibleArgumentParser(
        description="""
Benchmark Machete GEMM.

    To run square GEMMs:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--act-type",
        action=ToTorchDtype,
        required=True,
        choices=['bfloat16', 'float16', 'int8', 'float8_e4m3fn'],
    )
    parser.add_argument(
        "--group-scale-type",
        action=ToTorchDtype,
        choices=['bfloat16', 'float16'],
    )
    parser.add_argument(
        "--group-zero-type",
        type=to_torch_dtype,
        choices=['bfloat16', 'float16'],
    )
    parser.add_argument(
        "--channel-scale-type",
        action=ToTorchDtype,
        choices=['float'],
    )
    parser.add_argument(
        "--token-scale-type",
        action=ToTorchDtype,
        choices=['float'],
    )
    parser.add_argument(
        "--out-type",
        action=ToTorchDtype,
        choices=['bfloat16', 'float16'],
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="Available options are ['None', '-1', '128'], default=128",
        default=128,
    )
    parser.add_argument(
        "--sweep-schedules",
        action="store_true",
        help="Run a sweep over all supported schedules",
    )
    parser.add_argument("--sweep-csv-out",
                        help="CSV to store sweep results",
                        default="sch_sweep_results.csv")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument(
        "--dim-start",
        type=str,
        required=True,
        help="Start value for M,K,N as common separated list")
    range_parser.add_argument(
        "--dim-end",
        type=str,
        required=True,
        help="End value (inclusive) for M,K,N as common separated list")
    range_parser.add_argument(
        "--dim-increment",
        type=str,
        required=True,
        help="Increment value for M,K,N as common separated list")
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()

    _SWEEP_SCHEDULES_RESULTS_CSV = args.sweep_csv_out
    args.func(args)

    if _SWEEP_SCHEDULES_RESULTS is not None:
        _SWEEP_SCHEDULES_RESULTS.to_csv(_SWEEP_SCHEDULES_RESULTS_CSV)
