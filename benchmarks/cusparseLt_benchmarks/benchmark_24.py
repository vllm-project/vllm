import argparse
import copy
import itertools
import pickle as pkl
import time
from typing import Callable, Iterable, List, Tuple

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from weight_shapes import WEIGHT_SHAPES

from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import (
    compress_to_torch_sparse_semi_structured_mat, dense_matmul, get_random_mat,
    is_semi_structured_supported, semi_structured_sparse_dense_gemm)
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
DEFAULT_BATCH_SIZES = [32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]


# helpers
def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = get_random_mat(m, k, dtype)
    b = get_random_mat(n, k, dtype).t()
    return a, b


# bench
def bench_fn(label: str, sub_label: str, description: str, fn: Callable, *args,
             **kwargs) -> TMeasurement:
    min_run_time = 1

    globals = {
        "args": args,
        "kwargs": kwargs,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(*args, **kwargs)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(m: int, k: int, n: int, label: str, sub_label: str,
          use_fp8: bool) -> Iterable[TMeasurement]:
    a, b = make_rand_tensors(torch.float16, m, n, k)

    timers = []
    # pytorch float16
    timers.append(
        bench_fn(label, sub_label, "pytorch_fp16_fp16_matmul", torch.mm,
                 a.to(dtype=torch.float16), b.to(dtype=torch.float16)))

    # pytorch bf16
    timers.append(
        bench_fn(label, sub_label, "pytorch_bf16_bf16_matmul", torch.mm,
                 a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda")))

    # cusparseLt fp16
    timers.append(
        bench_fn(label, sub_label, "cusparseLt_fp16_fp16_2_4",
                 semi_structured_sparse_dense_gemm,
                 compress_to_torch_sparse_semi_structured_mat(a), b))

    # cusparseLt bf16
    timers.append(
        bench_fn(
            label, sub_label, "cusparseLt_bf16_bf16_2_4",
            semi_structured_sparse_dense_gemm,
            compress_to_torch_sparse_semi_structured_mat(
                a.to(dtype=torch.bfloat16)), b.to(torch.bfloat16)))

    a, b = make_rand_tensors(torch.int8, m, n, k)
    # cutlass i8
    timers.append(
        bench_fn(label, sub_label, "cutlass_i8_i8_matmul-w-scales",
                 dense_matmul, a, b, torch.int8))

    # cusparseLt i8
    timers.append(
        bench_fn(label, sub_label, "cusparseLt_i8_i8_2_4",
                 semi_structured_sparse_dense_gemm,
                 compress_to_torch_sparse_semi_structured_mat(a), b))

    if use_fp8:
        a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
        # cutlass fp8
        timers.append(
            bench_fn(label, sub_label, "cutlass_fp8_fp8_matmul-w-scales",
                     dense_matmul, a, b, torch.float8_e4m3fn))

        # cusparseLt fp8
        timers.append(
            bench_fn(label, sub_label, "cusparseLt_fp8_fp8_2_4",
                     semi_structured_sparse_dense_gemm,
                     compress_to_torch_sparse_semi_structured_mat(a), b))

    return timers


# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(MKNs: Iterable[Tuple[int, int, int]],
        use_fp8: bool) -> Iterable[TMeasurement]:
    results = []
    for m, k, n in MKNs:
        timers = bench(m, k, n, "gemm", f"MKN=({m}x{k}x{n})", use_fp8)
        print_timers(timers)
        results.extend(timers)

    return results


def make_output(data: Iterable[TMeasurement],
                MKNs: Iterable[Tuple[int, int, int]],
                base_description: str,
                timestamp=None):
    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


def run_model_bench(args):
    if not is_semi_structured_supported():
        raise ValueError("Device does not support semi-structured sparsity")

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
            assert m % 32 == 0, "Batch size has to be a multiple of 32"
            for k, n in KNs:
                if k % 32 or n % 32:
                    continue
                MKNs.append((m, k, n))

        data = run(MKNs, args.use_fp8)
        model_bench_data.append(data)

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results cuSparseLt {model}-TP{tp_size} ====")
        print_timers(data)

    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)


if __name__ == '__main__':

    parser = FlexibleArgumentParser(
        description="""
Benchmark cuSparseLt 2:4 GEMMs.

    To run dimensions from a model:
        python3 ./benchmarks/cusparseLt_benchmarks/benchmark_24.py --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cusparseLt implementations for the various GEMMs.
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--models",
                        nargs="+",
                        type=str,
                        default=DEFAULT_MODELS,
                        choices=WEIGHT_SHAPES.keys())
    parser.add_argument("--tp-sizes",
                        nargs="+",
                        type=int,
                        default=DEFAULT_TP_SIZES)
    parser.add_argument("--batch-sizes",
                        nargs="+",
                        type=int,
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        '--use-fp8',
        action='store_true',
        help='Add benchmarking fp8 matmul (on supporting fp8 platforms)')

    args = parser.parse_args()
    run_model_bench(args)
