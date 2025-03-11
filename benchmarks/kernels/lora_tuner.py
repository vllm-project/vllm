# SPDX-License-Identifier: Apache-2.0
import argparse
from itertools import product
from typing import List

import torch
from benchmark_lora import (BenchmarkContext, BenchmarkTensors, OpType,
                            as_benchmark_contexts)
from weight_shapes import WEIGHT_SHAPES

from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
DEFAULT_TP_SIZES = [1]
DEFAULT_BATCH_SIZES = [
    1, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
    2048, 3072, 4096, 5120, 6144, 7168, 8192
]
DEFAULT_HIDDEN_SIZES = [1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]
DEFAULT_LORA_RANKS = [16]
DEFAULT_NUM_LORAS = [1, 4, 8]
DEFAULT_SORT_BY_LORA_IDS = [False]
DEFAULT_SEQ_LENGTHS = [1]
DEFAULT_EXPAND_FN_ADD_INPUTS = [True, False]


def tune_optype(ctx: BenchmarkContext,
                op_type: OpType,
                add_input_arg: bool,
                test_correctness: bool = False):
    tensors = BenchmarkTensors.make(ctx, op_type)
    tensors.sanity_check()

    # Test correctness of our implementation.
    if test_correctness:
        tensors.test_correctness(op_type, add_input_arg)

    bench_kwargs = tensors.bench_fn_kwargs(op_type, add_inputs=add_input_arg)

    _LORA_A_PTR_DICT.clear()
    _LORA_B_PTR_DICT.clear()

    describe_args = (f"add_inputs={add_input_arg}"
                     if add_input_arg is not None else "")
    description = (f"{op_type.name}({describe_args}) ({tensors.io_types()})")

    print(f"Running :: {ctx.bench_sublabel(op_type)} -- {description} ...")
    op_type.bench_fn()(**bench_kwargs)


def run(args: argparse.Namespace, bench_ctxs: List[BenchmarkContext]):

    for bench_ctx in bench_ctxs:
        for seq_len in args.seq_lengths:
            bench_ops: List[OpType] = [OpType.V1_EXPAND, OpType.V1_SHRINK]
            for bench_op in bench_ops:
                for num_slices in bench_op.num_slices():
                    _ctx = bench_ctx.with_seq_length(seq_len).with_num_slices(
                        num_slices)
                    # Benchmark bench_op
                    expand_fn_add_inputs = [
                        None
                    ] if bench_op.is_shrink_fn() else args.expand_fn_add_inputs
                    for add_input_arg in expand_fn_add_inputs:
                        tune_optype(_ctx, bench_op, add_input_arg,
                                    args.test_correctness)


def run_list_bench(args: argparse.Namespace):
    print(args)

    print("List bench :\n"
          f"  Hidden Sizes {args.hidden_sizes}"
          f"  LoRA Ranks {args.lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: List[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=args.hidden_sizes, lora_ranks=args.lora_ranks, args=args)

    run(args, bench_contexts)


def run_range_bench(args: argparse.Namespace):
    print(args)

    hidden_sizes = list(
        range(args.hidden_sizes_start, args.hidden_sizes_end + 1,
              args.hidden_sizes_increment))
    lora_ranks = list(
        range(args.lora_ranks_start, args.lora_ranks_end + 1,
              args.lora_ranks_increment))

    print("Range bench :\n"
          f" Hidden Sizes {hidden_sizes}"
          f" LoRA Ranks {lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: List[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=hidden_sizes, lora_ranks=lora_ranks, args=args)

    run(args, bench_contexts)


def run_model_bench(args: argparse.Namespace):
    print(args)

    def hidden_sizes_from_model(model: str, tp_size: int) -> set[int]:
        hidden_sizes = set()
        for KN, tp_split_dim in WEIGHT_SHAPES[model]:
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            hidden_sizes.add(KN[1])
        return hidden_sizes

    # Get all hidden sizes
    hidden_sizes: set[int] = set()
    for model_name, tp_size in product(args.models, args.tp_sizes):
        hidden_sizes = hidden_sizes.union(
            hidden_sizes_from_model(model_name, tp_size))

    print("Model bench :\n"
          f" Hidden Sizes {hidden_sizes}"
          f" LoRA Ranks {args.lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: List[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=hidden_sizes, lora_ranks=args.lora_ranks, args=args)

    run(args, bench_contexts)


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "torch.float16":
            return torch.float16
        if dt == "torch.bfloat16":
            return torch.bfloat16
        raise ValueError("unsupported dtype")

    def get_bool(s: str) -> bool:
        return s.lower() in ['true', '1']

    def add_common_command_args(p: argparse.ArgumentParser):
        p.add_argument(
            "--dtype",
            type=to_torch_dtype,
            required=True,
            help="Available options are ['torch.float16', 'torch.bfloat16']")

        p.add_argument(
            "--arg-pool-size",
            type=int,
            default=1,
            help="Run profiles with a pool of input/output/meta tensors instead"
            "of simply reusing the same tensors for all runs. A bigger arg-pool"
            "mitigates hardware caching effects during benchmarking.")

        p.add_argument(
            "--cuda-graph-nops",
            type=int,
            help=("when set profiling is done using cudagraph, "
                  "with the given number of operations in a graph."
                  "Note that the measurement returned is the time "
                  "taken for N consecutive executions of the benchmarking "
                  "functions, where N is the value of this argument."))
        p.add_argument("--num-loras",
                       nargs="+",
                       type=int,
                       default=DEFAULT_NUM_LORAS)
        p.add_argument("--num-active-loras",
                       type=int,
                       default=None,
                       help="Active LoRAs. When None, all LoRAs are active")
        p.add_argument("--sort-by-lora-id",
                       nargs="+",
                       type=get_bool,
                       default=DEFAULT_SORT_BY_LORA_IDS)
        p.add_argument("--op-types",
                       nargs="+",
                       type=OpType.from_str,
                       default=list(OpType))
        p.add_argument('--seq-lengths',
                       nargs="+",
                       type=int,
                       default=DEFAULT_SEQ_LENGTHS)
        p.add_argument("--batch-sizes",
                       nargs="+",
                       type=int,
                       default=DEFAULT_BATCH_SIZES)
        p.add_argument("--expand-fn-add-inputs",
                       nargs="+",
                       type=get_bool,
                       default=DEFAULT_EXPAND_FN_ADD_INPUTS)
        p.add_argument(
            '-o',
            '--output-directory',
            type=str,
            help=("Output directory to store a the list of benchmarking"
                  "TMeasurement objects as a pickle file"))

        p.add_argument(
            "--test-correctness",
            action='store_true',
            help=("When enabled, the benchmarking functions are tested"
                  "for correctness before the actual benchmarking"))

    parser = FlexibleArgumentParser(
        description="""
Benchmark LoRA kernels:

    list_bench example:
        python3 benchmarks/kernels/benchmark_lora.py list_bench --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16 --hidden-sizes 2048 --lora-ranks 16 --num-loras 1 4 --op-types bgmv_shrink bgmv_expand sgmv_shrink sgmv_expand bgmv_expand_slice --seq-lengths 1 16 --sort-by-lora-id 1 --cuda-graph-nops 32

    model_bench example:
        python3 benchmarks/kernels/benchmark_lora.py model_bench --models meta-llama/Llama-3-8b  --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16  --lora-ranks 16 --num-loras 1 4 --op-types bgmv_shrink bgmv_expand sgmv_shrink sgmv_expand bgmv_expand_slice --seq-lengths 1 16 --sort-by-lora-id 1 --cuda-graph-nops 32 

    range_bench example:
        python3 benchmarks/kernels/benchmark_lora.py range_bench  --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16   --num-loras 1 4 --op-types bgmv_shrink bgmv_expand sgmv_shrink sgmv_expand bgmv_expand_slice --seq-lengths 1 16 --sort-by-lora-id 1 --cuda-graph-nops 32 --hidden-sizes-start 1024 --hidden-sizes-end 4096 --hidden-sizes-increment 1024 --lora-ranks-start 8 --lora-ranks-end 24 --lora-ranks-increment 8 
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    list_parser = subparsers.add_parser("list_bench")
    list_parser.add_argument("--hidden-sizes",
                             nargs="+",
                             type=int,
                             default=DEFAULT_HIDDEN_SIZES)
    list_parser.add_argument("--lora-ranks",
                             nargs="+",
                             type=int,
                             default=DEFAULT_LORA_RANKS)
    add_common_command_args(list_parser)
    list_parser.set_defaults(func=run_list_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--hidden-sizes-start", type=int, required=True)
    range_parser.add_argument("--hidden-sizes-end", type=int, required=True)
    range_parser.add_argument("--hidden-sizes-increment",
                              type=int,
                              required=True)
    range_parser.add_argument("--lora-ranks-start", type=int, required=True)
    range_parser.add_argument("--lora-ranks-end", type=int, required=True)
    range_parser.add_argument("--lora-ranks-increment",
                              type=int,
                              required=True)
    add_common_command_args(range_parser)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument("--models",
                              nargs="+",
                              type=str,
                              default=DEFAULT_MODELS,
                              choices=WEIGHT_SHAPES.keys())
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--lora-ranks",
                              nargs="+",
                              type=int,
                              default=DEFAULT_LORA_RANKS)
    add_common_command_args(model_parser)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)