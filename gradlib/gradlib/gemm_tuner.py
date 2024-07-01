import argparse
import json
import os
from pathlib import Path

import torch  # isort: split
import hipbsolidxgemm
import pandas as pd
import rocsolidxgemm

from gradlib.GemmTuner import GemmTuner

rocsolidxgemm.rocb_create_extension()
hipbsolidxgemm.hipb_create_extension()


def generate_mk_sets(model_dir, tp=1):
    with open(f'{model_dir}/config.json') as f:
        data = json.load(f)
        hidden_size = data['hidden_size']
        intermediate_size = data['intermediate_size']
        total_num_heads = data['num_attention_heads']
        total_num_kv_heads = data['num_key_value_heads']
        head_dim = hidden_size // total_num_heads
    return [((total_num_heads + (2 * total_num_kv_heads)) * head_dim // tp,
             hidden_size), (hidden_size, hidden_size // tp),
            (intermediate_size * 2 // tp, hidden_size),
            (hidden_size, intermediate_size // tp)], hidden_size


def get_dtype(dtype_str):
    dtype = torch.float16
    if dtype_str == 'f32':
        dtype = torch.float32
    elif dtype_str == 'bf16':
        dtype = torch.bfloat16
    elif dtype_str == 'f16':
        dtype = torch.float16
    elif dtype_str == 'fp8':
        dtype = torch.float8_e4m3fnuz
    else:
        print('>>> Warning! Invalid dtype', dtype_str,
              'using default dtype f16')
    return dtype


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def load_input_gemms(input_file):
    if Path(input_file).is_file():
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        default=os.getenv('GTUNE_MODEL', ""),
                        help="Enter the location of your model directory")
    parser.add_argument("--tuned_file",
                        type=str,
                        default=os.getenv('GTUNE_TUNED', "tuned.csv"),
                        help="output file for tuned gemm solutions")
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.getenv('GTUNE_INPUT', None),
        help="list of gemms to tune for, mutually exclusive with model_dir")
    parser.add_argument("--tp",
                        type=int,
                        default=os.getenv('GTUNE_TP', 1),
                        help="Tensor parallelism to be used.")
    parser.add_argument("--indtype",
                        type=str,
                        default='f16',
                        choices=["f32", "f16", "bf16", "fp8"],
                        help="dtype f32 f16 bf16 fp8")
    parser.add_argument("--outdtype",
                        type=str,
                        choices=["f32", "f16", "bf16", "fp8"],
                        help="dtype f32 f16 bf16 fp8")
    parser.add_argument("--rocblas-decode",
                        action="store_true",
                        default=False,
                        help="forces rocblas solution on decode N=1")
    parser.add_argument("--batch_size",
                        type=int,
                        default=os.getenv('GTUNE_BATCH_SIZE', 1),
                        help="Batch size to tune for")
    parser.add_argument("--nsets",
                        type=list_of_ints,
                        default=[1, 512, 1024, 2048, 3072, 4096, 8192, 16384],
                        help="N sizes to tune for: 1,128,2048")
    args = parser.parse_args()

    if args.outdtype is None:
        args.outdtype = args.indtype
    indtype = get_dtype(args.indtype)
    outdtype = get_dtype(args.outdtype)

    gtuner = GemmTuner(indtype, outdtype, args.tuned_file, args.rocblas_decode)
    nsets = [i * args.batch_size for i in args.nsets]
    if args.input_file:
        print(f">>> Loading {args.input_file}")
        if not Path(args.input_file).is_file():
            print(f">>> ERROR: {args.input_file} does not exist.  Exiting")
            exit(1)
        shapes = pd.read_csv(args.input_file)
        for i in range(len(shapes)):
            ds = shapes.iloc[i]
            gtuner.add_gemm(ds['M'], ds['N'], ds['K'])
    else:
        if not args.model_dir:
            print(">>> Warning! NO MODEL SPECIFIED. Tuning for LL2 13B TP1")
            #LL2 13B sizes
            mksets = [(15360, 5120), (5120, 5120), (27648, 5120),
                      (5120, 13824)]
            gtuner.add_gemm(m=32000, n=1, k=5120)  # logits gemm
        else:
            mksets, hidden_size = generate_mk_sets(args.model_dir, args.tp)
            gtuner.add_gemm(
                m=32000 // args.tp, n=1 * args.batch_size, k=hidden_size
            )  #TODO: Handle cases where vocab_size is not divisible by tp

        for n in sorted(nsets):
            for m, k in mksets:
                gtuner.add_gemm(m, n, k)

    gtuner.find_best_sols()
