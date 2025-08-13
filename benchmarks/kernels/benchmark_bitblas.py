# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from packaging import version

from vllm.model_executor.layers.quantization.utils.bitblas_utils import (
    MINIMUM_BITBLAS_VERSION,
)

try:
    import bitblas

    if version.parse(bitblas.__version__) < version.parse(MINIMUM_BITBLAS_VERSION):
        raise ImportError(
            "bitblas version is wrong. Please "
            f"install bitblas>={MINIMUM_BITBLAS_VERSION}"
        )
except ImportError as e:
    bitblas_import_exception = e
    raise ValueError(
        "Trying to use the bitblas backend, but could not import"
        f"with the following error: {bitblas_import_exception}. "
        "Please install bitblas through the following command: "
        f"`pip install bitblas>={MINIMUM_BITBLAS_VERSION}`"
    ) from bitblas_import_exception

from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target

from vllm.utils import FlexibleArgumentParser

parser = FlexibleArgumentParser(
    description="Benchmark BitBLAS int4 on a specific target."
)

# Add arguments to the parser
parser.add_argument(
    "--target",
    type=str,
    default=auto_detect_nvidia_target(),
    help="Specify the target device for benchmarking.",
)
parser.add_argument(
    "--group_size", type=int, default=None, help="Group size for grouped quantization."
)
parser.add_argument(
    "--A_dtype",
    type=str,
    default="float16",
    choices=["float16", "float32", "float64", "int32", "int8"],
    help="Data type of activation A.",
)
parser.add_argument(
    "--W_dtype",
    type=str,
    default="int4",
    choices=[
        "float16",
        "float32",
        "float64",
        "int32",
        "int8",
        "int4",
        "int2",
        "int1",
        "nf4",
        "fp4_e2m1",
    ],
    help="Data type of weight W.",
)
parser.add_argument(
    "--accum_dtype",
    type=str,
    default="float16",
    choices=["float16", "int32"],
    help="Data type for accumulation.",
)
parser.add_argument(
    "--out_dtype",
    type=str,
    default="float16",
    choices=["float16", "float32", "int32", "int8"],
    help="Data type for output.",
)
parser.add_argument(
    "--layout",
    type=str,
    default="nt",
    choices=["nt", "nn"],
    help="Matrix layout, 'nt' for non-transpose A and transpose W.",
)
parser.add_argument(
    "--with_bias", action="store_true", help="Include bias in the benchmark."
)
parser.add_argument(
    "--with_scaling",
    action="store_true",
    help="Include scaling factor in the quantization.",
)
parser.add_argument(
    "--with_zeros", action="store_true", help="Include zeros in the quantization."
)
parser.add_argument(
    "--zeros_mode",
    type=str,
    default=None,
    choices=["original", "rescale", "quantized"],
    help="Specify the mode for calculating zeros.",
)

# Parse the arguments
args = parser.parse_args()

# Assign arguments to variables
target = args.target
A_dtype = args.A_dtype
W_dtype = args.W_dtype
accum_dtype = args.accum_dtype
out_dtype = args.out_dtype
layout = args.layout
with_bias = args.with_bias
group_size = args.group_size
with_scaling = args.with_scaling
with_zeros = args.with_zeros
zeros_mode = args.zeros_mode

# Define a list of shared arguments that repeat in every config
shared_args = [
    A_dtype,
    W_dtype,
    out_dtype,
    accum_dtype,
    layout,
    with_bias,
    group_size,
    with_scaling,
    with_zeros,
    zeros_mode,
]

# Define just the (M, K, N) shapes in a more compact list
shapes = [
    # square test
    (1, 16384, 16384),
    # BLOOM-176B
    (1, 43008, 14336),
    (1, 14336, 14336),
    (1, 57344, 14336),
    (1, 14336, 57344),
    # OPT-65B
    (1, 9216, 9216),
    (1, 36864, 9216),
    (1, 9216, 36864),
    (1, 22016, 8192),
    # LLAMA-70B/65B
    (1, 8192, 22016),
    (1, 8192, 8192),
    (1, 28672, 8192),
    (1, 8192, 28672),
    # square test
    (16384, 16384, 16384),
    # BLOOM-176B
    (8192, 43008, 14336),
    (8192, 14336, 14336),
    (8192, 57344, 14336),
    (8192, 14336, 57344),
    # OPT-65B
    (8192, 9216, 9216),
    (8192, 36864, 9216),
    (8192, 9216, 36864),
    (8192, 22016, 8192),
    # LLAMA-70B/65B
    (8192, 8192, 22016),
    (8192, 8192, 8192),
    (8192, 28672, 8192),
    (8192, 8192, 28672),
]

# Build test shapes with all the shared arguments
test_shapes = [(MatmulConfig, Matmul, (*shape, *shared_args)) for shape in shapes]

benchmark_sets = []
benchmark_sets.extend(test_shapes)

benchmark_results = {}
for config_class, operator, input_args in benchmark_sets:
    config = config_class(*input_args)
    matmul = operator(config, target=target, enable_tuning=True)
    kernel_latency = matmul.profile_latency()

    print("Time cost is: {:.3f} ms".format(kernel_latency))

    profile_config = {
        f"{operator.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "BitBLAS_top20_latency": kernel_latency,
        }
    }

    benchmark_results.update(profile_config)

# Define headers for the table
headers = [
    "PrimFunc",
    "Input Arguments",
    "BitBLAS Top20 Latency",
]

# Calculate column widths for pretty printing
col_widths = [0, 0, 0]
for config_key, values in benchmark_results.items():
    args_split = config_key.split("-")
    func_name = args_split[0]
    input_args_str = "-".join(args_split[1:])
    col_widths[0] = max(col_widths[0], len(func_name) + 2, len(headers[0]) + 2)
    col_widths[1] = max(col_widths[1], len(input_args_str) + 2, len(headers[1]) + 2)
    col_widths[2] = max(
        col_widths[2],
        len(f"{values['BitBLAS_top20_latency']:.3f} ms") + 2,
        len(headers[2]) + 2,
    )
    # break only if you want to measure widths from a single example;
    # otherwise, let it loop over all items.

# Print header
for i, header in enumerate(headers):
    headers[i] = header.ljust(col_widths[i])
print("".join(headers))
print("-" * sum(col_widths))

# Print rows
for config_key, values in benchmark_results.items():
    args_split = config_key.split("-")
    func_name = args_split[0]
    input_args_str = "-".join(args_split[1:])
    row = [
        func_name,
        input_args_str,
        f"{values['BitBLAS_top20_latency']:.3f} ms",
    ]
    row_str = "".join(
        [str(cell).ljust(col_widths[idx]) for idx, cell in enumerate(row)]
    )
    print(row_str)
