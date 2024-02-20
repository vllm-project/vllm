#!/usr/bin/env python3

import argparse
import os

from torch.utils.hipify.hipify_python import hipify

if __name__ == '__main__':
    print(f"CWD {os.getcwd()}")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--build_dir",
        help="The build directory.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help="The output directory.",
    )

    parser.add_argument(
        "-i",
        "--include_dir",
        help="Include directory",
        action="append",
        default=[],
    )

    parser.add_argument(
        "sources",
        help="Source files to hipify.",
        nargs="*",
        default=[]
    )

    args = parser.parse_args()

    print(args.output_dir)

    # limit scope to build_dir only
    includes = [os.path.join(args.build_dir, '*')]
    print(f"includes {includes}")

    extra_files = [os.path.abspath(s) for s in args.sources]
    print(f"extra_files {extra_files}")

    hipify_result = hipify(
        project_directory=args.build_dir,
        output_directory=args.output_dir,
        header_include_dirs=[],
        includes=includes,
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True)

    #print(hipify_result)

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (hipify_result[s_abs].hipified_path if (s_abs in hipify_result and
                          hipify_result[s_abs].hipified_path is not None) else s_abs)
        if True:
            hipified_sources.append(hipified_s_abs)
        else:
            hipified_sources.append(
                os.path.relpath(hipified_s_abs,
                                os.path.abspath(os.path.join(args.build_dir, os.pardir))))

    assert(len(hipified_sources) == len(args.sources))

    #    print("\n".join(hipified_sources))

#    print(f"got here {args.output_dir}")
#    os.system(f"find {args.output_dir} -name '*.hip'")
#    print("end got here")

#    print(f"got here root")
#    os.system(f"find /app/vllm -name '*.hip'")
#    print("end got here root")

# project_directory /app/vllm
# show_detailed True
# extensions ('.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.in', '.hpp')
# header_extensions ('.cuh', '.h', '.hpp')
# output_directory /app/vllm
# header_include_dirs []
# includes ['/app/vllm/*']
# extra_files [
#     '/app/vllm/csrc/cache_kernels.cu',
#     '/app/vllm/csrc/attention/attention_kernels.cu',
#     '/app/vllm/csrc/pos_encoding_kernels.cu',
#     '/app/vllm/csrc/activation_kernels.cu',
#     '/app/vllm/csrc/layernorm_kernels.cu',
#     '/app/vllm/csrc/quantization/squeezellm/quant_cuda_kernel.cu',
#     '/app/vllm/csrc/quantization/gptq/q_gemm.cu',
#     '/app/vllm/csrc/cuda_utils_kernels.cu',
#     '/app/vllm/csrc/moe_align_block_size_kernels.cu',
#     '/app/vllm/csrc/pybind.cpp'
# ]
# out_of_place_only False
# ignores ()
# show_progress True
# hip_clang_launch False
# is_pytorch_extension True
# hipify_extra_files_only True
