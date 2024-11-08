import pprint
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import List, Optional

import autogen_manifest
from autogen_manifest import Cutlass3xArgs
from generator_types import GeneratorType, GeneratorTypes
from kernel_compiler import KernelCompiler
from kernel_generator import GeneratorOutput, KernelGenerator
from tqdm import tqdm


@dataclass
class GenerateFromArgInput:
    generator_type: Optional[GeneratorType] = None
    args: Optional[Cutlass3xArgs] = None
    kernel_compiler: Optional[KernelCompiler] = None


def generate_from_arg(input: GenerateFromArgInput) -> GeneratorOutput:
    """
    Kernel generation for a single Cutlass3xArg
    """
    generator_type, args, kernel_compiler = (input.generator_type, input.args,
                                             input.kernel_compiler)
    return KernelGenerator.generate(generator_type, args, kernel_compiler)


def generate_from_args_mt(generator_type: GeneratorType,
                          args: List[Cutlass3xArgs],
                          kernel_compiler: KernelCompiler,
                          num_threads: int = 32) -> GeneratorOutput:
    """
    Kernel generator for a list of Cutlass3xArgs with multi-threading.
    """
    generator_outputs = GeneratorOutput()
    # create thread pool with {num_threads} threads
    pool = ThreadPool(processes=num_threads)
    inputs = [
        GenerateFromArgInput(generator_type, x, kernel_compiler) for x in args
    ]
    result = pool.map_async(generate_from_arg, inputs)
    for r in result.get():
        generator_outputs.merge(r)
    return generator_outputs


def main(args):
    pprint.pprint(args)

    cutlass_args_list = getattr(autogen_manifest, args.cutlass_args_list)
    print(f"Generating {len(cutlass_args_list)} cuda files ...")

    generator_type: GeneratorType = GeneratorType.from_str(args.generator_type)

    additional_compile_args = [x.strip() for x in args.additional_compile_args]
    kernel_compiler: KernelCompiler = KernelCompiler(
        vllm_root_dir=args.vllm_root_dir,
        py_venv_dir=args.py_venv_dir,
        cuda_dir=args.cuda_dir,
        py_version=args.py_version,
        additional_args=additional_compile_args,
        test_compile=args.test_compile)
    kernel_compiler.init_compile_cache()

    generator_outputs = GeneratorOutput()
    batch_size = 100  # Compile-and-Generate batch_size items at a time
    for idx in tqdm(range(0, len(cutlass_args_list), batch_size)):
        print(f"Total {len(cutlass_args_list)}"
              f" | Success {len(generator_outputs.success_file_names)}"
              f"| Fail {len(generator_outputs.failed_file_names)}")

        chunk_generator_output = generate_from_args_mt(
            generator_type, cutlass_args_list[idx:idx + batch_size],
            kernel_compiler)
        generator_outputs.merge(chunk_generator_output)

        # Store intermediate results
        # fill-out ops.h
        KernelGenerator.write_ops(generator_type, generator_outputs.file_paths,
                                  generator_outputs.fn_names,
                                  generator_outputs.fn_decls)
        # store result batch
        kernel_compiler.cache.add(generator_outputs.success_file_names,
                                  generator_outputs.failed_file_names)
        kernel_compiler.cache.store()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='''
            Autogen cutlass kernels
            Example: 
            python3 csrc/quantization/cutlass_w8a8/generator/generator.py \
                 --generator-type scaled_sparse_mm \
                 --vllm-root-dir ${HOME}/code/nm-vllm-ent/nm-vllm-ent/ \
                 --py-venv-dir ${HOME}/code/nm-vllm-ent/nm-vllm-ent/vllm-test \
                 --cuda-dir /usr/local/cuda-12.5
            ''')

    parser.add_argument("--generator-type",
                        required=True,
                        choices=[x.description() for x in GeneratorTypes])
    parser.add_argument("--cutlass-args-list",
                        required=True,
                        type=str,
                        default=None,
                        help='''
                        The cutlass args list variable name constructed in
                        autogen_manifest.py. The variable name is imported
                        as,
                        getattr(autogen_manifest, args.cutlass_args_list)
                        ''')
    parser.add_argument('--test-compile',
                        action='store_true',
                        help='''
                        Runs as usual but,
                            - Prints compiler errors
                            - Doesn't update the kernel compiler cache.
                        ''')
    parser.add_argument("--vllm-root-dir",
                        required=True,
                        type=str,
                        default=None,
                        help="Root directory of vllm source code")
    parser.add_argument("--py-venv-dir",
                        required=True,
                        type=str,
                        default=None,
                        help="py venv root directory")
    parser.add_argument("--cuda-dir",
                        type=str,
                        default=None,
                        help="CUDA dir example: /usr/local/cuda-12.5")
    parser.add_argument(
        "--py-version",
        type=str,
        default="3.10",
        help="Python version to use. Used in fetching the python includes")
    parser.add_argument("--additional-compile-args", nargs='*', default=[])

    args = parser.parse_args()
    main(args)
