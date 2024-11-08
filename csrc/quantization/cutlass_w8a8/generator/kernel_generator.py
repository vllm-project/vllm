"""
Kernel Generator classes / functions.
"""

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import jinja2
import utils
from autogen_manifest import Cutlass3xArgs
from generator_types import GeneratorType
from kernel_compiler import KernelCompiler


@dataclass
class GeneratorOutput:
    # Used in torch_bindings generation
    file_paths: List[str] = field(default_factory=lambda: [])
    fn_names: List[str] = field(default_factory=lambda: [])
    fn_decls: List[str] = field(default_factory=lambda: [])
    # Used in cache update
    failed_file_names: List[str] = field(default_factory=lambda: [])
    success_file_names: List[str] = field(default_factory=lambda: [])

    def merge(self, output: "GeneratorOutput"):
        self.file_paths.extend(output.file_paths)
        self.fn_names.extend(output.fn_names)
        self.fn_decls.extend(output.fn_decls)
        self.failed_file_names.extend(output.failed_file_names)
        self.success_file_names.extend(output.success_file_names)


## Abstract generator


class KernelGenerator_(ABC):
    SCRIPT_DIR = utils.get_script_dir()
    GENERATE_DIR = SCRIPT_DIR / "generated"

    @staticmethod
    def write_torch_bindings(generator_type: GeneratorType,
                             fn_names: List[str], fn_decls: List[str],
                             ops_macro: str, dir_path: str):
        s = "#pragma once\n"
        s += "#include<torch/torch.h>\n"
        s += f"#define {ops_macro} \\\n"
        for fn_name in fn_names:
            s += generator_type.ops_def(fn_name) + '\\\n'
            s += generator_type.ops_impl(fn_name) + '\\\n'
        s += "\n"

        for fn_decl in fn_decls:
            s += f'{fn_decl}\n'

        # write ops.h
        file_path = Path(dir_path) / "ops.h"
        with open(str(file_path), 'w+') as f:
            f.write(s)

        # write torch_bindings.cpp
        s = ""
        s += '\n#include "core/registration.h"'
        s += '\n#include <torch/library.h>'
        s += '\n#include "ops.h"'
        s += '\nTORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {'
        s += f'\n {ops_macro}'
        s += '\n}'
        s += '\nREGISTER_EXTENSION(TORCH_EXTENSION_NAME)'
        s += '\n'

        tb_path = Path(dir_path) / "torch_bindings.cpp"
        with open(str(tb_path), 'w+') as f:
            f.write(s)

    @staticmethod
    def write_ops(generator_type: GeneratorType,
                  file_paths: List[str],
                  fn_names: List[str],
                  fn_decls: List[str],
                  ops_macro: str,
                  batch_size: int = 100):
        """
        batch_size defines the number of files per .so.
        If there are a 1000 filenames, then with batch_size 100, we generate
        10 directories, each directory containing 100 kernels. Each directory
        is converted into a .so during vllm compile.
        """

        assert len(file_paths) == len(fn_names)
        assert len(file_paths) == len(fn_decls)

        dir_name = 0
        for i in range(0, len(file_paths), batch_size):

            dir_path: Path = KernelGenerator_.GENERATE_DIR / f'{dir_name}'
            dir_path.mkdir(exist_ok=True)

            # Move files to dir
            for file_path in file_paths[i:i + batch_size]:
                if Path(file_path).exists():
                    shutil.move(file_path, str(dir_path))

            KernelGenerator_.write_torch_bindings(generator_type,
                                                  fn_names[i:i + batch_size],
                                                  fn_decls[i:i + batch_size],
                                                  ops_macro, dir_path)

            dir_name += 1  #noqa

    @staticmethod
    def last_namespace(s):
        return s.split('::')[-1]

    @staticmethod
    @abstractmethod
    def generate(generator_type: GeneratorType, args: Cutlass3xArgs,
                 kernel_compiler: KernelCompiler) -> GeneratorOutput:
        ...


class KernelGenerator(KernelGenerator_):
    OPS_MACRO = "CUTLASS_DEFS"

    @staticmethod
    def generate_name(description: str, args: Cutlass3xArgs):

        return 'autogen_{}_{}_{}x{}x{}_{}x{}x{}_{}_{}_{}_{}_{}_{}'.format(
            description, args.arch, args.tile_shape[0], args.tile_shape[1],
            args.tile_shape[2], args.cluster_shape[0], args.cluster_shape[1],
            args.cluster_shape[2],
            KernelGenerator_.last_namespace(args.kernel_schedule),
            KernelGenerator_.last_namespace(args.epilogue_schedule),
            KernelGenerator_.last_namespace(args.tile_schedule),
            KernelGenerator_.last_namespace(args.gemm_mode),
            KernelGenerator_.last_namespace(args.acc_type), args.dtype_str)

    @staticmethod
    def generate_filename(description: str, args: Cutlass3xArgs):

        f = '{}/autogen_{}_{}x{}x{}_{}x{}x{}_{}_{}_{}_{}_{}_{}_{}'.format(
            KernelGenerator_.GENERATE_DIR, description, args.tile_shape[0],
            args.tile_shape[1], args.tile_shape[2], args.cluster_shape[0],
            args.cluster_shape[1], args.cluster_shape[2],
            KernelGenerator_.last_namespace(args.kernel_schedule),
            KernelGenerator_.last_namespace(args.epilogue_schedule),
            KernelGenerator_.last_namespace(args.tile_schedule),
            KernelGenerator_.last_namespace(args.gemm_mode),
            KernelGenerator_.last_namespace(args.acc_type), args.dtype_str,
            args.arch)

        f = f + ".cu"
        return f

    @staticmethod
    def generate_kernel_file(generator_type: GeneratorType,
                             args: Cutlass3xArgs) -> Tuple[str, str]:
        """
        Generate a .cu file that respects args and return,
         - The function name of the generated function.
         - The c++ function declaration of the generated function.
        The return values are used in generating the torch bindings.
        """

        # Make the generate dir
        KernelGenerator_.GENERATE_DIR.mkdir(exist_ok=True)

        # Get jinja templates
        jenv = jinja2.Environment(loader=jinja2.FileSystemLoader("/"))
        fn_defn_template = jenv.get_template(
            str(generator_type.fn_defn_jinja_filepath()))
        fn_decl_template = jenv.get_template(
            str(generator_type.fn_decl_jinja_filepath()))

        # Generate code
        fn_name = KernelGenerator.generate_name(generator_type.description(),
                                                args)
        fn_decl = fn_decl_template.render(_name=fn_name)
        code: str = fn_defn_template.render(
            _name=fn_name,
            _torch_input_dtype=utils.to_torch_dtype_str(args.dtype_str),
            _cutlass_input_dtype=utils.to_cutlass_dtype_str(args.dtype_str),
            _tile_shape=utils.get_as_cutlass3x_gemm_shape(args.tile_shape),
            _cluster_shape=utils.get_as_cutlass3x_gemm_shape(
                args.cluster_shape),
            _kernel_schedule=args.kernel_schedule,
            _epilogue_schedule=args.epilogue_schedule,
            _tile_schedule=args.tile_schedule,
            _gemm_mode=args.gemm_mode,
            _acc_type=args.acc_type)

        filename = KernelGenerator.generate_filename(
            generator_type.description(), args)
        if utils.file_contents_same(filename, code):
            return (fn_name, fn_decl)

        # write code
        with open(filename, "w+") as f:
            f.write(code)

        return (fn_name, fn_decl)

    @staticmethod
    def generate(generator_type: GeneratorType, args: Cutlass3xArgs,
                 kernel_compiler: KernelCompiler) -> GeneratorOutput:
        generator_output = GeneratorOutput()

        filepath = KernelGenerator.generate_filename(
            generator_type.description(), args)
        filename = Path(filepath).name

        if kernel_compiler.cache.is_bad_kernel(filename):
            # We know that this kernel wouldn't compile. Abort
            return generator_output

        fn_name, fn_decl = KernelGenerator.generate_kernel_file(
            generator_type, args)

        if not kernel_compiler.cache.is_good_kernel(filename):
            # We dont have any information about this kernel in the cache.
            # try compiling
            compile_success = kernel_compiler.compile(filepath,
                                                      gencode_arch=args.arch)
            if compile_success:
                generator_output.success_file_names.append(filename)
            else:
                generator_output.failed_file_names.append(filename)
                if not kernel_compiler.test_compile:
                    # Remove generated file
                    Path(filepath).unlink()
                    return generator_output

        generator_output.file_paths.append(filepath)
        generator_output.fn_names.append(fn_name)
        generator_output.fn_decls.append(fn_decl)

        return generator_output

    @staticmethod
    def write_ops(generator_type: GeneratorType, file_paths: List[str],
                  fn_names: List[str], fn_decls: List[str]):
        return KernelGenerator_.write_ops(generator_type, file_paths, fn_names,
                                          fn_decls, KernelGenerator.OPS_MACRO)
