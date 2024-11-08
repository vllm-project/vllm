"""
Utilities to invoke the kernel compiler.
When generating cutlass kernels, we attempt an nvcc compile to make sure that
there won't be any issues at vllm build time.
"""

import pickle as pkl
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Global compile cache path that stores information about which kernels
# compiled successfully and which failed.
CACHE_FILE_PATH = Path('./kernels_compile_cache.pkl')


class KernelCompileCache:

    def __init__(self, test_compile=False):
        # If test_compile is true, we override the cache operations so it
        # is a no-op.
        self.test_compile = test_compile

        # self.bad_kernels are kernels that failed compilation
        # self.good_kernels are kernels that succeeded compilation
        if not CACHE_FILE_PATH.exists() or self.test_compile:
            self.bad_kernels = []
            self.good_kernels = []
        else:
            # Load from cache
            data = None
            with open(str(CACHE_FILE_PATH), 'rb') as f:
                data = pkl.load(f)
            self.bad_kernels, self.good_kernels = data
        print(f"#bad kernels {len(self.bad_kernels)},"
              f"#good kernels {len(self.good_kernels)} loaded from cache ...")

    def is_bad_kernel(self, kernel_file_name: str):
        if self.test_compile:
            return False
        return kernel_file_name in self.bad_kernels

    def is_good_kernel(self, kernel_file_name: str):
        if self.test_compile:
            return False
        return kernel_file_name in self.good_kernels

    def add(self, success: List[str], fail: List[str]):
        self.good_kernels.extend(success)
        self.bad_kernels.extend(fail)
        # Remove duplicates
        self.good_kernels = list(set(self.good_kernels))
        self.bad_kernels = list(set(self.bad_kernels))

    def store(self):
        if self.test_compile:
            return
        print(f"Storing #badkernels {len(self.bad_kernels)}, "
              f"#goodkernels {len(self.good_kernels)}")
        with open(str(CACHE_FILE_PATH), 'wb+') as f:
            pkl.dump((self.bad_kernels, self.good_kernels), f)


@dataclass
class KernelCompiler:
    # vllm source code directory path
    vllm_root_dir: Optional[str] = None
    # python venv directory path
    py_venv_dir: Optional[str] = None
    # cuda directory path. example : /usr/local/cuda-12.5
    cuda_dir: Optional[str] = None
    #python version
    py_version: str = '3.10'
    # any additional flags
    additional_args: List[str] = field(default_factory=lambda: [])
    # kernel compile cache. Cache that holds history of which kernels
    # succeeded and failed compilation.
    cache: Optional[KernelCompileCache] = None
    # Print nvcc compile information and override cache updates.
    test_compile: bool = False

    def init_compile_cache(self):
        self.cache = KernelCompileCache(self.test_compile)

    def compile(self, cu_file: str, gencode_arch: str) -> bool:
        compile_command_base = [
            'nvcc',
            '-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1',
            f'-I{self.vllm_root_dir}/csrc',
            f'-I{self.vllm_root_dir}/.deps/cutlass-src/include',  #noqa
            '-isystem',
            f'/usr/include/python{self.py_version}',
            '-isystem',
            f'{self.py_venv_dir}/lib/python3.10/site-packages/torch/include',
            '-isystem',
            f'{self.py_venv_dir}/lib/python3.10/site-packages/torch/include/torch/csrc/api/include',  #noqa
            '-isystem',
            f'{self.cuda_dir}/include',
            '-gencode',
            f'arch=compute_{gencode_arch},code=sm_{gencode_arch}',
            '-DONNX_NAMESPACE=onnx_c2',
            '-Xcudafe',
            '-DNDEBUG',
            '-std=c++17',
            '-Xcompiler=-fPIC',
            '--expt-relaxed-constexpr',
            '--threads=1',
            '-D_GLIBCXX_USE_CXX11_ABI=0'] + self.additional_args
        if gencode_arch == 90:
            compile_command_base += ['-gencode', 'arch=compute_90a,code=sm_90a']

        result = subprocess.run(compile_command_base + ['-c', cu_file],
                                capture_output=True)

        if self.test_compile:
            print(f"Compiling {cu_file} : \n"
                  f"   Successful compilation: {result.returncode == 0}\n"
                  f"   stdout : {result.stdout}\n"
                  f"   stderr : {result.stderr}\n")

        if result.returncode == 0:
            # Cleanup generated object code on successful compile.
            object_file_path = Path("./" + Path(cu_file).stem + '.o')
            assert object_file_path.exists(), object_file_path
            object_file_path.unlink()

        return result.returncode == 0
