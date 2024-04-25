from setuptools import Extension, setup
import torch
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from typing import List
import sys

import pprint

def build_extension(lib_name: str, sources: List[str]) -> str:
    return torch.utils.cpp_extension.load(
        name=lib_name,
        sources=sources,
        #extra_cflags=['-O2',f'-DLIBRARY_NAME={lib_name}'],
        extra_cflags=['-g',f'-DLIBRARY_NAME={lib_name}'],
        verbose=True,
        is_python_module=False,
    )

    sys.argv.append("build_ext")
    sys.argv.append("-q")
    sys.argv.append("--inplace")

    CXX_FLAGS = []
    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    CXX_FLAGS += [f'-DLIBRARY_NAME={lib_name}']

    ext = CppExtension(
        name=f"{lib_name}",
        sources=sources,
        extra_compile_args=CXX_FLAGS
    )

    #ext = CUDAExtension(
    #    name=lib_name,
    #    sources=sources,
    #    extra_compile_args={
    #        'cxx': [f'-DLIBRARY_NAME={lib_name}'],
    #        'nvcc': []
    #    }
    #)

    res = setup(ext_modules=[ext])
    pprint.pprint(vars(ext))
    return ext._file_name


#def import_ops():
#    import torch
#    from pathlib import Path
#    script_dir = Path(__file__).parent.resolve()
#    torch.ops.load_library(f"{script_dir}/nm_ops.so")

def load_extension(lib_name: str):
    if lib_name:
        print(f"loading {lib_name}")
        torch.ops.load_library(lib_name)


if __name__ == '__main__':
    ext = build_extension("foo", ["foo.cpp"])
    print(ext)

    load_extension(ext)

    print(torch.ops.foo)
    print(dir(torch.ops.foo))
    #print(torch.ops.foo.func("world"))
    #print(torch.ops.foo.module_version())
    print(torch.ops.foo.func2(torch.tensor([4.0, 4.0])))
