import setuptools
from torch.utils import cpp_extension

CXX_FLAGS = ['-g']
NVCC_FLAGS = ['-O2']


ext_modules = []

# Cache operations.
cache_extension = cpp_extension.CUDAExtension(
    name='cacheflow.ops',
    sources=['csrc/cache.cpp', 'csrc/cache_kernel.cu'],
    extra_compile_args={'cxx': CXX_FLAGS, 'nvcc': NVCC_FLAGS},
)
ext_modules.append(cache_extension)

setuptools.setup(
    name='cacheflow',
    requires_python='>=3.9',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
