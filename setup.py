import setuptools
from torch.utils import cpp_extension

CXX_FLAGS = ['-g']
NVCC_FLAGS = ['-O2']


ext_modules = []

# Cache operations.
cache_extension = cpp_extension.CUDAExtension(
    name='cacheflow.cache_ops',
    sources=['csrc/cache.cpp', 'csrc/cache_kernels.cu'],
    extra_compile_args={'cxx': CXX_FLAGS, 'nvcc': NVCC_FLAGS},
)
ext_modules.append(cache_extension)

# Attention kernels.
attention_extension = cpp_extension.CUDAExtension(
    name='cacheflow.attention_ops',
    sources=['csrc/attention.cpp', 'csrc/attention_kernels.cu'],
    extra_compile_args={'cxx': CXX_FLAGS, 'nvcc': NVCC_FLAGS},
)
ext_modules.append(attention_extension)

setuptools.setup(
    name='cacheflow',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
