from typing import List

import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME


# Build custom operators.
CXX_FLAGS = ["-g"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2"]

if not torch.cuda.is_available():
    raise RuntimeError(
        f"Cannot find CUDA at CUDA_HOME: {CUDA_HOME}. "
        "CUDA must be available in order to build the package.")

# FIXME(woosuk): Consider the case where the machine has multiple GPUs with
# different compute capabilities.
compute_capability = torch.cuda.get_device_capability()
major, minor = compute_capability
# Enable bfloat16 support if the compute capability is >= 8.0.
if major >= 8:
    NVCC_FLAGS.append("-DENABLE_BF16")

ext_modules = []

# Cache operations.
cache_extension = CUDAExtension(
    name="cacheflow.cache_ops",
    sources=["csrc/cache.cpp", "csrc/cache_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(cache_extension)

# Attention kernels.
attention_extension = CUDAExtension(
    name="cacheflow.attention_ops",
    sources=["csrc/attention.cpp", "csrc/attention/attention_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(attention_extension)

# Positional encoding kernels.
positional_encoding_extension = CUDAExtension(
    name="cacheflow.pos_encoding_ops",
    sources=["csrc/pos_encoding.cpp", "csrc/pos_encoding_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(positional_encoding_extension)

# Layer normalization kernels.
layernorm_extension = CUDAExtension(
    name="cacheflow.layernorm_ops",
    sources=["csrc/layernorm.cpp", "csrc/layernorm_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(layernorm_extension)

# Activation kernels.
activation_extension = CUDAExtension(
    name="cacheflow.activation_ops",
    sources=["csrc/activation.cpp", "csrc/activation_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(activation_extension)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open("requirements.txt") as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="cacheflow",
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
