import subprocess
from typing import List, Set

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

# Compiler flags.
CXX_FLAGS = ["-g", "-O2"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2"]


if not torch.cuda.is_available():
    raise RuntimeError(
        f"Cannot find CUDA at CUDA_HOME: {CUDA_HOME}. "
        "CUDA must be available in order to build the package.")


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


# Collect the compute capabilities of all available GPUs.
device_count = torch.cuda.device_count()
compute_capabilities: Set[int] = set()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 7:
        raise RuntimeError(
            "GPUs with compute capability less than 7.0 are not supported.")
    compute_capabilities.add(major * 10 + minor)
# If no GPU is available, add all supported compute capabilities.
if not compute_capabilities:
    compute_capabilities = {70, 75, 80, 86, 90}
# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    NVCC_FLAGS += ["-gencode", f"arch=compute_{capability},code=sm_{capability}"]

# Validate the NVCC CUDA version.
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if 86 in compute_capabilities and nvcc_cuda_version < Version("11.1"):
    raise RuntimeError(
        "CUDA 11.1 or higher is required for GPUs with compute capability 8.6.")
if 90 in compute_capabilities and nvcc_cuda_version < Version("11.8"):
    raise RuntimeError(
        "CUDA 11.8 or higher is required for GPUs with compute capability 9.0.")

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
