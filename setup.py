import io
import os
import re
import subprocess
from typing import List, Set
import warnings

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


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


def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime-compiled
    # and executed on the 8.6 or newer architectures. While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    valid_arch_strs = SUPPORTED_ARCHS + [s + "+PTX" for s in SUPPORTED_ARCHS]
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if arch_list is None:
        return set()

    # List are separated by ; or space.
    arch_list = arch_list.replace(" ", ";").split(";")
    for arch in arch_list:
        if arch not in valid_arch_strs:
            raise ValueError(
                f"Unsupported CUDA arch ({arch}). "
                f"Valid CUDA arch strings are: {valid_arch_strs}.")
    return set(arch_list)


ext_modules = []

# Cache operations.
cache_extension = CUDAExtension(
    name="vllm.cache_ops",
    sources=["csrc/cache.cpp", "csrc/cache_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(cache_extension)

# Attention kernels.
attention_extension = CUDAExtension(
    name="vllm.attention_ops",
    sources=["csrc/attention.cpp", "csrc/attention/attention_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(attention_extension)

# Positional encoding kernels.
positional_encoding_extension = CUDAExtension(
    name="vllm.pos_encoding_ops",
    sources=["csrc/pos_encoding.cpp", "csrc/pos_encoding_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(positional_encoding_extension)

# Layer normalization kernels.
layernorm_extension = CUDAExtension(
    name="vllm.layernorm_ops",
    sources=["csrc/layernorm.cpp", "csrc/layernorm_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(layernorm_extension)

# Activation kernels.
activation_extension = CUDAExtension(
    name="vllm.activation_ops",
    sources=["csrc/activation.cpp", "csrc/activation_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(activation_extension)

# Quantization kernels.
quantization_extension = CUDAExtension(
    name="vllm.quantization_ops",
    sources=[
        "csrc/quantization.cpp",
        "csrc/quantization/awq/gemm_kernels.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(quantization_extension)

# Misc. CUDA utils.
cuda_utils_extension = CUDAExtension(
    name="vllm.cuda_utils",
    sources=["csrc/cuda_utils.cpp", "csrc/cuda_utils_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(cuda_utils_extension)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="vllm",
    version=find_version(get_path("vllm", "__init__.py")),
    author="vLLM Team",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm",
        "Documentation": "https://vllm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("benchmarks", "csrc", "docs",
                                               "examples", "tests")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
