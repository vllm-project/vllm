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

MAIN_CUDA_VERSION = "12.1"

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package."
    )


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
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
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by ; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}."
        )
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}."
        )
    return arch_list


# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()
if not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported."
            )
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    # If no GPU is specified nor available, add all supported architectures
    # based on the NVCC CUDA version.
    compute_capabilities = SUPPORTED_ARCHS.copy()
    if nvcc_cuda_version < Version("11.1"):
        compute_capabilities.remove("8.6")
    if nvcc_cuda_version < Version("11.8"):
        compute_capabilities.remove("8.9")
        compute_capabilities.remove("9.0")

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if nvcc_cuda_version < Version("11.1"):
    if any(cc.startswith("8.6") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 11.1 or higher is required for compute capability 8.6."
        )
if nvcc_cuda_version < Version("11.8"):
    if any(cc.startswith("8.9") for cc in compute_capabilities):
        # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
        # However, GPUs with compute capability 8.9 can also run the code generated by
        # the previous versions of CUDA 11 and targeting compute capability 8.0.
        # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
        # instead of 8.9.
        warnings.warn(
            "CUDA 11.8 or higher is required for compute capability 8.9. "
            "Targeting compute capability 8.0 instead."
        )
        compute_capabilities = set(
            cc for cc in compute_capabilities if not cc.startswith("8.9")
        )
        compute_capabilities.add("8.0+PTX")
    if any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 11.8 or higher is required for compute capability 9.0."
        )

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    num = capability[0] + capability[2]
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

# Use NVCC threads to parallelize the build.
if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 8)
    NVCC_FLAGS += ["--threads", str(num_threads)]

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
        "csrc/quantization/squeezellm/quant_cuda_kernel.cu",
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


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_vllm_version() -> str:
    version = find_version(get_path("vllm", "__init__.py"))
    cuda_version = str(nvcc_cuda_version)
    if cuda_version != MAIN_CUDA_VERSION:
        cuda_version_str = cuda_version.replace(".", "")[:3]
        version += f"+cu{cuda_version_str}"
    return version


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="vllm",
    version=get_vllm_version(),
    author="vLLM Team",
    license="Apache 2.0",
    description=(
        "A high-throughput and memory-efficient inference and "
        "serving engine for LLMs"
    ),
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
    packages=setuptools.find_packages(
        exclude=("benchmarks", "csrc", "docs", "examples", "tests")
    ),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    package_data={"vllm": ["py.typed"]},
)
