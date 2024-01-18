import io
import os
import re
import subprocess
from typing import List, Set
import warnings

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, ROCM_HOME

ROOT_DIR = os.path.dirname(__file__)

MAIN_CUDA_VERSION = "12.1"

# Supported NVIDIA GPU architectures.
NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}
ROCM_SUPPORTED_ARCHS = {"gfx90a", "gfx908", "gfx906", "gfx1030", "gfx1100"}
# SUPPORTED_ARCHS = NVIDIA_SUPPORTED_ARCHS.union(ROCM_SUPPORTED_ARCHS)


def _is_hip() -> bool:
    return torch.version.hip is not None


def _is_neuron() -> bool:
    torch_neuronx_installed = True
    try:
        subprocess.run(["neuron-ls"], capture_output=True, check=True)
    except FileNotFoundError as e:
        torch_neuronx_installed = False
    return torch_neuronx_installed


def _is_cuda() -> bool:
    return (torch.version.cuda is not None) and not _is_neuron()


# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2", "-std=c++17"]

if _is_hip():
    if ROCM_HOME is None:
        raise RuntimeError(
            "Cannot find ROCM_HOME. ROCm must be available to build the package."
        )
    NVCC_FLAGS += ["-DUSE_ROCM"]

if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


def get_amdgpu_offload_arch():
    command = "/opt/rocm/llvm/bin/amdgpu-offload-arch"
    try:
        output = subprocess.check_output([command])
        return output.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        error_message = f"Error: {e}"
        raise RuntimeError(error_message) from e
    except FileNotFoundError as e:
        # If the command is not found, print an error message
        error_message = f"The command {command} was not found."
        raise RuntimeError(error_message) from e

    return None


def get_hipcc_rocm_version():
    # Run the hipcc --version command
    result = subprocess.run(['hipcc', '--version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)

    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Error running 'hipcc --version'")
        return None

    # Extract the version using a regular expression
    match = re.search(r'HIP version: (\S+)', result.stdout)
    if match:
        # Return the version string
        return match.group(1)
    else:
        print("Could not find HIP version in the output")
        return None


def get_neuronxcc_version():
    import sysconfig
    site_dir = sysconfig.get_paths()["purelib"]
    version_file = os.path.join(site_dir, "neuronxcc", "version", "__init__.py")

    # Check if the command was executed successfully
    with open(version_file, "rt") as fp:
        content = fp.read()

    # Extract the version using a regular expression
    match = re.search(r"__version__ = '(\S+)'", content)
    if match:
        # Return the version string
        return match.group(1)
    else:
        raise RuntimeError("Could not find HIP version in the output")


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
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by ; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = NVIDIA_SUPPORTED_ARCHS.union(
        {s + "+PTX"
         for s in NVIDIA_SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA/ROCM architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA/ROCM architectures are: {valid_archs}.")
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA/ROCM architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA/ROCM architectures are: "
            f"{valid_archs}.",
            stacklevel=2)
    return arch_list


# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()
if _is_cuda() and not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

if _is_cuda():
    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    if not compute_capabilities:
        # If no GPU is specified nor available, add all supported architectures
        # based on the NVCC CUDA version.
        compute_capabilities = NVIDIA_SUPPORTED_ARCHS.copy()
        if nvcc_cuda_version < Version("11.1"):
            compute_capabilities.remove("8.6")
        if nvcc_cuda_version < Version("11.8"):
            compute_capabilities.remove("8.9")
            compute_capabilities.remove("9.0")
    # Validate the NVCC CUDA version.
    if nvcc_cuda_version < Version("11.0"):
        raise RuntimeError(
            "CUDA 11.0 or higher is required to build the package.")
    if (nvcc_cuda_version < Version("11.1")
            and any(cc.startswith("8.6") for cc in compute_capabilities)):
        raise RuntimeError(
            "CUDA 11.1 or higher is required for compute capability 8.6.")
    if nvcc_cuda_version < Version("11.8"):
        if any(cc.startswith("8.9") for cc in compute_capabilities):
            # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
            # However, GPUs with compute capability 8.9 can also run the code generated by
            # the previous versions of CUDA 11 and targeting compute capability 8.0.
            # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
            # instead of 8.9.
            warnings.warn(
                "CUDA 11.8 or higher is required for compute capability 8.9. "
                "Targeting compute capability 8.0 instead.",
                stacklevel=2)
            compute_capabilities = set(cc for cc in compute_capabilities
                                       if not cc.startswith("8.9"))
            compute_capabilities.add("8.0+PTX")
        if any(cc.startswith("9.0") for cc in compute_capabilities):
            raise RuntimeError(
                "CUDA 11.8 or higher is required for compute capability 9.0.")

    # Add target compute capabilities to NVCC flags.
    for capability in compute_capabilities:
        num = capability[0] + capability[2]
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += [
                "-gencode", f"arch=compute_{num},code=compute_{num}"
            ]

    # Use NVCC threads to parallelize the build.
    if nvcc_cuda_version >= Version("11.2"):
        nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
        num_threads = min(os.cpu_count(), nvcc_threads)
        NVCC_FLAGS += ["--threads", str(num_threads)]

elif _is_hip():
    amd_arch = get_amdgpu_offload_arch()
    if amd_arch not in ROCM_SUPPORTED_ARCHS:
        raise RuntimeError(
            f"Only the following arch is supported: {ROCM_SUPPORTED_ARCHS}"
            f"amdgpu_arch_found: {amd_arch}")

elif _is_neuron():
    neuronxcc_version = get_neuronxcc_version()

ext_modules = []

vllm_extension_sources = [
    "csrc/cache_kernels.cu",
    "csrc/attention/attention_kernels.cu",
    "csrc/pos_encoding_kernels.cu",
    "csrc/activation_kernels.cu",
    "csrc/layernorm_kernels.cu",
    "csrc/quantization/squeezellm/quant_cuda_kernel.cu",
    "csrc/quantization/gptq/q_gemm.cu",
    "csrc/cuda_utils_kernels.cu",
    "csrc/pybind.cpp",
]

if _is_cuda():
    vllm_extension_sources.append("csrc/quantization/awq/gemm_kernels.cu")

if not _is_neuron():
    vllm_extension = CUDAExtension(
        name="vllm._C",
        sources=vllm_extension_sources,
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(vllm_extension)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_vllm_version() -> str:
    version = find_version(get_path("vllm", "__init__.py"))

    if _is_hip():
        # Get the HIP version
        hipcc_version = get_hipcc_rocm_version()
        if hipcc_version != MAIN_CUDA_VERSION:
            rocm_version_str = hipcc_version.replace(".", "")[:3]
            version += f"+rocm{rocm_version_str}"
    elif _is_neuron():
        # Get the Neuron version
        neuron_version = str(neuronxcc_version)
        if neuron_version != MAIN_CUDA_VERSION:
            neuron_version_str = neuron_version.replace(".", "")[:3]
            version += f"+neuron{neuron_version_str}"
    else:
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
    if _is_hip():
        with open(get_path("requirements-rocm.txt")) as f:
            requirements = f.read().strip().split("\n")
    elif _is_neuron():
        with open(get_path("requirements-neuron.txt")) as f:
            requirements = f.read().strip().split("\n")
    else:
        with open(get_path("requirements.txt")) as f:
            requirements = f.read().strip().split("\n")
    return requirements


package_data = {"vllm": ["py.typed"]}
if os.environ.get("VLLM_USE_PRECOMPILED"):
    ext_modules = []
    package_data["vllm"].append("*.so")

setuptools.setup(
    name="vllm",
    version=get_vllm_version(),
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
    cmdclass={"build_ext": BuildExtension} if not _is_neuron() else {},
    package_data=package_data,
)
