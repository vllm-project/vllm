import io
import os
import re
import subprocess
from typing import List

from packaging.version import parse, Version
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from shutil import which
import torch
from torch.utils.cpp_extension import CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

# vLLM only supports Linux platform
assert sys.platform.startswith(
    "linux"), "vLLM only supports Linux platform (including WSL)."

MAIN_CUDA_VERSION = "12.1"


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


class CMakeExtension(Extension):

    def __init__(self, name, cmake_lists_dir='.', **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):

    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        for ext in self.extensions:

            extdir = os.path.abspath(
                os.path.dirname(self.get_ext_fullpath(ext.name)))

            # Note: optimization level + debug info set by the build type
            cfg = os.getenv("VLLM_BUILD_TYPE", "RelWithDebInfo")

            cmake_args = [
                '-DCMAKE_BUILD_TYPE=%s' % cfg,
                # Ask CMake to place the resulting library in the directory
                # containing the extension
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir),
                # Other intermediate static libraries are placed in a
                # temporary build directory instead
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), self.build_temp),
            ]

            verbose = bool(int(os.getenv('VERBOSE', '0')))
            if verbose:
                cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

            if is_ccache_available():
                cmake_args += [
                    '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                    '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
                ]

            #
            # Setup parallelism
            #
            num_jobs = os.cpu_count()
            nvcc_cuda_version = get_nvcc_cuda_version()
            if nvcc_cuda_version >= Version("11.2"):
                nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
                num_jobs = max(1, round(num_jobs / (nvcc_threads / 4)))
                cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            ext_target_name = remove_prefix(ext.name, "vllm.")

            if is_ninja_available():
                build_tool = ['-G', 'Ninja']
                cmake_args += [
                    '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                    '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
                ]
                build_jobs = []
            else:
                build_tool = ['-G', 'Unix Makefiles']
                build_jobs = ['-j', str(num_jobs)]

            # Config
            # TODO: this only needs to happen once
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + build_tool +
                                  cmake_args,
                                  cwd=self.build_temp)

            # Build
            build_args = [
                '--build', '.', '--config', cfg, '--target', ext_target_name
            ]
            subprocess.check_call(['cmake'] + build_args + build_jobs,
                                  cwd=self.build_temp)


def _is_cuda() -> bool:
    return torch.version.cuda is not None


def _is_hip() -> bool:
    return torch.version.hip is not None


def _is_neuron() -> bool:
    torch_neuronx_installed = True
    try:
        subprocess.run(["neuron-ls"], capture_output=True, check=True)
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        torch_neuronx_installed = False
    return torch_neuronx_installed


def _is_cuda() -> bool:
    return (torch.version.cuda is not None) and not _is_neuron()


def _install_punica() -> bool:
    install_punica = bool(int(os.getenv("VLLM_INSTALL_PUNICA_KERNELS", "0")))
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            install_punica = False
            break
    return install_punica


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
    version_file = os.path.join(site_dir, "neuronxcc", "version",
                                "__init__.py")

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


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_vllm_version() -> str:
    version = find_version(get_path("vllm", "__init__.py"))

    if _is_cuda():
        cuda_version = str(nvcc_cuda_version)
        if cuda_version != MAIN_CUDA_VERSION:
            cuda_version_str = cuda_version.replace(".", "")[:3]
            version += f"+cu{cuda_version_str}"
    elif _is_hip():
        # Get the HIP version
        hipcc_version = get_hipcc_rocm_version()
        if hipcc_version != MAIN_CUDA_VERSION:
            rocm_version_str = hipcc_version.replace(".", "")[:3]
            version += f"+rocm{rocm_version_str}"
    elif _is_neuron():
        # Get the Neuron version
        neuron_version = str(get_neuronxcc_version())
        if neuron_version != MAIN_CUDA_VERSION:
            neuron_version_str = neuron_version.replace(".", "")[:3]
            version += f"+neuron{neuron_version_str}"
    elif _is_cuda():
        cuda_version = str(get_nvcc_cuda_version())
        if cuda_version != MAIN_CUDA_VERSION:
            cuda_version_str = cuda_version.replace(".", "")[:3]
            version += f"+cu{cuda_version_str}"
    else:
        raise RuntimeError("Unknown runtime environment")

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
    if _is_cuda():
        with open(get_path("requirements.txt")) as f:
            requirements = f.read().strip().split("\n")
        if nvcc_cuda_version <= Version("11.8"):
            # replace cupy-cuda12x with cupy-cuda11x for cuda 11.x
            for i in range(len(requirements)):
                if requirements[i].startswith("cupy-cuda12x"):
                    requirements[i] = "cupy-cuda11x"
                    break
    elif _is_hip():
        with open(get_path("requirements-rocm.txt")) as f:
            requirements = f.read().strip().split("\n")
    elif _is_neuron():
        with open(get_path("requirements-neuron.txt")) as f:
            requirements = f.read().strip().split("\n")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCM or Neuron.")

    return requirements


ext_modules = []

if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm._moe_C"))

    if _install_punica():
        ext_modules.append(CMakeExtension(name="vllm._punica_C"))

if not _is_neuron():
    ext_modules.append(CMakeExtension(name="vllm._C"))

package_data = {
    "vllm": ["py.typed", "model_executor/layers/fused_moe/configs/*.json"]
}
if os.environ.get("VLLM_USE_PRECOMPILED"):
    package_data["vllm"].append("*.so")

setup(
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
    packages=find_packages(exclude=("benchmarks", "csrc", "docs", "examples",
                                    "tests")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext} if not _is_neuron() else {},
    package_data=package_data,
)
