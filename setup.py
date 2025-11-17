# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

# cannot import envs directly because it depends on vllm,
#  which is not installed yet
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "vllm", "envs.py"))

VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE

if sys.platform.startswith("darwin") and VLLM_TARGET_DEVICE != "cpu":
    logger.warning("VLLM_TARGET_DEVICE automatically set to `cpu` due to macOS")
    VLLM_TARGET_DEVICE = "cpu"
elif not (sys.platform.startswith("linux") or sys.platform.startswith("darwin")):
    logger.warning(
        "vLLM only supports Linux platform (including WSL) and MacOS."
        "Building on %s, "
        "so vLLM may not be able to run correctly",
        sys.platform,
    )
    VLLM_TARGET_DEVICE = "empty"
elif (
    sys.platform.startswith("linux")
    and torch.version.cuda is None
    and os.getenv("VLLM_TARGET_DEVICE") is None
    and torch.version.hip is None
):
    # if cuda or hip is not available and VLLM_TARGET_DEVICE is not set,
    # fallback to cpu
    VLLM_TARGET_DEVICE = "cpu"


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(
        int(os.getenv("VLLM_DISABLE_SCCACHE", "0"))
    )


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_url_available(url: str) -> bool:
    from urllib.request import urlopen

    status = None
    try:
        with urlopen(url) as f:
            status = f.status
    except Exception:
        return False
    return status == 200


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = envs.NVCC_THREADS
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info(
                    "Using NVCC_THREADS=%d as the number of nvcc threads.", nvcc_threads
                )
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DVLLM_TARGET_DEVICE={}".format(VLLM_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ["-DVLLM_PYTHON_EXECUTABLE={}".format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ["-DVLLM_PYTHON_PATH={}".format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ["-DNVCC_THREADS={}".format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        # Make sure we use the nvcc from CUDA_HOME
        if _is_cuda():
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]
        elif _is_hip():
            cmake_args += [f"-DROCM_PATH={ROCM_HOME}"]

        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args += other_cmake_args.split()

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("vllm.").removeprefix("vllm_flash_attn.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake",
                "--install",
                ".",
                "--prefix",
                prefix,
                "--component",
                target_name(ext.name),
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()

        # copy vllm/vllm_flash_attn/**/*.py from self.build_lib to current
        # directory so that they can be included in the editable build
        import glob

        files = glob.glob(
            os.path.join(self.build_lib, "vllm", "vllm_flash_attn", "**", "*.py"),
            recursive=True,
        )
        for file in files:
            dst_file = os.path.join(
                "vllm/vllm_flash_attn", file.split("vllm/vllm_flash_attn/")[-1]
            )
            print(f"Copying {file} to {dst_file}")
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            self.copy_file(file, dst_file)


class precompiled_build_ext(build_ext):
    """Disables extension building when using precompiled binaries."""

    def run(self) -> None:
        assert _is_cuda(), "VLLM_USE_PRECOMPILED is only supported for CUDA builds"

    def build_extensions(self) -> None:
        print("Skipping build_ext: using precompiled extensions.")
        return


class precompiled_wheel_utils:
    """Extracts libraries and other files from an existing wheel."""

    @staticmethod
    def extract_precompiled_and_patch_package(wheel_url_or_path: str) -> dict:
        import tempfile
        import zipfile

        temp_dir = None
        try:
            if not os.path.isfile(wheel_url_or_path):
                wheel_filename = wheel_url_or_path.split("/")[-1]
                temp_dir = tempfile.mkdtemp(prefix="vllm-wheels")
                wheel_path = os.path.join(temp_dir, wheel_filename)
                print(f"Downloading wheel from {wheel_url_or_path} to {wheel_path}")
                from urllib.request import urlretrieve

                urlretrieve(wheel_url_or_path, filename=wheel_path)
            else:
                wheel_path = wheel_url_or_path
                print(f"Using existing wheel at {wheel_path}")

            package_data_patch = {}

            with zipfile.ZipFile(wheel_path) as wheel:
                files_to_copy = [
                    "vllm/_C.abi3.so",
                    "vllm/_moe_C.abi3.so",
                    "vllm/_flashmla_C.abi3.so",
                    "vllm/_flashmla_extension_C.abi3.so",
                    "vllm/_sparse_flashmla_C.abi3.so",
                    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
                    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
                    "vllm/cumem_allocator.abi3.so",
                ]

                compiled_regex = re.compile(
                    r"vllm/vllm_flash_attn/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py"
                )
                file_members = list(
                    filter(lambda x: x.filename in files_to_copy, wheel.filelist)
                )
                file_members += list(
                    filter(lambda x: compiled_regex.match(x.filename), wheel.filelist)
                )

                for file in file_members:
                    print(f"[extract] {file.filename}")
                    target_path = os.path.join(".", file.filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with (
                        wheel.open(file.filename) as src,
                        open(target_path, "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                    pkg = os.path.dirname(file.filename).replace("/", ".")
                    package_data_patch.setdefault(pkg, []).append(
                        os.path.basename(file.filename)
                    )

            return package_data_patch
        finally:
            if temp_dir is not None:
                print(f"Removing temporary directory {temp_dir}")
                shutil.rmtree(temp_dir)

    @staticmethod
    def get_base_commit_in_main_branch() -> str:
        # Force to use the nightly wheel. This is mainly used for CI testing.
        if envs.VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL:
            return "nightly"

        try:
            # Get the latest commit hash of the upstream main branch.
            resp_json = subprocess.check_output(
                [
                    "curl",
                    "-s",
                    "https://api.github.com/repos/vllm-project/vllm/commits/main",
                ]
            ).decode("utf-8")
            upstream_main_commit = json.loads(resp_json)["sha"]

            # In Docker build context, .git may be immutable or missing.
            if envs.VLLM_DOCKER_BUILD_CONTEXT:
                return upstream_main_commit

            # Check if the upstream_main_commit exists in the local repo
            try:
                subprocess.check_output(
                    ["git", "cat-file", "-e", f"{upstream_main_commit}"]
                )
            except subprocess.CalledProcessError:
                # If not present, fetch it from the remote repository.
                # Note that this does not update any local branches,
                # but ensures that this commit ref and its history are
                # available in our local repo.
                subprocess.check_call(
                    ["git", "fetch", "https://github.com/vllm-project/vllm", "main"]
                )

            # Then get the commit hash of the current branch that is the same as
            # the upstream main commit.
            current_branch = (
                subprocess.check_output(["git", "branch", "--show-current"])
                .decode("utf-8")
                .strip()
            )

            base_commit = (
                subprocess.check_output(
                    ["git", "merge-base", f"{upstream_main_commit}", current_branch]
                )
                .decode("utf-8")
                .strip()
            )
            return base_commit
        except ValueError as err:
            raise ValueError(err) from None
        except Exception as err:
            logger.warning(
                "Failed to get the base commit in the main branch. "
                "Using the nightly wheel. The libraries in this "
                "wheel may not be compatible with your dev branch: %s",
                err,
            )
            return "nightly"


def _no_device() -> bool:
    return VLLM_TARGET_DEVICE == "empty"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return VLLM_TARGET_DEVICE == "cuda" and has_cuda and not _is_tpu()


def _is_hip() -> bool:
    return (
        VLLM_TARGET_DEVICE == "cuda" or VLLM_TARGET_DEVICE == "rocm"
    ) and torch.version.hip is not None


def _is_tpu() -> bool:
    return VLLM_TARGET_DEVICE == "tpu"


def _is_cpu() -> bool:
    return VLLM_TARGET_DEVICE == "cpu"


def _is_xpu() -> bool:
    return VLLM_TARGET_DEVICE == "xpu"


def _build_custom_ops() -> bool:
    return _is_cuda() or _is_hip() or _is_cpu()


def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    # see https://github.com/ROCm/rocm-core/blob/d11f5c20d500f729c393680a01fa902ebf92094b/rocm_version.cpp#L21
    try:
        librocm_core_file = Path(ROCM_HOME) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(librocm_core_file)
        VerErrors = ctypes.c_uint32
        get_rocm_core_version = librocm_core.getROCmVersion
        get_rocm_core_version.restype = VerErrors
        get_rocm_core_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()

        if (
            get_rocm_core_version(
                ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
            )
            == 0
        ):
            return f"{major.value}.{minor.value}.{patch.value}"
        return None
    except Exception:
        return None


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_gaudi_sw_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run(
        "hl-smi",
        shell=True,
        text=True,
        capture_output=True,
        env={"ENABLE_CONSOLE": "true"},
    )
    if output.returncode == 0 and output.stdout:
        return (
            output.stdout.split("\n")[2]
            .replace(" ", "")
            .split(":")[1][:-1]
            .split("-")[0]
        )
    return "0.0.0"  # when hl-smi is not available


def get_vllm_version() -> str:
    # Allow overriding the version. This is useful to build platform-specific
    # wheels (e.g. CPU, TPU) without modifying the source.
    if env_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        print(f"Overriding VLLM version with {env_version} from VLLM_VERSION_OVERRIDE")
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = env_version
        return get_version(write_to="vllm/_version.py")

    version = get_version(write_to="vllm/_version.py")
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _no_device():
        if envs.VLLM_TARGET_DEVICE == "empty":
            version += f"{sep}empty"
    elif _is_cuda():
        if envs.VLLM_USE_PRECOMPILED:
            version += f"{sep}precompiled"
        else:
            cuda_version = str(get_nvcc_cuda_version())
            if cuda_version != envs.VLLM_MAIN_CUDA_VERSION:
                cuda_version_str = cuda_version.replace(".", "")[:3]
                # skip this for source tarball, required for pypi
                if "sdist" not in sys.argv:
                    version += f"{sep}cu{cuda_version_str}"
    elif _is_hip():
        # Get the Rocm Version
        rocm_version = get_rocm_version() or torch.version.hip
        if rocm_version and rocm_version != envs.VLLM_MAIN_CUDA_VERSION:
            version += f"{sep}rocm{rocm_version.replace('.', '')[:3]}"
    elif _is_tpu():
        version += f"{sep}tpu"
    elif _is_cpu():
        if envs.VLLM_TARGET_DEVICE == "cpu":
            version += f"{sep}cpu"
    elif _is_xpu():
        version += f"{sep}xpu"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    if _no_device():
        requirements = _read_requirements("common.txt")
    elif _is_cuda():
        requirements = _read_requirements("cuda.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if "vllm-flash-attn" in req and cuda_major != "12":
                # vllm-flash-attn is built only for CUDA 12.x.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
        requirements = modified_requirements
    elif _is_hip():
        requirements = _read_requirements("rocm.txt")
    elif _is_tpu():
        requirements = _read_requirements("tpu.txt")
    elif _is_cpu():
        requirements = _read_requirements("cpu.txt")
    elif _is_xpu():
        requirements = _read_requirements("xpu.txt")
    else:
        raise ValueError("Unsupported platform, please use CUDA, ROCm, or CPU.")
    return requirements


ext_modules = []

if _is_cuda() or _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._moe_C"))
    ext_modules.append(CMakeExtension(name="vllm.cumem_allocator"))

if _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._rocm_C"))

if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa2_C"))
    if envs.VLLM_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.3"):
        # FA3 requires CUDA 12.3 or later
        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
        # Optional since this doesn't get built (produce an .so file) when
        # not targeting a hopper system
        ext_modules.append(CMakeExtension(name="vllm._flashmla_C", optional=True))
        ext_modules.append(
            CMakeExtension(name="vllm._flashmla_extension_C", optional=True)
        )

if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="vllm._C"))

package_data = {
    "vllm": [
        "py.typed",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
    ]
}

# If using precompiled, extract and patch package_data (in advance of setup)
if envs.VLLM_USE_PRECOMPILED:
    assert _is_cuda(), "VLLM_USE_PRECOMPILED is only supported for CUDA builds"
    wheel_location = os.getenv("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
    if wheel_location is not None:
        wheel_url = wheel_location
    else:
        import platform

        arch = platform.machine()
        if arch == "x86_64":
            wheel_tag = "manylinux1_x86_64"
        elif arch == "aarch64":
            wheel_tag = "manylinux2014_aarch64"
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        base_commit = precompiled_wheel_utils.get_base_commit_in_main_branch()
        wheel_url = f"https://wheels.vllm.ai/{base_commit}/vllm-1.0.0.dev-cp38-abi3-{wheel_tag}.whl"
        nightly_wheel_url = (
            f"https://wheels.vllm.ai/nightly/vllm-1.0.0.dev-cp38-abi3-{wheel_tag}.whl"
        )
        from urllib.request import urlopen

        try:
            with urlopen(wheel_url) as resp:
                if resp.status != 200:
                    wheel_url = nightly_wheel_url
        except Exception as e:
            print(f"[warn] Falling back to nightly wheel: {e}")
            wheel_url = nightly_wheel_url

    patch = precompiled_wheel_utils.extract_precompiled_and_patch_package(wheel_url)
    for pkg, files in patch.items():
        package_data.setdefault(pkg, []).extend(files)

if _no_device():
    ext_modules = []

if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {
        "build_ext": precompiled_build_ext
        if envs.VLLM_USE_PRECOMPILED
        else cmake_build_ext
    }

setup(
    # static metadata should rather go in pyproject.toml
    version=get_vllm_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "matplotlib", "seaborn", "datasets"],
        "tensorizer": ["tensorizer==2.10.1"],
        "fastsafetensors": ["fastsafetensors >= 0.1.10"],
        "runai": ["runai-model-streamer[s3,gcs] >= 0.15.0"],
        "audio": [
            "librosa",
            "soundfile",
            "mistral_common[audio]",
        ],  # Required for audio processing
        "video": [],  # Kept for backwards compatibility
        "flashinfer": [],  # Kept for backwards compatibility
        # Optional deps for AMD FP4 quantization support
        "petit-kernel": ["petit-kernel"],
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
