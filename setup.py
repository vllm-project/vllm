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
import sysconfig
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
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
elif sys.platform.startswith("linux") and os.getenv("VLLM_TARGET_DEVICE") is None:
    if torch.version.hip is not None:
        VLLM_TARGET_DEVICE = "rocm"
        logger.info("Auto-detected ROCm")
    elif torch.version.cuda is not None:
        VLLM_TARGET_DEVICE = "cuda"
        logger.info("Auto-detected CUDA")
    else:
        VLLM_TARGET_DEVICE = "cpu"


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(
        int(os.getenv("VLLM_DISABLE_SCCACHE", "0"))
    )


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_freethreaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def compile_grpc_protos():
    """Compile gRPC protobuf definitions during build.

    This generates *_pb2.py, *_pb2_grpc.py, and *_pb2.pyi files from
    the vllm_engine.proto definition.
    """
    try:
        from grpc_tools import protoc
    except ImportError:
        logger.warning(
            "grpcio-tools not installed, skipping gRPC proto compilation. "
            "gRPC server functionality will not be available."
        )
        return False

    proto_file = ROOT_DIR / "vllm" / "grpc" / "vllm_engine.proto"
    if not proto_file.exists():
        logger.warning("Proto file not found at %s, skipping compilation", proto_file)
        return False

    logger.info("Compiling gRPC protobuf: %s", proto_file)

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={ROOT_DIR}",
            f"--python_out={ROOT_DIR}",
            f"--grpc_python_out={ROOT_DIR}",
            f"--pyi_out={ROOT_DIR}",
            str(proto_file),
        ]
    )

    if result != 0:
        logger.error("protoc failed with exit code %s", result)
        return False

    # Add SPDX headers and mypy ignore to generated files
    spdx_header = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "# mypy: ignore-errors\n"
    )

    grpc_dir = ROOT_DIR / "vllm" / "grpc"
    for generated_file in [
        grpc_dir / "vllm_engine_pb2.py",
        grpc_dir / "vllm_engine_pb2_grpc.py",
        grpc_dir / "vllm_engine_pb2.pyi",
    ]:
        if generated_file.exists():
            content = generated_file.read_text()
            if not content.startswith("# SPDX-License-Identifier"):
                generated_file.write_text(spdx_header + content)

    logger.info("gRPC protobuf compilation successful")
    return True


class BuildPyAndGenerateGrpc(build_py):
    """Build Python modules and generate gRPC stubs from proto files."""

    def run(self):
        compile_grpc_protos()
        super().run()


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=not is_freethreaded(), **kwa)
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
        if _is_cuda() and CUDA_HOME is not None:
            try:
                nvcc_version = get_nvcc_cuda_version()
                if nvcc_version >= Version("11.2"):
                    # `nvcc_threads` is either the value of the NVCC_THREADS
                    # environment variable (if defined) or 1.
                    # when it is set, we reduce `num_jobs` to avoid
                    # overloading the system.
                    nvcc_threads = envs.NVCC_THREADS
                    if nvcc_threads is not None:
                        nvcc_threads = int(nvcc_threads)
                        logger.info(
                            "Using NVCC_THREADS=%d as the number of nvcc threads.",
                            nvcc_threads,
                        )
                    else:
                        nvcc_threads = 1
                    num_jobs = max(1, num_jobs // nvcc_threads)
            except Exception as e:
                logger.warning("Failed to get NVCC version: %s", e)

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
        if _is_cuda() and CUDA_HOME is not None:
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]
        elif _is_hip() and ROCM_HOME is not None:
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

        if _is_cuda() or _is_hip():
            # copy vllm/third_party/triton_kernels/**/*.py from self.build_lib
            # to current directory so that they can be included in the editable
            # build
            print(
                f"Copying {self.build_lib}/vllm/third_party/triton_kernels "
                "to vllm/third_party/triton_kernels"
            )
            shutil.copytree(
                f"{self.build_lib}/vllm/third_party/triton_kernels",
                "vllm/third_party/triton_kernels",
                dirs_exist_ok=True,
            )


class precompiled_build_ext(build_ext):
    """Disables extension building when using precompiled binaries."""

    def run(self) -> None:
        return

    def build_extensions(self) -> None:
        print("Skipping build_ext: using precompiled extensions.")
        return


class precompiled_wheel_utils:
    """Extracts libraries and other files from an existing wheel."""

    @staticmethod
    def fetch_metadata_for_variant(
        commit: str, variant: str | None
    ) -> tuple[list[dict], str]:
        """
        Fetches metadata for a specific variant of the precompiled wheel.
        """
        variant_dir = f"{variant}/" if variant is not None else ""
        repo_url = f"https://wheels.vllm.ai/{commit}/{variant_dir}vllm/"
        meta_url = repo_url + "metadata.json"
        print(f"Trying to fetch nightly build metadata from {meta_url}")
        from urllib.request import urlopen

        with urlopen(meta_url) as resp:
            # urlopen raises HTTPError on unexpected status code
            wheels = json.loads(resp.read().decode("utf-8"))
        return wheels, repo_url

    @staticmethod
    def is_rocm_system() -> bool:
        """Detect ROCm without relying on torch (for build environment)."""
        if os.getenv("ROCM_PATH"):
            return True
        if os.path.isdir("/opt/rocm"):
            return True
        if which("rocminfo") is not None:
            return True
        try:
            import torch

            return torch.version.hip is not None
        except ImportError:
            return False

    @staticmethod
    def find_local_rocm_wheel() -> str | None:
        """Search for a local vllm wheel in common locations."""
        import glob

        for pattern in ["/vllm-workspace/dist/vllm-*.whl", "./dist/vllm-*.whl"]:
            wheels = glob.glob(pattern)
            if wheels:
                return sorted(wheels)[-1]
        return None

    @staticmethod
    def fetch_wheel_from_pypi_index(index_url: str, package: str = "vllm") -> str:
        """Fetch the latest wheel URL from a PyPI-style simple index."""
        import platform
        from html.parser import HTMLParser
        from urllib.parse import urljoin
        from urllib.request import urlopen

        arch = platform.machine()

        class WheelLinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.wheels = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for name, value in attrs:
                        if name == "href" and value.endswith(".whl"):
                            self.wheels.append(value)

        simple_url = f"{index_url.rstrip('/')}/{package}/"
        print(f"Fetching wheel list from {simple_url}")
        with urlopen(simple_url) as resp:
            html = resp.read().decode("utf-8")

        parser = WheelLinkParser()
        parser.feed(html)

        for wheel in reversed(parser.wheels):
            if arch in wheel:
                if wheel.startswith("http"):
                    return wheel
                return urljoin(simple_url, wheel)

        raise ValueError(f"No compatible wheel found for {arch} at {simple_url}")

    @staticmethod
    def determine_wheel_url_rocm() -> tuple[str, str | None]:
        """Determine the precompiled wheel for ROCm."""
        # Search for local wheel first
        local_wheel = precompiled_wheel_utils.find_local_rocm_wheel()
        if local_wheel is not None:
            print(f"Found local ROCm wheel: {local_wheel}")
            return local_wheel, None

        # Fall back to AMD's PyPI index
        index_url = os.getenv(
            "VLLM_ROCM_WHEEL_INDEX", "https://pypi.amd.com/vllm-rocm/simple"
        )
        print(f"Fetching ROCm precompiled wheel from {index_url}")
        wheel_url = precompiled_wheel_utils.fetch_wheel_from_pypi_index(index_url)
        download_filename = wheel_url.split("/")[-1].split("#")[0]
        print(f"Using ROCm precompiled wheel: {wheel_url}")
        return wheel_url, download_filename

    @staticmethod
    def determine_wheel_url() -> tuple[str, str | None]:
        """
        Try to determine the precompiled wheel URL or path to use.
        The order of preference is:
        1. user-specified wheel location (can be either local or remote, via
           VLLM_PRECOMPILED_WHEEL_LOCATION)
        2. user-specified variant (VLLM_PRECOMPILED_WHEEL_VARIANT) from nightly repo
        3. the variant corresponding to VLLM_MAIN_CUDA_VERSION from nightly repo
        4. the default variant from nightly repo

        If downloading from the nightly repo, the commit can be specified via
        VLLM_PRECOMPILED_WHEEL_COMMIT; otherwise, the head commit in the main branch
        is used.
        """
        wheel_location = os.getenv("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
        if wheel_location is not None:
            print(f"Using user-specified precompiled wheel location: {wheel_location}")
            return wheel_location, None
        else:
            # ROCm: use local wheel or AMD's PyPI index
            # TODO: When we have ROCm nightly wheels, we can update this logic.
            if precompiled_wheel_utils.is_rocm_system():
                return precompiled_wheel_utils.determine_wheel_url_rocm()

            import platform

            arch = platform.machine()
            # try to fetch the wheel metadata from the nightly wheel repo
            main_variant = "cu" + envs.VLLM_MAIN_CUDA_VERSION.replace(".", "")
            variant = os.getenv("VLLM_PRECOMPILED_WHEEL_VARIANT", main_variant)
            commit = os.getenv("VLLM_PRECOMPILED_WHEEL_COMMIT", "").lower()
            if not commit or len(commit) != 40:
                print(
                    f"VLLM_PRECOMPILED_WHEEL_COMMIT not valid: {commit}"
                    ", trying to fetch base commit in main branch"
                )
                commit = precompiled_wheel_utils.get_base_commit_in_main_branch()
            print(f"Using precompiled wheel commit {commit} with variant {variant}")
            try_default = False
            wheels, repo_url, download_filename = None, None, None
            try:
                wheels, repo_url = precompiled_wheel_utils.fetch_metadata_for_variant(
                    commit, variant
                )
            except Exception as e:
                logger.warning(
                    "Failed to fetch precompiled wheel metadata for variant %s: %s",
                    variant,
                    e,
                )
                try_default = True  # try outside handler to keep the stacktrace simple
            if try_default:
                print("Trying the default variant from remote")
                wheels, repo_url = precompiled_wheel_utils.fetch_metadata_for_variant(
                    commit, None
                )
                # if this also fails, then we have nothing more to try / cache
            assert wheels is not None and repo_url is not None, (
                "Failed to fetch precompiled wheel metadata"
            )
            # The metadata.json has the following format:
            # see .buildkite/scripts/generate-nightly-index.py for details
            """[{
    "package_name": "vllm",
    "version": "0.11.2.dev278+gdbc3d9991",
    "build_tag": null,
    "python_tag": "cp38",
    "abi_tag": "abi3",
    "platform_tag": "manylinux1_x86_64",
    "variant": null,
    "filename": "vllm-0.11.2.dev278+gdbc3d9991-cp38-abi3-manylinux1_x86_64.whl",
    "path": "../vllm-0.11.2.dev278%2Bgdbc3d9991-cp38-abi3-manylinux1_x86_64.whl"
    },
    ...]"""
            from urllib.parse import urljoin

            for wheel in wheels:
                # TODO: maybe check more compatibility later? (python_tag, abi_tag, etc)
                if wheel.get("package_name") == "vllm" and arch in wheel.get(
                    "platform_tag", ""
                ):
                    print(f"Found precompiled wheel metadata: {wheel}")
                    if "path" not in wheel:
                        raise ValueError(f"Wheel metadata missing path: {wheel}")
                    wheel_url = urljoin(repo_url, wheel["path"])
                    download_filename = wheel.get("filename")
                    print(f"Using precompiled wheel URL: {wheel_url}")
                    break
            else:
                raise ValueError(
                    f"No precompiled vllm wheel found for architecture {arch} "
                    f"from repo {repo_url}. All available wheels: {wheels}"
                )

        return wheel_url, download_filename

    @staticmethod
    def extract_precompiled_and_patch_package(
        wheel_url_or_path: str, download_filename: str | None
    ) -> dict:
        import tempfile
        import zipfile

        temp_dir = None
        try:
            if not os.path.isfile(wheel_url_or_path):
                # use provided filename first, then derive from URL
                wheel_filename = download_filename or wheel_url_or_path.split("/")[-1]
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
                    # ROCm-specific libraries
                    "vllm/_rocm_C.abi3.so",
                ]

                flash_attn_regex = re.compile(
                    r"vllm/vllm_flash_attn/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py"
                )
                triton_kernels_regex = re.compile(
                    r"vllm/third_party/triton_kernels/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py"
                )
                file_members = list(
                    filter(lambda x: x.filename in files_to_copy, wheel.filelist)
                )
                file_members += list(
                    filter(lambda x: flash_attn_regex.match(x.filename), wheel.filelist)
                )
                file_members += list(
                    filter(
                        lambda x: triton_kernels_regex.match(x.filename), wheel.filelist
                    )
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
            print(f"Upstream main branch latest commit: {upstream_main_commit}")

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
        if ROCM_HOME is None:
            return None
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
        if envs.VLLM_USE_PRECOMPILED and not envs.VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX:
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
    # Optional since this doesn't get built (produce an .so file). This is just
    # copying the relevant .py files from the source repository.
    ext_modules.append(CMakeExtension(name="vllm.triton_kernels", optional=True))

if _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._rocm_C"))

if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa2_C"))
    if envs.VLLM_USE_PRECOMPILED or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.3")
    ):
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
        "entrypoints/serve/instrumentator/static/*.js",
        "entrypoints/serve/instrumentator/static/*.css",
    ]
}


# If using precompiled, extract and patch package_data (in advance of setup)
if envs.VLLM_USE_PRECOMPILED:
    wheel_url, download_filename = precompiled_wheel_utils.determine_wheel_url()
    patch = precompiled_wheel_utils.extract_precompiled_and_patch_package(
        wheel_url, download_filename
    )
    for pkg, files in patch.items():
        package_data.setdefault(pkg, []).extend(files)

if _no_device():
    ext_modules = []

if not ext_modules:
    cmdclass = {"build_py": BuildPyAndGenerateGrpc}
else:
    cmdclass = {
        "build_ext": precompiled_build_ext
        if envs.VLLM_USE_PRECOMPILED
        else cmake_build_ext,
        "build_py": BuildPyAndGenerateGrpc,
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
        "runai": ["runai-model-streamer[s3,gcs] >= 0.15.3"],
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
