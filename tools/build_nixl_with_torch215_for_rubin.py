# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# TODO(arpera):
# This script is temporary. It builds nixl from source code against torch 2.15
# because currently there is no nixl wheel built against torch 2.15 which
# vLLM docker image for Rubin requires.
# Remove this script from vLLM once there is nixl wheel built against torch 2.15.

import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request

# --- Configuration ---
WHEELS_CACHE_HOME = os.environ.get("WHEELS_CACHE_HOME", "/tmp/wheels_cache")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UCX_DIR = os.path.join("/tmp", "ucx_source")
NIXL_DIR = os.path.join("/tmp", "nixl_source")
UCX_INSTALL_DIR = os.path.join("/tmp", "ucx_install")
NIXL_BUILD_DIR = os.path.join("/tmp", "nixl_build")
BUILD_TOOLS_DIR = os.path.join("/tmp", "nixl_build_tools")
UCX_REPO_URL = "https://github.com/openucx/ucx.git"
NIXL_REPO_URL = "https://github.com/ai-dynamo/nixl.git"
UCX_REF = os.environ.get("UCX_REF", "v1.23.x")
TEMP_WHEEL_DIR = os.path.join("/tmp", "nixl_temp_wheelhouse")
NIXL_REQUIREMENTS_FILE = os.environ.get(
    "NIXL_REQUIREMENTS_FILE",
    os.path.join(ROOT_DIR, os.pardir, "requirements", "kv_connectors.txt"),
)
MAX_JOBS = int(os.environ.get("MAX_JOBS") or os.cpu_count() or 1)
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
# SM targets NIXL EP can run on (it requires sm_90 or newer).
NIXL_EP_CUDA_ARCHS = ("90", "100", "103", "107", "110", "120")

# Libraries that must come from the host or from torch, not from the wheel.
AUDITWHEEL_EXCLUDES = [
    "libcuda*",
    "libcufile*",
    "libcuobjclient*",
    "libssl*",
    "libcrypto*",
    "libefa*",
    "libhwloc*",
    "libfabric*",
    "libtorch*",
    "libc10*",
    "libdoca*",
    "libred_client*",
    "libred_async*",
    "liblz4*",
]


# --- Helper Functions ---
def get_latest_nixl_version():
    """Helper function to get latest release version of NIXL.

    Raises:
        RuntimeError: If the latest release cannot be determined.

    """
    # /releases/latest redirects to the tag page. Unlike the REST API it is not
    # rate limited for anonymous clients, which are shared on compute nodes.
    request = urllib.request.Request(
        f"{NIXL_REPO_URL.removesuffix('.git')}/releases/latest", method="HEAD"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.geturl().rsplit("/", 1)[-1]
    except OSError as e:
        raise RuntimeError(
            "Cannot determine the latest NIXL release, set NIXL_VERSION."
        ) from e


def get_pinned_nixl_version():
    """Returns the NIXL version pinned in the vLLM requirements, if there is one."""
    try:
        with open(NIXL_REQUIREMENTS_FILE) as requirements:
            for line in requirements:
                requirement = line.split("#")[0].split(";")[0]
                name, separator, version = requirement.partition("==")
                if separator and name.strip() == "nixl" and version.split():
                    return version.split()[0]
    except OSError:
        pass
    return None


def resolve_nixl_version():
    """Returns the NIXL version to build and where it comes from."""
    if os.environ.get("NIXL_VERSION"):
        return os.environ["NIXL_VERSION"], "the NIXL_VERSION environment variable"
    pinned_version = get_pinned_nixl_version()
    if pinned_version:
        return pinned_version, os.path.normpath(NIXL_REQUIREMENTS_FILE)
    return get_latest_nixl_version(), "the latest NIXL release"


NIXL_VERSION, NIXL_VERSION_SOURCE = resolve_nixl_version()
# Version as it appears in wheel file names (tags may carry a "v" prefix).
NIXL_WHEEL_VERSION = NIXL_VERSION.removeprefix("v")


def run_command(command, cwd=".", env=None):
    """Helper function to run a shell command and check for errors."""
    print(f"--> Running command: {' '.join(command)} in '{cwd}'", flush=True)
    subprocess.check_call(command, cwd=cwd, env=env)


def is_pip_package_installed(package_name):
    """Checks if a package is installed via pip without raising an exception."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def find_nixl_wheel_in_cache(cache_dir):
    """Finds the nixl-cuXX wheel file in the specified cache directory."""
    # The repaired wheel will have a 'manylinux' tag, but this glob still works.
    search_pattern = os.path.join(cache_dir, f"nixl_*{NIXL_WHEEL_VERSION}*.whl")
    wheels = glob.glob(search_pattern)
    if wheels:
        # Sort to get the most recent/highest version if multiple exist
        wheels.sort()
        return wheels[-1]
    return None


def find_nixl_meta_wheel(wheel_dir):
    """Finds the `nixl` meta package wheel (the nixl_cuXX dispatcher)."""
    wheels = glob.glob(os.path.join(wheel_dir, f"nixl-{NIXL_WHEEL_VERSION}-*.whl"))
    return wheels[-1] if wheels else None


def get_cuda_version():
    """Returns the (major, minor) CUDA toolkit version, or None without CUDA."""
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    if not os.path.exists(nvcc):
        return None
    output = subprocess.check_output([nvcc, "--version"], text=True)
    _, separator, release = output.partition("release ")
    try:
        if not separator:
            raise ValueError
        major, minor = release.split(",")[0].split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        raise RuntimeError(
            f"Cannot parse the CUDA version from `{nvcc} --version`."
        ) from None


def get_default_cuda_arch_list():
    """Returns the SM targets of NIXL EP that the installed nvcc can compile."""
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    supported = subprocess.check_output([nvcc, "--list-gpu-code"], text=True).split()
    return ",".join(sm for sm in NIXL_EP_CUDA_ARCHS if f"sm_{sm}" in supported)


def is_torch_available():
    """Checks whether torch can be imported by the current interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def is_nixl_installed(build_nixl_ep):
    """Checks that the requested NIXL version, and its EP extension for the
    installed torch when requested, is already installed."""
    cuda_version = get_cuda_version()
    if cuda_version is None:
        return is_pip_package_installed("nixl")
    checks = [
        "import importlib, importlib.metadata as m",
        f"assert m.version('nixl') == '{NIXL_WHEEL_VERSION}'",
        f"assert m.version('nixl-cu{cuda_version[0]}') == '{NIXL_WHEEL_VERSION}'",
    ]
    if build_nixl_ep:
        # The package picks its extension by torch version and fails to import
        # when none was built for the installed torch.
        checks.append(f"importlib.import_module('nixl_ep_cu{cuda_version[0]}')")
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(checks)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def install_system_dependencies(build_nixl_ep=False):
    """Installs required system packages using apt-get if run as root."""
    if shutil.which("apt-get") is None:
        print("--> apt-get not found, relying on the build tools in PATH.", flush=True)
        return
    if os.geteuid() != 0:
        print("\n---", flush=True)
        print(
            "WARNING: Not running as root. \
            Skipping system dependency installation.",
            flush=True,
        )
        print(
            "Please ensure the listed packages are installed on your system:",
            flush=True,
        )
        print(
            "  patchelf build-essential git cmake ninja-build \
            autotools-dev automake libtool libtool-bin \
            libibverbs-dev",
            flush=True,
        )
        print("---\n", flush=True)
        return

    print("--- Running as root. Installing system dependencies... ---", flush=True)
    apt_packages = [
        "patchelf",  # <-- Add patchelf here
        "build-essential",
        "git",
        "cmake",
        "ninja-build",
        "autotools-dev",
        "automake",
        "libtool",
        "libtool-bin",
        "pkg-config",
        "libibverbs-dev",
    ]
    # -base CUDA images lack the headers UCX (NVML) and the torch extension
    # (cuSPARSE/cuSOLVER/cuBLAS, included by ATen/cuda/CUDAContext.h) need.
    cuda_version = get_cuda_version()
    if cuda_version is not None:
        major, minor = cuda_version
        if not os.path.exists(os.path.join(CUDA_HOME, "include", "nvml.h")):
            apt_packages.append(f"cuda-nvml-dev-{major}-{minor}")
        if build_nixl_ep:
            apt_packages += [
                f"{package}-{major}-{minor}"
                for package in ("libcusparse-dev", "libcublas-dev", "libcusolver-dev")
            ]
    run_command(["apt-get", "update"])
    run_command(["apt-get", "install", "-y"] + apt_packages)
    print("--- System dependencies installed successfully. ---\n", flush=True)


def install_python_build_dependencies():
    """Installs the Python tools needed to build and repair the NIXL wheel."""
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "auditwheel",
            "meson-python",
            "ninja",
            "patchelf",
            "pybind11",
            "pyyaml",
            "setuptools>=80.9.0",
            "tomlkit",
        ]
    )
    # An older meson already on the system (e.g. from the distribution) makes
    # meson-python fail to package NIXL and pip would keep it, so use our own.
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"--target={BUILD_TOOLS_DIR}",
            "meson",
        ]
    )


def build_ucx(ucx_install_path):
    """Builds UCX from source, with CUDA support if CUDA is installed."""
    if not os.path.exists(UCX_DIR):
        run_command(["git", "clone", UCX_REPO_URL, UCX_DIR])
    ucx_source_path = os.path.abspath(UCX_DIR)
    run_command(["git", "checkout", UCX_REF], cwd=ucx_source_path)
    run_command(["./autogen.sh"], cwd=ucx_source_path)
    configure_command = [
        "./contrib/configure-release-mt",
        f"--prefix={ucx_install_path}",
        "--enable-shared",
        "--disable-static",
        "--disable-doxygen-doc",
        "--enable-experimental-api",
        "--enable-optimizations",
        "--without-avx",
        "--enable-cma",
        "--enable-devel-headers",
        *([f"--with-cuda={CUDA_HOME}"] if get_cuda_version() else []),
        "--with-verbs",
        "--without-gdrcopy",
        "--without-dc",
        "--without-rdmacm",
        "--without-gga",
        "--with-ze=no",
    ]
    run_command(configure_command, cwd=ucx_source_path)
    run_command(["make", "-j", str(MAX_JOBS)], cwd=ucx_source_path)
    run_command(["make", "install"], cwd=ucx_source_path)


def build_nixl_wheel(ucx_path, build_env, wheel_dir, args):
    """Builds the (unrepaired) NIXL wheel against the UCX installed in ucx_path."""
    if not os.path.exists(NIXL_DIR):
        run_command(["git", "clone", NIXL_REPO_URL, NIXL_DIR])
    else:
        run_command(["git", "fetch", "--tags"], cwd=NIXL_DIR)
    nixl_source_path = os.path.abspath(NIXL_DIR)
    # Tags are "v<version>" for current releases and bare for older ones.
    for tag in (f"v{NIXL_WHEEL_VERSION}", NIXL_WHEEL_VERSION):
        tag_exists = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
            cwd=nixl_source_path,
            stdout=subprocess.DEVNULL,
        )
        if tag_exists.returncode == 0:
            run_command(["git", "checkout", "--force", tag], cwd=nixl_source_path)
            print(f"--> Checked out NIXL version: {tag}", flush=True)
            break
    else:
        raise RuntimeError(f"NIXL has no tag for version {NIXL_WHEEL_VERSION}.")

    cuda_version = get_cuda_version()
    if cuda_version is not None:
        # The wheel name selects which of nixl-cu12 / nixl-cu13 is built.
        run_command(
            [
                sys.executable,
                "contrib/tomlutil.py",
                "--wheel-name",
                f"nixl-cu{cuda_version[0]}",
                "pyproject.toml",
            ],
            cwd=nixl_source_path,
        )

    meson_options = {
        "build_tests": "false",
        "ucx_path": ucx_path,
    }
    cuda_arch_list = args.cuda_arch_list
    if args.build_nixl_ep:
        meson_options["build_nixl_ep"] = "true"
        meson_options["build_examples"] = "true"
        cuda_arch_list = cuda_arch_list or get_default_cuda_arch_list()
    if cuda_arch_list:
        meson_options["nixl_cuda_arch_list"] = cuda_arch_list
    if args.nixl_plugins:
        meson_options["enable_plugins"] = args.nixl_plugins
    config_settings = [f"-Csetup-args=-D{k}={v}" for k, v in meson_options.items()]
    # Keep the build tree: auditwheel needs the subproject libraries it holds.
    config_settings.append(f"-Cbuild-dir={NIXL_BUILD_DIR}")
    config_settings.append(f"-Ccompile-args=-j{MAX_JOBS}")

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            # Build against the torch already installed, not a fresh isolated one.
            "--no-build-isolation",
            f"--wheel-dir={wheel_dir}",
            *config_settings,
        ],
        cwd=nixl_source_path,
        env=build_env,
    )


def repair_nixl_wheel(unrepaired_wheel, repaired_dir, ucx_install_path, build_env):
    """Bundles the shared libraries and UCX modules into a self-contained wheel."""
    auditwheel_command = ["auditwheel", "repair"]
    for pattern in AUDITWHEEL_EXCLUDES:
        auditwheel_command += ["--exclude", pattern]
    auditwheel_command += [unrepaired_wheel, f"--wheel-dir={repaired_dir}"]
    # The wheel ships prometheus-cpp as libcore.so.1.3.0 but its users ask for the
    # soname libcore.so.1.3, which only exists as a symlink in the build tree.
    repair_env = build_env.copy()
    prometheus_dir = os.path.join(NIXL_BUILD_DIR, "subprojects", "prometheus-cpp")
    repair_env["LD_LIBRARY_PATH"] = f"{prometheus_dir}:{build_env['LD_LIBRARY_PATH']}"
    run_command(auditwheel_command, env=repair_env)

    repaired_wheel = find_nixl_wheel_in_cache(repaired_dir)
    if not repaired_wheel:
        raise RuntimeError("Failed to find the repaired NIXL wheel.")
    # UCX loads its transports (CUDA, IB, ...) with dlopen, so auditwheel does not
    # see them as dependencies; the NIXL plugin itself is already in the wheel.
    run_command(
        [
            sys.executable,
            "contrib/wheel_add_ucx_plugins.py",
            "--skip-nixl-plugins",
            f"--ucx-plugins-dir={os.path.join(ucx_install_path, 'lib', 'ucx')}",
            repaired_wheel,
        ],
        cwd=os.path.abspath(NIXL_DIR),
        env=build_env,
    )
    return repaired_wheel


def install_nixl_wheels(wheels):
    """Installs the NIXL wheels and drops the nixl-cuXX of other CUDA majors."""
    # --force-reinstall: the wheels usually have the version of the release that
    # is already installed, which pip would otherwise keep.
    # w/o "no-deps", it will install cuda-torch
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            *wheels,
        ]
    )

    cuda_version = get_cuda_version()
    if cuda_version is None:
        return
    # The dispatcher only loads nixl-cu<installed CUDA major>, and the meta
    # package built here does not depend on the other variant.
    for cuda_major in (12, 13):
        package = f"nixl-cu{cuda_major}"
        if cuda_major != cuda_version[0] and is_pip_package_installed(package):
            run_command([sys.executable, "-m", "pip", "uninstall", "-y", package])


def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source, creating a self-contained wheel."""
    print(f"--> NIXL version {NIXL_WHEEL_VERSION} (from {NIXL_VERSION_SOURCE})")
    if args.build_nixl_ep is None:
        args.build_nixl_ep = get_cuda_version() is not None and is_torch_available()
    reuse_existing = not (args.force_reinstall or args.build_only)
    if reuse_existing and is_nixl_installed(args.build_nixl_ep):
        print("--> NIXL is already installed. Nothing to do.", flush=True)
        return

    cached_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if reuse_existing and cached_wheel:
        print(
            f"\n--> Found self-contained wheel: \
                {os.path.basename(cached_wheel)}.",
            flush=True,
        )
        print("--> Installing from cache, skipping all source builds.", flush=True)
        cached_wheels = [cached_wheel]
        cached_meta_wheel = find_nixl_meta_wheel(WHEELS_CACHE_HOME)
        if cached_meta_wheel:
            cached_wheels.append(cached_meta_wheel)
        install_nixl_wheels(cached_wheels)
        print("\n--- Installation from cache complete. ---", flush=True)
        return

    print(
        "\n--> No installed package or cached wheel found. \
         Starting full build process...",
        flush=True,
    )
    if not args.skip_system_deps:
        install_system_dependencies(args.build_nixl_ep)
    print("\n--> Installing Python build dependencies...", flush=True)
    install_python_build_dependencies()
    print(f"--> Using wheel cache directory: {WHEELS_CACHE_HOME}", flush=True)
    os.makedirs(WHEELS_CACHE_HOME, exist_ok=True)

    # -- Step 1: Build UCX from source, unless an existing install is given --
    # With an external UCX (e.g. HPC-X) the wheel keeps depending on it at runtime.
    external_ucx = args.ucx_path is not None
    if external_ucx:
        ucx_install_path = os.path.abspath(args.ucx_path)
        print(f"\n[1/3] Using the existing UCX in {ucx_install_path}", flush=True)
    else:
        ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
        print("\n[1/3] Configuring and building UCX from source...", flush=True)
        build_ucx(ucx_install_path)
        print("--- UCX build and install complete ---", flush=True)

    # -- Step 2: Build NIXL wheel from source --
    print("\n[2/3] Building NIXL wheel from source...", flush=True)
    ucx_lib_path = os.path.join(ucx_install_path, "lib")
    ucx_plugin_path = os.path.join(ucx_lib_path, "ucx")
    build_env = os.environ.copy()
    # pip-installed build tools live next to the running interpreter.
    build_env["PATH"] = os.pathsep.join(
        [
            os.path.join(BUILD_TOOLS_DIR, "bin"),
            os.path.join(CUDA_HOME, "bin"),
            os.path.dirname(sys.executable),
            build_env["PATH"],
        ]
    )
    build_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [BUILD_TOOLS_DIR, build_env.get("PYTHONPATH")])
    )
    build_env["PKG_CONFIG_PATH"] = os.path.join(ucx_lib_path, "pkgconfig")
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    build_env["LD_LIBRARY_PATH"] = (
        f"{ucx_lib_path}:{ucx_plugin_path}:{existing_ld_path}".strip(":")
    )
    print(f"--> Using LD_LIBRARY_PATH: {build_env['LD_LIBRARY_PATH']}", flush=True)

    build_nixl_wheel(ucx_install_path, build_env, TEMP_WHEEL_DIR, args)

    # -- Step 3: Repair the wheel by copying UCX libraries --
    unrepaired_wheel = find_nixl_wheel_in_cache(TEMP_WHEEL_DIR)
    if not unrepaired_wheel:
        raise RuntimeError("Failed to find the NIXL wheel after building it.")
    if external_ucx:
        print("\n[3/3] Skipping the repair step: UCX is provided externally.")
        built_wheels = [unrepaired_wheel]
    else:
        print("\n[3/3] Repairing NIXL wheel to include UCX libraries...", flush=True)
        built_wheels = [
            repair_nixl_wheel(
                unrepaired_wheel,
                os.path.join(TEMP_WHEEL_DIR, "repaired"),
                ucx_install_path,
                build_env,
            )
        ]

    meta_wheel = find_nixl_meta_wheel(
        os.path.join(NIXL_BUILD_DIR, "src", "bindings", "python", "nixl-meta")
    )
    if meta_wheel:
        built_wheels.append(meta_wheel)
    else:
        print("WARNING: the nixl meta package was not built.", flush=True)

    new_wheels = []
    for wheel in built_wheels:
        run_command(["cp", wheel, WHEELS_CACHE_HOME])
        new_wheels.append(os.path.join(WHEELS_CACHE_HOME, os.path.basename(wheel)))
    run_command(["rm", "-rf", TEMP_WHEEL_DIR])

    wheel_names = ", ".join(os.path.basename(w) for w in new_wheels)
    if args.build_only:
        print(f"--> Built wheels in {WHEELS_CACHE_HOME}: {wheel_names}", flush=True)
        return
    print(
        f"--> Successfully built wheels: {wheel_names}. Now installing...", flush=True
    )
    install_nixl_wheels(new_wheels)
    print("--- NIXL installation complete ---", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and install UCX and NIXL dependencies."
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force rebuild and reinstall of UCX and NIXL \
        even if they are already installed.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the wheels into WHEELS_CACHE_HOME and stop, without \
        installing them or looking at what is already installed.",
    )
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Do not install system packages with apt-get; the build tools \
        (git, autotools, cmake, patchelf, ...) must already be on PATH.",
    )
    parser.add_argument(
        "--build-nixl-ep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build the NIXL EP extension for the installed torch version. \
        Defaults to on when CUDA and torch are available.",
    )
    parser.add_argument(
        "--cuda-arch-list",
        default=os.environ.get("NIXL_CUDA_ARCH_LIST"),
        help="Comma-separated CUDA SM targets for NIXL, e.g. 100,103,107. \
        Defaults to the NIXL EP targets the installed nvcc supports.",
    )
    parser.add_argument(
        "--nixl-plugins",
        default=os.environ.get("NIXL_PLUGINS", "UCX"),
        help="Comma-separated NIXL plugins to build. Pass an empty string to \
        build every plugin whose dependencies are available.",
    )
    parser.add_argument(
        "--ucx-path",
        default=os.environ.get("UCX_PATH"),
        help="Prefix of an existing UCX to build against instead of building one. \
        The resulting wheel is not self-contained.",
    )
    args = parser.parse_args()
    build_and_install_prerequisites(args)
