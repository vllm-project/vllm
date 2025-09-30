# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# install_prerequisites.py
import argparse
import os
import subprocess
import sys

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UCX_DIR = "ucx_source"
UCX_REPO_URL = "https://github.com/openucx/ucx.git"
UCX_INSTALL_DIR = os.path.join(ROOT_DIR, "ucx_install")
NIXL_DIR = "nixl_source"
NIXL_REPO_URL = "https://github.com/ai-dynamo/nixl.git"


# --- Helper Functions ---
def run_command(command, cwd=".", env=None):
    """Helper function to run a shell command and check for errors."""
    print(f"--> Running command: {' '.join(command)} in '{cwd}'")
    subprocess.check_call(command, cwd=cwd, env=env)


def is_pip_package_installed(package_name):
    """Checks if a package is installed via pip without raising an exception."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def install_system_dependencies():
    """Installs required system packages using apt-get if run as root."""
    if os.geteuid() != 0:
        print("\n---")
        print("WARNING: Not running as root. \
            Skipping system dependency installation.")
        print(
            "Please ensure the following packages are installed on your system:"
        )
        print("  build-essential git cmake ninja-build autotools-dev \
                automake meson libtool libtool-bin")
        print("---\n")
        return

    print("--- Running as root. Installing system dependencies... ---")
    apt_packages = [
        "build-essential",
        "git",
        "cmake",
        "ninja-build",
        "autotools-dev",
        "automake",
        "meson",
        "libtool",
        "libtool-bin",
    ]
    run_command(["apt-get", "update"])
    run_command(["apt-get", "install", "-y"] + apt_packages)
    print("--- System dependencies installed successfully. ---\n")


def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source."""
    install_system_dependencies()
    print("--- Starting prerequisite build and install process ---")
    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)

    # -- Step 1: Build and Install UCX from source --
    ucx_check_file = os.path.join(ucx_install_path, "bin", "ucx_info")
    if not args.force_reinstall and os.path.exists(ucx_check_file):
        print("\n--> UCX already found. Skipping build. \
                Use --force-reinstall to override.")
    else:
        print("\n[1/2] Configuring and building UCX from source...")
        if not os.path.exists(UCX_DIR):
            run_command(["git", "clone", UCX_REPO_URL, UCX_DIR])

        ucx_source_path = os.path.abspath(UCX_DIR)
        run_command(["git", "checkout", "v1.19.x"], cwd=ucx_source_path)
        run_command(["./autogen.sh"], cwd=ucx_source_path)

        configure_command = [
            "./configure",
            f"--prefix={ucx_install_path}",
            "--enable-shared",
            "--disable-static",
            "--disable-doxygen-doc",
            "--enable-optimizations",
            "--enable-cma",
            "--enable-devel-headers",
            "--with-verbs",
            "--enable-mt",
        ]
        run_command(configure_command, cwd=ucx_source_path)
        run_command(["make", "-j", str(os.cpu_count() or 1)],
                    cwd=ucx_source_path)
        run_command(["make", "install"], cwd=ucx_source_path)
        print("--- UCX build and install complete ---")

    # -- Step 2: Build and Install NIXL from source --
    if not args.force_reinstall and is_pip_package_installed("nixl"):
        print("\n--> NIXL is already installed. Skipping build. \
                Use --force-reinstall to override.")
    else:
        print("\n[2/2] Configuring and building NIXL from source...")
        if not os.path.exists(NIXL_DIR):
            run_command(["git", "clone", NIXL_REPO_URL, NIXL_DIR])

        build_env = os.environ.copy()
        pkg_config_path = os.path.join(ucx_install_path, "lib", "pkgconfig")
        build_env["PKG_CONFIG_PATH"] = pkg_config_path

        ucx_lib_path = os.path.join(ucx_install_path, "lib")
        existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        build_env[
            "LD_LIBRARY_PATH"] = f"{ucx_lib_path}:{existing_ld_path}".strip(
                ":")

        print(f"--> Using PKG_CONFIG_PATH: {build_env['PKG_CONFIG_PATH']}")
        print(f"--> Using LD_LIBRARY_PATH: {build_env['LD_LIBRARY_PATH']}")

        nixl_install_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            ".",
        ]
        if args.force_reinstall:
            nixl_install_command.insert(-1, "--force-reinstall")

        run_command(nixl_install_command,
                    cwd=os.path.abspath(NIXL_DIR),
                    env=build_env)
        print("--- NIXL installation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and install UCX and NIXL dependencies.")
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force rebuild and reinstall of UCX and NIXL \
            even if they are already installed.",
    )
    args = parser.parse_args()
    build_and_install_prerequisites(args)
