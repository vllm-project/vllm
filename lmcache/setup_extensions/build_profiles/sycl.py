# SPDX-License-Identifier: Apache-2.0
"""SYCL / Intel XPU GPU backend profile.

Builds ``lmcache.xpu_ops`` using the DPC++ compiler (icpx).
Requires Intel oneAPI environment to be sourced before building.
"""

# Standard
from typing import TYPE_CHECKING, Optional
import os
import shutil

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile


class SyclProfile(BuildProfile):
    """SYCL / Intel XPU GPU extension build profile."""

    name = "sycl"
    env_var = "BUILD_WITH_SYCL"

    def detect(self) -> bool:
        """Detect SYCL by checking for the icpx compiler."""
        return shutil.which("icpx") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build SYCL/XPU extensions via DPC++."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building SYCL/XPU extensions")
        if shutil.which("icpx") is None:
            raise RuntimeError("icpx not found. Please source oneAPI setvars.sh first")
        os.environ["CXX"] = "icpx"
        oneapi_root = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
        include_dirs = ["%s/include" % oneapi_root]
        library_dirs = ["%s/lib" % oneapi_root]

        sycl_sources = [
            "csrc/sycl/pybind_sycl.cpp",
            "csrc/sycl/mem_kernels_sycl.cpp",
            "csrc/sycl/cal_cdf_sycl.cpp",
            "csrc/sycl/pos_kernels_sycl.cpp",
            "csrc/sycl/ac_enc_sycl.cpp",
            "csrc/sycl/ac_dec_sycl.cpp",
        ]
        ext_modules = [
            cpp_extension.SyclExtension(
                "lmcache.xpu_ops",
                sources=sycl_sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                extra_compile_args={
                    "cxx": [
                        "-std=c++17",
                        "-D_GLIBCXX_USE_CXX11_ABI=1",
                        "-O3",
                        "-fsycl",
                        "-fno-sycl-id-queries-fit-in-int",
                        "-ffast-math",
                        "-funroll-loops",
                        "-Wno-deprecated-declarations",
                        "-Wno-nan-infinity-disabled",
                    ],
                },
                extra_link_args=["-fsycl"],
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def extra_cxx_flags_for(self, spec) -> list[str]:
        """SYCL uses CXX11 ABI for all common extensions except
        ``lmcache_fs``, which omits ABI flags to preserve pre-refactor
        behaviour."""
        if spec.name == "lmcache_fs":
            return []
        return ["-D_GLIBCXX_USE_CXX11_ABI=1"]

    def default_cxx_flags(self) -> list[str]:
        """SYCL downstream consumers use the CXX11 ABI."""
        return ["-D_GLIBCXX_USE_CXX11_ABI=1"]

    def requirements_file(self) -> Optional[str]:
        """SYCL core requirements file."""
        return "xpu_core.txt"
