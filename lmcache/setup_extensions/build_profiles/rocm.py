# SPDX-License-Identifier: Apache-2.0
"""ROCm/HIP GPU backend profile.

Hipifies CUDA sources via ``torch.utils.hipify``, then builds
``lmcache.c_ops`` with hipcc as the C++ compiler.
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import os

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile

ROOT_DIR = Path(__file__).parent.parent.parent
HIPIFY_DIR = os.path.join(ROOT_DIR, "csrc/")
HIPIFY_OUT_DIR = os.path.join(ROOT_DIR, "csrc_hip/")


def _hipify_wrapper() -> list[str]:
    """Run torch hipify on csrc/ and return hipified source paths."""
    # Third Party
    from torch.utils.hipify.hipify_python import hipify

    print("Hipifying sources")
    extra_files = [
        os.path.abspath(os.path.join(HIPIFY_DIR, item))
        for item in os.listdir(HIPIFY_DIR)
        if os.path.isfile(os.path.join(HIPIFY_DIR, item))
    ]
    hipify_result = hipify(
        project_directory=HIPIFY_DIR,
        output_directory=HIPIFY_OUT_DIR,
        header_include_dirs=[],
        includes=[],
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    hipified_sources: list[str] = []
    for source in extra_files:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (
                s_abs in hipify_result
                and hipify_result[s_abs].hipified_path is not None
            )
            else s_abs
        )
        hipified_sources.append(hipified_s_abs)

    if len(hipified_sources) != len(extra_files):
        raise RuntimeError(
            "Hipify failed: expected %d sources, got %d"
            % (len(extra_files), len(hipified_sources))
        )
    return hipified_sources


class RocmProfile(BuildProfile):
    """ROCm/HIP GPU extension build profile."""

    name = "rocm"
    env_var = "BUILD_WITH_HIP"

    def detect(self) -> bool:
        """Detect ROCm by checking for hipcc."""
        # Standard
        import shutil

        return shutil.which("hipcc") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build ROCm/HIP extensions via hipcc."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building ROCM extensions")
        _hipify_wrapper()
        hip_sources = [
            "csrc/pybind_hip.cpp",
            "csrc/mem_kernels.hip",
            "csrc/mp_mem_kernels.hip",
            "csrc/blend_kernels.hip",
            "csrc/cal_cdf.hip",
            "csrc/ac_enc.hip",
            "csrc/ac_dec.hip",
            "csrc/pos_kernels.hip",
            "csrc/mem_alloc_hip.cpp",
            "csrc/utils_hip.cpp",
            "csrc/event_recorder.cpp",
            "csrc/completion_recorder.cpp",
        ]
        define_macros = [("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1")]
        ext_modules = [
            cpp_extension.CppExtension(
                "lmcache.c_ops",
                sources=hip_sources,
                extra_compile_args={
                    "cxx": [
                        "-O3",
                        "-std=c++17",
                    ],
                },
                include_dirs=[
                    os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include")
                ],
                library_dirs=[
                    os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
                ],
                define_macros=define_macros,
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def requirements_file(self) -> Optional[str]:
        """ROCm core requirements file."""
        return "rocm_core.txt"
