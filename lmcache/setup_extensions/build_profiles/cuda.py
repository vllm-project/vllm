# SPDX-License-Identifier: Apache-2.0
"""CUDA GPU backend profile.

Builds the ``lmcache.c_ops`` extension containing memory kernels, lookup
kernels, Cascade-AC encode/decode, position kernels, and event recorders.
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

ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"


class CudaProfile(BuildProfile):
    """CUDA GPU extension build profile."""

    name = "cuda"
    env_var = "BUILD_WITH_CUDA"

    def detect(self) -> bool:
        """Detect CUDA by locating the ``nvcc`` compiler in PATH.

        Build-time detection deliberately avoids ``torch.cuda.is_available``
        because that probes the runtime driver, which is typically absent
        on headless CI build hosts that nevertheless ship a full CUDA
        toolchain.
        """
        return shutil.which("nvcc") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build CUDA extensions (kernels, allocator, recorders)."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building CUDA extensions")
        flag_cxx_abi = (
            "-D_GLIBCXX_USE_CXX11_ABI=1"
            if ENABLE_CXX11_ABI
            else "-D_GLIBCXX_USE_CXX11_ABI=0"
        )
        cuda_sources = [
            "csrc/pybind.cpp",
            "csrc/mem_kernels.cu",
            "csrc/mp_mem_kernels.cu",
            "csrc/blend_kernels.cu",
            "csrc/cal_cdf.cu",
            "csrc/ac_enc.cu",
            "csrc/ac_dec.cu",
            "csrc/pos_kernels.cu",
            "csrc/mem_alloc.cpp",
            "csrc/utils.cpp",
            "csrc/event_recorder.cpp",
            "csrc/completion_recorder.cpp",
        ]
        ext_modules = [
            cpp_extension.CUDAExtension(
                "lmcache.c_ops",
                sources=cuda_sources,
                extra_compile_args={
                    "cxx": [flag_cxx_abi, "-std=c++17"],
                    "nvcc": [flag_cxx_abi],
                },
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def extra_cxx_flags_for(self, spec) -> list[str]:
        """All common extensions share the same ABI flag under CUDA."""
        return self.default_cxx_flags()

    def default_cxx_flags(self) -> list[str]:
        """ABI-aware default flags for downstream consumers."""
        if ENABLE_CXX11_ABI:
            return ["-D_GLIBCXX_USE_CXX11_ABI=1"]
        return ["-D_GLIBCXX_USE_CXX11_ABI=0"]

    def requirements_file(self) -> Optional[str]:
        """Return the CUDA version-specific requirements file."""
        return "cuda%s_core.txt" % self._cuda_major()

    def extras_requirements(self) -> dict[str, str]:
        """Return the CUDA optional extras.

        Returns:
            Mapping with the ``"nixl"`` extra (``pip install lmcache[nixl]``).
            The extra depends on the ``nixl`` meta-package, which selects the
            CUDA backend at runtime, so it is identical for CUDA 12 and 13.
        """
        return {"nixl": "nixl.txt"}

    def _cuda_major(self) -> str:
        """Resolve the target CUDA major version from ``LMCACHE_CUDA_MAJOR``.

        Returns:
            ``"12"`` or ``"13"`` (default ``"13"``, matching the PyPI build).

        Raises:
            ValueError: If ``LMCACHE_CUDA_MAJOR`` is set to anything else.
        """
        cuda_major = os.environ.get("LMCACHE_CUDA_MAJOR", "13")
        if cuda_major not in ("12", "13"):
            raise ValueError(
                "LMCACHE_CUDA_MAJOR must be '12' or '13', got '%s'" % cuda_major
            )
        return cuda_major
