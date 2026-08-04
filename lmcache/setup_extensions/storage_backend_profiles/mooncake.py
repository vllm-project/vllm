# SPDX-License-Identifier: Apache-2.0
"""Mooncake L2 storage backend profile.

Builds the ``lmcache.lmcache_mooncake`` extension when the Mooncake SDK
is available.  Enabled via ``BUILD_WITH_MOONCAKE=1`` or auto-detected
through ``MOONCAKE_INCLUDE_DIR`` / ``BUILD_MOONCAKE`` env vars.
"""

# Standard
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile


class MooncakeStorageBackend(StorageBackendProfile):
    """Mooncake distributed KV-cache storage backend."""

    name = "mooncake"
    env_var = "BUILD_WITH_MOONCAKE"

    def detect(self) -> bool:
        """Detect Mooncake SDK via legacy env vars."""
        mc_env = os.environ.get("BUILD_MOONCAKE")
        if mc_env is not None:
            return mc_env == "1"
        return os.environ.get("MOONCAKE_INCLUDE_DIR", "") != ""

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the mooncake CppExtension."""
        # Third Party
        from torch.utils import cpp_extension

        mc_include = os.environ.get("MOONCAKE_INCLUDE_DIR", "")
        mc_lib = os.environ.get("MOONCAKE_LIB_DIR", "")
        mc_include_dirs = [
            "csrc/storage_backends",
            "csrc/storage_backends/mooncake",
        ]
        if mc_include:
            mc_include_dirs.extend(mc_include.split(";"))
        mc_library_dirs: list[str] = []
        if mc_lib:
            mc_library_dirs.extend(mc_lib.split(";"))
        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_mooncake",
                sources=[
                    "csrc/storage_backends/mooncake/pybind.cpp",
                    "csrc/storage_backends/mooncake/connector.cpp",
                ],
                include_dirs=mc_include_dirs,
                library_dirs=mc_library_dirs,
                libraries=["mooncake_store"],
                runtime_library_dirs=mc_library_dirs,
                extra_compile_args={
                    "cxx": extra_cxx_flags + ["-O3", "-std=c++20", "-DYLT_ENABLE_IBV"],
                },
            ),
        ]
