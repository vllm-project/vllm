# SPDX-License-Identifier: Apache-2.0
"""Aerospike L2 storage backend profile.

Builds the ``lmcache.lmcache_aerospike`` extension against a libaerospike
development install.  Enabled via ``BUILD_WITH_AEROSPIKE=1`` (or the legacy
``BUILD_AEROSPIKE=1``), or auto-detected through ``AEROSPIKE_INCLUDE_DIR``.
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile

# Repo root: setup_extensions/storage_backend_profiles/aerospike.py -> parents[2]
ROOT_DIR = Path(__file__).resolve().parents[2]


class AerospikeStorageBackend(StorageBackendProfile):
    """Optional native Aerospike L2 storage backend."""

    name = "aerospike"
    env_var = "BUILD_WITH_AEROSPIKE"

    def detect(self) -> bool:
        """Detect Aerospike via the legacy ``BUILD_AEROSPIKE`` flag or by the
        presence of ``AEROSPIKE_INCLUDE_DIR``."""
        as_env = os.environ.get("BUILD_AEROSPIKE")
        if as_env is not None:
            return as_env == "1"
        return os.environ.get("AEROSPIKE_INCLUDE_DIR", "") != ""

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the Aerospike CppExtension."""
        # Standard
        import ctypes.util

        # Third Party
        from torch.utils import cpp_extension

        as_include = os.environ.get("AEROSPIKE_INCLUDE_DIR", "")
        as_lib = os.environ.get("AEROSPIKE_LIBRARY_DIR", "")
        deps_yaml_lib = (
            ROOT_DIR / ".deps" / "libyaml-install" / "usr" / "lib" / "x86_64-linux-gnu"
        )
        include_dirs = [
            "csrc/storage_backends",
            "csrc/storage_backends/aerospike",
        ]
        if as_include:
            include_dirs.extend(as_include.split(";"))
        library_dirs: list[str] = []
        if as_lib:
            library_dirs.extend(as_lib.split(";"))
        extra_objects: list[str] = []
        yaml_shared = deps_yaml_lib / "libyaml.so"
        yaml_static = deps_yaml_lib / "libyaml.a"
        if yaml_shared.exists() or yaml_static.exists():
            library_dirs.append(str(deps_yaml_lib))

        libraries = ["aerospike"]
        if yaml_shared.exists() or ctypes.util.find_library("yaml"):
            libraries.append("yaml")
        elif yaml_static.exists():
            extra_objects.append(str(yaml_static))
        libraries.extend(["ssl", "crypto", "pthread", "z", "rt"])
        if os.environ.get("AEROSPIKE_EVENT_LIB", "libuv") == "libuv":
            libraries.append("uv")

        runtime_library_dirs = list(library_dirs)

        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_aerospike",
                sources=[
                    "csrc/storage_backends/aerospike/pybind.cpp",
                    "csrc/storage_backends/aerospike/connector.cpp",
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                extra_objects=extra_objects,
                runtime_library_dirs=runtime_library_dirs,
                extra_compile_args={
                    "cxx": extra_cxx_flags + ["-O3", "-std=c++17"],
                },
                extra_link_args=["-Wl,--no-as-needed"],
            ),
        ]
