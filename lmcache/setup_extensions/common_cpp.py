# SPDX-License-Identifier: Apache-2.0
"""Common C++ extension builders shared by all backends.

These extensions (storage manager, Redis, filesystem)
are always compiled regardless of which backend is selected.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

    # First Party
    from setup_extensions.build_profiles import BuildProfile


@dataclass(frozen=True)
class CommonExtSpec:
    """Declarative spec for a common C++ extension."""

    name: str
    sources: list[str]
    include_dirs: list[str]


COMMON_EXTENSIONS: list[CommonExtSpec] = [
    CommonExtSpec(
        name="native_storage_ops",
        sources=[
            "csrc/storage_manager/bitmap.cpp",
            "csrc/storage_manager/fold.cpp",
            "csrc/storage_manager/periodic_event_notifier.cpp",
            "csrc/storage_manager/pybind.cpp",
            "csrc/storage_manager/ttl_lock.cpp",
            "csrc/storage_manager/utils.cpp",
        ],
        include_dirs=["csrc/storage_manager"],
    ),
    CommonExtSpec(
        name="lmcache_redis",
        sources=[
            "csrc/storage_backends/redis/pybind.cpp",
            "csrc/storage_backends/redis/connector.cpp",
        ],
        include_dirs=[
            "csrc/storage_backends",
            "csrc/storage_backends/redis",
        ],
    ),
    CommonExtSpec(
        name="lmcache_fs",
        sources=[
            "csrc/storage_backends/fs/pybind.cpp",
            "csrc/storage_backends/fs/connector.cpp",
        ],
        include_dirs=[
            "csrc/storage_backends",
            "csrc/storage_backends/fs",
        ],
    ),
]


def build_common_cpp(
    profile: "BuildProfile | None" = None,
) -> tuple[list["Extension"], dict]:
    """Build pure C++ extensions that do not depend on any backend.

    Args:
        profile: Resolved backend profile (or ``None``).  Each spec in
            :data:`COMMON_EXTENSIONS` queries ``profile.extra_cxx_flags_for``
            to obtain its per-extension extra flags.

    Returns:
        ``(ext_modules, cmdclass)`` tuple.
    """
    # Third Party
    from torch.utils import cpp_extension

    ext_modules = [
        cpp_extension.CppExtension(
            "lmcache." + spec.name,
            sources=spec.sources,
            include_dirs=spec.include_dirs,
            extra_compile_args={
                "cxx": (
                    (profile.extra_cxx_flags_for(spec) if profile else [])
                    + ["-O3", "-std=c++17"]
                ),
            },
        )
        for spec in COMMON_EXTENSIONS
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass
