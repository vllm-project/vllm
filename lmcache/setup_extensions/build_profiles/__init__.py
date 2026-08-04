# SPDX-License-Identifier: Apache-2.0
"""Base classes for the platform extension build pattern.

Each hardware platform (CUDA, ROCm, SYCL, MUSA, ...) implements
:class:`BuildProfile`.
The :class:`BuildPolicy` orchestrates auto-detection, fallback, and building.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

    # First Party
    from setup_extensions.common_cpp import CommonExtSpec


class BuildProfile(ABC):
    """Build profile for platform-specific extensions.

    Subclasses must define:
        name     – unique identifier string.
        env_var  – ``BUILD_WITH_*`` environment variable name for explicit
                   selection.

    Subclasses must implement:
        detect() – auto-detect if this profile's hardware/compiler is present.
        build()  – return ``(ext_modules, cmdclass)`` for extensions.

    Subclasses may override:
        extra_cxx_flags_for(spec) – per-extension extra C++ flags for the
                                    common extensions defined in
                                    ``COMMON_EXTENSIONS``.
        requirements_file()        – core requirements file name.
        extras_requirements()      – optional-extra requirements files.
    """

    name: str = ""
    env_var: str = ""

    # ------------------------------------------------------------------
    # Build-mode flags (owned by the profile, not the policy)
    # ------------------------------------------------------------------

    @classmethod
    def is_building_sdist(cls) -> bool:
        """Return True when building a source distribution."""
        # Standard
        import sys

        return "sdist" in sys.argv

    @classmethod
    def is_native_ext_disabled(cls) -> bool:
        """Return True when native extensions are disabled.

        Controlled by the ``NO_NATIVE_EXT`` environment variable.
        When ``True``, all native C++ extensions are skipped — including
        both common extensions (Redis, filesystem, storage manager)
        and GPU backend extensions.

        ``NO_CUDA_EXT`` is the legacy alias kept for backwards
        compatibility (widely used by CI / packaging); it has always
        controlled all native extensions, not just CUDA ones, and is
        therefore treated as equivalent to ``NO_NATIVE_EXT``.
        Use ``NO_GPU_EXT=1`` instead when only the GPU backend should
        be skipped.
        """
        # Standard
        import os
        import sys

        if os.environ.get("NO_CUDA_EXT", "0") == "1":
            print(
                "warning: NO_CUDA_EXT is deprecated; use NO_NATIVE_EXT=1 instead.",
                file=sys.stderr,
            )
        return (
            os.environ.get("NO_NATIVE_EXT", "0") == "1"
            or os.environ.get("NO_CUDA_EXT", "0") == "1"
        )

    @classmethod
    def is_gpu_ext_disabled(cls) -> bool:
        """Return True when GPU extensions are disabled.

        Controlled by the ``NO_GPU_EXT`` environment variable.
        When ``True``, GPU-specific extensions (CUDA kernels, ROCm
        hipified sources, SYCL kernels, etc.) are skipped even when
        a platform is detected or explicitly requested.  Common C++
        extensions are still built.
        """
        # Standard
        import os

        return os.environ.get("NO_GPU_EXT", "0") == "1"

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def is_explicitly_requested(self) -> bool:
        """Return True when this profile was selected via env var."""
        if not self.env_var:
            return False
        # Standard
        import os

        return os.environ.get(self.env_var, "0") == "1"

    @abstractmethod
    def detect(self) -> bool:
        """Auto-detect if this profile's toolchain / hardware is available."""
        ...

    @abstractmethod
    def build(self) -> tuple[list["Extension"], dict]:
        """Build profile-specific extension modules.

        Returns:
            ``(ext_modules, cmdclass)`` tuple.
        """
        ...

    def extra_cxx_flags_for(self, spec: "CommonExtSpec") -> list[str]:
        """Extra C++ compile flags for a given common extension.

        Called once per spec in :data:`COMMON_EXTENSIONS`.  Default
        returns an empty list; subclasses override to inject per-extension
        flags (e.g. ABI defines).
        """
        return []

    def default_cxx_flags(self) -> list[str]:
        """Default C++ flags to hand to downstream consumers (e.g. optional
        L2 storage backends) that need a representative set of flags
        compatible with this profile.

        Subclasses override when their default ABI differs from the empty
        baseline.
        """
        return []

    def requirements_file(self) -> Optional[str]:
        """Core requirements file name, relative to ``requirements/``.

        Return ``None`` when this profile has no extra deps.
        """
        return None

    def extras_requirements(self) -> dict[str, str]:
        """Optional-extra requirements files, relative to ``requirements/``.

        Returns:
            Mapping from pip extra name (e.g. ``"nixl"``, installed via
            ``pip install lmcache[nixl]``) to the requirements file that
            lists that extra's dependencies. Empty when this profile
            exposes no optional extras.
        """
        return {}
