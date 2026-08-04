# SPDX-License-Identifier: Apache-2.0
"""Base classes for the optional L2 storage backend build profile pattern.

Each optional storage backend (Mooncake, ...) implements
:class:`StorageBackendProfile`.  Unlike GPU backends, multiple storage
backends can be selected simultaneously.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension


class StorageBackendProfile(ABC):
    """Build profile for an optional L2 storage backend extension.

    Subclasses must define:
        name     – unique identifier string.
        env_var  – ``BUILD_WITH_*`` environment variable name for explicit
                   selection.

    Subclasses must implement:
        detect() – auto-detect if this backend's SDK / toolchain is present.
        build()  – return ``list[Extension]`` for the backend extension.
    """

    name: str = ""
    env_var: str = ""

    def is_explicitly_requested(self) -> bool:
        """Return True when this backend was selected via env var."""
        if not self.env_var:
            return False
        # Standard
        import os

        return os.environ.get(self.env_var, "0") == "1"

    @abstractmethod
    def detect(self) -> bool:
        """Auto-detect if this backend's SDK / toolchain is available."""
        ...

    @abstractmethod
    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the storage backend extension module.

        Args:
            extra_cxx_flags: Additional C++ compiler flags from the
                selected GPU backend profile.

        Returns:
            List of ``Extension`` objects for this backend.
        """
        ...
