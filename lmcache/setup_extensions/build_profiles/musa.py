# SPDX-License-Identifier: Apache-2.0
"""MUSA GPU backend profile (placeholder).

MUSA fused GPU extensions are not built yet; this profile is a stub
ready for future implementation.
"""

# Standard
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile


class MusaProfile(BuildProfile):
    """MUSA GPU extension build profile (stub)."""

    name = "musa"
    env_var = "BUILD_WITH_MUSA"

    def detect(self) -> bool:
        """MUSA detection is not implemented yet."""
        return False

    def build(self) -> tuple[list["Extension"], dict]:
        """MUSA extensions are not built yet."""
        print("MUSA GPU extensions are not built yet; skipping GPU extensions")
        return [], {}

    def requirements_file(self) -> Optional[str]:
        """MUSA has no extra deps yet."""
        return None
