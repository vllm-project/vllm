# SPDX-License-Identifier: Apache-2.0
"""Extension build framework using strategy pattern.

Usage from setup.py::

    from setup_extensions import BuildPolicy
    policy = BuildPolicy()
    profile = policy.resolve_profile()
    ext_modules, cmdclass, req_file = policy.collect_extensions(profile)
"""

# First Party
from setup_extensions.build_profiles import BuildProfile  # noqa: F401
from setup_extensions.common_cpp import build_common_cpp  # noqa: F401
from setup_extensions.policy import BuildPolicy, discover_subclasses  # noqa: F401
from setup_extensions.storage_backend_profiles import (  # noqa: F401
    StorageBackendProfile,
)

# Re-export build-mode flags from BuildProfile for callers who prefer
# module-level access (e.g. setup.py).
BUILDING_SDIST = BuildProfile.is_building_sdist()
NO_NATIVE_EXT = BuildProfile.is_native_ext_disabled()
NO_GPU_EXT = BuildProfile.is_gpu_ext_disabled()
