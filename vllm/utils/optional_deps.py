# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-fast validation for platform optional dependencies.

If a platform advertises a feature (e.g. mxfp4 on ROCm), the corresponding
package must be importable or we raise at startup with a clear message.

WIP: Draft for team review. Only ROCm + amd-quark (mxfp4/mxfp6) for now;
other platforms can be added in the same way.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import Any

from packaging.version import Version

# Platform enum -> list of (feature_name, pip_package, import_name, min_version?).
# If supported_quantization includes the feature, we require the package.
_PLATFORM_OPTIONAL_DEPS: dict[str, list[tuple[str, str, str, str | None]]] = {
    "ROCM": [
        # mxfp4/mxfp6 use quark.torch.kernel
        ("mxfp4", "amd-quark", "quark", "0.8.99"),
    ],
}


def _check_package(
    feature_name: str,
    pip_package: str,
    import_name: str,
    min_version: str | None,
) -> None:
    """Import package and optionally check version. Raises ImportError if missing."""
    try:
        importlib.import_module(import_name)
    except ImportError as err:
        raise ImportError(
            f"The package `{pip_package}` is required for {feature_name} on this "
            f"platform. Install with: pip install {pip_package}"
        ) from err

    if min_version is not None:
        try:
            installed = version(pip_package)
            if Version(installed) < Version(min_version):
                raise ImportError(
                    f"`{pip_package}>={min_version}` required for {feature_name}; "
                    f"found {installed}. Upgrade: pip install -U {pip_package}"
                )
        except Exception as e:
            if isinstance(e, ImportError):
                raise
            # version() can fail for non-PyPI installs; allow through
            pass


def validate_platform_optional_deps(platform_enum: Any) -> None:
    """Validate optional deps for current platform.

    Call when platform is first initialized (e.g. from
    RocmPlatform.import_kernels()). Raises ImportError if dep missing.

    Args:
        platform_enum: PlatformEnum value (e.g. ROCM). Only validates that
            platform; others are no-ops.
    """
    name = getattr(platform_enum, "name", None) or str(platform_enum)
    deps = _PLATFORM_OPTIONAL_DEPS.get(name)
    if not deps:
        return

    for feature_name, pip_package, import_name, min_version in deps:
        _check_package(feature_name, pip_package, import_name, min_version)
