# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake import canary for CUDA wheel selection."""

import importlib.metadata as metadata


def test_mooncake_engine_imports():
    """Fail here rather than as "Mooncake is not available" at engine startup.

    The PyPI mooncake-transfer-engine wheel is built against CUDA 12, so on the
    CUDA 13 image install-kv-connectors.sh replaces it with the cuda13 variant.
    A mismatch surfaces as an ImportError on the compiled extension.
    """
    for package_name in ("mooncake-transfer-engine", "mooncake-transfer-engine-cuda13"):
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package_name}: {version}")

    import mooncake.engine  # noqa: F401
