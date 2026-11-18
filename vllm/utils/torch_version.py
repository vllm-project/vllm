# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch version checks.

Kept free of other vLLM imports so that early importers such as
`vllm.env_override` don't pull in `vllm.utils.torch_utils`, which resolves
`current_platform` at import time.
"""

import importlib.metadata

import torch
from packaging import version
from packaging.version import Version


# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    return version.parse(torch_version) >= version.parse(target)


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version
    (a string like "2.6.0").
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)


def _is_torch_equal(target: str) -> bool:
    assert target.count(".") == 2
    torch_version = str(torch.__version__)
    torch_version = version.parse(torch_version)
    # torch version is like "2.6.0.dev20240101" or "2.6.0.dev20240101+cpu"
    # or "2.6.0+cu128" but never "2.6.0.1"
    return (
        torch_version >= version.parse(target)
        and version.parse(target + ".1") > torch_version
    )


def is_torch_equal(target: str) -> bool:
    """Check if the installed torch version is == the target version
    (a string like "2.6.0").
    """
    try:
        return _is_torch_equal(target)
    except Exception:
        return Version(importlib.metadata.version("torch")) == Version(target)
