# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend choice for FlashInfer's Muse Spark fused norm kernels."""

import functools
import importlib.util
import os
from collections.abc import Iterable

from vllm.logger import init_logger

logger = init_logger(__name__)

_CAKE_PACKAGE = "cake_muse_spark_norm_backend"
_CAKE_MANIFEST = "build.json"


def select_norm_backend(package_dirs: Iterable[str]) -> str:
    """`"cute"` when a Cake norm package is installed without its manifest.

    FlashInfer's `"auto"` backend picks Cake for some row counts and only fails
    at launch when the package lacks its `build.json` manifest.
    """
    for package_dir in package_dirs:
        cake_dir = os.path.join(package_dir, _CAKE_PACKAGE)
        if os.path.isdir(cake_dir) and not os.path.isfile(
            os.path.join(cake_dir, _CAKE_MANIFEST)
        ):
            return "cute"
    return "auto"


@functools.cache
def get_norm_backend() -> str:
    """The `backend` to pass to FlashInfer's fused norms, decided once."""
    spec = importlib.util.find_spec("flashinfer")
    package_dirs = () if spec is None else spec.submodule_search_locations or ()
    backend = select_norm_backend(package_dirs)
    if backend == "cute":
        logger.warning_once(
            "FlashInfer's Cake norm package has no %s manifest; "
            "using the CuTe backend for the fused norms.",
            _CAKE_MANIFEST,
        )
    return backend
