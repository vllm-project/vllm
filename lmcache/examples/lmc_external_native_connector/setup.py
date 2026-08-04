# SPDX-License-Identifier: Apache-2.0
"""
Build script for the example native connector plugin.

Compiles the C++ pybind11 extension that implements
ExampleFSConnector and ExampleMemoryConnector.  The
LMCache connector_base.h headers are resolved from the
LMCache source tree, assuming this example is located
within the repository.

NOTE: The C++ extension requires Linux (eventfd).  On
other platforms the extension is skipped and only the
Python wrapper module is installed.
"""

# Standard
from pathlib import Path
import os
import platform

# Third Party
from setuptools import Extension, find_packages, setup
import pybind11

ROOT_DIR = Path(__file__).resolve().parent

# Use relative paths -- setuptools forbids absolute ones.
CSRC_REL = os.path.join("csrc")

# Resolve LMCache csrc headers (connector_base.h, etc.)
# Walk up from this example to the repo root.
LMCACHE_ROOT = ROOT_DIR.parent.parent
LMCACHE_CSRC = str(LMCACHE_ROOT / "csrc" / "storage_backends")

ext_modules = []
if platform.system() == "Linux":
    ext_modules = [
        Extension(
            "lmc_external_native_connector._native",
            sources=[
                os.path.join(CSRC_REL, "pybind.cpp"),
                os.path.join(CSRC_REL, "connector.cpp"),
            ],
            include_dirs=[
                str(ROOT_DIR / "csrc"),
                LMCACHE_CSRC,
                pybind11.get_include(),
            ],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        ),
    ]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
)
