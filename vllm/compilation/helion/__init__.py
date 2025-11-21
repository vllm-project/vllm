# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helion kernel compilation and benchmarking infrastructure.

This module automatically imports all Helion kernel implementations
to ensure they are registered with the benchmark framework.
"""

import contextlib
import importlib
from pathlib import Path

from vllm.compilation.helion.benchmark import KernelBenchmark
from vllm.compilation.helion.custom_op import HelionCustomOp

# Automatically import all kernel modules to trigger registration
# This allows new kernels to be added without modifying this file
_helion_dir = Path(__file__).parent
_kernel_files = [
    f.stem
    for f in _helion_dir.glob("*.py")
    if f.stem not in ("__init__", "benchmark", "custom_op")
]

# Import all kernel modules (this triggers benchmark registration)
for module_name in _kernel_files:
    with contextlib.suppress(ImportError):
        importlib.import_module(f"vllm.compilation.helion.{module_name}")

__all__ = [
    "HelionCustomOp",
    "KernelBenchmark",
]
