# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helion kernel compilation and benchmarking infrastructure.

This module automatically imports all Helion kernel implementations
to ensure they are registered with the benchmark framework.
All HelionCustomOp subclasses are automatically exported.
"""

import contextlib
import importlib
import inspect
from pathlib import Path

from vllm.compilation.helion.benchmark import KernelBenchmark
from vllm.compilation.helion.custom_op import HelionCustomOp
from vllm.compilation.helion.register import register_kernel

# Automatically import all kernel modules to trigger registration
# This allows new kernels to be added without modifying this file
_helion_dir = Path(__file__).parent
_kernel_files = [
    f.stem
    for f in _helion_dir.glob("*.py")
    if f.stem not in ("__init__", "benchmark", "custom_op")
]

# Import all kernel modules and collect HelionCustomOp subclasses
_helion_ops = {}
for module_name in _kernel_files:
    with contextlib.suppress(ImportError):
        module = importlib.import_module(f"vllm.compilation.helion.{module_name}")

        # Find all HelionCustomOp subclasses in this module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a HelionCustomOp subclass (but not HelionCustomOp itself)
            if (
                issubclass(obj, HelionCustomOp)
                and obj is not HelionCustomOp
                and obj.__module__ == module.__name__
            ):
                # Add to module namespace
                globals()[name] = obj
                _helion_ops[name] = obj

# Build __all__ dynamically
__all__ = [
    "HelionCustomOp",
    "KernelBenchmark",
    "register_kernel",
] + sorted(_helion_ops.keys())
