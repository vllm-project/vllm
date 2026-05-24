# SPDX-License-Identifier: Apache-2.0
"""Genesis vLLM patches — modular package.

This is the v7.0 architecture replacing monolithic patch_genesis_unified.py.
All patches migrate to modular kernel-level professional replacements под vendor-safe
defensive guards.

Mission: МЫ ЧИНИМ, НЕ ЛОМАЕМ (we fix, we don't break).
Each kernel works on NVIDIA CUDA / AMD ROCm / Intel XPU / CPU with graceful skip.

Sub-packages:
  guards    — canonical vendor/chip/model detection helpers
  prealloc  — safe pre-allocation framework (graph-safe, profiler-visible)
  kernels   — professional drop-in replacements for upstream-weak code paths
  patches   — thin monkey-patch bridges to upstream (legacy overlay)
  tests     — TDD-first pytest suite

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Project: github.com/Sandermage/genesis-vllm-patches
Version: see vllm/_genesis/__version__.py — single source of truth
"""

# Single source of truth lives in __version__.py — re-exported here.
from vllm._genesis.__version__ import __version__, VERSION  # noqa: F401
__author__ = "Sandermage(Sander)-Barzov Aleksandr"
__location__ = "Ukraine, Odessa"
__project__ = "https://github.com/Sandermage/genesis-vllm-patches"

# Public API — what gets imported from vllm._genesis.
#
# G-002 fix (audit 2026-05-02): `prealloc` is NOT eagerly imported here
# because it imports `torch` at module level. Eager import broke every
# CLI / pre-commit / static-analysis tool that doesn't have torch
# installed (schema_validator, dispatcher --json, lifecycle_audit_cli,
# self-test, pytest collection — all failed on `ModuleNotFoundError:
# torch` before they could even start). `guards` is fine — kept eager
# (light-weight, no torch on import path beyond optional probes).
#
# Consumers needing prealloc should import it explicitly:
#     from vllm._genesis import prealloc
# This makes the torch dependency explicit at the call site and lets
# torch-less tools import vllm._genesis cleanly.
from vllm._genesis import guards


def __getattr__(name):
    """Lazy import for prealloc — avoids torch dependency at package load.

    Triggered when a consumer does `from vllm._genesis import prealloc`
    or `vllm._genesis.prealloc`. Module is imported only on first access.

    Uses `importlib.import_module` (NOT `from vllm._genesis import ...`)
    to avoid re-triggering this very __getattr__ in an infinite loop —
    the from-import form first calls __getattr__ on the parent package.
    """
    if name == "prealloc":
        import importlib
        return importlib.import_module("vllm._genesis.prealloc")
    raise AttributeError(
        f"module 'vllm._genesis' has no attribute {name!r}"
    )


__all__ = [
    "guards",
    "prealloc",  # accessible via lazy __getattr__ — see above
    "__version__",
    "__author__",
]
