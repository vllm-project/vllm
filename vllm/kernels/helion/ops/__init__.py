# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.kernels/helion/ops/ -> vllm.backends.compute.kernels.helion.ops (lazy __getattr__ delegation)."""
import importlib as _importlib

_real = _importlib.import_module("vllm.backends.compute.kernels.helion.ops")

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

__all__ = getattr(_real, "__all__", [])
