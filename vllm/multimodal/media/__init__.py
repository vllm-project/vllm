# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.multimodal/media/ -> vllm.frontend.processing.multimodal.media (lazy __getattr__ delegation)."""
import importlib as _importlib

_real = _importlib.import_module("vllm.frontend.processing.multimodal.media")

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

__all__ = getattr(_real, "__all__", [])
