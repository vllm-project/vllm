# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.kernels/triton/ -> vllm.backends.compute.kernels.triton (lazy __getattr__ delegation)."""

import importlib as _importlib

_real = _importlib.import_module("vllm.backends.compute.kernels.triton")


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)


__all__ = getattr(_real, "__all__", [])
