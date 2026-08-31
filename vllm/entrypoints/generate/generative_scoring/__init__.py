# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.entrypoints/generate/generative_scoring/ -> vllm.frontend.entrypoints.generate.generative_scoring (lazy __getattr__ delegation)."""

import importlib as _importlib

_real = _importlib.import_module(
    "vllm.frontend.entrypoints.generate.generative_scoring"
)


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)


__all__ = getattr(_real, "__all__", [])
