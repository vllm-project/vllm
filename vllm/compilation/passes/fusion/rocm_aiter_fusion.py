# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.compilation/passes/fusion/rocm_aiter_fusion -> vllm.backends.compiler.passes.fusion.rocm_aiter_fusion (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module(
    "vllm.backends.compiler.passes.fusion.rocm_aiter_fusion"
)
sys.modules[__name__] = _real
