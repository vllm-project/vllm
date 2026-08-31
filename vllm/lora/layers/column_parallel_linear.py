# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.lora/layers/column_parallel_linear -> vllm.runtime.modeling.lora.layers.column_parallel_linear (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module(
    "vllm.runtime.modeling.lora.layers.column_parallel_linear"
)
sys.modules[__name__] = _real
