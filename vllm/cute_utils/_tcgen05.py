# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.cute_utils/_tcgen05 -> vllm.backends.compute.dsl.cute_utils._tcgen05 (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.backends.compute.dsl.cute_utils._tcgen05")
sys.modules[__name__] = _real
