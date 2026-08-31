# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.utils/tensor_schema -> vllm.foundation.utilities.tensor_schema (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.foundation.utilities.tensor_schema")
sys.modules[__name__] = _real
