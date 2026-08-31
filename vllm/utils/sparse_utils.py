# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.utils/sparse_utils -> vllm.foundation.utilities.sparse_utils (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.foundation.utilities.sparse_utils")
sys.modules[__name__] = _real
