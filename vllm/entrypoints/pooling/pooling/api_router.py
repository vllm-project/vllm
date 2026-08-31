# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.entrypoints/pooling/pooling/api_router -> vllm.frontend.entrypoints.pooling.pooling.api_router (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.entrypoints.pooling.pooling.api_router")
sys.modules[__name__] = _real
