# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.tracing/utils -> vllm.foundation.observability.tracing.utils (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.foundation.observability.tracing.utils")
sys.modules[__name__] = _real
