# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.parser/kimi_k2 -> vllm.frontend.processing.parser.kimi_k2 (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.processing.parser.kimi_k2")
sys.modules[__name__] = _real
