# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.tool_parsers/gigachat3_tool_parser -> vllm.frontend.processing.tool_parsers.gigachat3_tool_parser (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.processing.tool_parsers.gigachat3_tool_parser")
sys.modules[__name__] = _real
