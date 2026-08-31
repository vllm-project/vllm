# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.reasoning/deepseek_r1_reasoning_parser -> vllm.frontend.processing.reasoning.deepseek_r1_reasoning_parser (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.processing.reasoning.deepseek_r1_reasoning_parser")
sys.modules[__name__] = _real
