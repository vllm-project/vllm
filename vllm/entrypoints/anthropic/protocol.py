# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.entrypoints/anthropic/protocol -> vllm.frontend.entrypoints.anthropic.protocol (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.entrypoints.anthropic.protocol")
sys.modules[__name__] = _real
