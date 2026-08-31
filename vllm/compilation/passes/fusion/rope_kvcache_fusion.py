# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.compilation/passes/fusion/rope_kvcache_fusion -> vllm.backends.compiler.passes.fusion.rope_kvcache_fusion (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.compiler.passes.fusion.rope_kvcache_fusion")
sys.modules[__name__] = _real
