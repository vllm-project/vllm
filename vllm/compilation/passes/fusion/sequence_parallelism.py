# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.compilation/passes/fusion/sequence_parallelism -> vllm.backends.compiler.passes.fusion.sequence_parallelism (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.compiler.passes.fusion.sequence_parallelism")
sys.modules[__name__] = _real
