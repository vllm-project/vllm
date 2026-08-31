# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.entrypoints/launchers/run_batch -> vllm.frontend.entrypoints.launchers.run_batch (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.entrypoints.launchers.run_batch")
sys.modules[__name__] = _real
