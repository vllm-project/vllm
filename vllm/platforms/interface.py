# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.platforms/interface -> vllm.backends.platform.interface (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.platform.interface")
sys.modules[__name__] = _real
