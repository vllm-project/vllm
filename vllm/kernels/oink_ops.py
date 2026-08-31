# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.kernels/oink_ops -> vllm.backends.compute.kernels.oink_ops (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.compute.kernels.oink_ops")
sys.modules[__name__] = _real
